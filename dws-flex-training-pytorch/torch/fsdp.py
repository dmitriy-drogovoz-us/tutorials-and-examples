# Copyright 2024 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is optimized for training large models on multi-GPU setups (like B200s)
and includes performance logging.
"""

# --- Core Libraries ---
import os
import time
import functools
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dc
import re

# --- PyTorch FSDP ---
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
import itertools
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    lambda_auto_wrap_policy,
    _or_policy
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.fsdp.api import StateDictType
# --- Data & Training ---
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from tqdm import tqdm

# --- Hugging Face Libraries ---
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoConfig
from torch.distributed.device_mesh import init_device_mesh

from datasets import load_dataset, load_from_disk

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True

# Helper function to handle boolean environment variables
def get_bool_env(var_name, default=False):
    return os.environ.get(var_name, str(default)).lower() in ('true', '1', 't')

# --- Model ---
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-70B")
model_id_lower = MODEL_ID.lower()

if 'llama' in model_id_lower:
    decoder_layer = LlamaDecoderLayer
elif 'gemma' in model_id_lower:
    decoder_layer = Gemma3DecoderLayer
else:
    # Raise an error if the model family is not supported
    raise ValueError(
        f"Unsupported model family. Only 'llama' or 'gemma3' are supported, but got: {MODEL_ID}"
    )


# --- Dataset ---
DATASET_ID = os.environ.get("DATASET_ID", "philschmid/gretel-synthetic-text-to-sql")
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", 20))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/home/esaarenvirta/gemma3-4b-sql-lora")
PROCESSED_DATASET_DIR = os.environ.get("PROCESSED_DATASET_DIR", "/home/esaarenvirta/gretel-sql-processed")
DATASET_SEED = int(os.environ.get("DATASET_SEED", 42))
SYSTEM_INSTRUCT = get_bool_env("SYSTEM_INSTRUCT", False)
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.1))
TRAIN_WITH_SAMPLE = get_bool_env("TRAIN_WITH_SAMPLE", True)
TRAIN_SAMPLE_SIZE = int(os.environ.get("TRAIN_SAMPLE_SIZE", 12500))
SHARED_STORAGE = get_bool_env("SHARED_STORAGE", False)

# --- Training Hyperparameters ---
NUM_TRAIN_EPOCHS = int(os.environ.get("NUM_TRAIN_EPOCHS", 12))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 8))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 8))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 0.3))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 512))

# --- LORA Hyperparameters ---
PEFT = get_bool_env("PEFT", False)
PEFT_R = int(os.environ.get("PEFT_R", 16))
PEFT_ALPHA = int(os.environ.get("PEFT_ALPHA", 16))
PEFT_DROPOUT = float(os.environ.get("PEFT_DROPOUT", 0.05))

# --- Misc Parameters ---
SYSTEM_INSTRUCT = get_bool_env("SYSTEM_INSTRUCT", False)
TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.1))
CHECKPOINT_EPOCHS = int(os.environ.get("CHECKPOINT_EPOCHS", 12))


# ==============================================================================
# 2. HELPERS & UTILITIES
# ==============================================================================

def find_most_recent_checkpoint(output_dir):
    """
    Scans the output directory for epoch checkpoints and returns the path to the
    most recent one and its epoch number.
    """
    if not os.path.isdir(output_dir):
        return None, 0

    epoch_dirs = [d for d in os.listdir(output_dir) if d.startswith('epoch_')]
    if not epoch_dirs:
        return None, 0

    max_epoch = 0
    for dir_name in epoch_dirs:
        match = re.search(r'epoch_(\d+)', dir_name)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num

    if max_epoch > 0:
        latest_checkpoint_path = os.path.join(output_dir, f"epoch_{max_epoch}")
        return latest_checkpoint_path, max_epoch

    return None, 0

def save_checkpoint(model, optimizer, scheduler, tokenizer, epoch, global_rank, local_rank, output_dir, peft_config=None):
    """Saves a sharded FSDP checkpoint, coordinated across all ranks."""

    # Create the checkpoint directory for the current epoch
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    if global_rank == 0:
        os.makedirs(epoch_dir, exist_ok=True)
    dist.barrier(device_ids=[local_rank]) # Ensure directory is created before any rank writes to it

    # Each rank saves its own checkpoint shard
    dc.save(
        state_dict={
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, 
    checkpoint_id=epoch_dir)

    # Rank 0 saves the tokenizer and PEFT config
    if global_rank == 0:
        tokenizer.save_pretrained(epoch_dir)
        if peft_config:
            peft_config.save_pretrained(epoch_dir)
        print(f"✅ Checkpoint for epoch {epoch + 1} saved to {epoch_dir}")

    dist.barrier(device_ids=[local_rank]) # Wait for all ranks to finish saving

def setup_distributed():
    """Initializes the distributed training environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class PerfLogger:
    """
    A simple logger that calculates and logs performance metrics per optimizer step.
    """
    def __init__(self, total_model_params, world_size, global_rank):
        self.total_model_params = total_model_params
        self.world_size = world_size
        self.global_rank = global_rank
        self.step_start_time = None
        
        # Tokens processed in one full optimizer step across all GPUs
        self.tokens_per_gradient_step = (
            PER_DEVICE_TRAIN_BATCH_SIZE * self.world_size * GRADIENT_ACCUMULATION_STEPS * MAX_SEQ_LENGTH
        )
        
        if self.global_rank == 0:
            print(f"[PerfLogger] Using {total_model_params / 1e9:.2f}B parameters for TFLOPs calculation.")

    def start_step(self):
        """Records the start time of an optimizer step."""
        self.step_start_time = time.time()

    def log_step_performance(self, loss_val, global_step_num):
        """Calculates and logs performance for the completed step."""
        if self.step_start_time is None or self.global_rank != 0:
            return

        step_time = time.time() - self.step_start_time
        
        # Calculate Tokens/s/device
        total_tokens_per_sec = self.tokens_per_gradient_step / step_time if step_time > 0 else 0
        tokens_per_sec_per_device = total_tokens_per_sec / self.world_size
        
        # Calculate TFLOPs/s/device
        # Formula: 6 * num_params * num_tokens for a forward/backward pass
        total_theoretical_flops = (6 * self.total_model_params) * self.tokens_per_gradient_step
        achieved_flops_total = total_theoretical_flops / step_time if step_time > 0 else 0
        achieved_tflops_per_device = (achieved_flops_total / self.world_size) / 1e12
        
        # Print the formatted log line
        print(
            f"Step: {global_step_num:<5} | "
            f"Time: {step_time:>5.2f}s | "
            f"TFLOPs/s/GPU: {achieved_tflops_per_device:>6.1f} | "
            f"Tokens/s/GPU: {tokens_per_sec_per_device:>6.0f} | "
            f"Loss: {loss_val:.4f}"
        )


def format_prompt(sample, tokenizer):
    """
    Creates a full conversation string using the tokenizer's chat template.
    """
    system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
    user_prompt = f"""Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command.\n<SCHEMA>\n{sample['sql_context']}\n</SCHEMA>\n<USER_QUERY>\n{sample['sql_prompt']}\n</USER_QUERY>"""

    messages = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": sample["sql"]}]
    if SYSTEM_INSTRUCT:
        messages.insert(0, {"role": "system", "content": system_message})

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token}


def get_data_loaders(dataset_id, tokenizer, local_rank, global_rank, world_size):
    """
    Loads and processes data on rank 0, saves it, and then loads on all ranks.
    """
    if global_rank == 0:
        if not os.path.exists(PROCESSED_DATASET_DIR):
                print(f"Dataset not found at {PROCESSED_DATASET_DIR}. Processing from scratch...")
                raw_dataset = load_dataset(dataset_id, split="train").shuffle(seed=DATASET_SEED).select(range(TRAIN_SAMPLE_SIZE))
                formatted_dataset = raw_dataset.map(lambda sample: format_prompt(sample, tokenizer), num_proc=os.cpu_count())
                filtered_dataset = formatted_dataset.filter(
                    lambda x: 0 < len(x['text']) and len(tokenizer(x["text"]).input_ids) <= MAX_SEQ_LENGTH,
                    num_proc=os.cpu_count()
                )
                dataset = filtered_dataset.train_test_split(test_size=TEST_SIZE, seed=DATASET_SEED)
                print(f"💾 Saving processed dataset to {PROCESSED_DATASET_DIR}")
                dataset.save_to_disk(PROCESSED_DATASET_DIR)

    dist.barrier(device_ids=[local_rank])
    
    if global_rank == 0:
        print(f"💿 Loading processed dataset from {PROCESSED_DATASET_DIR}")
    dataset = load_from_disk(PROCESSED_DATASET_DIR)
    train_dataset = dataset['train']

    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        inputs = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['labels'][inputs['input_ids'] == tokenizer.pad_token_id] = -100
        return inputs

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=DATALOADER_NUM_WORKERS, pin_memory=True, drop_last=True
    )

    if global_rank == 0:
        print("✅ Data loaders are ready.")
        
    return train_loader

# ==============================================================================
# 3. MAIN TRAINING WORKFLOW
# ==============================================================================

def main():
    # --- Step 1: Initialize Distributed Environment ---
    setup_distributed()
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # --- Step 2: Load Model and Tokenizer ---
    if global_rank == 0: print("🚀 Preparing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Step 3: Prepare DataLoaders ---
    if global_rank == 0: print("🚀 Preparing data loaders...")
    train_loader = get_data_loaders(DATASET_ID, tokenizer, local_rank, global_rank, world_size)

    # --- Step 4: Prepare Model ---
    if global_rank == 0: print("🚀 Preparing model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    if PEFT:
        if global_rank == 0: print("🚀 PEFT Enabled, preparing PEFT model...")
        peft_config = LoraConfig(
            r=PEFT_R, lora_alpha=PEFT_ALPHA, lora_dropout=PEFT_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
    model = model.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    
    # --- Step 5: Shard Model with FSDP ---
    if global_rank == 0: print("🚀 Sharding model with FSDPv2...")
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    
    for module in model.modules():
        if isinstance(module, decoder_layer):
            fully_shard(module, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)
    
    dist.barrier(device_ids=[local_rank])

    # --- Step 6: Setup Optimizer, Scheduler, and Compilation ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = (len(train_loader) * NUM_TRAIN_EPOCHS) // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * WARMUP_RATIO), num_training_steps=num_training_steps
    )

     # -- Step 7: Model compilation ---
    if global_rank == 0: print("🚀 Compiling model...")
    model = torch.compile(model, dynamic=True)
    if global_rank == 0: print("✅ Done compiling model")

    # --- Step 8: Checkpoint auto resuming ---
    starting_epoch = 0
    resume_from_checkpoint, last_epoch = find_most_recent_checkpoint(OUTPUT_DIR)
    if resume_from_checkpoint:
        if global_rank == 0:
            print(f"✅ Resuming training from checkpoint: {resume_from_checkpoint}")
        dist.barrier()

        # Load the state on all ranks
        checkpoint_shard_path = os.path.join(OUTPUT_DIR, f"epoch_{last_epoch}")
        if global_rank == 0: print(f"Loading checkpoint shard at: {checkpoint_shard_path}")

        # Set the state dict type for loading sharded states
        state_to_load = {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler.state_dict(), 
                'epoch': -1,
            }
        
        dc.load(
                state_dict=state_to_load,
                checkpoint_id=checkpoint_shard_path # Directory to load from
            )
        scheduler.load_state_dict(state_to_load['scheduler'])
        loaded_epoch = state_to_load['epoch'] 
        starting_epoch = loaded_epoch + 1

        if global_rank == 0:
            print(f"✅ Resumed from epoch {loaded_epoch}. Starting next epoch at {starting_epoch}.")
    else:
        if global_rank == 0:
            print("✅ Starting training from scratch.")

    
    # --- START: Model Warmup ---
    if global_rank == 0: print("\n🔥 Starting model warmup to finalize compilation...")
    warmup_steps = 50
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(itertools.islice(train_loader, warmup_steps)):
            if global_rank == 0: print(f"  Warmup step {i + 1}/{warmup_steps}")
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            model(**batch)
    if global_rank == 0: print("✅ Model warmup complete.\n")
    dist.barrier(device_ids=[local_rank])
    # --- END: Model Warmup ---
    
    # --- Step 9: Start Training Loop ---
    if global_rank == 0: 
        print(f"🚀 Starting FSDP training for {NUM_TRAIN_EPOCHS} epoch(s)...")
        print("-" * 80)

    model.train()
    perf_logger = PerfLogger(total_params, world_size, global_rank)
    
    total_dataloader_steps = 0
    global_optimizer_step = 0
    for epoch in range(starting_epoch, NUM_TRAIN_EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        
        if global_rank == 0:
            print(f"\n--- Starting Epoch {epoch + 1}/{NUM_TRAIN_EPOCHS} ---")
            
        for step, batch in enumerate(train_loader):
            # Start timer at the beginning of an accumulation cycle
            if total_dataloader_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                perf_logger.start_step()

            batch = {k: v.to(local_rank) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_dataloader_steps += 1
            
            # Optimizer step at the end of an accumulation cycle
            if total_dataloader_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                global_optimizer_step += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Rescale loss to get the average loss for the full batch and log
                scaled_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                perf_logger.log_step_performance(scaled_loss, global_optimizer_step)

        if (epoch + 1) % CHECKPOINT_EPOCHS == 0:
            save_checkpoint(model=model, 
                            optimizer=optimizer, 
                            scheduler=scheduler, 
                            tokenizer=tokenizer, 
                            epoch=epoch,
                            global_rank=global_rank,
                            local_rank=local_rank,
                            output_dir=OUTPUT_DIR,
                            peft_config=model.peft_config.get("default") if PEFT else None)


    # --- Step 10: Final Cleanup ---
    dist.barrier(device_ids=[local_rank])

    if global_rank == 0:
        print("\n🎉 Training complete!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()