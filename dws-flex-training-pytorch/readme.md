# Sample LLM training on GCP with B200s on DWS Flex or Spot 

Thie repo is meant to be a basic instruction on how to set up a foundation for distributed ML training on either H200 or B200 accelerators on GCP on Kubernetes. In this repo we setup: 
- High performance networking using Google's RoCE (RDMA over converged ethernet) implementation
- A Google Kubernetes Engine (GKE) cluster that supports Dynamic Workload Scheduling (on demand GPU provisioning) and Spot for H200 and B200
- Kueue for orchestrating the job submission and hooking into DWS to get H200s/B200 GPUs on demand 
- A PyTorch job using FSDP2 as a sample training job 

It also provides some sample utilities for running NCCL tests and doing GPU health scanning.

This repo can easily be adapted to using a reservation (we omit here because most users testing won't have access to a reservation) by changing some of the setup commands to something from here: https://cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom#create-with-rdma under the reservation bound path. The performance with DWS should be similar because DWS will compact place the GPUs by default. Spot will have lower performance

# Setup 

Clone repo 
```
git clone https://github.com/esaaren/torch-distributed-training-gke.git && cd torch-distributed-training-gke 
```

Set root 
```
export REPO_ROOT=`git rev-parse --show-toplevel`
```

Source environment and then navigate to setup 
```
source $REPO_ROOT/.env 
cd $REPO_ROOT/setup && ./setup.sh 
```

Once cluster is setup you can build the image from the torch dir 
```
cd $REPO_ROOT/torch 
gcloud artifacts repositories create $REPOSITORY --repository-format=docker --location=${REGION} --project=${PROJECT}
gcloud builds submit . \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --config=cloudbuild.yaml \
    --substitutions="_ARTIFACT_REGISTRY=${ARTIFACT_REGISTRY},_IMAGE_NAME=${IMAGE_NAME}" \
    --timeout="2h" \
    --machine-type="e2-highcpu-32"
```

Quick NCCL test to validate RDMA/networking deployment on the spot nodepool. Wait a couple minutes for the spot nodes to come online 

```
cd $REPO_ROOT/utils 
kubectl apply -f nccl_test.yaml

kubectl exec nccl-test-host-1 -it -- /bin/bash -c " /usr/local/gib/scripts/run_nccl_tests.sh -t all_gather -b 1K -e 8G nccl-host-1 nccl-host-2"
```

The output should look like:
```
NCCL version 2.26.6+cuda12.8
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
        1024            16     float    none      -1    43.34    0.02    0.02      0    42.48    0.02    0.02      0
        2048            32     float    none      -1    42.80    0.05    0.04      0    42.42    0.05    0.05      0
        4096            64     float    none      -1    43.10    0.10    0.09      0    42.85    0.10    0.09      0
        8192           128     float    none      -1    44.48    0.18    0.17      0    44.14    0.19    0.17      0
       16384           256     float    none      -1    46.17    0.35    0.33      0    45.51    0.36    0.34      0
       32768           512     float    none      -1    47.93    0.68    0.64      0    47.57    0.69    0.65      0
       65536          1024     float    none      -1    48.31    1.36    1.27      0    48.15    1.36    1.28      0
      131072          2048     float    none      -1    50.06    2.62    2.45      0    50.38    2.60    2.44      0
      262144          4096     float    none      -1    50.99    5.14    4.82      0    52.18    5.02    4.71      0
      524288          8192     float    none      -1    54.63    9.60    9.00      0    55.29    9.48    8.89      0
     1048576         16384     float    none      -1    63.77   16.44   15.41      0    63.85   16.42   15.40      0
     2097152         32768     float    none      -1    84.46   24.83   23.28      0    79.28   26.45   24.80      0
     4194304         65536     float    none      -1    94.15   44.55   41.77      0    94.28   44.49   41.71      0
     8388608        131072     float    none      -1    107.9   77.76   72.90      0    107.1   78.35   73.46      0
    16777216        262144     float    none      -1    140.4  119.46  111.99      0    137.4  122.13  114.50      0
    33554432        524288     float    none      -1    152.6  219.83  206.09      0    154.7  216.88  203.33      0
    67108864       1048576     float    none      -1    236.2  284.13  266.37      0    234.7  285.96  268.09      0
   134217728       2097152     float    none      -1    415.7  322.90  302.72      0    411.0  326.59  306.18      0
   268435456       4194304     float    none      -1    742.9  361.35  338.77      0    762.0  352.27  330.25      0
   536870912       8388608     float    none      -1   1412.5  380.09  356.34      0   1403.5  382.53  358.62      0
  1073741824      16777216     float    none      -1   2767.0  388.05  363.80      0   2749.2  390.57  366.16      0
  2147483648      33554432     float    none      -1   5429.5  395.52  370.80      0   5413.8  396.67  371.87      0
  4294967296      67108864     float    none      -1    10757  399.29  374.33      0    10736  400.06  375.05      0
  8589934592     134217728     float    none      -1    21395  401.49  376.40      0    21381  401.75  376.64      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 135.094 
```

Deploy a training job via helm. Feel free to experiment with changing some of the values.  On DWS: 
```
cd $REPO_ROOT/training 
helm upgrade --install torch-training-job . --set infra.nodepool_name="gpu-nodepool-dws" --set training_params.model_id="meta-llama/Llama-3.1-70B" --set training_params.per_device_train_batch_size=8 --set training.parallelism=2 --set image.name=${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${IMAGE_NAME}:latest --set fuse.bucket=${GSBUCKET}
```

You can follow the job status with 
```
kubectl describe job torch-job`
```

on DWS, Kueue will leave your job in a suspended state until the resources are provisioned. It will wait until everything is ready at once and resume the job. 

on Spot: 
```
cd $REPO_ROOT/training 
helm upgrade --install torch-training-job . --set infra.nodepool_name="gpu-nodepool-spot" --set infra.spot="true" --set training_params.model_id="meta-llama/Llama-3.1-70B" --set training_params.per_device_train_batch_size=8 --set training.parallelism=2 --set image.name=${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${IMAGE_NAME}:latest --set fuse.bucket=${GSBUCKET}
```


If you want to cancel the training job at any point:
```
helm uninstall torch-training-job 
```

You can follow the job logs with:
```
kubectl logs -l app=torch-job -c job -f
```

The training job output should look like:
```
INFO 2025-09-09T16:10:21.372601224Z [resource.labels.containerName: job] 🚀 Starting FSDP training for 10 epoch(s)...
INFO 2025-09-09T16:10:21.372619683Z [resource.labels.containerName: job] --------------------------------------------------------------------------------
INFO 2025-09-09T16:10:21.376110182Z [resource.labels.containerName: job] [PerfLogger] Using 70.55B parameters for TFLOPs calculation.
INFO 2025-09-09T16:10:21.376116804Z [resource.labels.containerName: job] {}
INFO 2025-09-09T16:10:21.376119014Z [resource.labels.containerName: job] --- Starting Epoch 1/10 ---
INFO 2025-09-09T16:10:48.625736211Z [resource.labels.containerName: job] Step: 1 | Time: 27.02s | TFLOPs/s/GPU: 513.3 | Tokens/s/GPU: 1213 | Loss: 1.3697
INFO 2025-09-09T16:11:03.741677921Z [resource.labels.containerName: job] Step: 2 | Time: 15.12s | TFLOPs/s/GPU: 917.7 | Tokens/s/GPU: 2168 | Loss: 1.3077
INFO 2025-09-09T16:11:19.025340931Z [resource.labels.containerName: job] Step: 3 | Time: 15.28s | TFLOPs/s/GPU: 907.6 | Tokens/s/GPU: 2144 | Loss: 1.1527
INFO 2025-09-09T16:11:33.552950048Z [resource.labels.containerName: job] Step: 4 | Time: 14.53s | TFLOPs/s/GPU: 954.9 | Tokens/s/GPU: 2256 | Loss: 0.9699
INFO 2025-09-09T16:11:48.093697184Z [resource.labels.containerName: job] Step: 5 | Time: 14.54s | TFLOPs/s/GPU: 954.0 | Tokens/s/GPU: 2254 | Loss: 0.6620
INFO 2025-09-09T16:12:02.639682457Z [resource.labels.containerName: job] Step: 6 | Time: 14.55s | TFLOPs/s/GPU: 953.7 | Tokens/s/GPU: 2253 | Loss: 0.6007
INFO 2025-09-09T16:12:17.163969264Z [resource.labels.containerName: job] Step: 7 | Time: 14.52s | TFLOPs/s/GPU: 955.1 | Tokens/s/GPU: 2256 | Loss: 0.5132
INFO 2025-09-09T16:12:31.728377069Z [resource.labels.containerName: job] Step: 8 | Time: 14.56s | TFLOPs/s/GPU: 952.4 | Tokens/s/GPU: 2250 | Loss: 0.5110
INFO 2025-09-09T16:12:46.268846051Z [resource.labels.containerName: job] Step: 9 | Time: 14.54s | TFLOPs/s/GPU: 954.0 | Tokens/s/GPU: 2254 | Loss: 0.4839
INFO 2025-09-09T16:13:00.804060769Z [resource.labels.containerName: job] Step: 10 | Time: 14.53s | TFLOPs/s/GPU: 954.4 | Tokens/s/GPU: 2254 | Loss: 0.4395
```

On B200 with a local batch size of 8 for llama 70b the MFU should be ~42.5% on DWS flex. 

# Tear down everything 
```
source $REPO_ROOT/.env 
cd $REPO_ROOT/setup
./cleanup.sh
```


