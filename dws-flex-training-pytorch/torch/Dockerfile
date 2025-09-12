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

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /app

# Copy all files into the application directory
COPY . .

# Combine all installation and cleanup steps into a single RUN command
RUN apt-get update && apt-get install -y python3 python3-pip wget && \
    pip3 install --no-cache-dir --break-system-packages packaging && \
    # Add --break-system-packages to override any externally-managed-environment error
    pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir --break-system-packages --trusted-host pypi.python.org -r requirements.torch && \
    # Install the CUDA keyring for Ubuntu 24.04
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-8 && \
    # Clean up apt caches and downloaded files to save space
    rm -rf /var/lib/apt/lists/* && \
    rm -f cuda-keyring_1.1-1_all.deb