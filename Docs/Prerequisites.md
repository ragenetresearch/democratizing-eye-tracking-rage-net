# Prerequisites

## 1. Downloading data from Drive

Drive url: https://drive.google.com/drive/folders/1RHs7xGCD-k13N2YD2P0-54d0tmD7_XKy?usp=share_link

1.1 Download `OpenCvDNN` content and add it to `/Models/OpenCvDNN` folder.

1.2 Download `ShapePredictors` content and add it to `/ShapePredictors` folder.

1.3 Download `rn_w_attention__tf_model` file and add it to `/Models/rn_w_attention__tf_model` folder.

1.4 Download and unzip `Study_2.zip` file and add it to `/Datasets/Study_2` folder.

## 2. Conda's environment for Windows using Wsl2 and NVIDIA GPU

Following setup covers installing conda environment for Windows device with NVIDIA GPU using Wsl2.

- Note: Using NVIDIA GPU is not necessary, code can run both on GPU and CPU. NVIDIA GPU is though recommended.

### 2.1 NVIDIA CUDA drivers

Download official drivers from NVIDIA website:

- https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl2

Check NVIDIA driver:

```bash
nvidia-smi
```

### 2.2 Miniconda

We will use Miniconda as our package manager for Python. Download it from official website:

- https://docs.conda.io/projects/miniconda/en/latest/
- Also add conda to PATH.

### 2.3 - Conda's environment

Create venv using dependecies from `environment.yml` file:

```bash
conda env create --name rage_net --file environment.yml 
```

To validate tensorflow CPU installation:

```bash
py -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

To validate tensorflow GPU installation:

```bash
py -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
