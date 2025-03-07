# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; <div align="left"><img src="https://canada1.discourse-cdn.com/flex035/uploads/forum11/original/1X/dfb6d71c9b8deb73aa10aa9bc47a0f8948d5304b.png" width="90px"/>&nbsp;
# **RAPIDS-singlecell Tutorial Notebooks**


---



This repository houses the notebooks made to run on [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a GPU accelerated library developed by [scverse®](https://github.com/scverse).
The goal is of this repository is to help users be able to try out and explore many different capabilities of RAPIDS-singlecell on cell datasets ranging from **250 thousand to 11 million cells** on their own CUDA capabile GPU systems or on an instance of the quick deploy capability of [Brev.dev](https://developer.nvidia.com/brev) Launchables.

These notebooks will be valuable for single-cell scientists who want to quickly evaluate ease of use as well as explore the biological interpretability of RAPIDS-singlecell results. Secondarily, scientists will find value in learning to apply these methods to very large data sets. This repository is also broadly useful for any data scientist or developer who wants to run and evaluate single cell methods leveraging RAPIDS-singlecell. Data sets used for this tutorial are were made [publicly available by 10X](https://www.10xgenomics.com/datasets) as well as [CZ cellxgene](https://cellxgene.cziscience.com/).  The base container is the [24.12 RAPIDSAI Notebooks Container](https://rapids.ai), which you can freely get from [NVIDIA's NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/notebooks) following the instructions below.

If you like these notebooks and this GPU accelerated capability, and want to support scverse's efforts, please [learn more about them here](https://scverse.org/about/) as well as [consider joining their community](https://scverse.org/join/).

# Overview


---


This repository contains a diverse set of notebooks to help get anyone started using RAPIDS-singlecell. 



![layout architecture](https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/assets/scdiagram.png)


The outline below is a suggested exploration flow.  Unless otherwise noted, you can choose any notebook to get started, as long as you have the GPU resources to run the notebook.

For those who are new to doing basic analysis for single cell data, the end to end analysis of [01_demo_gpu_e2e](01_demo_gpu_e2e.ipynb) is the best place to start, where you are walked through the steps of data preprocessing, cleanup, visualization, and investigation.

| Notebook         | Description |Min GPU Size /<br>Instance |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| [01_demo_gpu_e2e](01_demo_gpu_e2e.ipynb)   | End to end workflow, where we understand the cells, run ETL on the data set then visiualize and explore the resiults. <br>This tutorial is good for all users | 24GB /<br>Standard RSC Instance | 
| [02_decoupler](02_decoupler.ipynb)   | This notebook continues from the outputs of [01_demo_gpu_e2e](01_demo_gpu_e2e.ipynb) as an overview of methods that <br>can be used to investigate transcriptional regulation | 24GB /<br>Standard RSC Instance |
| [demo_gpu_e2e_with_PR](demo_gpu_e2e_with_PR.ipynb)  | End to end workflow, like [01_demo_gpu_e2e](01_demo_gpu_e2e.ipynb), but uses pearson residuals for normalization. | 24GB /<br>Standard RSC Instance |
| [spatial_autocorr](spatial_autocorr.ipynb) | An introduction to spatial transcriptomics analysis and visualization | 24GB / Standard RSC Instance |
| [out-of-core_processing](out-of-core_processing.ipynb) | In this notebook, we show the scalability of the analysis toof up to 11M cells easily by using Dask.<br>**Requires a 48GB GPU** | 48GB /<br>Standard RSC Instance |
| [multi_gpu_large_data_showcase](multi_gpu_large_data_showcase.ipynb) | This notebook enhances the 11M cell dataset analysis with dask without exceeding memory limits.  <br>It fully scales to utilize all available GPUs, uses chunk-based execution, and efficiently manages memory<br>**Requires 8x H100s or better.  For all other GPUs systems, please run [out-of-core_processing](out-of-core_processing.ipynb) instead**| 8x 80GB /<br>Large RSC Instance |
| [demo_gpu-seuratv3](demo_gpu-seuratv3.ipynb) | In this notebook, show diversity in capability by run a similar workflow to [01_demo_gpu_e2e](01_demo_gpu_e2e.ipynb), but on brain cells | 24GB /<br>Standard RSC Instance |
| [demo_gpu-seuratv3-brain-1M](demo_gpu-seuratv3-brain-1M.ipynb) | In this notebook, we scale up the analysis of [demo_gpu-seuratv3](demo_gpu-seuratv3.ipynb) to 1 million brain cells.<br>**Requires an 80GB GPU, like an H100** |  80GB /<br>Large RSC Instance |

**Note:** To ensure you have the maximum GPU memory available, **please remember to shut down your completed notebook's kernel before starting a new notebook**.  If you don't, you may experience Out Of Memory (OOM) based errors.  To fix that, simply kill all the kernels, and the restart only the kernel for the notebook you want to run.

<div align="center"><img src="https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/assets/kernel.png" width="460px"/>&nbsp</div>
<br>

# Deploying this Repository

---

This repo is made to run as Brev.dev's Launchables, or a machine you own with a local CUDA compatible GPU.  
## Deploy Using [Brev](https://brev.dev)
Please click this button to deploy this Repo using Brev.dev's Launchables

### Using our Brev Launchable (super easy mode - highly recommended)

1. For a **Standard RSC Instance** (L40s), click here: [![ Click here to deploy the RAPIDS Singlecell Launchable.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://nvda.ws/4kr01um)  
   For a **Large RSC Instance** (8x H100),click here: [![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://nvda.ws/4h6vbo1).

2. Click **Deploy Launchable** on the Brev.dev Launchable page

   ![deploy_brev](https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/assets/deploy_brev.png)
   
3. Wait for the Container status show **Ready** (can take up to 8 minutes).  Then, click **Access GPU**

   ![go_on_green](https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/assets/go_on_green.png)
   
4. On the **Instance** page, click **Open Notebook**

   ![open_notebook](https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/assets/open_notebook.png)

You should drop into a fully installed and populated JupyterLab environment.  Open up your desired notebook, and have a great time!

### Making this Launchable Portable on Brev (knowledgeable users only)
If using Brev to create a new Launchable or Standalone Compute Instance, so that you can use your desired cloud provider: 
- In **Compute**, remember the GPU requirements when picking an instance
- When adding the **Container**, use Docker Compose.  In most cases, please use the docker-compose file, **[docker/brev/docker-compose-nb-2412.yaml](https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks/raw/main/docker/brev/docker-compose-nb-2412.yaml)** as your base.  You *can* make edits, but will not be a recommended activity.  Instead use and install additional packages using the `requirements.txt` found in this repo as a base.  **Do not** Preinstall Jupyter.
- For **Files**, please clone this repo, or any repo of your choosing
- In **Ports**, please open ports **8888, 8787, and 8786**.
Git is not installed in this container, but can be installed into the container when it is running using
```
apt update
apt install git -y
```

Of course, if all you want is to install additional packages, it is easiest to just use the [![ Click here to deploy the RAPIDS Singlecell Launchable.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://nvda.ws/4kr01um)  and install your additional software from there!

## Deploy on a CUDA compatible GPU system (knowledgeable users only)

Some people wmay want to have this expereince off of Brev and take it with you.  Great!  Here's a (somewhat) easy way how!  

If you are provisioning this on a non-brev cloud instance, workstation, or local machine, please follow the steps below:

### 1. Provision a System Meeting These Requirements
All provisioned systems need to be RAPIDS capable. Here's what is required:

<i class="fas fa-microchip"></i> **GPU:** NVIDIA Volta™ or higher with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
- For most of the notebooks, we recommend a **GPU with 24 GB VRAM or more**, due to the dataset size, such as the **L40S**, which can be quickly deployed above.  Some other common GPU options found in your favorite cloud service providers, your workstations, or your PC are:

| Cloud/DGX | Workstation GPU | Consumer GPU |
|---|---|---|
| A/H/B100  | A4000 ADA or better | 5090 |
| L40s/L40 | A5000 or better | 4090 |
| A10/A10s | RTX6000 or better | 3090 |

The [multi_gpu_large_data_showcase](multi_gpu_large_data_showcase.ipynb) and the [demo_gpu-seuratv3-brain-1M](demo_gpu-seuratv3-brain-1M.ipynb) requires a large multigpu system.  The [out-of-core_processing](out-of-core_processing.ipynb) notebook, even using the 11 million cell dataset, and the [demo_gpu-seuratv3](demo_gpu-seuratv3.ipynb) are respectively similar and can be run on one of the GPUs above, but a 48GB GPU is recommended.


<i class="fas fa-desktop"></i> **OS:**
- <i class="fas fa-check-circle"></i> Linux distributions with `glibc>=2.28` (released in August 2018), which include the following:
  - [Arch Linux](https://archlinux.org/), minimum version 2018-08-02
  - [Debian](https://www.debian.org/), minimum version 10.0
  - [Fedora](https://fedoraproject.org/), minimum version 29
  - [Linux Mint](https://linuxmint.com/), minimum version 20
  - [Rocky Linux](https://rockylinux.org/) / [Alma Linux](https://almalinux.org/) / [RHEL](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), minimum version 8
  - [Ubuntu](https://ubuntu.com/), minimum version 20.04
- <i class="fas fa-check-circle"></i> Windows 11 using a [WSL2 specific install](https://docs.rapids.ai/install/#wsl2)

<i class="fas fa-download text-purple"></i> **CUDA 12 & latest NVIDIA Drivers:** Install the latest drivers for your system [HERE](https://www.nvidia.com/en-us/drivers/)

 **Note:** RAPIDS is tested with and officially supports the versions listed above. Newer CUDA and driver versions may also work with RAPIDS. See [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for details.

### 2. Clone This Repo:
```
git clone https://github.com/clara-parabricks-workflows/RAPIDS-singlecell-introduction-notebooks.git
```

### 3. Install Packaging Environment Software:
RAPIDS can be installed using Pip, Conda, or Docker.  To replicate the same expereince as in the Launchable, it is recommended to use Docker for installation.  
<i class="fas fa-download text-purple"></i> **3.1 Get Docker:** Use the code below to download and install Docker on your system
```
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```
<i class="fas fa-download text-purple"></i> **3.2 Get RAPIDS:** Use the code below to pull the same RAPIDS container used in the Launchable
```
docker run --gpus all --pull always --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786
    -v ~/RAPIDS-singlecell-introduction-notebooks:/home/rapids/notebooks/RAPIDS-singlecell-introduction-notebooks
    nvcr.io/nvidia/rapidsai/notebooks:24.12-cuda12.5-py3.12
```
**Note:** The volume is currently assuming that you cloned the `RAPIDS-singlecell-introduction-notebooks` repository into your `HOME` folder (`~/RAPIDS-singlecell-introduction-notebooks`).  If you did not clone it there, please change the `~/RAPIDS-singlecell-introduction-notebooks` portion of the above to the correct path before running the command.


### 4. Install the Extra RAPIDS Singlecell Libraries into the Running Container
Once the container is running, 
- if the provisioned system is an external cloud or workstation, please use a web browser to navigate to the system's IP address and it's port 8888.  Example `http://192.168.1.2:8888`
- if the provisioned system is your sustem, then use `http://127.0.0.1:8888`.

Once the JupyterLab instance loads, open the terminal form the JupyterLab GUI, and run:
```
cd /home/rapids/notebooks/RAPIDS-singlecell-introduction-notebooks
pip install -r requirements.txt
```

For additional information on RAPIDS Singlecell please visit the [RAPIDS Singlecell Docs](https://rapids-singlecell.readthedocs.io/)
For alternate installation instructions, please refer to the [RAPIDS Singlecell Install Guide](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html) to install using [pip](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#pypi), [Conda](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#conda), or [Docker](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#docker)

## Tips and Tricks
1. If after clicking a "Healthy" Port 8888 link in `Deploy with Brev: Step 4`, JupyterLab does not start, or the notebooks don't show, please try again in a few seconds.  There is a known issue where there system needs a minute or two Also, sometimes, the page needs to be refreshed to update the status.
2. Currently, restarting an instance, after stopping it, will start up far faster than starting a new instance.
3. If you **Stop** the instance, all the data on the main storage will be retained for the next time you start it.
4. To conserve GPU memory, please remember to shut down your completed notebook's kernel before starting a new notebook.
5. If your data download gets interrupted, please delete the files you intended to download and try again
6. The Standard RSC Instance (L40s) has 128GB of space (~90GB user usable).    
7. The Large RSC Instance (8x H100) has the same 128GB of space as the Standard RSC Instance, but there is a special `data` folder is mapped to a 1.12TB disk.  Data retention on that disk (in that folder) is not guaranteed.  YMMV.


# Detailed Overview of the Notebooks


---


### **Notebook 01_demo_gpu_e2e : Basic Demo Workflow**

By the completion of this notebook, a user will be able to do the following:

- Load and Preprocess the data
    - Load a sparse matrix in h5ad format using Scanpy
    - Preprocess the data, implementing standard QC metrics to assess cell and gene quality per cell, as well as per gen

- QC cells visually to understand the data
    - Users will learn how to visually inspect 5 different plots that help reflect quality control metrics for single cell data to:
        - Identify stressed or dying cells undergoing apoptosis
        - Empty droplets or dead cells
        - Cells with abnormal gene counts
        - Low quality or overly dominant cells

- Filter unusual cells
    - Users will learn how to remove cells with an extreme number of genes expressed
    - Users will filter out cells with an unusual amount of mitochondrial content

- Remove unwanted sources of variation
    - Select most variable genes to better inform analysis and increase computational efficiency
    - Regress out additional technical variation that we observed in the visual plots (Note, this can actually remove biologically relevant information, and would need to be carefully considered with a more complex data set)
    - Standardize by using a z-score transformation

- Cluster and visualize data
    - Implement PCA to reduce computational complexity. We use the GPU-accelerated PCA implementation from cuML, which significantly speeds up computation compared to CPU-based methods.
    - Identify batch effects visually by generating a UMAP plot with graph-based clustering

- Batch Correction and analysis
    - Remove assay-specific batch effects using Harmony
    - Re-compute the k-nearest neighbors graph and visualize using the UMAP.
    - Perform graph-based clustering
    - Visualize using other methods (tSNE)

- Explore the biological information from the data
    - Differential expression analysis: Identifying marker genes for cell types
        - Implement logistic regression
        - Rank genes that distinguish cell types
    - Trajectory analysis
        - Implement a diffusion map to understand the progress of cell types

### **Notebook 02_decoupler - Decoupler-GPU: Accelerated Transcriptional Regulatory Analysis**

This is an overview of methods that can be used to investigate transcriptional regulation. Although the tutorial does not dive deeply into these models, the notebook is used to reflect a difference in speed that is benefitted by GPU.

### **Notebook demo_gpu_e2e_with_PR - A different way to normalize with Pearson Residuals**

By the completion of this notebook, a user will be able to remove unwanted sources of variation using Pearson Residuals to normalize. This is a different approach to analysis from Notebook 01. After filtering, we introduce a normalization step toa address the potential issues in how we previously removed unwanted sources of variation.

### **Notebook spatial_autocorr - Visualizing and investigating spatial transcriptomics data**

By the completion of this notebook, a user will be able to do the following:
- Compute spatial autocorrelation, which represents how gene expression levels are spatially distributed across tissue sections.
- Use Squidpy to compute a graph on the coordinates
- Compute two metrics using Moran's I (better for global structures) and Geary's C (better for local structures)
- Explore the results visually by plotting the expression of genes Mbp (myelin-associated) and Nrgn (neuronal marker)

### **Notebook out-of-core_processing - Scale analysis to 11M cells easily and quickly leveraging Dask**

By the completion of this notebook, a user will be able to do the following:
- Create a local DASK cluster
- Load data directly into an AnnData object using a Dask array
- Filter cells and genes without additional computation
- Log normalize the data and identify highly variable genes
- Scale gene expression
- Compute PCA using GPU acceleration

### **Notebook multi_gpu_large_data_showcase - Multi-GPU showcase using persist**

By the completion of this notebook, a user will be able to perform the same steps as Notebook 05, but with the following:
- Process massive single-cell datasets without exceeding memory limits
- Fully utilize all available GPUs, scaling performance across multiple devices
- Enable chunk-based execution, efficiently managing memory by loading only necessary data

### **Notebook demo_gpu-seuratv3 - End to end example on Brain Cells**

By the completion of this noteobok, a user will be able to perform similar steps to Notebook 01 (end to end demo) but on brain cells.

### **Notebook demo_gpu-seuratv3-brain-1M - End to end example on 1M Brain Cells**

By the completion of this noteobok, a user will be able to perform similar steps to demo_gpu-seuratv3 but on 1M brain cells.
