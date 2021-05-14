# Installation
The original installation is not suitable for gpu cluster with 3090. Follows are the updated one:
```
conda create --name 3drecon python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
conda install opencv
conda install scikit-image
pip install \
  open3d==0.12.0 \
  trimesh==3.9.15 \
  pyquaternion==0.9.9 \
  pytorch-lightning==1.2.10 \
  pyrender==0.1.45
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```
- Issues:
    1. the torchvision version is too low, which can not import `torchvision.ops`.
    - [x] Solution: re-install torch and torchvision: `conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`.
    2. the pytorch version is incompatible with detectron2, which causes `ImportError: libc10_cuda.so: cannot open shared object file: No such file or directory`. 
    - [x] Solution: install from source: `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`.
    3. `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`. This issue seems be caused by scipy being compiled against a different version of numpy than the one installed. 
    - [x] Solution: `pip uninstall numpy && pip install numpy`

- Useful links:
  * [Managing CUDA dependencies with Conda](https://towardsdatascience.com/managing-cuda-dependencies-with-conda-89c5d817e7e1)
  * [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-installation)

Final installation command (correct) for cluster with 3090
```
conda create --name 3drecon python=3.7
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install \
  open3d==0.12.0 \
  trimesh==3.9.15 \
  pyquaternion==0.9.9 \
  pytorch-lightning==1.2.10 \
  pyrender==0.1.45 \
  scikit-image==0.18.1
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
pip uninstall numpy
pip install numpy
```

Note that `apex` is turned off, as `amp_backend='native'` in the newer version of `pytorch-lightning`.
Furthermore, this part of code in `train.py` needs to be updated:

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_path, '{epoch:03d}'),
        save_top_k=-1,
        period=cfg.TRAINER.CHECKPOINT_PERIOD)

    trainer = pl.Trainer(
        logger=logger,
        check_val_every_n_epoch=cfg.TRAINER.CHECKPOINT_PERIOD,
        callbacks=[CudaClearCacheCallback(), checkpoint_callback],
        distributed_backend='ddp',
        benchmark=True,
        gpus=cfg.TRAINER.NUM_GPUS,
        precision=cfg.TRAINER.PRECISION,
        amp_level='O0')


##
New env installed on DGX1-3 (python 3.6.7, cuda 10.2, torch 1.5.0)
```
conda create --name env_3drecon python=3.6.7
conda install -y pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -c pytorch
conda install opencv
pip install \
  open3d==0.12.0 \
  trimesh==3.9.15 \
  pyquaternion==0.9.9 \
  pytorch-lightning==1.2.10 \
  pyrender==0.1.45 \
  PyOpenGL_accelerate==3.1.5 \ 
  PyOpenGL_accelerate==3.1.5 \ 
  scikit-image
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html
```
- Issue:
    1. torch.utils.data.DataLoader multiprocess hanging when run `prepare_data.py`
       - ~~possible reason 1~~: matplotlib/trimesh imported after torch [c.f. [here](https://github.com/pytorch/pytorch/issues/36375)]
       - [x] ~~solution 1~~: import matplotlib/trimesh before importing torch.
       - ~~possible reason 2~~: there might be some limitation of queue to be < 2^31 (i.e., the total size of a batch is not greater than 2GB) [c.f. [here](https://github.com/pytorch/pytorch/issues/1595)]
       - [x] ~~solution 2~~: reduce the num_worker from `4` to `2`.
       - ~~possible reason 3~~: use cuda in multiprocessing? [c.f. [here](https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing)] 
       - [x] ~~solution 3~~: set `torch.multiprocessing.set_start_method('spawn')`
       - [x] **naive solution**: set the num_worker as `0`.
    2. pytorch-lightning version is too high.
        > pytorch_lightning.utilities.exceptions.MisconfigurationException: You have asked for native AMP but your PyTorch version does not support it. Consider upgrading with `pip install torch>=1.6`
        - [x] solution: pip uninstall pytorch-lightning && pip install pytorch-lightning==0.8.5
    3. ModuleNotFoundError: You set `use_amp=True` but do not have apex installed.
        > Install apex first using this guide and rerun with use_amp=True:https://github.com/NVIDIA/apex#linux his run will NOT use 16 bit precision
        - [x] Can not install apex successfully. Final solution: set `use_amp=False`.
    4. When running `evaluate.py`, raise `raise NoSuchDisplayException` issue. The error info is "pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None""
        - [x] Add `os.environ['PYOPENGL_PLATFORM'] = 'egl'` in `evaluate.py` 
    5. After that, get another issue: `TypeError: startswith first arg must be bytes or a tuple of bytes, not str`.
        This may due to thw NVIDIA Driver and OpenGL during rendering. Potential solution can be modify the `GL` to `OpenGL` in `egl.py` under `OpenGL/platform`. Pls refer to this discussion [Headless rendering with nvidia and using libOpenGL.so](https://github.com/mcfletch/pyopengl/issues/27).
        
## 
Setup env with docker 
1. Build image from Dockerfile:
    ```
    docker build --tag atlas-docker - < Dockerfile 
    ```
2. Run the image as a container:
    + To initialize a container with name `atlas-docker` based on docker image `atlas-docker` in the foreground mode.  
        ```
        docker run --gpus all -it -p [host port]:[container port] --name atlas-container --mount type=bind,source="/storage_fast/nzhao/workspace/Atlas",target=/Atlas atlas-docker /bin/bash
        ```
        By default, the container port is `5000`. 
    + Use `docker ps -a` to list all the container info. Since the container is running in foreground mode, make sure tmux is open when run the container.
    + Use `Ctrl+D` or `exit` to close container.
    + To restart an existing container:
        ```
        docker restart atlas-container
        ```  
    + To execute an interactive bash shell on the container:
       ```
       docker exec -it atlas-container /bin/bash
       ```
    + To exit container without stopping process, enter `Ctrl+p` then `Ctrl+q`.
    
3. Install dependencies:
    ```
    sudo apt install htop libusb-1.0-0
    ```  
    The reason to install `libusb-1.0-0` is because it is needed when importing `open3d`. 
4. Set `os.environ["PYOPENGL_PLATFORM"] = "osmesa"` to solve the bug `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"` when running `evaluate.py`.
5. Downgrade the version of pytorch-lightning if its version is larger than 1.2.10.

     
Refer to [here](https://docs.docker.com/language/python/build-images/) fore more info on docker.