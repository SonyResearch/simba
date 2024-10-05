FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-12.2
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      rsync \
      tree \
      curl \
      wget \
      unzip \
      htop \
      tmux \
      xvfb \
      patchelf \
      ca-certificates \
      bash-completion \
      libjpeg-dev \
      libpng-dev \
      ffmpeg \
      cmake \
      swig \
      libssl-dev \
      libcurl4-openssl-dev \
      libopenmpi-dev \
      python3-dev \
      zlib1g-dev \
      qtbase5-dev \
      qtdeclarative5-dev \
      libglib2.0-0 \
      libglu1-mesa-dev \
      libgl1-mesa-dev \
      libvulkan1 \
      libgl1-mesa-glx \
      libosmesa6 \
      libosmesa6-dev \
      libglew-dev \
      mesa-utils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /root/.ssh

# python
RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv

## kubernetes authorization
# kubernetes needs a numeric user apparently
# Ensure the user has write permissions
RUN useradd --create-home \
    --shell /bin/bash \
    --base-dir /home \
    --groups dialout,audio,video,plugdev \
    --uid 1000 \
    user
USER root
WORKDIR /home/user
RUN chown -R user:user /home/user && \
    chmod -R u+rwx /home/user
USER 1000

# Install packages
COPY deps/requirements.txt /home/user/requirements.txt
RUN python3.10 -m venv /home/user/venv && \
    . /home/user/venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install -U "jax[cuda12]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
ENV VIRTUAL_ENV=/home/user/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false 

# install mujoco 2.1.0, humanoid-bench and myosuite
ENV MUJOCO_GL egl
ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
     tar -xzf mujoco210-linux-x86_64.tar.gz && \
     rm mujoco210-linux-x86_64.tar.gz && \
     mkdir /home/user/.mujoco && \
     mv mujoco210 /home/user/.mujoco/mujoco210 && \
     find /home/user/.mujoco -exec chown user:user {} \; && \
     python -c "import mujoco_py" && \
     git clone https://github.com/joonleesky/humanoid-bench /home/user/humanoid-bench && \
     pip install -e /home/user/humanoid-bench && \
     git clone --recursive https://github.com/joonleesky/myosuite /home/user/myosuite && \
     pip install -e /home/user/myosuite
