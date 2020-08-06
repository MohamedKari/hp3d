##### CUDA #####
FROM nvidia/cuda:10.2-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"] 

##### PYENV & PYTHON #####
# Install pyenv dependencies & fetch pyenv
# see: https://github.com/pyenv/pyenv/wiki/common-build-problems
# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git && \
    git clone --single-branch --depth 1  https://github.com/pyenv/pyenv.git /.pyenv

ENV PYENV_ROOT="/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
ENV PATH="$PYENV_ROOT/shims:$PATH"

ARG PYTHON_VERSION=3.6.4

# Use PYTHON_CONFIGURE_OPTS="--enable-shared" to avoid 
# 'relocation R_X86_64_PC32 against symbol `_Py_NoneStruct' can not be used when making a shared object; recompile with -fPIC' error
# https://stackoverflow.com/questions/42582712/relocation-r-x86-64-32s-against-py-notimplementedstruct-can-not-be-used-when
RUN PYTHON_CONFIGURE_OPTS="--enable-shared"  pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION}

##### NON-PYTHON DEPENDENCIES #####
WORKDIR /app

RUN apt-get update && apt-get install -y \ 
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \ 
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    qt5-default \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev

# Install Open CV - Warning, this takes absolutely forever
RUN mkdir -p ~/opencv cd ~/opencv && \
    wget https://github.com/opencv/opencv/archive/4.0.0.zip && \
    unzip 4.0.0.zip && \
    rm 4.0.0.zip && \
    mv opencv-4.0.0 OpenCV && \
    cd OpenCV && \
    mkdir build && \ 
    cd build && \
    cmake \
    -DWITH_QT=OFF \ 
    -DWITH_OPENGL=OFF \ 
    -DFORCE_VTK=OFF \
    -DWITH_TBB=OFF \
    -DWITH_GDAL=OFF \
    -DWITH_XINE=OFF \
    -DWITH_GTK=OFF \
    -DBUILD_EXAMPLES=OFF .. && \
    make -j4 && \
    make install && \ 
    ldconfig

##### PYTHON PACKAGE DEPENDENCIES #####
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

##### APPLICATION #####
COPY . .

ENV PYTHONPATH=hp3d/pose_extractor/build/:$PYTHONPATH
ENTRYPOINT python setup.py build_ext && python -m hp3d.app raw.mp4 share
# ENTRYPOINT python setup.py build_ext && python -m hp3d.rpc.server

# make docker-ssh
# python setup.py build_ext
# export 
# python
# from pose_extractor import extract_poses
# python demo.py --model human-pose-estimation-3d.pth --video 0
# python demo.py --model human-pose-estimation-3d.pth --video waymo.mp4
# python demo.py --model human-pose-estimation-3d.pth --video raw.mp4


# sudo /.pyenv/shims/python demo.py --model human-pose-estimation-3d.pth --video 0