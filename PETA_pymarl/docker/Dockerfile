#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04
MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y
RUN apt-get install software-properties-common -y --allow-unauthenticated
RUN add-apt-repository -y multiverse
RUN apt-get update -y
RUN apt-get upgrade -y --allow-unauthenticated
RUN apt-get install -y apt-utils nano vim man build-essential wget sudo --allow-unauthenticated
RUN rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y
RUN apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git --allow-unauthenticated
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get install -y python3

RUN apt-get -y install python3-pip --allow-unauthenticated
RUN pip3 install --upgrade pip

RUN sudo apt-get purge python-pip
RUN wget https://bootstrap.pypa.io/pip/3.5/get-pip.py
RUN python3 get-pip.py
RUN python3 get-pip.py

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame
RUN ls
RUN mkdir /install
WORKDIR /install

RUN pip3 install wrapt
RUN pip3 install munch==2.0.2
RUN pip3 install docopt==0.3
RUN pip3 install jsonpickle==0.9.6
# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install
#RUN ls -l --progress=plain --no-cache
RUN cd /install/sacred
RUN python3 setup.py install

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

## -- SMAC
#RUN pip3 install dlb
#RUN apt-get install bazel
RUN #pip install git+git://github.com/deepmind/tree.git

RUN pip install dm-tree==0.1.1
RUN git clone https://github.com/deepmind/pysc2.git
RUN pip3 install --upgrade pysc2/
#RUN pip3 install pysc2
#RUN pip3 install --upgrade pip

#RUN git clone https://github.com/oxwhirl/smac.git
#RUN pip3 install -e smac/
RUN pip3 install git+http://github.com/oxwhirl/smac.git
ENV SC2PATH /pymarl/3rdparty/StarCraftII

WORKDIR /pymarl
