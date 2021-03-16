FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python-dev \
        python-numpy \
        libglib2.0-0 \
        libgl1-mesa-dev \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get install -y 
#RUN apt-get install -y 

# Install python libraries
RUN pip install pytorch-lightning  
RUN pip install jupyterlab
RUN pip install pandas
RUN pip install numpy
RUN pip install scipy 
RUN pip install matplotlib 
RUN pip install ipython 
RUN pip install jupyter 
RUN pip install pandas 
RUN pip install sympy 
RUN pip install nose
RUN pip install einops
RUN pip install opencv-python  
RUN pip install wandb  
RUN pip install kornia  
RUN pip install torchfunc torchsummary torchlayers  
RUN pip install hydra-core
RUN pip install python-language-server[all]
RUN pip install jedi-language-server
RUN pip install black



# Add user
ARG USERNAME=ayb  
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME




#Enable gpu
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

CMD sh
