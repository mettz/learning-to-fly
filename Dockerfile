FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget gnupg
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-mkl-devel-2023.1.0
RUN apt-get update && apt-get install -y cmake build-essential git libhdf5-dev libboost-all-dev protobuf-compiler libprotobuf-dev python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install tensorboard==2.12.2
RUN git clone https://github.com/arplaboratory/learning_to_fly_in_seconds
RUN mkdir build
WORKDIR /build
RUN cmake ../learning_to_fly_in_seconds -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON