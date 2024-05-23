FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install -y ca-certificates apt-transport-https software-properties-common lsb-release \
    && apt install -y --no-install-recommends wget curl git make gfortran swig libssl-dev \
    && apt install -y gcc-12 g++-12 \
    && apt install -y python3.11 python3.11-dev python3.11-distutils \
    && apt install -y python3-setuptools \
    && cd /usr/bin \
    && unlink gcc && ln -s gcc-12 gcc && ln -s g++-12 g++ \
    && unlink python3 && ln -s python3.11 python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && export PATH=$PATH:$HOME/.local/bin \
    && pip3 install wheel \
    && apt remove --purge -y \
    && rm -rf /var/lib/apt/lists/*

# install cmake and ccache
RUN wget https://github.com/Kitware/CMake/archive/refs/tags/v3.28.5.tar.gz \
    && tar xvfz v3.28.5.tar.gz \
    && cd CMake-3.28.5 \
    && mkdir build && cd build \
    && ../bootstrap \
    && make -j3 && make install \
    && cd .. && rm -rf CMake-3.28.5 && rm v3.28.5.tar.gz \
    && wget https://github.com/ccache/ccache/archive/refs/tags/v4.9.1.tar.gz \
    && tar xvfz v4.9.1.tar.gz \
    && cd ccache-4.9.1 \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release .. \
    && make -j3 && make install \
    && cd .. && rm -rf ccache-4.9.1 && rm v4.9.1.tar.gz

# install knowhere dependancies
RUN apt install -y libopenblas-dev libcurl4-openssl-dev libaio-dev libevent-dev lcov \
    && pip3 install conan==1.61.0 \
    && conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# clone knowhere repo and build to update .conan
RUN git clone https://github.com/zilliztech/knowhere.git \
    && cd knowhere \
    && mkdir build && cd build \
    && conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release \
    && conan build .. \
    && cd ../.. \
    && rm -rf knowhere
