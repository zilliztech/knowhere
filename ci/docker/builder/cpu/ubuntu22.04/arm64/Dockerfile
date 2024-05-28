FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_VERSION="v3.28.5"
ENV CMAKE_TAR="cmake-3.28.5-linux-aarch64.tar.gz"
ENV CCACHE_DIR="ccache-4.9.1"
ENV CCACHE_SRC_TAR="v4.9.1.tar.gz"
ENV BFLOAT16_WHL="bfloat16-1.4.0-cp311-cp311-linux_aarch64.whl"

RUN apt update \
    && apt install -y ca-certificates apt-transport-https software-properties-common lsb-release \
    && apt install -y --no-install-recommends wget curl git make gfortran gcc g++ swig libssl-dev \
    && apt install -y gcc-12 g++-12 \
    && apt install -y python3.11 python3.11-dev python3.11-distutils \
    && apt install -y python3-setuptools \
    && cd /usr/bin \
    && unlink gcc && ln -s gcc-12 gcc \
    && unlink g++ && ln -s g++-12 g++ \
    && unlink python3 && ln -s python3.11 python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && export PATH=$PATH:$HOME/.local/bin \
    && pip3 install wheel \
    && apt remove --purge -y \
    && rm -rf /var/lib/apt/lists/*

# install cmake, ccache and bfloat16
RUN cd /tmp \
    && wget https://github.com/Kitware/CMake/releases/download/${CMAKE_VERSION}/${CMAKE_TAR} \
    && tar --strip-components=1 -xz -C /usr/local -f ${CMAKE_TAR} \
    && rm -f ${CMAKE_TAR} \
    && wget https://github.com/ccache/ccache/archive/refs/tags/${CCACHE_SRC_TAR} \
    && tar xvfz ${CCACHE_SRC_TAR} \
    && cd ${CCACHE_DIR} \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release .. \
    && make -j3 && make install \
    && cd ../.. && rm -rf ${CCACHE_DIR} ${CCACHE_SRC_TAR} \
    && wget https://github.com/zilliztech/knowhere/releases/download/v2.3.1/${BFLOAT16_WHL} \
    && pip3 install ${BFLOAT16_WHL} \
    && rm -f ${BFLOAT16_WHL}

# install knowhere dependancies
RUN apt update \
    && apt install -y libopenblas-dev libcurl4-openssl-dev libaio-dev libevent-dev lcov \
    && pip3 install conan==1.61.0 \
    && conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# clone knowhere repo and build to update .conan
RUN git clone https://github.com/zilliztech/knowhere.git \
    && cd knowhere \
    && mkdir build && cd build \
    && conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release \
    && conan build .. \
    && cd ../.. && rm -rf knowhere
