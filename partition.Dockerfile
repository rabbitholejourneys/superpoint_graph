FROM python:3.9-slim-bullseye as builder


RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    cmake \
    libboost-all-dev \
    python3-numpy \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install numpy laspy[lazrs]
RUN python -m pip install -r requirements.txt

COPY ./partition /partition
RUN cd /partition/cut-pursuit && \
    ls && \
    mkdir build &&\
    cd build && \
    #cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/ -DPYTHON_LIBRARY=/usr/local/lib/libpython3.10.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.10 -DBOOST_INCLUDEDIR=/usr/include/  -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -Dboost_numpy310_DIR=/usr/lib/x86_64-linux-gnu/cmake/boost_numpy-1.74.0/ && \
    cmake .. -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/ -DPYTHON_LIBRARY=/usr/local/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.9 -DBOOST_INCLUDEDIR=/usr/include/  -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -Dboost_numpy39_DIR=/usr/lib/x86_64-linux-gnu/cmake/boost_numpy-1.74.0/ && \
    make 
RUN cd /partition/ply_c && \
    ls && \
    cmake . -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/ -DPYTHON_LIBRARY=/usr/local/lib/libpython3.9.so -DPYTHON_INCLUDE_DIR=/usr/local/include/python3.9 -DBOOST_INCLUDEDIR=/usr/include/  -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -Dboost_numpy39_DIR=/usr/lib/x86_64-linux-gnu/cmake/boost_numpy-1.74.0/ && \
    make


