# proto

Repository containing protobuf definitions

Structured as follows:
| *.proto
| *_pb2.py

Once changes are made to any of the .proto files, to regenerate the required python proto files, run:
`protoc --python_out=./ ./*.proto` in the root of this repository.
Uses protobuf v3.

# Compile Env
## Sample Dockerfile for compiling env
    FROM nvidia/cuda:9.0-devel-ubuntu16.04
    RUN apt-get update
    
    RUN apt-get install -y apt-utils
    RUN apt-get install -y \
    unzip \
    cmake \
    g++ \
    make \
    vim \
    wget 
    
    WORKDIR /root
    
    RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.10.0/protobuf-cpp-3.10.0.tar.gz
    RUN tar -xvf protobuf-cpp-3.10.0.tar.gz


    WORKDIR /root/protobuf-3.10.0
    RUN ./configure
    RUN make -j && make install
    RUN ldconfig

## Docker
registry.aibee.cn/aibee/hjlu/protoc:0.0.1

## Command
    docker run --rm  -v ~/proto:/root/proto registry.aibee.cn/aibee/hjlu/protoc:0.0.1 \
    bash -c "cd /root/proto && \
    protoc --python_out=./ ./*.proto"

## Trouble Shooting
When running python code, make sure to include this directory in your PYTHONPATH environment variable!
Example:

    export PYTHONPATH=$PYTHONPATH:/root/proto

