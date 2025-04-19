install docker-ce with nvidia cuda container toolkit, then nvidia cuda container and run. cuda version on your pc and container's version must be same. 

container run command:
sudo docker run --gpus all -it \
  -p 8080:8080 \
  -v $(pwd):/workspace \
  -w /workspace \
  --name my-cuda-container \
  nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  bash

download libtorch accordingly with your cuda version. unzip it and open

echo 'export CMAKE_PREFIX_PATH=/workspace/libtorch' >> ~/.bashrc
source ~/.bashrc

create cmake lists txt file. 

create main.cpp and do whatever you want

mkdir build then open

cd build && cmake .. && make -j && ./main

you have to execute this every time you want to 

execute vscode server:
code-server --bind-addr 0.0.0.0:8080 --auth password

go in working container:
docker exec -it my-cuda-container bash