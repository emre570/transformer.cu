# 🚀 CUDA Transformer: Modular Transformer Components with LibTorch and CUDA Kernels

This project is a modular, from-scratch Transformer implementation using **LibTorch** (PyTorch C++ API) and **custom CUDA kernels** for acceleration.

You’ll find pure C++ modules for:
- Input Embedding
- Positional Encoding
- Layer Normalization
- Residual Connection
- FeedForward
- Softmax (custom CUDA kernel)

---

## ⚙️ Prerequisites

- Docker CE
- NVIDIA Container Toolkit
- Compatible CUDA driver installed **on your host** (must match container CUDA version)

---

## 🐳 Setting Up Your Environment

### 1. **Pull and Run CUDA Docker Container**

Your **host CUDA driver version** must match the container’s CUDA version (e.g., `12.4.0`):

```bash
sudo docker run --gpus all -it \
  -p 8080:8080 \
  -v $(pwd):/workspace \
  -w /workspace \
  --name my-cuda-container \
  nvidia/cuda:12.4.0-devel-ubuntu22.04 \
  bash
```

> 🔁 Re-enter your container with:
```bash
docker exec -it my-cuda-container bash
```

You may install some packages like `nano`, `code-server` for working inside of container etc.

---

## 🔧 Installing LibTorch

1. [Download LibTorch](https://pytorch.org/get-started/locally/) for your CUDA version.
2. Extract and place it inside the project directory.

Then add it to your environment:

```bash
echo 'export CMAKE_PREFIX_PATH=/workspace/libtorch' >> ~/.bashrc
source ~/.bashrc
```
You can also look at CMakeLists.txt for reference.

---

## 🧱 Build the Project

1. Create a `CMakeLists.txt` file (already included if cloned).
2. Create a build directory:

```bash
mkdir build
cd build
cmake ..
make -j
```

3. Run the executable:

```bash
./main
```

Also look at `build_and_run.sh` for building and running the script without executing multiple lines.

---

## 🧠 Structure

Each module (e.g., `feed_forward.cpp`, `layer_norm.cpp`, etc.) is implemented as a separate class. 
CUDA kernels (e.g., `softmax.cu`) are invoked using custom launcher functions.

See `main.cpp` for usage examples and test cases for each component.

---

## 📌 Notes

- This project is **GPU-only**.
- All tensors are allocated on `torch::kCUDA`.
- You’ll need to manually uncomment desired test blocks in `main.cpp` for validation.

---

## 🧪 Example Output

```cpp
Input shape: [2, 4, 512]
Output shape: [2, 4, 512]
```

---

## 📊 Future Work

- Implement full **Encoder** and **Decoder** blocks using the completed modules.
- Introduce further **CUDA kernel optimizations** for Attention and other bottlenecks.
- Train a **Transformer model end-to-end** using this modular repo as the base.
