# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目用于测试和对比多种特征提取与匹配算法（SuperPoint、LightGlue、ALIKED、XFeat、SIFT、ORB 等），使用 OpenCV DNN 模块和 ONNX Runtime 进行推理。代码库同时包含 C++ 和 Python 实现。

## 构建与运行

```bash
# 构建所有 C++ 目标
cd build && cmake .. && make

# 构建单个目标（文件名即目标名）
cd build && make test_lightglue

# 覆盖 OpenCV 路径
cmake -DOpenCV_DIR=/path/to/opencv/build ..

# 运行可执行文件（从 build 目录执行，因为路径引用使用 ../model/ 和 ../images/）
cd build && ./test_lightglue
```

CMakeLists.txt 自动将每个 `.cpp` 文件编译为独立的可执行文件。OpenCV 默认路径为 `/home/christylinux/opencv_build/opencv/build`。

所有 C++ 可执行文件通过 `../model/` 和 `../images/` 的相对路径引用模型和图片，因此**必须从 `build/` 目录运行**。

## 架构

### 模型文件 (`model/`)

ONNX 模型文件，通过 OpenCV `cv::dnn::readNetFromONNX()` 或 ONNX Runtime `ort.InferenceSession` 加载：
- `superpoint.onnx` / `superpoint_fixed_480x640.onnx` — SuperPoint 特征提取器
- `superpoint_lightglue.onnx` / `superpoint_lightglue_fixed.onnx` 等 — 联合 SuperPoint+LightGlue 流水线
- `aliked-n16rot-top1k-640.onnx` — ALIKED 特征提取器
- `xfeat.onnx` — XFeat 特征提取器

### C++ 源码 (`*.cpp`)

每个 `.cpp` 文件编译为独立的可执行文件。当前活跃的（非存根）文件：

- **`test_lightglue.cpp`** — 使用 `cv::dnn::readNetFromONNX` 加载 LightGlue 模型，生成随机虚拟关键点和描述子，运行推理并打印匹配结果。作为纯模型测试使用。
- **`test_aliked.cpp`** — 使用 `ENGINE_ORT` 引擎加载 ALIKED ONNX 模型，对真实图片进行推理，解析关键点坐标并可视化。
- **`test_xfeat.cpp`** — 使用默认引擎加载 XFeat ONNX 模型，预处理灰度图（resize + padding 到 640×640），解析 score_map 提取关键点。
- **`example_for_featur_extractor_matcher.cpp`** — 完整特征提取与匹配演示：SIFT/ORB/FAST + Shi-Tomasi/Harris 角点，多种匹配策略（BF+KNN+ratio、Flann+KNN、CrossCheck），RANSAC 几何过滤，匹配可视化。
- **`in_memory_loading_demo.cpp`** — 演示如何从内存缓冲区（`std::vector<char>`）使用 `readNetFromONNX(data, size)` 加载 ONNX 模型，适用于无文件系统的嵌入式/Android 环境。
- **`superpoint_lightglue_only_demo.cpp`** — 当前为存根（返回 0），大部分代码已注释。原用于演示 OpenCV `cv::features::SuperPoint` 和 `cv::features::LightGlue` API。
- **`feature_extraction_example.cpp`** — 存根，原用于演示多种传统特征检测器。

### Python 源码 (`*.py`)

- **`test_superpoint.py`** — 使用 OpenCV DNN（`cv2.dnn.readNetFromONNX`）加载 SuperPoint，强制指定输出节点名 `["keypoints", "scores", "descriptors"]`（跳过不支持的 Gather/NonZero 算子），解析输出并绘制关键点。
- **`test_xfeat.py`** — 使用 ONNX Runtime 直接加载 XFeat，预处理（resize+padding→归一化），解析 score_map 提取关键点。
- **`test_aliked_onnx.py`** — 使用 ONNX Runtime 加载 ALIKED，自动检测归一化坐标并恢复为像素坐标，绘制关键点。
- **`validate_lightglue.py`** — 使用 OpenCV DNN 加载 LightGlue，生成随机虚拟特征，运行推理并验证输出。
- **`run_lightglue_fixed.py`** — 针对 `superpoint_lightglue_fixed.onnx` 模型的 Python 验证脚本，使用 `DNN_BACKEND_OPENCV` + `DNN_TARGET_CPU`。
- **`comparison/compare_features.py`** — SIFT vs SuperPoint+LightGlue 性能对比（ONNX Runtime），多次重复测试计算平均值与标准差。
- **`comparison/plot_comparation.py`** — 扩展版性能对比，支持不同关键点数量（256→4096），生成 matplotlib 对数坐标延迟图表。

### 调试工具 (`debug/`)

- **`debuglayers.cpp`** — 使用 MNIST 模型演示逐层前向传播，打印每层名称、类型和输出形状。用于调试 ONNX 模型图层结构。

## OpenCV DNN 关键 API 模式

从代码库中观察到的常见模式：

```cpp
// 加载模型（三种方式）
dnn::Net net = dnn::readNetFromONNX(path);                        // 默认引擎
dnn::Net net = dnn::readNetFromONNX(path, dnn::ENGINE_ORT);       // ONNX Runtime 引擎
dnn::Net net = dnn::readNetFromONNX(data_ptr, data_size);         // 内存加载

// 禁用 Winograd（LightGlue 需要）
net.enableWinograd(false);

// 设置命名输入
net.setInput(blob, "input_name");

// 指定输出节点名称进行推理
vector<Mat> outs;
net.forward(outs, vector<string>{"out1", "out2"});
```

## 图片与输入

测试图片位于 `images/` 目录（`image1.jpg`、`image2.jpg`）。C++ 代码相对于 `build/` 目录引用，Python 代码相对项目根目录引用。输出结果（特征可视化图片、匹配可视化图片）写入当前工作目录。
