# Model Summary: SuperPoint / DISK / ALIKED + LightGlue

## 模型分类

### 独立特征提取器（输入图片，输出关键点+描述子）

| 模型 | 描述子维度 | 输入 | 输出 |
|------|-----------|------|------|
| `superpoint.onnx` | 256-D float32 | `image` [1,1,H,W] float32 灰度 | `keypoints` [1,N,2] int64, `scores` [1,N] float32, `descriptors` [1,N,256] float32 |
| `disk.onnx` | 128-D float32 | `image` [1,3,H,W] float32 RGB | `keypoints` [1,N,2] int16, `scores` [1,N] float32, `descriptors` [1,N,128] float32 |
| `disk_1024.onnx` | 128-D float32 | `image` [1,3,H,W] float32 RGB | 同上 (max 1024 keypoints) |
| `aliked-n16rot-top1k-640.onnx` | 128-D float32 | `image` [1,3,640,640] float32 RGB | `keypoints` [1,N,2] float32 (归一化坐标), `scores` [1,N] float32, `descriptors` [1,N,128] float32 |

### LightGlue 匹配器（输入两图的关键点+描述子，输出匹配对）

| 模型 | 描述子维度 | 输出格式 | 输入节点 |
|------|-----------|----------|----------|
| `superpoint_lightglue.onnx` | 256-D | **Standard**: `matches0` [1,N] int64 (-1=无匹配), `mscores0` [1,N] float32, `matches1` [1,M], `mscores1` [1,M] | `kpts0/kpts1` [1,N,2] float32, `desc0/desc1` [1,N,256] float32 |
| `disk_lightglue.onnx` | 128-D | **Standard**: 同上 | `kpts0/kpts1` [1,N,2] float32, `desc0/desc1` [1,N,128] float32 |
| `disk_lightglue_flash.onnx` | 128-D | **Standard**: 同上 | 同上 |
| `disk_lightglue_fused.onnx` | 128-D | **Fused**: `matches0` [M,2] int64 (配对索引), `mscores0` [M] float32 | 同上 |
| `aliked_lightglue.onnx` | 128-D | **Fused**: `matches0` [M,2] int64, `mscores0` [M] float32 | `kpts0/kpts1` [1,N,2] float32, `desc0/desc1` [1,N,128] float32 |
| `lightglue_for_aliked.onnx` | 128-D | **Fused**: 同上 | 同上（功能等价于 `aliked_lightglue.onnx`，内部图结构不同） |
| `aliked_lightglue_fused.onnx` | 128-D | **Fused**: `matches0` [M,2] int64, `mscores0` [M] float32 | 同上（含 MultiHeadAttention，需 GPU） |

---

## 两种输出格式

### Standard（标准格式）
```
matches0  shape=[1, N]    # 每元素=-1(无匹配) 或 index(匹配到的目标关键点)
mscores0  shape=[1, N]    # 对应置信度
matches1  shape=[1, M]    # 反向匹配
mscores1  shape=[1, M]
```

### Fused（融合格式）
```
matches0  shape=[M, 2]    # 直接给出 M 对 (kpt0_idx, kpt1_idx)
mscores0  shape=[M]       # 每对置信度
# 无 matches1/mscores1
```

---

## 引擎兼容性矩阵

| 模型 | ENGINE_NEW | ENGINE_ORT | 备注 |
|------|------------|------------|------|
| `superpoint_lightglue.onnx` | 通过 | 通过 | |
| `disk.onnx` | **不支持** | 通过 | ENGINE_NEW 无法解析 DISK ONNX 格式 |
| `disk_1024.onnx` | **不支持** | 通过 | 同上 |
| `disk_lightglue.onnx` | 通过 | 通过 | |
| `disk_lightglue_flash.onnx` | 通过 | 通过 | |
| `disk_lightglue_fused.onnx` | **不支持** | **需 GPU** | MultiHeadAttention CPU 不支持 |
| `aliked_lightglue.onnx` | **不支持** | 通过 | |
| `lightglue_for_aliked.onnx` | **不支持** | 通过 | 功能等价于 `aliked_lightglue.onnx` |
| `aliked_lightglue_fused.onnx` | **不支持** | **需 GPU** | MultiHeadAttention CPU 不支持 |

---

## SuperPoint vs DISK vs ALIKED 对比

| 特性 | SuperPoint | DISK | ALIKED |
|------|-----------|------|--------|
| 描述子维度 | 256-D | 128-D | 128-D |
| 输入图像 | 灰度 [1,1,H,W] | RGB [1,3,H,W] | RGB [1,3,640,640] |
| 关键点类型 | int64 | int16 (像素坐标) | float32 (归一化坐标 [-1,1]) |
| LightGlue 输出格式 | Standard | Standard / Fused | Fused |
| ENGINE_NEW 支持 | 是 | 提取器不支持，匹配器支持 | 不支持 |
| ENGINE_ORT 支持 | 是 | 是 | 是 |

---

## 测试文件对应关系

| 测试文件 | 引擎 | 测试模型 |
|----------|------|---------|
| `test_lightglue.cpp` | ENGINE_NEW | `superpoint_lightglue.onnx` |
| `test_disk_extractor_new.cpp` | ENGINE_NEW | `disk.onnx`, `disk_1024.onnx` (不支持) |
| `test_disk_extractor_ort.cpp` | ENGINE_ORT | `disk.onnx`, `disk_1024.onnx` |
| `test_disk_lightglue_new.cpp` | ENGINE_NEW | `disk_lightglue.onnx`, `disk_lightglue_flash.onnx` |
| `test_disk_lightglue_ort.cpp` | ENGINE_ORT | 上两个 + `disk_lightglue_fused.onnx` (需 GPU) |
| `test_aliked_lightglue_new.cpp` | ENGINE_NEW | `aliked_lightglue.onnx`, `lightglue_for_aliked.onnx`, `aliked_lightglue_fused.onnx` (均不支持) |
| `test_aliked_lightglue_ort.cpp` | ENGINE_ORT | 同上 (`aliked_lightglue_fused.onnx` 需 GPU) |
