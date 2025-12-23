import cv2
import numpy as np
import sys
import os

print(f"正在使用 OpenCV 版本: {cv2.__version__}")

# --- 1. 定义模型路径和关键参数 ---


LG_MODEL_PATH = 'model/superpoint_lightglue_fixed_overwrite.onnx' 

# - 'superpoint_lightglue.onnx' -> DESCRIPTOR_DIM = 256
DESCRIPTOR_DIM = 256  # LightGlue 描述子维度

# --- 2. 定义假数据的参数 ---
NUM_KPTS_0 = 500  # 图像 0 的假特征点数量
NUM_KPTS_1 = 500  # 图像 1 的假特征点数量
IMG_SHAPE_0_HW = (480, 640) # 假图像 0 的 (H, W)
IMG_SHAPE_1_HW = (480, 640) # 假图像 1 的 (H, W)



def normalize_keypoints(kpts, shape_hw):
    """
    将关键点坐标从像素空间归一化到 [0, 1] 范围。
    """
    # shape_hw 是 (height, width)
    h, w = shape_hw
    
    kpts_norm = kpts.copy().astype(np.float32)
    
    # kpts[:, 0] 是 x (width)
    # kpts[:, 1] 是 y (height)
    kpts_norm[:, 0] = kpts_norm[:, 0] / (w - 1)
    kpts_norm[:, 1] = kpts_norm[:, 1] / (h - 1)
    
    # 添加批处理维度 (1, N, 2)
    return np.expand_dims(kpts_norm, 0)

def run_lightglue(net, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw):
    """
    运行 LightGlue ONNX 模型
    
    参数:
        kpts0, kpts1: (N, 2) 和 (M, 2) 像素坐标
        desc0, desc1: (N, D) 和 (M, D) 描述子
        shape0_hw, shape1_hw: 图像的 (height, width)
    """
    
    # 1. 归一化关键点
    kpts0_norm = normalize_keypoints(kpts0, shape0_hw) # (1, N, 2)
    kpts1_norm = normalize_keypoints(kpts1, shape1_hw) # (1, M, 2)
    
    # 2. 扩展描述子维度 (1, N, D)
    desc0_batch = np.expand_dims(desc0, 0) # (1, N, D)
    desc1_batch = np.expand_dims(desc1, 0) # (1, M, D)

    # 3. 设置模型输入
    try:
        net.setInput(kpts0_norm, "kpts0")
        net.setInput(desc0_batch, "desc0")
        net.setInput(kpts1_norm, "kpts1")
        net.setInput(desc1_batch, "desc1")
    except cv2.error as e:
        print("\n--- 关键错误 ---")
        print("cv2.dnn.setInput 失败。这几乎总是因为描述子维度不匹配。")
        print(f"  你设置的 DESCRIPTOR_DIM 是: {DESCRIPTOR_DIM}")
        print(f"  你的描述子数据形状是: {desc0_batch.shape}")
        print(f"  请检查你的 ONNX 模型 ('{LG_MODEL_PATH}') 是否期望这个维度！")
        print("原始错误信息:", e)
        print("---------------\n")
        sys.exit(1) # 退出程序
 
    
    # 4. 定义输出节点
    output_names = ["matches0", "mscores0", "matches1", "mscores1"]
    
    # 5. 执行推理
    print(f"正在运行 LightGlue 推理... (输入0: {kpts0.shape[0]}点, 输入1: {kpts1.shape[0]}点)")
    outputs = net.forward(output_names)
    print("LightGlue 推理完成。")
    
    # 6. 解析输出
    matches_indices = outputs[0][0] # (N,)
    match_scores = outputs[1][0]    # (N,)
    
    return matches_indices, match_scores

def generate_dummy_features(num_kpts, descriptor_dim, img_shape_hw):
    """
    生成随机的、符合 LightGlue 输入格式的假数据。
    """
    print(f"正在为 shape={img_shape_hw} 生成 {num_kpts} 个 D={descriptor_dim} 的假特征...")
    
    rng = np.random.default_rng()
    
    h, w = img_shape_hw
    
    # 1. 生成 (N, 2) 关键点坐标
    # kpts_x 在 [0, width-1]
    kpts_x = rng.uniform(0, w - 1, (num_kpts, 1))
    # kpts_y 在 [0, height-1]
    kpts_y = rng.uniform(0, h - 1, (num_kpts, 1))
    
    # (N, 2) 格式 (x, y)
    kpts_coords = np.hstack((kpts_x, kpts_y)).astype(np.float32)
    
    # 2. 生成 (N, D) 描述子 (float32)
    # SIFT/SuperPoint 描述子通常是 L2 归一化的
    descriptors = rng.random((num_kpts, descriptor_dim), dtype=np.float32)
    
    # L2 归一化
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors_normalized = descriptors / (norm + 1e-6) # 避免除以零
    
    return kpts_coords, descriptors_normalized


# --- 4. 主执行流程 ---
try:
    # 0. 检查文件是否存在
    if not os.path.exists(LG_MODEL_PATH):
        raise FileNotFoundError(f"模型文件未找到: {LG_MODEL_PATH}。")
    
    # 1. 加载 LightGlue ONNX 模型
    print(f"加载 LightGlue 模型: {LG_MODEL_PATH}...")
    net_lg = cv2.dnn.readNet(LG_MODEL_PATH)
    print("ONNX 模型加载成功！")


    # 2. 生成随机假数据
    # kpts_coords: (N, 2) numpy 数组
    # descriptors: (N, D) numpy 数组
    kpts0_coords, desc0 = generate_dummy_features(
        NUM_KPTS_0, DESCRIPTOR_DIM, IMG_SHAPE_0_HW
    )
    kpts1_coords, desc1 = generate_dummy_features(
        NUM_KPTS_1, DESCRIPTOR_DIM, IMG_SHAPE_1_HW
    )
    
    print("\n--- 假数据形状检查 ---")
    print(f"Kpts 0: {kpts0_coords.shape}, Type: {kpts0_coords.dtype}")
    print(f"Desc 0: {desc0.shape}, Type: {desc0.dtype}")
    print(f"Kpts 1: {kpts1_coords.shape}, Type: {kpts1_coords.dtype}")
    print(f"Desc 1: {desc1.shape}, Type: {desc1.dtype}")
    print("------------------------\n")

    # 4. 运行 LightGlue 进行匹配
    matches_indices, match_scores = run_lightglue(
        net_lg, 
        kpts0_coords, desc0, 
        kpts1_coords, desc1, 
        IMG_SHAPE_0_HW, IMG_SHAPE_1_HW
    )
        
    # 5. 显示结果
    print("\n--- 推理结果 ---")
    print("LightGlue 模型已成功运行！")
    print(f"输出 matches_indices 形状: {matches_indices.shape}") # 应为 (NUM_KPTS_0,)
    print(f"输出 match_scores 形状: {match_scores.shape}")   # 应为 (NUM_KPTS_0,)
    
    # 计算有多少个“假”匹配（因为数据是随机的，这个数字没有意义，但能证明输出了）
    valid_matches = np.sum(matches_indices > -1)
    print(f"在 {NUM_KPTS_0} 个点中，有 {valid_matches} 个点找到了一个匹配（无论置信度如何）。")
    
    print("\n--- 示例输出 (前10个) ---")
    print("索引 (-1 表示未匹配):")
    print(matches_indices[:10].astype(int))
    print("\n置信度:")
    print(match_scores[:10])


except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
except cv2.error as e:
    print(f"OpenCV 错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
    import traceback
    traceback.print_exc()