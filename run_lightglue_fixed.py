# 用来验证固定尺寸的 LightGlue 模型能否够正确运行
import cv2
import numpy as np
import sys
import os

print(f"正在使用 OpenCV 版本: {cv2.__version__}")

# --- 1. 模型配置 ---
LG_MODEL_PATH = "model/superpoint_lightglue_fixed.onnx"
DESCRIPTOR_DIM = 256  # LightGlue 描述子维度
NUM_KPTS_0 = 500
NUM_KPTS_1 = 500
IMG_SHAPE_0_HW = (480, 640)
IMG_SHAPE_1_HW = (480, 640)

# --- 2. 函数定义 ---
def normalize_keypoints(kpts, shape_hw):
    """将关键点归一化到 [0,1]"""
    h, w = shape_hw
    kpts_norm = kpts.astype(np.float32)
    kpts_norm[:, 0] /= (w - 1)
    kpts_norm[:, 1] /= (h - 1)
    return np.expand_dims(kpts_norm, 0)  # (1, N, 2)

def generate_dummy_features(num_kpts, descriptor_dim, img_shape_hw):
    """生成假特征点和描述子"""
    rng = np.random.default_rng()
    h, w = img_shape_hw
    kpts_x = rng.uniform(0, w - 1, (num_kpts, 1))
    kpts_y = rng.uniform(0, h - 1, (num_kpts, 1))
    kpts_coords = np.hstack((kpts_x, kpts_y)).astype(np.float32)

    desc = rng.random((num_kpts, descriptor_dim), dtype=np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6
    return kpts_coords, desc

def run_lightglue(net, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw):
    """执行 LightGlue 推理"""
    kpts0_norm = normalize_keypoints(kpts0, shape0_hw)
    kpts1_norm = normalize_keypoints(kpts1, shape1_hw)
    desc0_batch = np.expand_dims(desc0, 0)
    desc1_batch = np.expand_dims(desc1, 0)

    # 设置模型输入
    net.setInput(kpts0_norm, "kpts0")
    net.setInput(kpts1_norm, "kpts1")
    net.setInput(desc0_batch, "desc0")
    net.setInput(desc1_batch, "desc1")

    # 模型的输出名称（固定）
    output_names = ["matches0", "matches1", "mscores0", "mscores1"]
    print(f"运行 LightGlue 推理... 输入点数: {kpts0.shape[0]} vs {kpts1.shape[0]}")
    outputs = net.forward(output_names)
    print("LightGlue 推理完成。")

    # 输出处理：模型输出形状可能为 (1, N)，(N,) 或空
    def safe_extract(out):
        if out.size == 0:
            return np.zeros((kpts0.shape[0],), dtype=np.float32)
        return np.squeeze(out)

    matches0 = safe_extract(outputs[0])
    mscores0 = safe_extract(outputs[2])

    return matches0, mscores0

# --- 3. 主执行 ---
if __name__ == "__main__":
    try:
        if not os.path.exists(LG_MODEL_PATH):
            raise FileNotFoundError(f"模型文件未找到: {LG_MODEL_PATH}")

        print(f"加载 LightGlue 模型: {LG_MODEL_PATH}")
        net_lg = cv2.dnn.readNetFromONNX(LG_MODEL_PATH)
        net_lg.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_lg.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("✅ 模型加载成功，后端设置为 CPU")

        # 生成假特征
        kpts0, desc0 = generate_dummy_features(NUM_KPTS_0, DESCRIPTOR_DIM, IMG_SHAPE_0_HW)
        kpts1, desc1 = generate_dummy_features(NUM_KPTS_1, DESCRIPTOR_DIM, IMG_SHAPE_1_HW)

        print("\n--- 假数据检查 ---")
        print(f"kpts0: {kpts0.shape}, desc0: {desc0.shape}")
        print(f"kpts1: {kpts1.shape}, desc1: {desc1.shape}")

        # 运行模型
        matches_indices, match_scores = run_lightglue(
            net_lg, kpts0, desc0, kpts1, desc1, IMG_SHAPE_0_HW, IMG_SHAPE_1_HW
        )

        # 打印结果
        print("\n--- 推理结果 ---")
        print(f"matches_indices 形状: {matches_indices.shape}")
        print(f"match_scores 形状: {match_scores.shape}")

        valid_matches = np.sum(matches_indices > -1)
        print(f"在 {NUM_KPTS_0} 个点中，有 {valid_matches} 个点找到匹配。")

        print("\n前10个匹配结果:")
        print("索引:", matches_indices[:10].astype(int))
        print("置信度:", match_scores[:10])

    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
