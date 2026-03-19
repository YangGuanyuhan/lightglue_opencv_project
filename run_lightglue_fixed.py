# Used to verify if fixed size LightGlue model runs correctly
import cv2
import numpy as np
import sys
import os

print(f"Using OpenCV version {cv2.__version__}")

# --- 1 Model configuration ---
LG_MODEL_PATH = "model/superpoint_lightglue_fixed.onnx"
DESCRIPTOR_DIM = 256  # LightGlue descriptor dimension
NUM_KPTS_0 = 500
NUM_KPTS_1 = 500
IMG_SHAPE_0_HW = (480, 640)
IMG_SHAPE_1_HW = (480, 640)

# --- 2 Function definitions ---
def normalize_keypoints(kpts, shape_hw):
    """Normalize keypoints to 0 1"""
    h, w = shape_hw
    kpts_norm = kpts.astype(np.float32)
    kpts_norm[:, 0] /= (w - 1)
    kpts_norm[:, 1] /= (h - 1)
    return np.expand_dims(kpts_norm, 0)  # (1, N, 2)

def generate_dummy_features(num_kpts, descriptor_dim, img_shape_hw):
    """Generate dummy keypoints and descriptors"""
    rng = np.random.default_rng()
    h, w = img_shape_hw
    kpts_x = rng.uniform(0, w - 1, (num_kpts, 1))
    kpts_y = rng.uniform(0, h - 1, (num_kpts, 1))
    kpts_coords = np.hstack((kpts_x, kpts_y)).astype(np.float32)

    desc = rng.random((num_kpts, descriptor_dim), dtype=np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-6
    return kpts_coords, desc

def run_lightglue(net, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw):
    """Execute LightGlue inference"""
    kpts0_norm = normalize_keypoints(kpts0, shape0_hw)
    kpts1_norm = normalize_keypoints(kpts1, shape1_hw)
    desc0_batch = np.expand_dims(desc0, 0)
    desc1_batch = np.expand_dims(desc1, 0)

    # Set model inputs
    net.setInput(kpts0_norm, "kpts0")
    net.setInput(kpts1_norm, "kpts1")
    net.setInput(desc0_batch, "desc0")
    net.setInput(desc1_batch, "desc1")

    # Fixed model output names
    output_names = ["matches0", "matches1", "mscores0", "mscores1"]
    print(f"Running LightGlue inference Input points {kpts0.shape[0]} vs {kpts1.shape[0]}")
    outputs = net.forward(output_names)
    print("LightGlue inference complete")

    # Output processing output shape might be 1 N N or empty
    def safe_extract(out):
        if out.size == 0:
            return np.zeros((kpts0.shape[0],), dtype=np.float32)
        return np.squeeze(out)

    matches0 = safe_extract(outputs[0])
    mscores0 = safe_extract(outputs[2])

    return matches0, mscores0

# --- 3 Main execution ---
if __name__ == "__main__":
    try:
        if not os.path.exists(LG_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found {LG_MODEL_PATH}")

        print(f"Loading LightGlue model {LG_MODEL_PATH}")
        net_lg = cv2.dnn.readNetFromONNX(LG_MODEL_PATH)
        net_lg.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net_lg.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Model loaded successfully Backend set to CPU")

        # Generate dummy features
        kpts0, desc0 = generate_dummy_features(NUM_KPTS_0, DESCRIPTOR_DIM, IMG_SHAPE_0_HW)
        kpts1, desc1 = generate_dummy_features(NUM_KPTS_1, DESCRIPTOR_DIM, IMG_SHAPE_1_HW)

        print("\n--- Dummy data check ---")
        print(f"kpts0 {kpts0.shape} desc0 {desc0.shape}")
        print(f"kpts1 {kpts1.shape} desc1 {desc1.shape}")

        # Run model
        matches_indices, match_scores = run_lightglue(
            net_lg, kpts0, desc0, kpts1, desc1, IMG_SHAPE_0_HW, IMG_SHAPE_1_HW
        )

        # Print results
        print("\n--- Inference results ---")
        print(f"matches_indices shape {matches_indices.shape}")
        print(f"match_scores shape {match_scores.shape}")

        valid_matches = np.sum(matches_indices > -1)
        print(f"Out of {NUM_KPTS_0} points {valid_matches} points found matches")

        print("\n前10个匹配结果:")
        print("索引:", matches_indices[:10].astype(int))
        print("置信度:", match_scores[:10])

    except Exception as e:
        print(f"❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
