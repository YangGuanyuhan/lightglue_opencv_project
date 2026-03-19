import cv2
import numpy as np
import sys
import os

print(f"Using OpenCV version {cv2.__version__}")

# --- 1 Define model path and key parameters ---


LG_MODEL_PATH = 'model/superpoint_lightglue_simplified_2.onnx' 

# - 'superpoint_lightglue.onnx' -> DESCRIPTOR_DIM = 256
DESCRIPTOR_DIM = 256  # LightGlue descriptor dimension

# --- 2 Define dummy data parameters ---
NUM_KPTS_0 = 500  # Number of dummy keypoints for image 0
NUM_KPTS_1 = 500  # Number of dummy keypoints for image 1
IMG_SHAPE_0_HW = (480, 640) # Dummy image 0 H W
IMG_SHAPE_1_HW = (480, 640) # Dummy image 1 H W



def normalize_keypoints(kpts, shape_hw):
    """
    Normalize keypoint coordinates from pixel space to 0 1 range
    """
    # shape_hw is height width
    h, w = shape_hw
    
    kpts_norm = kpts.copy().astype(np.float32)
    
    # kpts 0 is x width
    # kpts 1 is y height
    kpts_norm[:, 0] = kpts_norm[:, 0] / (w - 1)
    kpts_norm[:, 1] = kpts_norm[:, 1] / (h - 1)
    
    # Add batch dimension 1 N 2
    return np.expand_dims(kpts_norm, 0)

def run_lightglue(net, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw):
    """
    Run LightGlue ONNX model
    
    Parameters
        kpts0 kpts1 N 2 and M 2 pixel coordinates
        desc0 desc1 N D and M D descriptors
        shape0_hw shape1_hw image height width
    """
    
    # 1 Normalize keypoints
    kpts0_norm = normalize_keypoints(kpts0, shape0_hw) # (1, N, 2)
    kpts1_norm = normalize_keypoints(kpts1, shape1_hw) # (1, M, 2)
    
    # 2 Expand descriptor dimension 1 N D
    desc0_batch = np.expand_dims(desc0, 0) # (1, N, D)
    desc1_batch = np.expand_dims(desc1, 0) # (1, M, D)

    # 3 Set model inputs
    try:
        net.setInput(kpts0_norm, "kpts0")
        net.setInput(desc0_batch, "desc0")
        net.setInput(kpts1_norm, "kpts1")
        net.setInput(desc1_batch, "desc1")
    except cv2.error as e:
        print("\n--- Key Error ---")
        print("cv2.dnn.setInput failed This is almost always due to descriptor dimension mismatch")
        print(f"  Your set DESCRIPTOR_DIM is {DESCRIPTOR_DIM}")
        print(f"  Your descriptor data shape is {desc0_batch.shape}")
        print(f"  Please check if your ONNX model {LG_MODEL_PATH} expects this dimension")
        print("Original error message", e)
        print("---------------\n")
        sys.exit(1) # Exit program
 
    
    # 4 Define output nodes
    output_names = ["matches0", "mscores0", "matches1", "mscores1"]
    
    # 5 Execute inference
    print(f"Running LightGlue inference Input 0 {kpts0.shape[0]} points Input 1 {kpts1.shape[0]} points")
    outputs = net.forward(output_names)
    print("LightGlue inference complete")
    
    # 6 Parse output
    matches_indices = outputs[0][0] # (N,)
    match_scores = outputs[1][0]    # (N,)
    
    return matches_indices, match_scores

def generate_dummy_features(num_kpts, descriptor_dim, img_shape_hw):
    """
    Generate random dummy data consistent with LightGlue input format
    """
    print(f"Generating {num_kpts} dummy features with D {descriptor_dim} for shape {img_shape_hw}")
    
    rng = np.random.default_rng()
    
    h, w = img_shape_hw
    
    # 1 Generate N 2 keypoint coordinates
    # kpts_x in 0 width-1
    kpts_x = rng.uniform(0, w - 1, (num_kpts, 1))
    # kpts_y in 0 height-1
    kpts_y = rng.uniform(0, h - 1, (num_kpts, 1))
    
    # N 2 format x y
    kpts_coords = np.hstack((kpts_x, kpts_y)).astype(np.float32)
    
    # 2 Generate N D descriptors float32
    # SIFT SuperPoint descriptors are usually L2 normalized
    descriptors = rng.random((num_kpts, descriptor_dim), dtype=np.float32)
    
    # L2 normalization
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors_normalized = descriptors / (norm + 1e-6) # Avoid division by zero
    
    return kpts_coords, descriptors_normalized


# --- 4 Main execution flow ---
try:
    # 0 Check if files exist
    if not os.path.exists(LG_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found {LG_MODEL_PATH}")
    
    # 1 Load LightGlue ONNX model
    print(f"Loading LightGlue model {LG_MODEL_PATH}")
    net_lg = cv2.dnn.readNet(LG_MODEL_PATH)
    print("ONNX model loaded successfully")


    # 2 Generate random dummy data
    # kpts_coords N 2 numpy array
    # descriptors N D numpy array
    kpts0_coords, desc0 = generate_dummy_features(
        NUM_KPTS_0, DESCRIPTOR_DIM, IMG_SHAPE_0_HW
    )
    kpts1_coords, desc1 = generate_dummy_features(
        NUM_KPTS_1, DESCRIPTOR_DIM, IMG_SHAPE_1_HW
    )
    
    print("\n--- Dummy data shape check ---")
    print(f"Kpts 0 {kpts0_coords.shape} Type {kpts0_coords.dtype}")
    print(f"Desc 0 {desc0.shape} Type {desc0.dtype}")
    print(f"Kpts 1 {kpts1_coords.shape} Type {kpts1_coords.dtype}")
    print(f"Desc 1 {desc1.shape} Type {desc1.dtype}")
    print("------------------------\n")

    # 4 Run LightGlue for matching
    matches_indices, match_scores = run_lightglue(
        net_lg, 
        kpts0_coords, desc0, 
        kpts1_coords, desc1, 
        IMG_SHAPE_0_HW, IMG_SHAPE_1_HW
    )
        
    # 5 Display results
    print("\n--- Inference results ---")
    print("LightGlue model ran successfully")
    print(f"Output matches_indices shape {matches_indices.shape}") # Should be NUM_KPTS_0
    print(f"Output match_scores shape {match_scores.shape}")   # Should be NUM_KPTS_0
    
    # Calculate how many dummy matches because data is random this number is meaningless but proves output
    valid_matches = np.sum(matches_indices > -1)
    print(f"Out of {NUM_KPTS_0} points {valid_matches} points found a match regardless of confidence")
    
    print("\n--- Example output first 10 ---")
    print("Index -1 indicates no match")
    print(matches_indices[:10].astype(int))
    print("\nConfidence")
    print(match_scores[:10])


except FileNotFoundError as e:
    print(f"File not found error {e}")
except cv2.error as e:
    print(f"OpenCV error {e}")
except Exception as e:
    print(f"Unknown error occurred {e}")
    import traceback
    traceback.print_exc()