import cv2
import numpy as np
import onnxruntime as ort
import os
import sys

print(f"Using OpenCV version {cv2.__version__}")
print(f"Using OnnxRuntime version {ort.__version__}")

# --- 1. Define model paths and parameters ---
SP_MODEL_PATH = 'model/superpoint.onnx' # Since ORT is used the original dynamic SuperPoint can also run
LG_MODEL_PATH = 'model/superpoint_lightglue_simplified_2.onnx'

IMG_0_PATH = 'images/image1.jpg' 
IMG_1_PATH = 'images/image2.jpg'

DESCRIPTOR_DIM = 256
MATCH_CONFIDENCE_THRESHOLD = 0.2


# --- 2. Core inference functions ---

def extract_superpoint_ort(sp_session, img_bgr, target_size=(640, 480)):
    """
    Use OnnxRuntime to run SuperPoint for keypoint and descriptor extraction
    """
    # Resize and convert to grayscale
    img_resized = cv2.resize(img_bgr, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize and add dimensions (1, 1, H, W)
    img_normalized = gray.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_normalized, 0), 0)
    
    # Get ORT input name and run inference
    input_name = sp_session.get_inputs()[0].name
    outputs = sp_session.run(None, {input_name: img_tensor})
    
    # Dynamically parse outputs (look for Nx2 keypoints and Nx256 descriptors)
    kpts, desc = None, None
    for out in outputs:
        out = np.array(out)
        if len(out.shape) == 3 and out.shape[-1] == 2:
            kpts = out[0]  # (N, 2)
        elif len(out.shape) == 3 and out.shape[-1] == DESCRIPTOR_DIM:
            desc = out[0]  # (N, 256)
            
    if kpts is None or desc is None:
        raise ValueError("Cannot parse required data from SuperPoint ORT output")
        
    return kpts, desc


def run_lightglue_cv2(net_lg, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw):
    """
    Use OpenCV DNN to run LightGlue for matching
    """
    # Internal normalization function
    def normalize_keypoints(kpts, shape_hw):
        h, w = shape_hw
        kpts_norm = kpts.copy().astype(np.float32)
        kpts_norm[:, 0] = kpts_norm[:, 0] / (w - 1)
        kpts_norm[:, 1] = kpts_norm[:, 1] / (h - 1)
        return np.expand_dims(kpts_norm, 0) # (1, N, 2)

    kpts0_norm = normalize_keypoints(kpts0, shape0_hw)
    kpts1_norm = normalize_keypoints(kpts1, shape1_hw)
    
    desc0_batch = np.expand_dims(desc0, 0) # (1, N, 256)
    desc1_batch = np.expand_dims(desc1, 0)

    # Feed into OpenCV DNN
    net_lg.setInput(kpts0_norm, "kpts0")
    net_lg.setInput(desc0_batch, "desc0")
    net_lg.setInput(kpts1_norm, "kpts1")
    net_lg.setInput(desc1_batch, "desc1")
    
    output_names = ["matches0", "mscores0", "matches1", "mscores1"]
    outputs = net_lg.forward(output_names)
    
    matches_indices = outputs[0][0] # (N,)
    match_scores = outputs[1][0]    # (N,)
    
    return matches_indices, match_scores


# --- 3. Result visualization ---

def draw_matches(img0, img1, kpts0, kpts1, matches_indices, match_scores, threshold):
    """Convert data format and draw OpenCV matches"""
    kp0 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpts0]
    kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpts1]
    
    dmatches = []
    for i, idx1 in enumerate(matches_indices):
        idx1 = int(idx1)
        score = float(match_scores[i])
        
        if idx1 > -1 and score > threshold:
            # The smaller the distance the better so use 1.0 - score
            dmatches.append(cv2.DMatch(_queryIdx=i, _trainIdx=idx1, _distance=1.0 - score))
            
    print(f"Number of valid matching points confidence > {threshold} {len(dmatches)}")

    vis_img = cv2.drawMatches(
        img0, kp0, img1, kp1, dmatches, None, 
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis_img


# --- 4. Main execution flow ---

if __name__ == "__main__":
    if not os.path.exists(LG_MODEL_PATH) or not os.path.exists(SP_MODEL_PATH):
        print("Error cannot find model files")
        sys.exit(1)
        
    # 1. Mixed model loading
    print("Loading SuperPoint model engine OnnxRuntime")
    session_sp = ort.InferenceSession(SP_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    print("Loading LightGlue model engine OpenCV DNN")
    net_lg = cv2.dnn.readNet(LG_MODEL_PATH)

    # 2. Load and preprocess images
    img0 = cv2.imread(IMG_0_PATH)
    img1 = cv2.imread(IMG_1_PATH)
    
    target_size = (640, 480)
    img0 = cv2.resize(img0, target_size)
    img1 = cv2.resize(img1, target_size)
    
    shape0_hw = img0.shape[:2]
    shape1_hw = img1.shape[:2]

    # 3. ORT feature extraction
    print("\n1/2 Extracting features SuperPoint")
    kpts0, desc0 = extract_superpoint_ort(session_sp, img0, target_size)
    kpts1, desc1 = extract_superpoint_ort(session_sp, img1, target_size)
    print(f" -> Extracted {kpts0.shape[0]} features from image 1 and {kpts1.shape[0]} features from image 2")

    # 4. OpenCV DNN feature matching
    print("\n2/2 Matching features LightGlue")
    matches_indices, match_scores = run_lightglue_cv2(
        net_lg, kpts0, desc0, kpts1, desc1, shape0_hw, shape1_hw
    )

    # 5. Visualize and save
    print("\nDrawing results")
    result_img = draw_matches(img0, img1, kpts0, kpts1, matches_indices, match_scores, MATCH_CONFIDENCE_THRESHOLD)

    cv2.imshow("SuperPoint(ORT) + LightGlue(OpenCV) Matches", result_img)
    cv2.imwrite("hybrid_matches_result.jpg", result_img)
    print("Successfully completed result saved press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()