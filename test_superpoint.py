import cv2
import numpy as np
import os
import sys

print(f"Using OpenCV version {cv2.__version__}")

# --- 1 Configuration paths ---
# Ensure this is the SuperPoint model path you want to test
SP_MODEL_PATH = 'model/superpoint_simplified.onnx' 
TEST_IMAGE_PATH = 'images/image1.jpg'

# --- 2 Core extraction and drawing function ---
def test_superpoint_cv2(model_path, image_path, target_size=(640, 480)):
    if not os.path.exists(model_path):
        print(f"Error Cannot find model file {model_path}")
        sys.exit(1)
        
    if not os.path.exists(image_path):
        print(f"Error Cannot find test image {image_path}")
        sys.exit(1)

    # 1 Load model
    print(f"Loading SuperPoint model OpenCV DNN {model_path}")
    net = cv2.dnn.readNetFromONNX(model_path)

    # 2 Read and preprocess image
    img_bgr = cv2.imread(image_path)
    img_resized = cv2.resize(img_bgr, target_size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Convert to blob 1 1 480 640 and normalize to 0 1
    blob = cv2.dnn.blobFromImage(gray, 1.0/255.0, target_size, swapRB=False, crop=False)
    print(f"Input blob shape {blob.shape}")

    # 3 Set input and inference
    net.setInput(blob)
    
    # Key modification Force fixed output nodes no longer using getUnconnectedOutLayersNames
    # This skips unsupported Gather NonZero operators at the end of the model
    out_names = ["keypoints", "scores", "descriptors"]
    print(f"Forcing output nodes to {out_names}")
    
    try:
        print("Executing forward inference")
        outputs = net.forward(out_names)
        print("Inference complete")
    except cv2.error as e:
        print("\nOpenCV DNN inference failed")
        print(f"Error message {e}")
        print("Hint If node not found error occurs use Netron to confirm node names in ONNX model")
        sys.exit(1)

    # 4 Dynamically parse output tensors
    kpts = None
    scores = None
    
    for i, out in enumerate(outputs):
        print(f"Output {i} {out_names[i]} shape {out.shape}")
        # Look for keypoint tensor with shape like 1 N 2 or N 2
        if len(out.shape) >= 2 and out.shape[-1] == 2:
            # Compatible with 1 N 2 and N 2 cases
            kpts = out[0] if len(out.shape) == 3 else out
        # Look for score tensor one dimensional or 1 N
        elif len(out.shape) == 2 or (len(out.shape) == 3 and out.shape[-1] == 1):
             scores = out.flatten() # Flatten to 1D array N

    if kpts is None:
        raise ValueError("Failed to identify keypoint tensor from model output expected last dimension length 2")

    # 5 Draw keypoints
    print(f"\nPreparing to draw {len(kpts)} extracted keypoints")
    cv_kpts = []
    for i in range(len(kpts)):
        pt_x = float(kpts[i][0])
        pt_y = float(kpts[i][1])
        # Add score threshold filtering if needed
        cv_kpts.append(cv2.KeyPoint(x=pt_x, y=pt_y, size=2))

    # Use DRAW_RICH_KEYPOINTS to draw colored points
    vis_img = cv2.drawKeypoints(
        img_resized, cv_kpts, None, 
        color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return vis_img

# --- 3 Execute test ---
if __name__ == "__main__":
    result_image = test_superpoint_cv2(SP_MODEL_PATH, TEST_IMAGE_PATH)
    
    # Display and save
    cv2.imshow("OpenCV DNN - SuperPoint Features", result_image)
    cv2.imwrite("cv2_superpoint_test.jpg", result_image)
    print("Result saved as cv2_superpoint_test.jpg Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()