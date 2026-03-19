import cv2
import numpy as np
import onnxruntime as ort

def test_aliked(model_path, image_path):
    
    # 1. Initialize ONNX Runtime session
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 2. Inspect the model's expected inputs
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    print(f"Model Input -> Name: '{input_name}', Shape: {input_shape}")
    
    # 3. Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}. Check the path.")
    
    # ALIKED typically expects RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Handle image resizing
    if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        target_h, target_w = input_shape[2], input_shape[3]
        img_resized = cv2.resize(img_rgb, (target_w, target_h))
    else:
        img_resized = cv2.resize(img_rgb, (640, 640))
        
    # Convert image from HWC to NCHW format [1, 3, H, W]
    img_tensor = img_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    img_tensor /= 255.0 
    
    # 4. Run Inference
    print("Running inference...")
    outputs = session.run(None, {input_name: img_tensor})
    
    # 5. Display the results and process outputs
    output_metadata = session.get_outputs()
    print("\n Inference successful! Extracted Outputs:")
    
    # Store outputs in a dictionary mapping name -> data
    out_dict = {}
    for meta, out_data in zip(output_metadata, outputs):
        print(f" 🔹 {meta.name:<15} | Shape: {out_data.shape} | Type: {out_data.dtype}")
        out_dict[meta.name] = out_data

    # 6. Draw Keypoints on the image
    vis_img = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) 
    
    # 寻找 keypoints
    kpts = None
    for name, data in out_dict.items():
        if len(data.shape) in [2, 3] and data.shape[-1] == 2:
            kpts = data
            break
    
    if kpts is not None:
        if len(kpts.shape) == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]
            
        print(f"Drawing {len(kpts)} keypoints on the image...")
        
        # 👇 【新增】打印前5个点，看看坐标到底长什么样！
        print(f"🔍 Sample keypoints (first 5):\n{kpts[:5]}")
        
        # 👇 【新增】自动检测并恢复归一化坐标
        if np.max(kpts) <= 1.0:
            print("💡 Detected normalized coordinates (<= 1.0). Scaling to image size...")
            # 如果坐标域是 [-1, 1]，先将其转换到 [0, 1]
            if np.min(kpts) < 0:
                kpts = (kpts + 1.0) / 2.0
            
            # 乘以图像的宽和高，还原到真实像素坐标
            # 注意：kpts[:, 0] 是 x (对应宽), kpts[:, 1] 是 y (对应高)
            kpts[:, 0] *= vis_img.shape[1] 
            kpts[:, 1] *= vis_img.shape[0] 
            print(f"🔍 Scaled keypoints (first 5):\n{kpts[:5]}")

        # 把每个点画上去，并记录实际画了多少个
        drawn_count = 0
        for pt in kpts:
            x, y = int(pt[0]), int(pt[1])
            # 确保坐标在图像范围内
            if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                cv2.circle(vis_img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                drawn_count += 1
            else:
                pass # 可以用来 debug 越界的点
                
        output_filename = "output_features.jpg"
        cv2.imwrite(output_filename, vis_img)
        print(f"✅ Actually drawn {drawn_count} points within boundaries.")
        print(f"✅ Image saved as '{output_filename}'")
    else:
        print("\n⚠️ 依然没有找到 keypoints。")

if __name__ == "__main__":
    MODEL_PATH = "model/aliked-n16rot-top1k-640.onnx"
    IMAGE_PATH = "image1.jpg" 
    
    test_aliked(MODEL_PATH, IMAGE_PATH)