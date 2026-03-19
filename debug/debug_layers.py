
import cv2
import numpy as np
import os

def debug_inference():
    # Correct model path, pointing to model/ directory
    model_path = "model/superpoint_lightglue_fixed_overwrite.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: Cannot find file {model_path}")
        print("Please confirm you are in the lightglue_opencv_project directory and the model file is in the model/ folder.")
        return

    print(f"Loading model: {model_path} ...")
    try:
        net = cv2.dnn.readNet(model_path, engine=cv2.dnn.DNN_BACKEND_OPENCV)
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # 准备输入数据
    kpts0 = np.random.rand(1, 500, 2).astype(np.float32)
    kpts1 = np.random.rand(1, 500, 2).astype(np.float32)
    desc0 = np.random.rand(1, 500, 256).astype(np.float32)
    desc1 = np.random.rand(1, 500, 256).astype(np.float32)

    net.setInput(kpts0, "kpts0")
    net.setInput(kpts1, "kpts1")
    net.setInput(desc0, "desc0")
    net.setInput(desc1, "desc1")

    layer_names = net.getLayerNames()
    print(f"Model has {len(layer_names)} layers in total. Starting layer-by-layer inspection...")

    for i, name in enumerate(layer_names):
        try:
            # Print current progress
            print(f"[{i+1}/{len(layer_names)}] Testing layer: {name}", end="\r")
            
            # Try to run to this layer
            _ = net.forward(name)
            
        except cv2.error as e:
            # Get layer information for error reporting
            layer_id = net.getLayerId(name)
            layer = net.getLayer(layer_id)
            print(f"\n\n{'='*40}")
            print(f"!!! Error node found !!!")
            print(f"{'='*40}")
            print(f"Error layer name: {name}")
            print(f"Error layer type: {layer.type}")
            print(f"Error message summary: {str(e).split('(')[0]}") 
            print(f"{'='*40}")
            break
        except Exception as e:
            print(f"\n\nUnknown error occurred: {e}")
            break
            
    print("\nDebug finished.")

if __name__ == "__main__":
    debug_inference()

