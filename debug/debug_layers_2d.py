import cv2
import numpy as np
import os

def debug_inference_2d():
    # 1. Here the path should be that patched model with modified dimensions
    model_path = "model/superpoint_lightglue_fixed_2d_patched.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error cannot find file {model_path}")
        return

    print(f"Loading 2D corrected model {model_path} ")
    try:
        net = cv2.dnn.readNet(model_path)
    except Exception as e:
        print(f"Failed to load model {e}")
        return

    # 2. Prepare 2D input data (500, X) instead of (1, 500, X)
    print("Generating 2D test data")
    kpts0 = np.random.rand(500, 2).astype(np.float32)
    kpts1 = np.random.rand(500, 2).astype(np.float32)
    desc0 = np.random.rand(500, 256).astype(np.float32)
    desc1 = np.random.rand(500, 256).astype(np.float32)

    net.setInput(kpts0, "kpts0")
    net.setInput(kpts1, "kpts1")
    net.setInput(desc0, "desc0")
    net.setInput(desc1, "desc1")

    # 3. Get all layers
    layer_names = net.getLayerNames()
    print(f"Model has {len(layer_names)} layers starting troubleshooting")

    # 4. Run layer by layer
    for i, name in enumerate(layer_names):
        try:
            # Here \r is used to refresh the same line for a clean look
            print(f"[{i+1}/{len(layer_names)}] Testing layer {name}", end="\r")
            
            # Run up to this layer
            _ = net.forward(name)
            
        except cv2.error as e:
            # Catch OpenCV error
            layer_id = net.getLayerId(name)
            layer = net.getLayer(layer_id)
            print(f"\n\n{'='*40}")
            print(f"!!! Error node found !!!")
            print(f"{'='*40}")
            print(f"Layer name {name}")
            print(f"Layer type {layer.type}")
            print(f"Error message {str(e).split('(')[0]}")
            # Try to print input information for this layer if available
            print(f"{'='*40}")
            break
        except Exception as e:
            print(f"\n\nUnknown error {e}")
            break

    print("\nDebug finished")

if __name__ == "__main__":
    debug_inference_2d()
