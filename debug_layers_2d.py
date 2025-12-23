import cv2
import numpy as np
import os

def debug_inference_2d():
    # 1. 这里的路径要是那个 patched (修改过维度) 的模型
    model_path = "model/superpoint_lightglue_fixed_2d_patched.onnx"
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到文件 {model_path}")
        return

    print(f"正在加载 2D 修正版模型: {model_path} ...")
    try:
        net = cv2.dnn.readNet(model_path)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 准备 2D 输入数据 (500, X) 而不是 (1, 500, X)
    print("生成 2D 测试数据...")
    kpts0 = np.random.rand(500, 2).astype(np.float32)
    kpts1 = np.random.rand(500, 2).astype(np.float32)
    desc0 = np.random.rand(500, 256).astype(np.float32)
    desc1 = np.random.rand(500, 256).astype(np.float32)

    net.setInput(kpts0, "kpts0")
    net.setInput(kpts1, "kpts1")
    net.setInput(desc0, "desc0")
    net.setInput(desc1, "desc1")

    # 3. 获取所有层
    layer_names = net.getLayerNames()
    print(f"模型总共有 {len(layer_names)} 层。开始排查...")

    # 4. 逐层运行
    for i, name in enumerate(layer_names):
        try:
            # 这里的 \r 用于刷新同一行，看起来整洁点
            print(f"[{i+1}/{len(layer_names)}] 测试层: {name}", end="\r")
            
            # 运行到这一层
            _ = net.forward(name)
            
        except cv2.error as e:
            # 捕获 OpenCV 错误
            layer_id = net.getLayerId(name)
            layer = net.getLayer(layer_id)
            print(f"\n\n{'='*40}")
            print(f"!!! 找到错误节点 !!!")
            print(f"{'='*40}")
            print(f"层名称: {name}")
            print(f"层类型: {layer.type}")
            print(f"错误信息: {str(e).split('(')[0]}")
            # 尝试打印该层的输入信息（如果有名字的话）
            print(f"{'='*40}")
            break
        except Exception as e:
            print(f"\n\n未知错误: {e}")
            break

    print("\n调试结束。")

if __name__ == "__main__":
    debug_inference_2d()
