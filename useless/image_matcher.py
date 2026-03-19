import cv2
import numpy as np

print(f"正在使用 OpenCV 版本: {cv2.__version__}")

# --- 1. 定义模型路径 ---
# 你需要两个模型：SuperPoint (提取) 和 LightGlue (匹配)
# 
LG_MODEL_PATH = 'model/superpoint_lightglue.onnx'
SP_MODEL_PATH = 'model/superpoint_simplified.onnx'

# --- 2. 定义图像路径 ---
# 你需要两张用于测试的图片
IMAGE_PATH_1 = 'image1.jpg'
IMAGE_PATH_2 = 'image2.jpg'

# --- 3. 定义辅助函数 ---

def load_image(image_path, target_size=(640, 480)):
    """加载并预处理图像以便 SuperPoint 使用"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
        
    # SuperPoint 通常需要 640x480 的输入
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 归一化并添加批处理和通道维度 (1, 1, H, W)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_normalized, 0), 0)
    return img, img_tensor, img_resized.shape

def normalize_keypoints(kpts, shape_hw):
    """将关键点坐标归一化到 [0, 1] 范围"""
    # shape_hw 是 (height, width)
    h, w = shape_hw
    
    # 我们创建一个副本
    kpts_norm = kpts.copy().astype(np.float32)
    
    # kpts[:, 0] 是 x 坐标, kpts[:, 1] 是 y 坐标
    # 归一化到 [0, 1]
    kpts_norm[:, 0] = kpts_norm[:, 0] / (w - 1)
    kpts_norm[:, 1] = kpts_norm[:, 1] / (h - 1)
    
    # 添加批处理维度 (1, N, 2)
    return np.expand_dims(kpts_norm, 0)

def run_superpoint(net, img_tensor):
    """运行 SuperPoint 模型"""
    # 
    # 
    # 这里的输入/输出名称 ("image", "keypoints", "scores", "descriptors")
    # 可能是特定于你的 SuperPoint ONNX 模型的。
    # 如果出错，请使用 Netron 检查你的 "superpoint.onnx" 文件！
    #
    
    # 1. 设置输入
    # 假设 SuperPoint 的输入节点名为 "image"
    net.setInput(img_tensor, "image")
    
    # 2. 定义输出节点
    # 假设输出节点名为 "keypoints", "scores", "descriptors"
    output_names = ["keypoints", "scores", "descriptors"]
    
    # 3. 执行推理
    try:
        outputs = net.forward(output_names)
    except cv2.error as e:
        print(f"OpenCV 模型推理失败: {e}")
        # 尝试备用输出名称
        print("正在尝试备用输出名称: 'keypoints', 'scores', 'descriptors'")
        output_names = ['keypoints', 'scores', 'descriptors']
        try:
            outputs = net.forward(output_names)
        except cv2.error as e2:
            print(f"使用备用名称再次失败: {e2}")
            raise
    
    # 4. 解析输出
    # 打印输出形状以进行调试
    print(f"SuperPoint outputs shape: {[o.shape for o in outputs]}")
    
    # [0] -> keypoints, 形状 (1, N, 2)
    # [1] -> scores, 形状 (1, N)
    # [2] -> descriptors, 形状 (1, N, 256)
    kpts = outputs[0][0]  # (N, 2)
    scores = outputs[1][0] # (N,)
    descriptors = outputs[2][0] # (N, 256)
    
    return kpts, scores, descriptors

def run_lightglue(net, kpts0, desc0, kpts1, desc1, shape0, shape1):
    """运行 LightGlue 模型"""
    
    # 1. 归一化关键点
    kpts0_norm = normalize_keypoints(kpts0, shape0) # (1, N, 2)
    kpts1_norm = normalize_keypoints(kpts1, shape1) # (1, M, 2)
    
    # 2. 扩展描述子维度 (1, N, 256)
    desc0_batch = np.expand_dims(desc0, 0) # (1, N, 256)
    desc1_batch = np.expand_dims(desc1, 0) # (1, M, 256)

  
    # - kpts0: [1, 0, 2]  <- kpts0_norm (1, N, 2) 匹配
    # - kpts1: [1, 0, 2]  <- kpts1_norm (1, M, 2) 匹配
    # - desc0: [1, 0, 256] <- desc0_batch (1, N, 256) 匹配
    # - desc1: [1, 0, 256] <- desc1_batch (1, M, 256) 匹配
    #
    net.setInput(kpts0_norm, "kpts0")
    net.setInput(desc0_batch, "desc0")
    net.setInput(kpts1_norm, "kpts1")
    net.setInput(desc1_batch, "desc1")
    
    # 3. 定义输出节点
    # 假设输出为 "matches0" 和 "mscores0"
    output_names = ["matches0", "mscores0"]
    
    # 4. 执行推理
    outputs = net.forward(output_names)
    
    # 5. 解析输出
    # matches0 形状 (1, N), 值为 -1 (未匹配) 或 kpts1 中的索引
    # mscores0 形状 (1, N), 匹配的置信度
    matches_indices = outputs[0][0] # (N,)
    match_scores = outputs[1][0]  # (N,)
    
    return matches_indices, match_scores


def filter_matches(kpts0, kpts1, matches_indices, match_scores, score_threshold=0.8):
    """根据置信度过滤匹配"""
    
    good_matches = []
    points1 = []
    points2 = []
    
    for i, match_index in enumerate(matches_indices):
        if match_index >= 0 and match_scores[i] >= score_threshold:
            # 这是一个好的匹配
            # i 是 kpts0 的索引
            # match_index 是 kpts1 的索引
            
            # 创建 DMatch 对象
            # DMatch(queryIdx, trainIdx, distance)
            # 我们用 (1.0 - score) 作为 distance
            match = cv2.DMatch(i, int(match_index), 1.0 - match_scores[i])
            
            good_matches.append(match)
            points1.append(kpts0[i])
            points2.append(kpts1[int(match_index)])
            
    # 将关键点转换为 cv2.KeyPoint 对象以便绘制
    cv_kpts1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts0]
    cv_kpts2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in kpts1]
            
    return good_matches, cv_kpts1, cv_kpts2, np.array(points1), np.array(points2)


def draw_matches(img0, kpts0, img1, kpts1, matches):
    """绘制匹配结果"""
    
    # img0 和 img1 是原始的、未缩放的灰度图
    # kpts0 和 kpts1 是在 *缩放后* 的图像上检测到的
    
    # 我们需要在缩放后的图像上绘制
    # 注意：这里我们重新加载图像以获取用于绘制的缩放后版本
    img0_resized, _, shape0 = load_image(IMAGE_PATH_1, (640, 480))
    img1_resized, _, shape1 = load_image(IMAGE_PATH_2, (640, 480))
    
    # 将灰度图转为 BGR 以便绘制彩色线条
    img0_bgr = cv2.cvtColor(img0_resized, cv2.COLOR_GRAY2BGR)
    img1_bgr = cv2.cvtColor(img1_resized, cv2.COLOR_GRAY2BGR)
    
    match_img = cv2.drawMatches(
        img0_bgr, kpts0, 
        img1_bgr, kpts1, 
        matches, 
        None, 
        matchColor=(0, 255, 0), # 绿色
        singlePointColor=(0, 0, 255), # 红色
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return match_img

# --- 4. 主执行流程 ---
try:
    # 1. 加载模型
    print("加载 SuperPoint 模型...")
    net_sp = cv2.dnn.readNetFromONNX(SP_MODEL_PATH)
    # print(net_sp.getLayerNames())
    print("加载 LightGlue 模型...")
    net_lg = cv2.dnn.readNetFromONNX(LG_MODEL_PATH)
    print("ONNX 模型加载成功！")
    # print(net_lg.getLayerNames())

    # 2. 设置后端 (使用 CPU)
    net_sp.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net_sp.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net_lg.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net_lg.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("计算后端设置为 CPU。")

    # 3. 加载并处理图像
    print(f"加载图像 {IMAGE_PATH_1} 和 {IMAGE_PATH_2}...")
    # original_imgX 是原始灰度图, img_tensorX 是预处理后的, shapeX 是 (H, W)
    original_img0, img_tensor0, shape0 = load_image(IMAGE_PATH_1, (640, 480))
    original_img1, img_tensor1, shape1 = load_image(IMAGE_PATH_2, (640, 480))
    
    # 4. 运行 SuperPoint 提取特征
    print("正在运行 SuperPoint (图像 1)...")
    kpts0, scores0, desc0 = run_superpoint(net_sp, img_tensor0)
    print("正在运行 SuperPoint (图像 2)...")
    kpts1, scores1, desc1 = run_superpoint(net_sp, img_tensor1)
    print(f"图像1 提取到 {len(kpts0)} 个特征点")
    print(f"图像2 提取到 {len(kpts1)} 个特征点")

    # 5. 运行 LightGlue 进行匹配
    print("正在运行 LightGlue...")
    matches_indices, match_scores = run_lightglue(net_lg, kpts0, desc0, kpts1, desc1, shape0, shape1)
    
    # 6. 过滤并格式化匹配结果
    # 阈值 (0.0 ~ 1.0), 越高越严格
    MATCH_THRESHOLD = 0.8 
    good_matches, cv_kpts0, cv_kpts1, pts0, pts1 = filter_matches(
        kpts0, kpts1, matches_indices, match_scores, MATCH_THRESHOLD
    )
    print(f"找到 {len(good_matches)} 个高质量匹配 (置信度 > {MATCH_THRESHOLD})")
    
    # 7. (可选) 计算单应性矩阵 (Homography)
    if len(good_matches) > 4:
        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        print("计算单应性矩阵 H:\n", H)

    # 8. 绘制匹配结果
    print("正在绘制匹配结果...")
    # 注意：我们将原始图像(original_img0)传递给 draw_matches，
    # 但函数 *内部* 应该重新加载和缩放它们，以匹配 kpts 的坐标系。
    # 或者，我们应该传递缩放后的图像。
    # 让我们修正 draw_matches 函数的调用（已在 draw_matches 内部修正）
    result_image = draw_matches(original_img0, cv_kpts0, original_img1, cv_kpts1, good_matches)
    
    # 9. 显示结果
    cv2.imshow("SuperPoint + LightGlue 匹配结果", result_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
except cv2.error as e:
    print(f"OpenCV 加载模型失败: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
    import traceback
    traceback.print_exc()