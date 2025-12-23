import cv2
import numpy as np

print(f"正在使用 OpenCV 版本: {cv2.__version__}")

# --- 1. 定义模型路径 ---
# (我们暂时不再需要 ONNX 模型)
# LG_MODEL_PATH = 'model/superpoint_lightglue_simplified_2.onnx'
# SP_MODEL_PATH = 'model/superpoint.onnx'

# --- 2. 定义图像路径 ---
IMAGE_PATH_1 = 'image1.jpg'
IMAGE_PATH_2 = 'image2.jpg'

# --- 3. 定义辅助函数 ---

def extract_sift_features(image_path):
    """
    使用 SIFT 提取关键点和描述子
    """
    # 读取灰度图用于特征提取
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
        
    # 读取 BGR 图像用于后续绘制
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 1. 初始化 SIFT 检测器
    # 注意：在某些 OpenCV 版本中 (例如 contrib)，你可能需要使用 cv2.xfeatures2d.SIFT_create()
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("警告：cv2.SIFT_create() 不可用，尝试 cv2.xfeatures2d.SIFT_create()")
        try:
            sift = cv2.xfeatures2d.SIFT_create()
        except Exception as e:
            print("错误：无法初始化 SIFT。请确保你安装了 opencv-contrib-python。")
            raise e
    
    # 2. 检测关键点并计算描述子
    # kpts 是 cv2.KeyPoint 对象的列表
    # descs 是 (N, 128) 的 numpy 数组
    kpts, descs = sift.detectAndCompute(img_gray, None)
    
    # 3. 将关键点坐标转换为 (N, 2) 的 numpy 数组 (为了兼容后续的 findHomography)
    kpts_np = np.array([kp.pt for kp in kpts], dtype=np.float32)
    
    return img_bgr, kpts, kpts_np, descs

def match_features_bf(desc0, desc1):
    """
    使用 Brute-Force (BF) 匹配器和 KNN 进行匹配
    """
    # 1. 初始化 BFMatcher
    # NORM_L2 适用于 SIFT (浮点数描述子)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # 2. 使用 k-NN 匹配 (k=2)，以便后续使用 Lowe's Ratio Test
    # raw_matches 是一个列表，每个元素包含 k=2 个最佳匹配
    raw_matches = bf.knnMatch(desc0, desc1, k=2)
    
    return raw_matches

def filter_matches_ratio_test(kpts0_np, kpts1_np, raw_matches, ratio_threshold=0.75):
    """
    使用 Lowe's Ratio Test 过滤原始匹配
    """
    good_matches = [] # 存储 cv2.DMatch 对象
    points1 = []      # 存储匹配上的点坐标
    points2 = []
    
    for m, n in raw_matches:
        # m 是最佳匹配, n 是次佳匹配
        # 如果最佳匹配的距离远小于次佳匹配的距离，说明这个匹配很“独特”
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            
            # m.queryIdx 是 desc0 (图像1) 中的索引
            # m.trainIdx 是 desc1 (图像2) 中的索引
            points1.append(kpts0_np[m.queryIdx])
            points2.append(kpts1_np[m.trainIdx])
            
    return good_matches, np.array(points1), np.array(points2)


def draw_matches(img_bgr0, kpts0, img_bgr1, kpts1, matches):
    """绘制匹配结果 (使用 BGR 图像)"""
    
    # kpts0 和 kpts1 是 cv2.KeyPoint 对象的列表
    # matches 是 cv2.DMatch 对象的列表
    
    match_img = cv2.drawMatches(
        img_bgr0, kpts0, 
        img_bgr1, kpts1, 
        matches, 
        None, 
        matchColor=(0, 255, 0), # 绿色
        singlePointColor=(0, 0, 255), # 红色
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return match_img

# --- 4. 主执行流程 ---
try:
    # 1. (不再加载 ONNX 模型)
    print("使用 OpenCV SIFT 提取器和 BFMatcher")

    # 2. (不再设置 ONNX 后端)

    # 3. & 4. 加载图像并提取特征
    print(f"加载图像 {IMAGE_PATH_1} 并提取 SIFT 特征...")
    # img_bgrX 用于绘制, kptsX 是 cv2.KeyPoint 列表, kpts_npX 是 (N,2) 坐标, descX 是 (N, 128) 描述子
    img_bgr0, kpts0, kpts0_np, desc0 = extract_sift_features(IMAGE_PATH_1)
    
    print(f"加载图像 {IMAGE_PATH_2} 并提取 SIFT 特征...")
    img_bgr1, kpts1, kpts1_np, desc1 = extract_sift_features(IMAGE_PATH_2)

    print(f"图像1 提取到 {len(kpts0)} 个特征点")
    print(f"图像2 提取到 {len(kpts1)} 个特征点")

    # 5. 运行 Brute-Force 匹配
    print("正在运行 BFMatcher (k=2)...")
    raw_matches = match_features_bf(desc0, desc1)
    
    # 6. 过滤匹配 (Lowe's Ratio Test)
    RATIO_THRESHOLD = 0.75 
    good_matches, pts0, pts1 = filter_matches_ratio_test(
        kpts0_np, kpts1_np, raw_matches, RATIO_THRESHOLD
    )
    print(f"找到 {len(good_matches)} 个高质量匹配 (Ratio Test < {RATIO_THRESHOLD})")
    
    # 7. (可选) 计算单应性矩阵 (Homography)
    if len(good_matches) > 4:
        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        print("计算单应性矩阵 H:\n", H)

    # 8. 绘制匹配结果
    print("正在绘制匹配结果...")
    result_image = draw_matches(img_bgr0, kpts0, img_bgr1, kpts1, good_matches)
    
    # 9. 显示结果
    cv2.imshow("SIFT + BFMatcher 匹配结果", result_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
except cv2.error as e:
    print(f"OpenCV 错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
    import traceback
    traceback.print_exc()