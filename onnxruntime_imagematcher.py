import cv2
import numpy as np
import onnxruntime as ort
import os
import traceback

print(f"正在使用 OpenCV 版本: {cv2.__version__}")
print(f"正在使用 OnnxRuntime 版本: {ort.__version__}")

# --- 1. 定义模型路径 ---
LG_MODEL_PATH = 'model/superpoint_lightglue.onnx'
SP_MODEL_PATH = 'model/superpoint.onnx'

# --- 2. 定义图像路径 ---
IMAGE_PATH_1 = 'image1.jpg'
IMAGE_PATH_2 = 'image2.jpg'

# --- 3. 辅助函数 ---
def load_image_as_color_and_tensor(image_path, target_size=(1280, 720)):
    """
    返回:
      color_resized_bgr: 用于绘图的彩色图 (H, W, 3) BGR
      img_tensor: 供 SuperPoint 的灰度归一化张量 (1,1,H,W) float32
      shape_hw: (H, W) 用于归一化关键点等
    """
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    # 按指定大小 resize（和模型推理时一致）
    color_resized = cv2.resize(color, target_size, interpolation=cv2.INTER_AREA)
    gray_resized = cv2.cvtColor(color_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = gray_resized.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(np.expand_dims(img_normalized, 0), 0).astype(np.float32)  # (1,1,H,W)
    h, w = gray_resized.shape
    return color_resized, img_tensor, (h, w)

def normalize_keypoints(kpts, shape_hw):
    """ 将关键点坐标 (N,2) 归一化到 [0,1] 范围，返回 (1,N,2) """
    h, w = shape_hw
    kpts_norm = kpts.copy().astype(np.float32)
    # 假设 kpts 的格式是 (x, y)
    kpts_norm[:, 0] = kpts_norm[:, 0] / (w - 1)
    kpts_norm[:, 1] = kpts_norm[:, 1] / (h - 1)
    return np.expand_dims(kpts_norm, 0)

# SuperPoint 推理 (ORT)
def run_superpoint_ort(sp_session, img_tensor):
    input_name = sp_session.get_inputs()[0].name
    input_feed = {input_name: img_tensor}
    # 输出名根据模型而定，这里使用你给的名字
    output_names = ["keypoints", "scores", "descriptors"]
    outputs = sp_session.run(output_names, input_feed)
    kpts = outputs[0][0]        # (N,2)
    scores = outputs[1][0]      # (N,)
    descriptors = outputs[2][0] # (N, D) 例如 256
    return kpts, scores, descriptors

# LightGlue 推理 (ORT)
def run_lightglue_ort(lg_session, kpts0, desc0, kpts1, desc1, shape0, shape1):
    kpts0_norm = normalize_keypoints(kpts0, shape0)
    kpts1_norm = normalize_keypoints(kpts1, shape1)
    desc0_batch = np.expand_dims(desc0, 0).astype(np.float32)
    desc1_batch = np.expand_dims(desc1, 0).astype(np.float32)
    input_feed = {
        "kpts0": kpts0_norm,
        "desc0": desc0_batch,
        "kpts1": kpts1_norm,
        "desc1": desc1_batch
    }
    output_names = ["matches0", "mscores0", "matches1", "mscores1"]
    outputs = lg_session.run(output_names, input_feed)
    matches_indices = outputs[0][0]  # (N,)
    match_scores = outputs[1][0]     # (N,)
    return matches_indices, match_scores

def filter_matches(kpts0, kpts1, matches_indices, match_scores, score_threshold=0.8):
    """
    根据置信度过滤匹配，返回:
      good_matches (cv2.DMatch 列表),
      cv_kpts0_list, cv_kpts1_list (供 drawMatches 使用),
      pts0 (Nx2), pts1 (Nx2) - 对应的点坐标数组 (float)
    """
    good_matches = []
    pts0 = []
    pts1 = []

    for i, match_index in enumerate(matches_indices):
        if match_index >= 0 and match_scores[i] >= score_threshold:
            # match: queryIdx=i (kpts0), trainIdx=match_index (kpts1)
            # DMatch(signature: queryIdx, trainIdx, imgIdx, distance)
            # 我们把 distance 设为 1.0 - score 以便可视化排序
            d = float(1.0 - match_scores[i])
            m = cv2.DMatch(_queryIdx=int(i), _trainIdx=int(match_index), _imgIdx=0, _distance=d)
            good_matches.append(m)
            pts0.append(kpts0[i])
            pts1.append(kpts1[int(match_index)])

    # 构建 cv2.KeyPoint 列表（注意 x,y 顺序）
    cv_kpts0_list = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in kpts0]
    cv_kpts1_list = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in kpts1]

    if len(pts0) == 0:
        pts0_arr = np.zeros((0, 2), dtype=np.float32)
        pts1_arr = np.zeros((0, 2), dtype=np.float32)
    else:
        pts0_arr = np.array(pts0, dtype=np.float32)
        pts1_arr = np.array(pts1, dtype=np.float32)

    return good_matches, cv_kpts0_list, cv_kpts1_list, pts0_arr, pts1_arr

# ---------- 主流程 ----------
try:
    # 检查文件
    if not os.path.exists(SP_MODEL_PATH):
        raise FileNotFoundError(f"SuperPoint 模型未找到: {SP_MODEL_PATH}")
    if not os.path.exists(LG_MODEL_PATH):
        raise FileNotFoundError(f"LightGlue 模型未找到: {LG_MODEL_PATH}")
    if not os.path.exists(IMAGE_PATH_1):
        raise FileNotFoundError(f"图像未找到: {IMAGE_PATH_1}")
    if not os.path.exists(IMAGE_PATH_2):
        raise FileNotFoundError(f"图像未找到: {IMAGE_PATH_2}")

    # 加载模型
    print("加载 SuperPoint 模型 (OnnxRuntime)...")
    session_sp = ort.InferenceSession(SP_MODEL_PATH, providers=['CPUExecutionProvider'])
    print("加载 LightGlue 模型 (OnnxRuntime)...")
    session_lg = ort.InferenceSession(LG_MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX 模型加载成功！")

    # 加载并准备图像（返回彩色用于绘图，tensor 用于推理）
    print(f"加载图像 {IMAGE_PATH_1} 和 {IMAGE_PATH_2} ...")
    color0, img_tensor0, shape0 = load_image_as_color_and_tensor(IMAGE_PATH_1, (640, 480))
    color1, img_tensor1, shape1 = load_image_as_color_and_tensor(IMAGE_PATH_2, (640, 480))

    # 运行 SuperPoint
    print("正在运行 SuperPoint (图像 1)...")
    kpts0, scores0, desc0 = run_superpoint_ort(session_sp, img_tensor0)
    print("正在运行 SuperPoint (图像 2)...")
    kpts1, scores1, desc1 = run_superpoint_ort(session_sp, img_tensor1)
    print(f"图像1 提取到 {len(kpts0)} 个特征点")
    print(f"图像2 提取到 {len(kpts1)} 个特征点")

    # ---------- 先显示关键点（在彩色图像上） ----------
    # 构造 cv2.KeyPoint 列表用于 drawKeypoints / drawMatches
    cv_kpts0_for_draw = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 2) for pt in kpts0]
    cv_kpts1_for_draw = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 2) for pt in kpts1]

    vis0 = cv2.drawKeypoints(color0, cv_kpts0_for_draw, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis1 = cv2.drawKeypoints(color1, cv_kpts1_for_draw, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Image 1 - SuperPoint Keypoints", vis0)
    cv2.imshow("Image 2 - SuperPoint Keypoints", vis1)
    print("已显示特征点，按任意键继续显示匹配结果...")
    cv2.waitKey(0)
    cv2.destroyWindow("Image 1 - SuperPoint Keypoints")
    cv2.destroyWindow("Image 2 - SuperPoint Keypoints")

    # ---------- 运行 LightGlue 并过滤匹配 ----------
    print("正在运行 LightGlue...")
    matches_indices, match_scores = run_lightglue_ort(session_lg, kpts0, desc0, kpts1, desc1, shape0, shape1)
    MATCH_THRESHOLD = 0.8
    good_matches, cv_kpts0, cv_kpts1, pts0, pts1 = filter_matches(kpts0, kpts1, matches_indices, match_scores, MATCH_THRESHOLD)
    print(f"找到 {len(good_matches)} 个高质量匹配 (置信度 > {MATCH_THRESHOLD})")

    # 如果有足够点，尝试求单应性并输出内点数（可选）
    if len(good_matches) > 4 and pts0.shape[0] >= 4:
        H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"单应性矩阵 H:\n{H}\nRANSAC 内点数: {inliers}/{len(pts0)}")
    else:
        H = None

    # ---------- 绘制匹配（在彩色图像上） ----------
    if len(good_matches) == 0:
        print("没有达到阈值的匹配可绘制。")
        # 仍然显示无匹配的拼接图以便检查
        blank = np.zeros_like(color0)
        result_image = np.hstack([color0, blank])
    else:
        # drawMatches 会在左右两张图上画连线
        result_image = cv2.drawMatches(color0, cv_kpts0, color1, cv_kpts1, good_matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("SuperPoint + LightGlue 匹配结果 (彩色)", result_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
except Exception as e:
    print("发生未知错误:", e)
    traceback.print_exc()
