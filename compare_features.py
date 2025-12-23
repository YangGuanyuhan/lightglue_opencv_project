import cv2
import numpy as np
import onnxruntime as ort
import time
import statistics
print(ort.get_available_providers())
# ------------------- 配置 -------------------
IMAGE_PATH_1 = 'image1.jpg'
IMAGE_PATH_2 = 'image2.jpg'
SP_MODEL_PATH = 'model/superpoint.onnx'
LG_MODEL_PATH = 'model/superpoint_lightglue.onnx'
TARGET_SIZE = (1280, 720)  # 保持输入一致
DEVICE = "CUDAExecutionProvider"
REPEAT = 10  # 连续测试次数
MAX_KPTS = 2048 
# --------------------------------------------

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3


# ============ 工具函数 ============
def load_gray_resized(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, TARGET_SIZE)

def load_color_resized(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, TARGET_SIZE)

# ============ SIFT ============
def test_sift(img1, img2):
    sift = cv2.SIFT_create(nfeatures=MAX_KPTS)
    kpts1, desc1 = sift.detectAndCompute(img1, None)


    start = time.perf_counter()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    t1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    t2 = (time.perf_counter() - start) * 1000

    bf = cv2.BFMatcher(cv2.NORM_L2)
    start = time.perf_counter()
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    t_match = (time.perf_counter() - start) * 1000

    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    return {
        "det1_time": t1, "det2_time": t2, "match_time": t_match,
        "num_kpts1": len(kpts1), "num_kpts2": len(kpts2), "num_good": len(good)
    }

# ============ SuperPoint + LightGlue ============
def normalize_keypoints(kpts, shape):
    h, w = shape
    kpts_norm = kpts.copy().astype(np.float32)
    kpts_norm[:, 0] /= (w - 1)
    kpts_norm[:, 1] /= (h - 1)
    return np.expand_dims(kpts_norm, 0)

def test_superpoint(img1, img2, sp_sess, lg_sess):
    def prep(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TARGET_SIZE)
        tensor = gray.astype(np.float32)[None, None, :, :] / 255.0
        return tensor, gray.shape

    inp1, shape1 = prep(img1)
    inp2, shape2 = prep(img2)
    input_name = sp_sess.get_inputs()[0].name

    # SuperPoint
    start = time.perf_counter()
    kpts1, scores1, desc1 = sp_sess.run(["keypoints", "scores", "descriptors"], {input_name: inp1})
    t1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    kpts2, scores2, desc2 = sp_sess.run(["keypoints", "scores", "descriptors"], {input_name: inp2})
    t2 = (time.perf_counter() - start) * 1000

    idxs1 = np.argsort(-scores1[0])[:MAX_KPTS]  
    kpts1, desc1 = kpts1[0][idxs1], desc1[0][idxs1]

    idxs2 = np.argsort(-scores2[0])[:MAX_KPTS]  
    kpts2, desc2 = kpts2[0][idxs2], desc2[0][idxs2]

    # LightGlue
    kpts0n = normalize_keypoints(kpts1, shape1)
    kpts1n = normalize_keypoints(kpts2, shape2)
    feed = {
        "kpts0": kpts0n,
        "desc0": np.expand_dims(desc1, 0).astype(np.float32),
        "kpts1": kpts1n,
        "desc1": np.expand_dims(desc2, 0).astype(np.float32)
    }

    start = time.perf_counter()
    matches, scores = lg_sess.run(["matches0", "mscores0"], feed)
    t_match = (time.perf_counter() - start) * 1000

    matches, scores = matches[0], scores[0]
    valid = (matches >= 0) & (scores > 0.8)
    num_good = int(np.sum(valid))

    return {
        "det1_time": t1, "det2_time": t2, "match_time": t_match,
        "num_kpts1": len(kpts1), "num_kpts2": len(kpts2), "num_good": num_good
    }

# ============ 主流程 ============
if __name__ == "__main__":
    img1_gray = load_gray_resized(IMAGE_PATH_1)
    img2_gray = load_gray_resized(IMAGE_PATH_2)
    img1_color = load_color_resized(IMAGE_PATH_1)
    img2_color = load_color_resized(IMAGE_PATH_2)

    print("加载模型（不计入测试时间）...")
    sp_sess = ort.InferenceSession(SP_MODEL_PATH, sess_options, providers=[DEVICE])
    lg_sess = ort.InferenceSession(LG_MODEL_PATH, sess_options, providers=[DEVICE])
    print("模型加载完成。\n")

    # ---------- SIFT ----------
    print(f"=== 测试 SIFT（重复 {REPEAT} 次）===")
    sift_res = []
    for i in range(REPEAT):
        res = test_sift(img1_gray, img2_gray)
        sift_res.append(res)
        print(f"  第 {i+1:02d} 次: {res}")

    def avg(field): return statistics.mean(r[field] for r in sift_res)
    def std(field): return statistics.stdev(r[field] for r in sift_res)

    print(f"\nSIFT 平均结果：")
    print(f"  det1_time: {avg('det1_time'):.2f} ± {std('det1_time'):.2f} ms")
    print(f"  det2_time: {avg('det2_time'):.2f} ± {std('det2_time'):.2f} ms")
    print(f"  match_time: {avg('match_time'):.2f} ± {std('match_time'):.2f} ms")
    print(f"  总时间: {(avg('det1_time')+avg('det2_time')+avg('match_time')):.2f} ms")
    print(f"  平均关键点数: {avg('num_kpts1'):.1f} / {avg('num_kpts2'):.1f}")
    print(f"  平均有效匹配数: {avg('num_good'):.1f}")

    # ---------- SuperPoint ----------
    print(f"\n=== 测试 SuperPoint + LightGlue（重复 {REPEAT} 次）===")
    sp_res = []
    for i in range(REPEAT):
        res = test_superpoint(img1_color, img2_color, sp_sess, lg_sess)
        sp_res.append(res)
        print(f"  第 {i+1:02d} 次: {res}")

    def avg_sp(field): return statistics.mean(r[field] for r in sp_res)
    def std_sp(field): return statistics.stdev(r[field] for r in sp_res)

    print(f"\nSuperPoint 平均结果：")
    print(f"  det1_time: {avg_sp('det1_time'):.2f} ± {std_sp('det1_time'):.2f} ms")
    print(f"  det2_time: {avg_sp('det2_time'):.2f} ± {std_sp('det2_time'):.2f} ms")
    print(f"  match_time: {avg_sp('match_time'):.2f} ± {std_sp('match_time'):.2f} ms")
    print(f"  总时间: {(avg_sp('det1_time')+avg_sp('det2_time')+avg_sp('match_time')):.2f} ms")
    print(f"  平均关键点数: {avg_sp('num_kpts1'):.1f} / {avg_sp('num_kpts2'):.1f}")
    print(f"  平均有效匹配数: {avg_sp('num_good'):.1f}")

    # ---------- 汇总 ----------
    print("\n=== 对比汇总 ===")
    print(f"SIFT 平均总时间: {(avg('det1_time')+avg('det2_time')+avg('match_time')):.2f} ms")
    print(f"SuperPoint 平均总时间: {(avg_sp('det1_time')+avg_sp('det2_time')+avg_sp('match_time')):.2f} ms")
    print(f"SIFT 平均匹配数: {avg('num_good'):.1f} vs SuperPoint: {avg_sp('num_good'):.1f}")
