import cv2
import numpy as np
import onnxruntime as ort
import time
import statistics
import json
import matplotlib.pyplot as plt

# ------------------- 配置 -------------------
IMAGE_PATH_1 = 'image1.jpg'
IMAGE_PATH_2 = 'image2.jpg'
SP_MODEL_PATH = 'model/superpoint.onnx'
LG_MODEL_PATH = 'model/superpoint_lightglue.onnx'
TARGET_SIZE = (1280, 720)
DEVICE = "CUDAExecutionProvider"
REPEAT = 10                  # 每轮测试重复次数
WARMUP = 3                   # 预热次数
KEYPOINT_LIST = [256, 512, 1024, 2048, 4096] # 模拟Benchmark的横坐标
# --------------------------------------------

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3

# ============ 工具函数 ============
def load_gray_resized(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    return cv2.resize(img, TARGET_SIZE)

def load_color_resized(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.resize(img, TARGET_SIZE)

# ============ SIFT (动态关键点数量) ============
def test_sift(img1, img2, top_k):
    # 动态设置 nfeatures
    sift = cv2.SIFT_create(nfeatures=top_k)
    
    # 检测 (Detect & Compute)
    start = time.perf_counter()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    t1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    t2 = (time.perf_counter() - start) * 1000

    # 匹配 (Match)
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return {"total_time": t1 + t2, "match_count": 0}

    bf = cv2.BFMatcher(cv2.NORM_L2)
    start = time.perf_counter()
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    t_match = (time.perf_counter() - start) * 1000

    # 筛选
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    return {
        "total_time": t1 + t2 + t_match,
        "det_time": t1 + t2,
        "match_time": t_match,
        "match_count": len(good)
    }

# ============ SuperPoint + LightGlue (动态关键点数量) ============
def normalize_keypoints(kpts, shape):
    h, w = shape
    kpts_norm = kpts.copy().astype(np.float32)
    kpts_norm[:, 0] /= (w - 1)
    kpts_norm[:, 1] /= (h - 1)
    return np.expand_dims(kpts_norm, 0)

def test_superpoint_lightglue(img1, img2, sp_sess, lg_sess, top_k):
    def prep(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TARGET_SIZE)
        tensor = gray.astype(np.float32)[None, None, :, :] / 255.0
        return tensor, gray.shape

    inp1, shape1 = prep(img1)
    inp2, shape2 = prep(img2)
    sp_input_name = sp_sess.get_inputs()[0].name

    # 1. SuperPoint Inference
    start = time.perf_counter()
    kpts1, scores1, desc1 = sp_sess.run(["keypoints", "scores", "descriptors"], {sp_input_name: inp1})
    t1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    kpts2, scores2, desc2 = sp_sess.run(["keypoints", "scores", "descriptors"], {sp_input_name: inp2})
    t2 = (time.perf_counter() - start) * 1000

    # 2. Filter Top-K Keypoints
    # 注意：这里根据 top_k 动态截取
    idxs1 = np.argsort(-scores1[0])[:top_k]
    kpts1, desc1 = kpts1[0][idxs1], desc1[0][idxs1]

    idxs2 = np.argsort(-scores2[0])[:top_k]
    kpts2, desc2 = kpts2[0][idxs2], desc2[0][idxs2]

    if len(kpts1) == 0 or len(kpts2) == 0:
        return {"total_time": t1 + t2, "match_count": 0}

    # 3. LightGlue Inference
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
    valid = (matches >= 0) & (scores > 0.0) # LightGlue输出通常已经经过阈值筛选，这里做个保底
    num_good = int(np.sum(valid))

    return {
        "total_time": t1 + t2 + t_match,
        "det_time": t1 + t2,
        "match_time": t_match,
        "match_count": num_good
    }

# ============ 绘图函数 (风格复刻) ============
def plot_benchmark(json_data):
    x_labels = json_data["x_axis"]
    methods = json_data["methods"]

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 颜色和样式映射，模仿截图风格
    styles = [
        {"color": "#1f77b4", "marker": "o", "label": "SIFT (OpenCV)"},
        {"color": "#ff7f0e", "marker": "o", "label": "SuperPoint + LightGlue"},
    ]

    for idx, (name, data) in enumerate(methods.items()):
        style = styles[idx % len(styles)]
        y_values = data["latency"]
        ax.plot(x_labels, y_values, 
                label=name, 
                color=style["color"], 
                marker=style["marker"], 
                linewidth=1.5,
                markersize=6)

    # 设置坐标轴风格
    ax.set_title("Latency Benchmark", fontsize=14)
    ax.set_xlabel("# keypoints", fontsize=12)
    ax.set_ylabel("Latency [ms]", fontsize=12)
    
    # 设置对数坐标 (Log Scale) - 这是截图风格的关键
    ax.set_yscale('log')
    
    # 设置X轴刻度为具体的点数，而不是自动线性
    ax.set_xticks(x_labels)
    ax.set_xticklabels([str(x) for x in x_labels])
    
    # 设置自定义的对数Y轴刻度 (可选，根据数据范围调整)
    # ax.set_yticks([5, 10, 20, 50, 100])
    # from matplotlib.ticker import ScalarFormatter
    # ax.yaxis.set_major_formatter(ScalarFormatter())

    # 网格线
    ax.grid(True, which="both", ls="-", alpha=0.4)
    
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# ============ 主流程 ============
if __name__ == "__main__":
    try:
        img1_gray = load_gray_resized(IMAGE_PATH_1)
        img2_gray = load_gray_resized(IMAGE_PATH_2)
        img1_color = load_color_resized(IMAGE_PATH_1)
        img2_color = load_color_resized(IMAGE_PATH_2)
        
        print(f"Provider: {ort.get_available_providers()}")
        print("加载模型...")
        sp_sess = ort.InferenceSession(SP_MODEL_PATH, sess_options, providers=[DEVICE])
        lg_sess = ort.InferenceSession(LG_MODEL_PATH, sess_options, providers=[DEVICE])
        
        # 结果容器
        benchmark_data = {
            "x_axis": KEYPOINT_LIST,
            "methods": {
                "SIFT": {"latency": []},
                "SuperPoint+LightGlue": {"latency": []}
            }
        }

        print(f"\n开始 Benchmark (Repeat={REPEAT})...")
        
        # --- 循环测试不同的 Keypoint 数量 ---
        for k in KEYPOINT_LIST:
            print(f"\n--- Testing Top-K = {k} ---")
            
            # 1. 测试 SIFT
            latencies = []
            for _ in range(REPEAT + WARMUP):
                res = test_sift(img1_gray, img2_gray, k)
                latencies.append(res["total_time"])
            # 去掉预热数据，取平均
            avg_sift = statistics.mean(latencies[WARMUP:])
            benchmark_data["methods"]["SIFT"]["latency"].append(avg_sift)
            print(f"SIFT: {avg_sift:.2f} ms")

            # 2. 测试 SP + LG
            latencies = []
            for _ in range(REPEAT + WARMUP):
                res = test_superpoint_lightglue(img1_color, img2_color, sp_sess, lg_sess, k)
                latencies.append(res["total_time"])
            avg_lg = statistics.mean(latencies[WARMUP:])
            benchmark_data["methods"]["SuperPoint+LightGlue"]["latency"].append(avg_lg)
            print(f"SP+LG: {avg_lg:.2f} ms")

        # --- 输出 JSON ---
        print("\n=== Benchmark Result JSON ===")
        json_str = json.dumps(benchmark_data, indent=2)
        print(json_str)

        # --- 画图 ---
        print("\n正在绘图...")
        plot_benchmark(benchmark_data)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()