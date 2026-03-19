import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------
# 1 Read image
# -----------------------------
img = cv2.imread("image1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape

# -----------------------------
# 2 Preprocess
# -----------------------------
input_size = 640
scale = input_size / max(h, w)

new_w = int(w * scale)
new_h = int(h * scale)

img_resized = cv2.resize(gray, (new_w, new_h))

# Padding to 640
padded = np.zeros((input_size, input_size), dtype=np.uint8)
padded[:new_h, :new_w] = img_resized

# Normalization
input_tensor = padded.astype(np.float32) / 255.0
input_tensor = input_tensor[None, None, :, :]  # (1,1,H,W)

# -----------------------------
# 3 ONNX inference
# -----------------------------
session = ort.InferenceSession("./model/xfeat.onnx")

input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

desc_map = outputs[0]
logits = outputs[1]
score_map = outputs[2]

score_map = score_map[0, 0]  # (80,80)

# -----------------------------
# 4 Extract keypoints
# -----------------------------
threshold = 0.5
ys, xs = np.where(score_map > threshold)

# feature map stride
stride = input_size / score_map.shape[0]

keypoints = []
for x, y in zip(xs, ys):

    px = int(x * stride / scale)
    py = int(y * stride / scale)

    if px < w and py < h:
        keypoints.append((px, py))

# -----------------------------
# 5 Draw keypoints
# -----------------------------
vis = img.copy()

for x, y in keypoints:
    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

print("keypoints", len(keypoints))

cv2.imshow("xfeat keypoints", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()