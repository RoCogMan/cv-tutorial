import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Lenna 이미지 불러오기 → 그레이 → 25×25 축소
img = cv2.imread("lenna.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (25, 25)).astype(np.float32)
# 3x3 커널 (예:평균값 필터)
kernel = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)
kh, kw = kernel.shape
h, w = img.shape
out_h, out_w = h - kh + 1, w - kw + 1
output = np.zeros((out_h, out_w), dtype=np.float32)
def draw_with_numbers(ax, data, title):
    ax.imshow(data, cmap="gray", vmin=data.min(), vmax=data.max())
    ax.set_title(title)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
plt.ion()  # interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
step = 1
for y in range(out_h):
    for x in range(out_w):
        roi = img[y:y+kh, x:x+kw]
        output[y, x] = np.sum(roi * kernel)
        # 프레임 초기화
        fig.clf()
        ax1, ax2 = fig.subplots(1, 2)
        # --- 왼쪽: 원본 이미지 + 커널 위치 표시 ---
        draw_with_numbers(ax1, img, f"Original 25×25 Lenna (step {step})")
        rect = Rectangle((x-0.5, y-0.5), kw, kh,
                         linewidth=2, edgecolor='yellow', facecolor='none')
        ax1.add_patch(rect)
        # --- 오른쪽: 현재까지의 출력 ---
        draw_with_numbers(ax2, output, "Convolution Output (so far)")
        plt.tight_layout()
        plt.pause(0.001)  # 움직임 확인용 속도
        step += 1
plt.ioff()
plt.show()
