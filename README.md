# 2023100068-ZiDongHua-LiuChongji
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取彩色图像（BGR格式）
img = cv2.imread("test.jpg")  # 替换为你的图片路径
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB用于matplotlib显示
h, w = img.shape[:2]

# 2. 转换到YCbCr色彩空间
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)  # 提取Y、Cr、Cb通道

# 3. 对Cb、Cr通道进行下采样（2倍下采样）
down_scale = 2
Cb_down = cv2.resize(Cb, (w//down_scale, h//down_scale), interpolation=cv2.INTER_NEAREST)
Cr_down = cv2.resize(Cr, (w//down_scale, h//down_scale), interpolation=cv2.INTER_NEAREST)

# 4. 插值恢复原尺寸（使用双线性插值）
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 5. 与Y通道重建图像并转回RGB
img_ycrcb_recon = cv2.merge([Y, Cr_up, Cb_up])
img_recon_rgb = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2RGB)

# 6. 计算PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

psnr_value = calculate_psnr(img_rgb, img_recon_rgb)
print(f"PSNR: {psnr_value:.2f} dB")

# 7. 结果可视化
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original RGB Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(Y, cmap="gray")
plt.title("Y Channel")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(Cb, cmap="gray")
plt.title("Original Cb Channel")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(Cb_up, cmap="gray")
plt.title("Reconstructed Cb Channel")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(Cr_up, cmap="gray")
plt.title("Reconstructed Cr Channel")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(img_recon_rgb)
plt.title(f"Reconstructed Image\nPSNR: {psnr_value:.2f} dB")
plt.axis("off")

plt.tight_layout()
plt.show()
