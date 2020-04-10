import numpy as np
import random
import matplotlib.pyplot as plt
import cv2


def GaussianNoise(img, mean=0, var=0.01):
    """
    添加高斯噪声
    mean : 均值
    var : 方差
    """
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)  # 标准正态分布
    output = img + noise
    output = np.clip(output, 0.0, 1.0)  # 防止像素溢出
    output = np.uint8(output * 255)

    return output


def SaltAndPepperNoise(img, prob):
    """
    添加椒盐噪声
    image:原始图像
    prob:噪声比例(在0到1之间)
    """
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()  # 返回随机的浮点数，在半开区间 [0.0, main)
            if rdn < prob:
                output[i, j] = 0
            elif rdn > thres:
                output[i, j] = 255
            else:
                output[i, j] = img[i, j]
    return output


img = cv2.imread(r'C:/Users/lenovo/Desktop/thing/project/image/chip.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


'''''
noise_1 = SaltAndPepperNoise(img, 0.01)
noise_2 = SaltAndPepperNoise(img, 0.05)
plt.figure("椒盐噪声")
ax = plt.subplot(131)
ax.set_title('prob=0')
plt.imshow(img), plt.axis('off')
ax = plt.subplot(132)
ax.set_title('prob=0.01')
plt.imshow(noise_1), plt.axis('off')
ax = plt.subplot(133)
ax.set_title('prob=0.05')
plt.imshow(noise_2), plt.axis('off')
plt.show()
'''''


'''''
noise_guass = GaussianNoise(img)

plt.figure("高斯噪声")
plt.subplot(121)
plt.imshow(img), plt.axis('off'), plt.title('origin')
plt.subplot(122)
plt.imshow(noise_guass), plt.axis('off'), plt.title('Gaussian noise')
plt.show()
'''''

noise_guass = GaussianNoise(img)
noise_salt = SaltAndPepperNoise(img, 0.05)
blur_origin = cv2.blur(img, (5, 5))
blur_gauss = cv2.blur(noise_guass, (5, 5))
blur_salt = cv2.blur(noise_salt, (5, 5))
plt.figure("均值滤波")
plt.subplot(221)
plt.imshow(noise_salt), plt.axis('off'), plt.title('noise_salt')
plt.subplot(222)
plt.imshow(blur_salt), plt.axis('off'), plt.title('blur_salt')
plt.subplot(223)
plt.imshow(noise_guass), plt.axis('off'), plt.title('noise_guass')
plt.subplot(224)
plt.imshow(blur_gauss), plt.axis('off'), plt.title('blur_gauss')
plt.show()


