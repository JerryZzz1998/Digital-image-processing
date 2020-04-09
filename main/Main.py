import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import cv2
import random
import numpy as np


class MainSystem(tkinter.Tk):
    def __init__(self):
        super().__init__()
        # 设置图片属性
        self.FilePath = None
        self.img_bgr = None
        self.img_gray = None
        self.img_b = None
        self.img_g = None
        self.img_r = None
        self.img_noise = None
        # 设置窗口属性
        self.SetMainWindow()
        # 创建主菜单
        self.BigMenu = tkinter.Menu(self)
        # 创建子菜单
        self.FileMenu = tkinter.Menu(self.BigMenu, tearoff=0)
        self.FileMenu.add_command(label='打开图片', command=self.OpenImg)
        self.FileMenu.add_command(label='保存图片', command=self.SaveImg)
        self.FileMenu.add_separator()
        self.FileMenu.add_command(label='退出')
        self.SetMenu = tkinter.Menu(self.BigMenu, tearoff=0)
        self.SetMenu.add_command(label='平滑处理选项')
        self.SetMenu.add_command(label='阈值处理选项')
        self.SetMenu.add_command(label='形态学操作选项')
        self.SetMenu.add_separator()
        self.SetMenu.add_command(label='添加椒盐噪声', command=self.SaltAndPepperNoise)
        self.SetMenu.add_command(label='添加高斯噪声', command=self.GaussianNoise)
        self.SetMenu.add_command(label='运动模糊')
        self.SetMenu.add_command(label='测试用例', command=self.MSRCP)  # ！~~~~~~~~~~~~~~~~测试用~~~~~~~~~~~~~~~~~！
        # 将子菜单添加到主菜单
        self.BigMenu.add_cascade(label='文件', menu=self.FileMenu)
        self.BigMenu.add_cascade(label='选项', menu=self.SetMenu)
        # 将主菜单加入到界面
        self.config(menu=self.BigMenu)

        # 图像的平滑操作
        self.SmoothImg = LabelFrame(self, text='平滑操作', labelanchor='n', padx=10, pady=10)
        self.SmoothImg.grid(column=0, row=0, sticky='n')
        # 均值滤波按钮
        self.BtnMeanFil = tkinter.Button(self.SmoothImg, text='均值滤波', command=self.MeanFilter)
        self.BtnMeanFil.pack()
        # 中值滤波按钮
        self.BtnMedianFil = tkinter.Button(self.SmoothImg, text='中值滤波', command=self.MedianFilter)
        self.BtnMedianFil.pack(pady=10)
        # 高斯滤波按钮
        self.BtnGaussFil = tkinter.Button(self.SmoothImg, text='高斯滤波', command=self.GaussFilter)
        self.BtnGaussFil.pack()
        # 维纳滤波按钮
        self.BtnWienerFil = tkinter.Button(self.SmoothImg, text='维纳滤波', command=self.WienerFilter)
        self.BtnWienerFil.pack(pady=10)

        # 图像的形态学操作
        self.Morphology = LabelFrame(self, text='形态学操作', labelanchor='n', padx=10, pady=10)
        self.Morphology.grid(column=1, row=0, sticky='n')
        # 腐蚀
        self.BtnEroImg = tkinter.Button(self.Morphology, text='腐蚀', command=self.EroImg)
        self.BtnEroImg.pack()
        # 膨胀
        self.BtnDilateImg = tkinter.Button(self.Morphology, text='膨胀', command=self.DilateImg)
        self.BtnDilateImg.pack(pady=10)
        # 开运算
        self.BtnOpen = tkinter.Button(self.Morphology, text='开运算', command=self.OpenProcess)
        self.BtnOpen.pack()
        # 闭运算
        self.BtnClose = tkinter.Button(self.Morphology, text='闭运算', command=self.CloseProcess)
        self.BtnClose.pack(pady=10)
        # 礼帽运算
        self.BtnTopHat = tkinter.Button(self.Morphology, text='礼帽运算', command=self.TopHat)
        self.BtnTopHat.pack()
        # 黑帽运算
        self.BtnBlackHat = tkinter.Button(self.Morphology, text='黑帽运算', command=self.BlackHat)
        self.BtnBlackHat.pack(pady=10)

        # 亮度增强
        self.Bright = LabelFrame(self, text='亮度增强', labelanchor='n', padx=10, pady=10)
        self.Bright.grid(column=2, row=0, sticky='n')
        # 直方图均衡
        self.BtnEquColor = tkinter.Button(self.Bright, text='直方图均衡', command=self.EquColor)
        self.BtnEquColor.pack()
        # 自适应直方图均衡
        self.BtnApEquColor = tkinter.Button(self.Bright, text='自适应直方图均衡', command=self.AdaptiveEquColor_2)
        self.BtnApEquColor.pack(pady=10)
        # 伽马变换
        self.BtnGammaTras = Button(self.Bright, text='伽马变换', command=self.GammaTransform)
        self.BtnGammaTras.pack()
        # MSRCR算法
        self.BtnMSRCR = Button(self.Bright, text='MSRCR算法', command=self.MSRCR)
        self.BtnMSRCR.pack(pady=10)

    def SetMainWindow(self):
        self.title("图像增强系统v1.2")
        self.geometry('500x400')
        self.iconbitmap(r'C:\Users\lenovo\Desktop\thing\project\image\toolbox.ico')

    def OpenImg(self):
        self.FilePath = filedialog.askopenfilename(parent=self, title='打开图片')
        self.img_bgr = cv2.imread(self.FilePath)
        self.img_gray = cv2.imread(self.FilePath, cv2.IMREAD_GRAYSCALE)
        self.img_b, self.img_g, self.img_r = cv2.split(self.img_bgr)  # 通道拆分

        if self.img_bgr is None:
            messagebox.showerror(title='提示信息', message='读取图像失败')
        else:
            cv2.imshow("image", self.img_bgr)

    def SaveImg(self):
        pass

    def SaltAndPepperNoise(self):
        """
        添加椒盐噪声
        image:原始图像
        prob:噪声比例(在0到1之间)
        """
        output = np.zeros(self.img_bgr.shape, np.uint8)
        thres = 1 - 0.05
        for i in range(self.img_bgr.shape[0]):
            for j in range(self.img_bgr.shape[1]):
                rdn = random.random()  # 返回随机的浮点数，在半开区间 [0.0, main)
                if rdn < 0.05:
                    output[i, j] = 0
                elif rdn > thres:
                    output[i, j] = 255
                else:
                    output[i, j] = self.img_bgr[i, j]
        self.img_noise = output
        cv2.imshow("dst", output)

    def GaussianNoise(self, mean=0, var=0.004):
        """
        添加高斯噪声
        mean : 均值
        var : 方差
        """
        img = np.array(self.img_bgr / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, img.shape)  # 标准正态分布
        output = img + noise   # 加性噪声
        output = np.clip(output, 0.0, 1.0)  # 防止像素溢出
        output = np.uint8(output * 255)
        self.img_noise = output
        cv2.imshow("dst", output)

    def MotionProcess(self):
        pass

    def MeanFilter(self):
        if self.img_noise is not None:
            output = cv2.blur(self.img_noise, (5, 5))
        else:
            output = cv2.blur(self.img_bgr, (5, 5))
        cv2.imshow("MeanFilter", output)

    def MedianFilter(self):
        if self.img_noise is not None:
            output = cv2.medianBlur(self.img_noise, 3)
        else:
            output = cv2.medianBlur(self.img_bgr, 3)
        cv2.imshow("MedianFilter", output)

    def GaussFilter(self):
        if self.img_noise is not None:
            output = cv2.GaussianBlur(self.img_noise, (7, 7), 0, 0)
        else:
            output = cv2.GaussianBlur(self.img_bgr, (7, 7), 0, 0)
        cv2.imshow("GaussFilter", output)

    def WienerFilter(self):
        pass

    def EroImg(self):
        kernel = np.ones((5, 5), dtype=np.uint8)  # 设置腐蚀核的大小
        erosion = cv2.erode(self.img_bgr, kernel)
        cv2.imshow("Erosion", erosion)

    def DilateImg(self):
        kernel = np.ones((5, 5), dtype=np.uint8)
        output = cv2.dilate(self.img_bgr, kernel)
        cv2.imshow("Dilate", output)

    def OpenProcess(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_OPEN, kernel)
        cv2.imshow("Open", output)

    def CloseProcess(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Close", output)

    def TopHat(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow("TopHat", output)

    def BlackHat(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow("BlackHat", output)

    def GammaTransform(self, gamma=0.5):
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        Value = np.power(Hsv[:, :, 2] / 255., gamma)
        Value = np.clip(Value * 255, 0, 255)  # 防止像素溢出
        Hsv[:, :, 2] = np.uint8(Value)
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Gamma Transform", output)

    def EquColor(self):  # 待画出直方图
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        v = cv2.equalizeHist(Hsv[:, :, 2])
        Hsv[:, :, 2] = v
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("dst", output)

    def AdaptiveEquColor(self):
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        v = clahe.apply(Hsv[:, :, 2])
        Hsv[:, :, 2] = v
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("dst", output)

    def AdaptiveEquColor_2(self):  # 有待改进
        ycrcb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)

        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])

        cv2.merge(channels, ycrcb)
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        cv2.imshow("dst", output)

    # RF:A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes
    def MSRCR(self, beta=46.0, alpha=125.0, G=5.0, b=25.0, low_clip=0.01, high_clip=0.99):

        sigma_list = [15, 80, 250]

        # multiScaleRetinex
        img = np.float64(self.img_bgr) + 1.0  # 防止0取对数的情况发生
        img_retinex = np.zeros_like(img)
        for sigma in sigma_list:
            img_retinex += np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))  # 主要是此步骤计算时间过长
            print(sigma)
        img_retinex = img_retinex / len(sigma_list)

        # colorRestoration
        img_sum = np.sum(img, axis=2, keepdims=True)
        img_color = beta * (np.log10(alpha * img) - np.log10(img_sum))

        # 经验公式
        img_msrcr = G * (img_retinex * img_color + b)

        for i in range(img_msrcr.shape[2]):
            img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                                 (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                                 255

        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
        # simplestColorBalance
        total = img_msrcr.shape[0] * img_msrcr.shape[1]
        for i in range(img_msrcr.shape[2]):
            unique, counts = np.unique(img_msrcr[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c

            img_msrcr[:, :, i] = np.maximum(np.minimum(img_msrcr[:, :, i], high_val), low_val)

        cv2.imshow("MSRCR", img_msrcr)

    def MSRCP(self, low_clip=0.01, high_clip=0.99):
        sigma_list = [15, 80, 250]
        img = np.float64(self.img_bgr) + 1.0

        intensity = np.sum(img, axis=2) / img.shape[2]

        img_retinex = np.zeros_like(intensity)
        for sigma in sigma_list:
            img_retinex += np.log10(intensity) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
            print(sigma)
        img_retinex = img_retinex / len(sigma_list)

        intensity = np.expand_dims(intensity, 2)
        img_retinex = np.expand_dims(img_retinex, 2)

        total = img_retinex.shape[0] * img_retinex.shape[1]

        for i in range(img_retinex.shape[2]):
            unique, counts = np.unique(img_retinex[:, :, i], return_counts=True)
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        intensity1 = img_retinex

        intensity1 = (intensity1 - np.min(intensity1)) / \
                     (np.max(intensity1) - np.min(intensity1)) * \
                     255.0 + 1.0

        img_msrcp = np.zeros_like(self.img_bgr)

        for y in range(img_msrcp.shape[0]):
            for x in range(img_msrcp.shape[1]):
                B = np.max(img[y, x])
                A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
                img_msrcp[y, x, 0] = A * img[y, x, 0]
                img_msrcp[y, x, 1] = A * img[y, x, 1]
                img_msrcp[y, x, 2] = A * img[y, x, 2]

        img_msrcp = np.uint8(img_msrcp - 1.0)

        cv2.imshow("MSRCP", img_msrcp)


if __name__ == "__main__":
    System = MainSystem()
    System.mainloop()
