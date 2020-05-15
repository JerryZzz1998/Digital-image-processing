import math
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
        self.img_rgb = None
        self.img_gray = None
        self.img_b = None
        self.img_g = None
        self.img_r = None
        self.img_noise = None
        self.img_noise_shake = None
        self.img_current = None  # 用于保存图片
        self.motionkernel = None
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
        self.SetMenu.add_command(label='二值化')
        self.SetMenu.add_separator()
        self.SetMenu.add_command(label='添加椒盐噪声', command=self.SaltAndPepperNoise)
        self.SetMenu.add_command(label='添加高斯噪声', command=self.GaussianNoise)
        self.SetMenu.add_command(label='仿运动模糊', command=self.makeMotionBlur)
        self.EvaluationMenu = tkinter.Menu(self.BigMenu, tearoff=0)
        self.EvaluationMenu.add_command(label='信息熵', command=self.getEntropy)
        self.EvaluationMenu.add_cascade(label='峰值信噪比', command=self.getPSNR)
        self.EvaluationMenu.add_cascade(label='标准差', command=self.getVariance)
        self.EvaluationMenu.add_cascade(label='亮度均值', command=self.getMean)
        # 将子菜单添加到主菜单
        self.BigMenu.add_cascade(label='文件', menu=self.FileMenu)
        self.BigMenu.add_cascade(label='选项', menu=self.SetMenu)
        self.BigMenu.add_cascade(label='评价指标', menu=self.EvaluationMenu)
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
        self.BtnApEquColor = tkinter.Button(self.Bright, text='自适应直方图均衡', command=self.AdaptiveEquColor)
        self.BtnApEquColor.pack(pady=10)
        # 伽马变换
        self.BtnGammaTras = Button(self.Bright, text='伽马变换', command=self.GammaTransform)
        self.BtnGammaTras.pack()
        # SSR算法
        self.BtnMSRCR = Button(self.Bright, text='SSR算法', command=self.MSRCR)
        self.BtnMSRCR.pack(pady=10)

    def SetMainWindow(self):
        self.title("图像增强系统v1.2")
        self.geometry('310x310')
        self.iconbitmap(r'C:\Users\lenovo\Desktop\thing\project\image\toolbox.ico')

    def OpenImg(self):
        self.FilePath = filedialog.askopenfilename(parent=self, title='打开图片')
        self.img_bgr = cv2.imread(self.FilePath)
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        self.img_gray = cv2.imread(self.FilePath, cv2.IMREAD_GRAYSCALE)
        self.img_b, self.img_g, self.img_r = cv2.split(self.img_bgr)  # 通道拆分

        if self.img_bgr is None:
            messagebox.showerror(title='提示信息', message='读取图像失败')
        else:
            self.img_current = self.img_gray
            cv2.imshow("image", self.img_bgr)

    def SaveImg(self):
        if self.img_current is None:
            messagebox.showerror(title='提示信息', message='请对图片进行变更后再保存图片')
        else:
            ret = cv2.imwrite(r'C:/Users/lenovo/Desktop/src/src.jpg', self.img_current)
            if ret == FALSE:
                messagebox.showerror(title='提示信息', message='保存图片失败')
            else:
                messagebox.showinfo(title='提示信息', message='保存图片成功')

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
        self.img_current = output
        cv2.imshow("dst", output)

    def GaussianNoise(self, mean=0, var=0.004):
        """
        添加高斯噪声
        mean : 均值
        var : 方差
        """
        img = np.array(self.img_bgr / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, img.shape)  # 标准正态分布
        output = img + noise  # 加性噪声
        output = np.clip(output, 0.0, 1.0)  # 防止像素溢出
        output = np.uint8(output * 255)
        self.img_noise = output
        self.img_current = output
        cv2.imshow("dst", output)

    def MeanFilter(self):
        if self.img_noise is not None:
            output = cv2.blur(self.img_noise, (5, 5))
        else:
            output = cv2.blur(self.img_bgr, (5, 5))
        self.img_current = output
        cv2.imshow("MeanFilter", output)

    def MedianFilter(self):
        if self.img_noise is not None:
            output = cv2.medianBlur(self.img_noise, 3)
        else:
            output = cv2.medianBlur(self.img_bgr, 3)
        self.img_current = output
        cv2.imshow("MedianFilter", output)

    def GaussFilter(self):
        if self.img_noise is not None:
            output = cv2.GaussianBlur(self.img_noise, (7, 7), 0, 0)
        else:
            output = cv2.GaussianBlur(self.img_bgr, (7, 7), 0, 0)
        self.img_current = output
        cv2.imshow("GaussFilter", output)

    def WienerFilter(self, K=0.01):
        img_fft = np.fft.fft2(self.img_gray)
        kernel_fft = np.fft.fft2(self.motionkernel) + 1e-3
        kernel_fft = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
        output = np.fft.ifft2(img_fft * kernel_fft)
        output = np.abs(np.fft.fftshift(output))
        self.img_noise_shake = output
        cv2.imshow("Motionblur", output)

    def getMotionKernel(self, angle=60):
        img_h = np.shape(self.img_gray)[0]
        img_w = np.shape(self.img_gray)[1]
        kernel = np.zeros((img_h, img_w))
        center_position = (img_h - 1) / 2
        slope_tan = math.tan(angle * math.pi / 180)
        slope_cot = 1 / slope_tan
        if slope_tan <= 1:
            for i in range(15):
                offset = round(i * slope_tan)     # ((center_position-i)*slope_tan)
                kernel[int(center_position + offset), int(center_position - offset)] = 1
                kernel = kernel / kernel.sum()  # 归一化
                self.motionkernel = kernel
            return kernel
        else:
            for i in range(15):
                offset = round(i * slope_cot)
                kernel[int(center_position - offset), int(center_position + offset)] = 1
                kernel = kernel / kernel.sum()
                self.motionkernel = kernel
            return kernel

    def makeMotionBlur(self):
        self.getMotionKernel()
        img_fft = np.fft.fft2(self.img_gray)  # 进行二维数组的傅里叶变换
        kernel_fft = np.fft.fft2(self.motionkernel) + 1e-3
        blurred = np.fft.ifft2(img_fft * kernel_fft)
        blurred = np.abs(np.fft.fftshift(blurred))
        self.img_noise_shake = blurred
        cv2.imshow("MotionBlur", blurred)



    def EroImg(self):
        kernel = np.ones((5, 5), dtype=np.uint8)  # 设置腐蚀核的大小
        output = cv2.erode(self.img_bgr, kernel)
        self.img_current = output
        cv2.imshow("Erosion", output)

    def DilateImg(self):
        kernel = np.ones((5, 5), dtype=np.uint8)
        output = cv2.dilate(self.img_bgr, kernel)
        self.img_current = output
        cv2.imshow("Dilate", output)

    def OpenProcess(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_OPEN, kernel)
        self.img_current = output
        cv2.imshow("Open", output)

    def CloseProcess(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_CLOSE, kernel)
        self.img_current = output
        cv2.imshow("Close", output)

    def TopHat(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_TOPHAT, kernel)
        self.img_current = output
        cv2.imshow("TopHat", output)

    def BlackHat(self):
        kernel = np.ones((10, 10), np.uint8)
        output = cv2.morphologyEx(self.img_bgr, cv2.MORPH_BLACKHAT, kernel)
        self.img_current = output
        cv2.imshow("BlackHat", output)

    def GammaTransform(self, gamma=0.5):
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        Value = np.power(Hsv[:, :, 2] / 255., gamma)
        Value = np.clip(Value * 255, 0, 255)  # 防止像素溢出
        Hsv[:, :, 2] = np.uint8(Value)
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        self.img_current = output
        cv2.imshow("Gamma Transform", output)

    def EquColor(self):  # 待画出直方图
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        v = cv2.equalizeHist(Hsv[:, :, 2])
        Hsv[:, :, 2] = v
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        self.img_current = output
        cv2.imshow("dst", output)

    def AdaptiveEquColor(self):
        Hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        v = clahe.apply(Hsv[:, :, 2])
        Hsv[:, :, 2] = v
        output = cv2.cvtColor(Hsv, cv2.COLOR_HSV2BGR)
        self.img_current = output
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

        self.img_current = img_msrcr
        cv2.imshow("MSRCR", img_msrcr)

    def MotionBlur(self, degree=12, angle=60):

        # 这里是得到任意角度的运动模糊卷积核矩阵，矩阵越大越模糊
        mat = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)  # 获得仿射变换矩阵

        kernel = np.diag(np.ones(degree))

        kernel = cv2.warpAffine(kernel, mat, (degree, degree))

        kernel = kernel / degree
        # 将卷积核应用到图像上，并转化为uint8
        output = cv2.filter2D(self.img_gray, -1, kernel)
        cv2.normalize(output, output, 0, 255, cv2.NORM_MINMAX)
        output = np.array(output, dtype=np.uint8)
        self.img_noise_shake = output
        self.img_current = output
        cv2.imshow("Motion Blur", output)

    def getEntropy(self):
        """
        param: img:narray 二维灰度图像
        return: float 图像约清晰越大,图像的信息熵
        """
        out = 0
        count = np.shape(self.img_current)[0] * np.shape(self.img_current)[1]
        p = np.bincount(np.array(self.img_current).flatten())
        for i in range(0, len(p)):
            if p[i] != 0:
                out -= p[i] * math.log(p[i] / count) / count
        print(out)

    def getPSNR(self):
        img_1 = cv2.cvtColor(self.img_current, cv2.COLOR_BGR2GRAY)
        img_2 = self.img_gray
        psnr = cv2.PSNR(img_1, img_2)
        print("PSNR IS: ", psnr)

    def getVariance(self):
        """
        :param img:narray 二维灰度图像
        :return: float 图像约清晰越大,标准差
        """
        out = 0
        if np.ndim(self.img_current)==3:
            cv2.cvtColor(self.img_current, cv2.COLOR_BGR2GRAY)
        u = np.mean(self.img_current)
        size = np.shape(self.img_current)[0] * np.shape(self.img_current)[1]
        shape = np.shape(self.img_current)
        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                out += (self.img_current[x, y] - u) ** 2
        out = np.sqrt(out /size)
        print(out)

    def getMean(self):
        if np.ndim(self.img_current)==3:
            cv2.cvtColor(self.img_current, cv2.COLOR_BGR2GRAY)
        mean = np.mean(self.img_current)
        print(mean)



if __name__ == "__main__":
    System = MainSystem()
    System.mainloop()
