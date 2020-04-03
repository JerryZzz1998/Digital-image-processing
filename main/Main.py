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
        self.SetMenu.add_command(label='测试用例',
                                 command=self.AdaptiveEquColor_2)  # ！~~~~~~~~~~~~~~~~测试用~~~~~~~~~~~~~~~~~！
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

        # 亮度提升
        self.Morphology = LabelFrame(self, text='形态学操作', labelanchor='n', padx=10, pady=10)
        self.Morphology.grid(column=1, row=0, sticky='n')

    def SetMainWindow(self):
        self.title("图像增强系统v1.2")
        self.geometry('500x500')
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
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        output = img + noise
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

    def SSR(self):
        pass




if __name__ == "__main__":
    System = MainSystem()
    System.mainloop()