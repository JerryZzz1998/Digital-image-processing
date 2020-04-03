import numpy as np
import cv2
import matplotlib.pyplot as plt

o = cv2.imread(r'C:\Users\lenovo\Desktop\thing\project\image\house.jpg', cv2.IMREAD_GRAYSCALE)
plt.hist(o.ravel(), 256)
plt.show()
