import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('sampleimage.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.medianBlur(image ,5)

ret,th1 = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold(image,120,255,cv2.THRESH_BINARY_INV)


plt.subplot(131),plt.imshow(image,  cmap="gray"),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(th1,  cmap="gray"),plt.title('binary')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(th2,  cmap="gray"),plt.title('binary inverse')
plt.xticks([]), plt.yticks([])
plt.show()
