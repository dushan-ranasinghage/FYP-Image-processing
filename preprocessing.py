import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('sampleimage.jpg')

plt.subplot(141),plt.imshow(image,  cmap="gray"),plt.title('original')
plt.xticks([]), plt.yticks([])

kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(image, -1, kernel)

#convert to gray
gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

#find contours
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#binary
ret,th1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

plt.subplot(142),plt.imshow(th1,  cmap="gray"),plt.title('binary')
plt.xticks([]), plt.yticks([])

#draw contours
cv2.drawContours(th1,contours,-1,(0,0,0),3)
plt.subplot(143),plt.imshow(th1,  cmap="gray"),plt.title('after draw contours')
plt.xticks([]), plt.yticks([])

plt.subplot(144),plt.imshow(th2,  cmap="gray"),plt.title('binary inverse')
plt.xticks([]), plt.yticks([])
plt.show()
