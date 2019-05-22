import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('image1.jpg')

#convert to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#binary
ret,th1 = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
ret,th2 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5),np.uint8)
#erosion = cv2.erode(th1,kernel,iterations = 1)

dilation = cv2.dilate(th1,kernel,iterations = 0)

#find contours
ret, thresh = cv2.threshold(dilation, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
for contour in contours:
  cv2.drawContours(dilation, contour, -1, (0, 0, 255), 3)
plt.figure()

plt.subplot(131),plt.imshow(image),plt.title('original')
plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(th1,  cmap="gray"),plt.title('binary')
plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(dilation,  cmap="gray"),plt.title('contours')
plt.xticks([]), plt.yticks([])
plt.show()


