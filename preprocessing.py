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

dilation = cv2.dilate(th2,kernel,iterations = 0)

# Calculate horizontal projection
proj = np.sum(dilation,1)

# Create output image same height as text, 500 px wide
m = np.max(proj)
w = 500
result = np.zeros((proj.shape[0],500))

# Draw a line for each row
for row in range(dilation.shape[0]):
   cv2.line(result, (0,row), (int(proj[row]*w/m),row), (255,255,255), 1)

plt.subplot(131),plt.imshow(image),plt.title('original')
plt.xticks([]), plt.yticks([])

plt.subplot(132),plt.imshow(th1,  cmap="gray"),plt.title('binary')
plt.xticks([]), plt.yticks([])

plt.subplot(133),plt.imshow(result,  cmap="gray"),plt.title('horizontal projection')
plt.xticks([]), plt.yticks([])
plt.show()

