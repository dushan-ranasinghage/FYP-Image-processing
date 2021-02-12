import cv2 as cv
import sys
import numpy as np


def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def main(argv):
    # [load_image]
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    img = cv.imread(argv[0], cv.IMREAD_COLOR)
    height, width = img.shape[:2]

    # Check if image is loaded fine
    if img is None:
        print ('Error opening image: ' + argv[0])
        return -1

    # [gray]
    # Transform source image to gray if it is not already
    if len(img.shape) != 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    (_, thresh) = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)
    closed = cv.erode(thresh, None, iterations=7)
    height, width = closed.shape[:2]

    # # binary
    # ret, th1 = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
    # ret, th2 = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
    z = [0] * height
    v = [0] * width
    hfg = [[0 for col in range(2)] for row in range(height)]
    lfg = [[0 for col in range(2)] for row in range(width)]
    box = [0, 0, 0, 0]

    # Horizontal projection
    a = 0
    emptyImage1 = np.zeros((height, width, 3), np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            cp = closed[y, x]
            # if np.any(closed[y,x]):
            if cp == 0:
                a = a + 1
            else:
                continue
        z[y] = a
        # print z[y]
        a = 0

    # Select line split point based on horizontal projection value
    inline = 1
    start = 0
    j = 0
    for i in range(0, height):
        if inline == 1 and z[i] >= 150:  # Enter the text area from the blank area
            start = i  # record starting line split point
            # print i
            inline = 0
        elif (i - start > 3) and z[i] < 150 and inline == 0:  # Enter the blank area from the text area
            inline = 1
            hfg[j][0] = start - 2  # Save line split position
            hfg[j][1] = i + 2
            j = j + 1

    # Vertically project and split each line
    a = 0
    for p in range(0, j):
        for x in range(0, width):
            for y in range(hfg[p][0], hfg[p][1]):
                cp1 = closed[y, x]
                if cp1 == 0:
                    a = a + 1
                else:
                    continue
            v[x] = a  # Save each column of pixel values
            a = 0
        # print width
        #
        incol = 1
        start1 = 0
        j1 = 0
        z1 = hfg[p][0]
        z2 = hfg[p][1]
        for i1 in range(0, width):
            if incol == 1 and v[i1] >= 20:  # Enter the text area from the blank area
                start1 = i1  # record starting column split point
                incol = 0
            elif (i1 - start1 > 3) and v[i1] < 20 and incol == 0:  # Enter the blank area from the text area
                incol = 1
                lfg[j1][0] = start1 - 2  # Save column split location
                lfg[j1][1] = i1 + 2
                l1 = start1 - 2
                l2 = i1 + 2
                j1 = j1 + 1
                cv.rectangle(img, (l1, z1), (l2, z2), (255, 0, 0), 2)

    # cv.namedWindow("main", cv.WINDOW_NORMAL)
    # show_wait_destroy('binary', th1)
    show_wait_destroy('result', img)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
