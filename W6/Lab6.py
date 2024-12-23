import cv2
import numpy as np
import math

def edge_detection(path):
    img = cv2.imread(path) 
    cv2.imshow('Original', img)
    cv2.waitKey(0)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    cv2.imshow('Sobel X', sobelx)
    cv2.imwrite("img\\sobel_X.jpg", sobelx)

    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.imwrite("img\\sobel_Y.jpg", sobely)

    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.imwrite("img\\sobel_XY.jpg", sobelxy)
    cv2.waitKey(0)
    
    candy = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) 
    cv2.imshow('Canny Edge Detection', candy)
    cv2.imwrite("img\\candy.jpg", candy)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def lines_Hough_transform(path):
    img = cv2.imread(path) 
    cv2.imshow('Original', img)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_P = np.copy(img_gray)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
    candy = cv2.Canny(img_blur, 50, 200, None, 3) 
    lines = cv2.HoughLines(candy, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(img_gray, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", img_gray)
    cv2.imwrite("img\\Standard_Hough_Line_Transform.jpg", img_gray)

    linesP = cv2.HoughLinesP(candy, 1, np.pi / 180, 60, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_gray_P, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", img_gray_P)
    cv2.imwrite("img\\Probabilistic_Hough_Line_Transform.jpg", img_gray_P)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def circles_Hough_transform(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR) 
    cv2.imshow('Original', img)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)

    rows = img_gray.shape[0]
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30, 
                               minRadius=1, maxRadius=30)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
            
    cv2.imshow("Detected Circles", img)
    cv2.imwrite("img\\Hough_Circle_Transform.jpg", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def sudoku_grabber(path):
    sudoku = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    outerBox = np.zeros(sudoku.shape, dtype=np.uint8)

    sudoku = cv2.GaussianBlur(sudoku, (11, 11), 0)
    outerBox = cv2.adaptiveThreshold(
        sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
    )
    outerBox = cv2.bitwise_not(outerBox)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    outerBox = cv2.dilate(outerBox, kernel)

    max_area = 0
    max_pt = (0, 0)
    for y in range(outerBox.shape[0]):
        for x in range(outerBox.shape[1]):
            if outerBox[y, x] >= 128:
                area = cv2.floodFill(outerBox, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    max_pt = (x, y)

    cv2.floodFill(outerBox, None, max_pt, 255)
    for y in range(outerBox.shape[0]):
        for x in range(outerBox.shape[1]):
            if outerBox[y, x] == 64:
                cv2.floodFill(outerBox, None, (x, y), 0)

    outerBox = cv2.erode(outerBox, kernel)

    lines = cv2.HoughLines(outerBox, 1, np.pi / 180, 200)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(outerBox, (x1, y1), (x2, y2), 128, 2)

    cv2.imshow("lines", outerBox)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    path_1 = "img\\sudoku_original.jpg"
    path_2 = "img\\hough_circles_demo_01.png"
    # edge_detection(path_1)
    # lines_Hough_transform(path_1)
    # circles_Hough_transform(path_2)
    sudoku_grabber(path_1)