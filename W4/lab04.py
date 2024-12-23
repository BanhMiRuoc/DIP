import cv2 
import matplotlib.pyplot as plt
import numpy as np

def ex1(path):
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    _, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    binary_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(1, 4, 2), plt.imshow(binary_otsu, cmap='gray'), plt.title('Otsu')
    plt.subplot(1, 4, 3), plt.imshow(binary_adaptive, cmap='gray'), plt.title('Adaptive')
    plt.show()

def ex2(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((10, 10), np.uint8)
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_colored, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Bounding Boxes', img_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
def ex3(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    clean_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    clean_image = cv2.morphologyEx(clean_image, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_box_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(binary_image, cmap='gray')
    ax[0].set_title('Binary Image')

    ax[1].imshow(bounding_box_image)
    ax[1].set_title('Bounding Boxes')

    plt.show()

if __name__ == "__main__":
    path1 = "pic01.png"
    path2 = "pic02.png"
    path3 = "pic03.png"
    ex1(path1)
    ex2(path2)
    ex3(path3)
    