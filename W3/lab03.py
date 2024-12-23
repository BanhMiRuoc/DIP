import cv2 as c
import numpy as np

def ex1(path):
    img = c.imread (path)
    Coor = {
        'left_man' : (250, 400),
        'middle_man' : (725, 300),
        'right_man' : (1200, 450)  
    }
    find_mask(img, Coor)
     
def find_mask(img, Coor):
    for man, Coor_m in Coor.items():
        mask = np.zeros(img.shape[:2], dtype="uint8")
        c.circle(mask, (Coor_m), 100, 255, -1)
        masked = c.bitwise_and(img, img, mask=mask)
        c.imwrite(f'dest/{man}.png', masked)
        img_show(masked)     
def ex2(path1, path2):
    img1 = c.imread(path1)
    img2 = c.imread(path2)
    img2 = c.resize(img2, img1.shape[1::-1])
    dst = c.addWeighted(img1, 0.5, img2, 0.5, 0)
    c.imwrite('dest/new_img.png', dst)
    img_show(dst)
def ex3(path):
    img = c.imread(path, c.IMREAD_GRAYSCALE)
    _, img_1 = c.threshold(img, 180, 255, c.THRESH_BINARY_INV)
    c.imwrite('dest/binary_1.png', img_1)
    img_show(img_1)
    
    _, img_2 = c.threshold(img, 30, 128, c.THRESH_BINARY)
    _, img_3 = c.threshold(img, 170, 255, c.THRESH_BINARY)
    maske = c.bitwise_not(img_3, img_3, mask=img_2)

    c.imwrite('dest/binary_2.png', maske)
    img_show(maske)    
def ex3_2(path):
    
    img = c.imread(path)
    _, img_new = c.threshold(img, 10,  255, c.THRESH_BINARY_INV)
    c.imwrite('dest/img_new.png', img_new)
    number_coor = {
        '32': (50, 100),
        '64': (200, 300),
        '100': (400, 170),
        '128': (200, 170),
        '180': (50, 300),
        '200': (300, 50),
        '255': (350, 300)
    }
    def find_mask(img, Coor):
        for man, Coor_m in Coor.items():
            mask = np.zeros(img.shape[:2], dtype="uint8")
            c.circle(mask, (Coor_m), 80, 255, -1)
            masked = c.bitwise_and(img, img, mask=mask)
            c.imwrite(f'dest/{man}.png', masked)
            img_show(masked)    
    find_mask(img_new, number_coor)
  
def img_show(img):
    c.imshow(f'{img}', img)
    c.waitKey(0)
    c.destroyAllWindows()  
    
def ex4(path):
    logo = c.imread(path)
    logo = c.resize(logo, (200, 200)) 
    cap = c.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        x_offset, y_offset = 10, 10
        y1, y2 = y_offset, y_offset + logo.shape[0]
        x1, x2 = x_offset, x_offset + logo.shape[1]
        roi = frame[y1:y2, x1:x2]
        gray_logo = c.cvtColor(logo, c.COLOR_BGR2GRAY)
        _, mask = c.threshold(gray_logo, 1, 255, c.THRESH_BINARY)
        mask_inv = c.bitwise_not(mask)
        frame_bg = c.bitwise_and(roi, roi, mask=mask_inv)
        logo_fg = c.bitwise_and(logo, logo, mask=mask)
        dst = c.add(frame_bg, logo_fg)
        frame[y1:y2, x1:x2] = dst
        c.imshow('Webcam Video with Logo', frame)
        
        if c.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    c.destroyAllWindows()
if __name__ == "__main__":
    path = 'src/pic_01.png'
    path_1 = 'src/pic_02.png'
    path_2 = 'src/pic_03.png'
    ex1(path)
    ex2(path, path_1)
    ex3(path_2)
    ex3_2(path_2)
    ex4(path_1)