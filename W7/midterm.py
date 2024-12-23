import cv2
import numpy as np

def slove(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2] 
    sub_img = []
    x = width // 3
    y = height // 2
    sub_img.append(gray[:, :x])
    sub_img.append(gray[:y, x:])
    sub_img.append(gray[y:, x:] )

    thesholds = [110, 20, 10]
    for i in range(3):
        _, binary = cv2.threshold(sub_img[i], thesholds[i], 255, cv2.THRESH_BINARY_INV)
        sub_img[i] = binary

    sub_img[2] = cv2.morphologyEx(sub_img[2], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    sub_img[0] = cv2.morphologyEx(sub_img[0], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    tmp_height, _ = sub_img[2].shape[:2]
    tmp_y = tmp_height // 3
    sub_img2_1 = sub_img[2][:2*tmp_y, :]
    sub_img2_2 = sub_img[2][2*tmp_y:, :]
    _, tmp2_width = sub_img2_2.shape[:2]
    tmp_x = tmp2_width // 4
    sub_img2_2_1 = sub_img2_2[:, :tmp_x]
    sub_img2_2_2 = sub_img2_2[:, tmp_x:tmp_x*2]
    sub_img2_2_3 = sub_img2_2[:, tmp_x*2:tmp_x*3]
    sub_img2_2_4 = sub_img2_2[:, tmp_x*3:]

    sub_img2_2_2 = cv2.erode(sub_img2_2_2, np.ones((4, 4), np.uint8))
    sub_img2_2_2 = cv2.dilate(sub_img2_2_2, np.ones((5, 5), np.uint8))
    sub_img2_2_3 = cv2.erode(sub_img2_2_3, np.ones((3, 3), np.uint8))
    sub_img2_2_3 = cv2.dilate(sub_img2_2_3, np.ones((3, 3), np.uint8))

    sub_img2_2 = cv2.hconcat([sub_img2_2_1, sub_img2_2_2, sub_img2_2_3, sub_img2_2_4])
    sub_img[2] = cv2.vconcat([sub_img2_1, sub_img2_2])
    concat_img = cv2.hconcat([sub_img[0], cv2.vconcat([sub_img[1], sub_img[2]])])
    cv2.imshow('Concatenated Image', concat_img)

    contour, _ = cv2.findContours(concat_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for contour in contour:
        x, y, w, h = cv2.boundingRect(contour)

        if 10 < w < 70 and 50 < h < 100:
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 0, 0), 2)

    cv2.imshow('Final output image', bounding_box_image)    

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "img\\input.png"
    slove(path)
    pass