import numpy as n
import cv2 as c

def ex1(path, channel):
    img_ori = c.imread(path)
    img_out = choose_channel(channel, img_ori)

    show_img(channel + ' channel', img_out)

def choose_channel(channel, img_ori):
    b, g, r = c.split(img_ori)
    if channel == 'red':
        return r
    elif channel == 'blue':
        return b
    elif channel == 'green':
        return g
    else:
        return img_ori
    
def show_img(text, img):
    c.imshow(text, img)
    c.waitKey(0)
    c.destroyAllWindows()

def ex2_3(path):
    image = c.imread(path)
    output_image = image.copy()

    hsv_image = c.cvtColor(image, c.COLOR_BGR2HSV)

    color_ranges = {
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'blue': ((90, 50, 50), (130, 255, 255)),
        'red': ((0, 100, 100), (5, 255, 255)), 
        'green': ((40, 50, 50), (90, 255, 255)),
        'orange': ((10, 100, 100), (25, 255, 255)) 
    }

    font = c.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 0, 0)
    font_thickness = 2

    for color, (lower, upper) in color_ranges.items():
        mask = c.inRange(hsv_image, n.array(lower), n.array(upper))
        kernel = n.ones((5, 5), n.uint8)
        mask = c.morphologyEx(mask, c.MORPH_CLOSE, kernel)
        
        contours, _ = c.findContours(mask, c.RETR_EXTERNAL, c.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = c.contourArea(contour)
            if area > 900:  # Adjust area threshold if necessary
                x, y, w, h = c.boundingRect(contour)
                c.putText(output_image, color, (x, y - 10), font, font_scale, font_color, font_thickness, c.LINE_AA)
                c.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    show_img('Balloons bounding', output_image)
def extract_yellow_balloon(path):
    image = c.imread(path)
    hsv_image = c.cvtColor(image, c.COLOR_BGR2HSV)

    yellow_range = ((20, 100, 100), (30, 255, 255))

    mask = c.inRange(hsv_image, n.array(yellow_range[0]), n.array(yellow_range[1]))
    kernel = n.ones((5, 5), n.uint8)
    mask = c.morphologyEx(mask, c.MORPH_CLOSE, kernel)

    contours, _ = c.findContours(mask, c.RETR_EXTERNAL, c.CHAIN_APPROX_SIMPLE)

    yellow_balloon_image = n.zeros_like(image)

    for contour in contours:
        area = c.contourArea(contour)
        if area > 900:
            x, y, w, h = c.boundingRect(contour)
            yellow_balloon_image[y:y + h, x:x + w] = image[y:y + h, x:x + w]

    yellow_balloon_image = c.bitwise_and(yellow_balloon_image, yellow_balloon_image, mask=mask)
    show_img('Extracted Yellow Balloon', yellow_balloon_image)
def extract_yellow_balloon_auto(path):
    image = c.imread(path)
    
    hsv_image = c.cvtColor(image, c.COLOR_BGR2HSV)

    lower_yellow = n.array([20, 100, 100]) 
    upper_yellow = n.array([30, 255, 255])

    mask = c.inRange(hsv_image, lower_yellow, upper_yellow)
    kernel = n.ones((5, 5), n.uint8)
    mask = c.morphologyEx(mask, c.MORPH_CLOSE, kernel)

    yellow_balloon_image = c.bitwise_and(image, image, mask=mask)
    show_img('Extracted Yellow Balloon', yellow_balloon_image)

def repaint_yellow_balloon(path):
    image = c.imread(path)
    
    hsv_image = c.cvtColor(image, c.COLOR_BGR2HSV)

    lower_yellow = n.array([20, 100, 100])
    upper_yellow = n.array([30, 255, 255])

    mask = c.inRange(hsv_image, lower_yellow, upper_yellow)

    result_image = image.copy()

    result_image[mask > 0] = [0, 255, 0]

    show_img('Repainted Yellow Balloon', result_image)
    
def rotate_first_balloon(path):
    image = c.imread(path)
    hsv_image = c.cvtColor(image, c.COLOR_BGR2HSV)
    lower_yellow = n.array([20, 100, 100])
    upper_yellow = n.array([30, 255, 255])
    mask = c.inRange(hsv_image, lower_yellow, upper_yellow)
    contours, _ = c.findContours(mask, c.RETR_EXTERNAL, c.CHAIN_APPROX_SIMPLE)

    if contours:
        first_contour = contours[0]
        area = c.contourArea(first_contour)
        if area > 900:
            x, y, w, h = c.boundingRect(first_contour)
            balloon_region = image[y:y+h, x:x+w]
            center = (w // 2, h // 2)
            rotation_matrix = c.getRotationMatrix2D(center, 20, 1.0)
            rotated_balloon = c.warpAffine(balloon_region, rotation_matrix, (w, h))
            result_image = image.copy()
            result_image[y:y+h, x:x+w] = rotated_balloon
            show_img('Rotated First Balloon', result_image)
def capture_face_image():
    cap = c.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def increase_brightness(image, value=50):
    hsv = c.cvtColor(image, c.COLOR_BGR2HSV)
    hsv[:, :, 2] = n.clip(hsv[:, :, 2] + value, 0, 255)
    return c.cvtColor(hsv, c.COLOR_HSV2BGR)

def histogram_equalization(image):
    gray = c.cvtColor(image, c.COLOR_BGR2GRAY)
    equalized = c.equalizeHist(gray)
    return c.cvtColor(equalized, c.COLOR_GRAY2BGR)

def adaptive_histogram_equalization(image):
    gray = c.cvtColor(image, c.COLOR_BGR2GRAY)
    clahe = c.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    return c.cvtColor(clahe_image, c.COLOR_GRAY2BGR)
def ex2_1():
    face_image = capture_face_image()
    if face_image is not None:
        bright_image = increase_brightness(face_image)
        hist_eq_image = histogram_equalization(face_image)
        clahe_image = adaptive_histogram_equalization(face_image)

        c.imshow('Original Image', face_image)
        c.imshow('Brightened Image', bright_image)
        c.imshow('Histogram Equalized Image', hist_eq_image)
        c.imshow('CLAHE Image', clahe_image)
        c.waitKey(0)
        c.destroyAllWindows()
    else:
        print("Failed to capture image.")
if __name__ == "__main__":
    path = "IMG_EX1.png"
    channel = 'red'
    ex1(path, channel)
    ex2_3(path)
    extract_yellow_balloon(path)
    extract_yellow_balloon_auto(path)
    repaint_yellow_balloon(path)
    rotate_first_balloon(path)
    ex2_1()
    
    

