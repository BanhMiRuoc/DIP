import cv2
import numpy as np


def filter_signs_by_color(frame):
    
    blurred = cv2.GaussianBlur(frame, (9, 9), 2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    colors = {
        "red1": {
            "lower": np.array([0, 100, 100]),
            "upper": np.array([10, 255, 255])
        },
        "red2": {
            "lower": np.array([160, 100, 100]),
            "upper": np.array([180, 255, 255])
        }, 
        "blue": {
            "lower": np.array([100, 150, 50]),
            "upper": np.array([140, 255, 255])
        },  
        "yellow": {
            "lower": np.array([20, 100, 100]),
            "upper": np.array([30, 255, 255])
        }
    }
    mask_red1 = cv2.inRange(hsv, colors["red1"]["lower"], colors["red1"]["upper"])
    mask_red2 =  cv2.inRange(hsv, colors["red2"]["lower"], colors["red2"]["upper"])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, colors["blue"]["lower"], colors["blue"]["upper"])
    mask_yellow = cv2.inRange(hsv, colors["yellow"]["lower"], colors["yellow"]["upper"])

    mask_final = mask_red | mask_blue | mask_yellow
    
    return mask_final

def detect_circles(mask):
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=100
    )
    
    positions = [] 
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            positions.append((x - r, y - r, 2 * r, 2 * r))
    return positions

def draw_bounding_boxes(frame, positions, roi_x, roi_y):
    for (x, y, w, h) in positions:
        cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + roi_x + w, y + roi_y + h), (0, 255, 255), 2)
    return frame

def detect_traffic_signs_left(current_time):
    roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    if 2000 > current_time > 0000:
        roi_x, roi_y, roi_width, roi_height = 800, 400, 200, 120
    elif 4000 > current_time >= 2000:
        roi_x, roi_y, roi_width, roi_height = 800, 350, 200, 170
    elif 5000 > current_time >= 4000:
        roi_x, roi_y, roi_width, roi_height = 800, 300, 200, 200
    elif 6000 > current_time >= 5000:
        roi_x, roi_y, roi_width, roi_height = 800, 250, 200, 250
    elif 7000 > current_time >= 6000:
        roi_x, roi_y, roi_width, roi_height = 800, 200, 200, 300
    elif 7500 > current_time >= 7000:
        roi_x, roi_y, roi_width, roi_height = 700, 100, 250, 450
    elif 9000 > current_time >= 7500:
        roi_x, roi_y, roi_width, roi_height = 300, 000, 920, 570
    elif 9500 > current_time >= 9000:
        roi_x, roi_y, roi_width, roi_height = 000, 000, 1120, 570
    elif 13000 > current_time >= 9500:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 320, 300
    elif 18000 > current_time >= 13000:
        roi_x, roi_y, roi_width, roi_height = 700, 200, 620, 300
    elif 20000 > current_time >= 18000:
        roi_x, roi_y, roi_width, roi_height = 200, 000, 1050, 550
    elif 23000 > current_time >= 20000:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 220, 200
    elif 26000 > current_time >= 23000:
        roi_x, roi_y, roi_width, roi_height = 500, 200, 620, 300
    elif 27000 > current_time >= 26000:
        roi_x, roi_y, roi_width, roi_height = 500, 000, 620, 500
    elif 28500 > current_time >= 27000:
        roi_x, roi_y, roi_width, roi_height = 100, 000, 1120, 500
    elif 30000 > current_time >= 28500:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 220, 300
    elif 33000 > current_time >= 30000:
        roi_x, roi_y, roi_width, roi_height = 750, 200, 370, 300
    elif 35000 > current_time >= 33000:
        roi_x, roi_y, roi_width, roi_height = 400, 000, 700, 400
    elif 40000 > current_time >= 35000:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 200, 300
    elif 42840 > current_time >= 40000:
        roi_x, roi_y, roi_width, roi_height = 600, 000, 500, 500
    elif 45510 > current_time >= 42840:
        roi_x, roi_y, roi_width, roi_height = 600, 000, 600, 570
    elif 48000 > current_time >= 45510:
        roi_x, roi_y, roi_width, roi_height = 600, 000, 600, 500
    elif 53000 > current_time >= 48000:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 220, 200
    elif 57000 > current_time >= 53000:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 220, 170
    elif 65000 > current_time >= 57000:
        roi_x, roi_y, roi_width, roi_height = 900, 200, 220, 270
    elif 69700 > current_time >= 65000:
        roi_x, roi_y, roi_width, roi_height = 500, 000, 620, 370
    elif 84000 > current_time >= 69700:
        roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    elif 89000 > current_time >= 84000:
        roi_x, roi_y, roi_width, roi_height = 900, 300, 220, 200
    elif 91700 > current_time >= 89000:
        roi_x, roi_y, roi_width, roi_height = 400, 000, 720, 500
    elif 92790 > current_time >= 91700:
        roi_x, roi_y, roi_width, roi_height = 400, 000, 420, 400     
    elif current_time > 92790:
        roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    return roi_x, roi_y, roi_width, roi_height

def detect_traffic_signs_right(current_time):
    roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    if 2000 > current_time > 0000:
        roi_x, roi_y, roi_width, roi_height = 1200, 400, 150, 120
    elif 4000 > current_time >= 2000:
        roi_x, roi_y, roi_width, roi_height = 1200, 350, 250, 170
    elif 5000 > current_time >= 4000:
        roi_x, roi_y, roi_width, roi_height = 1400, 300, 300, 200
    elif 6000 > current_time >= 5000:
        roi_x, roi_y, roi_width, roi_height = 1400, 250, 300, 250
    elif 7000 > current_time >= 6000:
        roi_x, roi_y, roi_width, roi_height = 1400, 200, 400, 300
    elif 7500 > current_time >= 7000:
        roi_x, roi_y, roi_width, roi_height = 1600, 100, 320, 470
    elif 35000 > current_time >= 7500:
        roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    elif 40000 > current_time >= 35000:
        roi_x, roi_y, roi_width, roi_height = 1250, 400, 300, 150
    elif 42840 > current_time >= 40000:
        roi_x, roi_y, roi_width, roi_height = 1200, 300, 720, 200
    elif 48000 > current_time >= 42840:
        roi_x, roi_y, roi_width, roi_height = 1200, 400, 200, 100
    elif 53000 > current_time >= 48000:
        roi_x, roi_y, roi_width, roi_height = 1100, 400, 300, 100
    elif 57000 > current_time >= 53000:
        roi_x, roi_y, roi_width, roi_height = 1100, 350, 300, 150
    elif 65000 > current_time >= 57000:
        roi_x, roi_y, roi_width, roi_height = 1200, 300, 400, 200
    elif 69700 > current_time >= 65000:
        roi_x, roi_y, roi_width, roi_height = 1300, 000, 620, 400
    elif 70670 > current_time >= 69700:
        roi_x, roi_y, roi_width, roi_height = 1400, 000, 520, 370
    elif 74500 > current_time >= 70670:
        roi_x, roi_y, roi_width, roi_height = 1000, 400, 220, 180
    elif 80000 > current_time >= 74500:
        roi_x, roi_y, roi_width, roi_height = 1100, 400, 220, 180
    elif 83000 > current_time >= 80000:
        roi_x, roi_y, roi_width, roi_height = 1200, 350, 720, 200
    elif 84000 > current_time >= 83000:
        roi_x, roi_y, roi_width, roi_height = 1400, 200, 520, 250
    elif 87000 > current_time >= 84000:
        roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    elif 89000 > current_time >= 87000:
        roi_x, roi_y, roi_width, roi_height = 1200, 300, 320, 250
    elif 91700 > current_time >= 89000:
        roi_x, roi_y, roi_width, roi_height = 1400, 300, 520, 200
    elif current_time >= 91700:
        roi_x, roi_y, roi_width, roi_height = 10, 10, 10, 10
    return roi_x, roi_y, roi_width, roi_height

def output_vid(cap):
    output_filename = 'videos\\task1_output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
    return out

def task1(path):
    cap = cv2.VideoCapture(path)
    out = output_vid(cap)

    while cap.isOpened():
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not ret:
            break
        elif (ret == True):
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            roi1_x, roi1_y, roi1_width, roi1_height = detect_traffic_signs_left (current_time)
            roi1 = frame[roi1_y:roi1_y + roi1_height, roi1_x:roi1_x + roi1_width]
            roi2_x, roi2_y, roi2_width, roi2_height = detect_traffic_signs_right(current_time)
            roi2 = frame[roi2_y:roi2_y + roi2_height, roi2_x:roi2_x + roi2_width]

            mask1 = filter_signs_by_color(roi1)
            positions1 = detect_circles(mask1)
            frame = draw_bounding_boxes(frame, positions1, roi1_x, roi1_y)

            mask2 = filter_signs_by_color(roi2)
            positions2 = detect_circles(mask2)
            frame = draw_bounding_boxes(frame, positions2, roi2_x, roi2_y)

            cv2.putText(frame,  
                        '52200205 - 52200051 - 52200033',  
                        (50, 50),  
                        font, 1,  
                        (0, 255, 255),  
                        2,  
                        cv2.LINE_4) 

            out.write(frame)
            cv2.imshow("Traffic Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def task2(path):
    # 1.2.1. Reading the image and converting it to grayscale
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1.2.2 Slicing image into 3 parts
    height, width = gray.shape[:2] 
    sub_img = []
    x = width // 3
    y = height // 2
    sub_img.append(gray[:, :x])
    sub_img.append(gray[:y, x:])
    sub_img.append(gray[y:, x:])

    # 1.2.3 Thresholding
    thesholds = [110, 20, 10]
    for i in range(3):
        _, binary = cv2.threshold(sub_img[i], thesholds[i], 255, cv2.THRESH_BINARY_INV)
        sub_img[i] = binary

    # 1.2.4 Morphological Operations
    sub_img[2] = cv2.morphologyEx(sub_img[2], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    sub_img[0] = cv2.morphologyEx(sub_img[0], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # 1.2.5 Slicing the bottom right sub-images to slove the noise problem of the number 9, 8, 3
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

    # 1.2.6 Applying erosion and dilation to reduce noise
    sub_img2_2_2 = cv2.erode(sub_img2_2_2, np.ones((4, 4), np.uint8))
    sub_img2_2_2 = cv2.dilate(sub_img2_2_2, np.ones((5, 5), np.uint8))
    sub_img2_2_3 = cv2.erode(sub_img2_2_3, np.ones((3, 3), np.uint8))
    sub_img2_2_3 = cv2.dilate(sub_img2_2_3, np.ones((3, 3), np.uint8))

    # 1.2.7 Concatenating the sub-images
    sub_img2_2 = cv2.hconcat([sub_img2_2_1, sub_img2_2_2, sub_img2_2_3, sub_img2_2_4])
    sub_img[2] = cv2.vconcat([sub_img2_1, sub_img2_2])
    concat_img = cv2.hconcat([sub_img[0], cv2.vconcat([sub_img[1], sub_img[2]])])
    cv2.imshow('Concatenated Image', concat_img)

    # 1.2.7 Detecting contours in an image and drawing bounding boxes
    contour, _ = cv2.findContours(concat_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_box_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for contour in contour:
        x, y, w, h = cv2.boundingRect(contour)

        if 10 < w < 70 and 50 < h < 100:
            cv2.rectangle(bounding_box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Bounding Boxes', bounding_box_image)    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def draw_name_on_frame(frame, name, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale=1, color=(255, 255, 255), thickness=2):
    """
    Hàm để viết tên lên khung hình.

    Tham số:
    - frame: Khung hình (ảnh) cần vẽ lên.
    - name: Chuỗi tên cần viết lên khung hình.
    - position: Vị trí (x, y) để viết tên trên khung hình.
    - font: Kiểu font chữ (mặc định là FONT_HERSHEY_SIMPLEX).
    - font_scale: Kích thước của chữ (mặc định là 1).
    - color: Màu của chữ, ở dạng BGR (mặc định là trắng).
    - thickness: Độ dày của nét chữ (mặc định là 2).

    Trả về:
    - frame với tên đã được viết lên.
    """
    cv2.putText(frame, name, position, font, font_scale, color, thickness)
    return frame
if __name__ == "__main__":
    path = "img\\input.png"
    path_vid = "videos\\task1.mp4"
    task1(path_vid)
    task2(path)
    pass