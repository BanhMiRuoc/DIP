import cv2
import numpy as np

def vehicle_motion_detection(video_source='traffic_video.mp4'):
    video = cv2.VideoCapture(video_source)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        foreground_mask = backgroundObject.apply(frame)
        _, foreground_mask = cv2.threshold(foreground_mask, 244, 255, cv2.THRESH_BINARY)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frameCopy = frame.copy()
        for cnt in contours:
            if cv2.contourArea(cnt) > 1500:
                x, y, width, height = cv2.boundingRect(cnt)
                cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)
                cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)
        stacked_frame = np.hstack((frame, foregroundPart, frameCopy))
        cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked_frame, None, fx=0.5, fy=0.5))
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

def webcam_motion_detector():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        fgmask = fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        cv2.imshow('Motion Detection', frame)
        cv2.imshow('Foreground Mask', fgmask)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vehicle_motion_detection('traffic_video.mp4')
    # webcam_motion_detector()