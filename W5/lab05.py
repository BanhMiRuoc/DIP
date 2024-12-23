import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def ex1(path):

    image = cv.imread(path)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    equalized_image = cv.equalizeHist(image_gray)

    hist_original = cv.calcHist([image_gray], [0], None, [256], [0, 256])
    hist_equalized = cv.calcHist([equalized_image], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Gray Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.plot(hist_original, color='black')
    plt.title('Histogram (Original)')
    plt.xlim([0, 256])

    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized, color='black')
    plt.title('Histogram (Equalized)')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()
def ex2(path):
    img = cv.imread(path)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    noise_img = np.zeros(gray_img.shape[:2])
    cv.randu(noise_img, 0, 256)
    noise_img = np.array(noise_img, dtype=np.uint8)
    
    noisy_gray = cv.add(gray_img, np.array(0.2*noise_img, dtype=np.uint8))
    
    plt.figure(figsize=(8, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(noise_img, cmap='gray')
    plt.title('Noise Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_gray, cmap='gray')
    plt.title('Gray Image with noise')
    plt.axis('off')
    
    
    plt.tight_layout()
    plt.show()
def ex2_1(path):
    image = cv.imread(path)
    
    blurred_images = []
    
    blurred_avg = cv.blur(image, (5, 5))
    blurred_images.append(blurred_avg)

    blurred_gaussian = cv.GaussianBlur(image, (5, 5), 0)
    blurred_images.append(blurred_gaussian)

    blurred_median = cv.medianBlur(image, 5)
    blurred_images.append(blurred_median)

    blurred_bilateral = cv.bilateralFilter(image, 9, 75, 75)
    blurred_images.append(blurred_bilateral)

    kernel = np.ones((5, 5), np.float32) / 25
    blurred_custom = cv.filter2D(image, -1, kernel)
    blurred_images.append(blurred_custom)

    plt.figure(figsize=(10, 5))
    titles = ['Original Image', 'Average Blur', 'Gaussian Blur', 'Median Blur', 'Bilateral Blur', 'Custom Kernel Blur']
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(titles[0])
    plt.axis('off')

    for i in range(5):
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv.cvtColor(blurred_images[i], cv.COLOR_BGR2RGB))
        plt.title(titles[i + 1])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
def ex3(path):
    image = cv.imread(path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Orginal Pic')
    plt.axis('off')

    smoothed = cv.GaussianBlur(image, (9, 9), 10)
    unsharped = cv.addWeighted(image, 1.25, smoothed, -0.5, 0)
    plt.subplot(1, 3, 2)
    plt.imshow(unsharped, cmap='gray')
    plt.title('addWeighted() method')
    plt.axis('off')

    kernel = np.array([ [0, -1, 0],
                        [-1, 5, -1], 
                        [0, -1, 0]])
    image_sharp = cv.filter2D(src=image, ddepth=-1, kernel=kernel)
    plt.subplot(1, 3, 3)
    plt.imshow(image_sharp, cmap='gray')
    plt.title('filter2D() method')
    plt.axis('off')

    plt.show()
      
def ex4(pathVid):
    video_path = pathVid
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_list = []
    frame_count = 100

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i}.")
            break
        frame_list.append(frame)

    if len(frame_list) > 0:
        median_frame = np.median(np.array(frame_list), axis=0).astype(dtype=np.uint8)
        gray_median_frame = cv.cvtColor(median_frame, cv.COLOR_BGR2GRAY)

        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            diff_frame = cv.absdiff(gray_frame, gray_median_frame)
            _, thresh_frame = cv.threshold(diff_frame, 25, 255, cv.THRESH_BINARY)

            combined_frame = np.hstack((
                frame,
                cv.cvtColor(diff_frame, cv.COLOR_GRAY2BGR),
                cv.cvtColor(thresh_frame, cv.COLOR_GRAY2BGR)
            ))

            cv.imshow("Video Window", combined_frame)

            if cv.waitKey(30) & 0xFF == ord('q'):
                break
    else:
        print("Error: No frames were captured from the video.")

    cap.release()
    cv.destroyAllWindows()
   
if __name__ == "__main__":
    path = 'pic01.png'
    pathVid = 'shortVideo.mp4'
    ex1(path)
    ex2(path)
    ex2_1(path)
    ex3(path)
    ex4(pathVid)