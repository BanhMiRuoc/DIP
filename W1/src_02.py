import cv2 as c
import matplotlib.pyplot as plt
import numpy as np

#ex1 - Arithmetic operations on Images
def ex1():
    img1 = c.imread('src_img/cat.jpg')
    img2 = c.imread('src_img/flower_fire.jpg')
    
    if img1.shape != img2.shape:
        img2 = c.resize(img2, (img1.shape[1], img1.shape[0]))

    img_add = c.addWeighted(img1, 0.5, img2, 0.2, 0)
    c.imshow('add', img_add)
    c.waitKey(0)
    img_sub = c.subtract(img1, img2)
    c.imshow('sub', img_sub)
    c.waitKey(0)
    c.destroyAllWindows()
#ex2 - Bitwise Operations on Binary Images
def ex2():
    img1 = c.imread('src_img/cat.jpg')
    img2 = c.imread('src_img/flower_fire.jpg')
    
    if img1.shape != img2.shape:
        img2 = c.resize(img2, (img1.shape[1], img1.shape[0]))

    dest_and = c.bitwise_and(img2, img1, mask=None)
    c.imshow('bitwise and', dest_and)
    c.waitKey(0)
    dest_or = c.bitwise_or(img2, img1, mask=None)
    c.imshow('bitwise or', dest_or)
    c.waitKey(0)
    dest_xor = c.bitwise_xor(img2, img1, mask=None)
    c.imshow('bitwise xor', dest_xor)
    c.waitKey(0)
    dest_not = c.bitwise_not(img2, img1, mask=None)
    c.imshow('bitwise not', dest_not)
    c.waitKey(0)
    c.destroyAllWindows()
#ex3 - Image Resizing
def ex3():
        
    image = c.imread('src_img/cat.jpg')
    # Loading the image

    half = c.resize(image, (0, 0), fx = 0.1, fy = 0.1)
    bigger = c.resize(image, (1050, 1610))

    stretch_near = c.resize(image, (780, 540), 
                interpolation = c.INTER_LINEAR)


    Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
    images =[image, half, bigger, stretch_near]
    count = 4

    for i in range(count):
        plt.subplot(2, 2, i + 1)
        plt.title(Titles[i])
        plt.imshow(images[i])
    plt.show()
#ex4 - Create Border around Images
def ex4(path):
    image = c.imread(path) 
   
    # Window name in which image is displayed 
    window_name = 'Image'
    
    # Using c.copyMakeBorder() method 
    image = c.copyMakeBorder(image, 10, 10, 10, 10, c.BORDER_CONSTANT, None, value = 0) 
    
    # Displaying the image  
    c.imshow(window_name, image) 
    c.waitKey(0)
    c.destroyAllWindows()
#ex5 - Grayscaling of Images
def ex5():
    # Load the input image
    image = c.imread('src_img/cat.jpg')
    c.imshow('Original', image)
    c.waitKey(0)

    # Use the cvtColor() function to grayscale the image
    gray_image = c.cvtColor(image, c.COLOR_BGR2GRAY)

    c.imshow('Grayscale', gray_image)
    c.waitKey(0)  
    c.destroyAllWindows()
#ex6 - Scaling, Rotating, Shifting and Edge Detection
def ex6():
    # Load the image
    image = c.imread('src_img/cat.jpg')

    # Convert BGR image to RGB
    image_rgb = c.cvtColor(image, c.COLOR_BGR2RGB)

    # Define the scale factor
    # Increase the size by 3 times
    scale_factor_1 = 3.0  
    # Decrease the size by 3 times
    scale_factor_2 = 1/3.0

    # Get the original image dimensions
    height, width = image_rgb.shape[:2]

    # Calculate the new image dimensions
    new_height = int(height * scale_factor_1)
    new_width = int(width * scale_factor_1)

    # Resize the image
    zoomed_image = c.resize(src =image_rgb, 
                            dsize=(new_width, new_height), 
                            interpolation=c.INTER_CUBIC)

    # Calculate the new image dimensions
    new_height1 = int(height * scale_factor_2)
    new_width1 = int(width * scale_factor_2)

    # Scaled image
    scaled_image = c.resize(src= image_rgb, 
                            dsize =(new_width1, new_height1), 
                            interpolation=c.INTER_AREA)

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    # Plot the original image
    axs[0].imshow(image_rgb)
    axs[0].set_title('Original Image Shape:'+str(image_rgb.shape))

    # Plot the Zoomed Image
    axs[1].imshow(zoomed_image)
    axs[1].set_title('Zoomed Image Shape:'+str(zoomed_image.shape))

    # Plot the Scaled Image
    axs[2].imshow(scaled_image)
    axs[2].set_title('Scaled Image Shape:'+str(scaled_image.shape))

    # Remove ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the subplots
    plt.tight_layout()
    plt.show()

#ex7 - Convert an image from one color space to another
def ex7(path): 
    # Reading an image in default mode
    src = c.imread(path)
    window_name = 'Image'
    image = c.cvtColor(src, c.COLOR_BGR2GRAY )

    # Displaying the image 
    c.imshow(window_name, image)
    c.waitKey(0)  
    c.destroyAllWindows()

#ex8 - Filter Color with OpenCV
def ex8():
    cap = c.VideoCapture(0) 
    
    while(1): 
        _, frame = cap.read() 
        # It converts the BGR color space of image to HSV color space 
        hsv = c.cvtColor(frame, c.COLOR_BGR2HSV) 
        
        # Threshold of blue in HSV space 
        lower_blue = np.array([60, 35, 140]) 
        upper_blue = np.array([180, 255, 255]) 
    
        # preparing the mask to overlay 
        mask = c.inRange(hsv, lower_blue, upper_blue) 
        
        # The black region in the mask has the value of 0, 
        # so when multiplied with original image removes all non-blue regions 
        result = c.bitwise_and(frame, frame, mask = mask) 
    
        c.imshow('frame', frame) 
        c.imshow('mask', mask) 
        c.imshow('result', result) 
        
        c.waitKey(0) 
    c.destroyAllWindows() 
    cap.release() 
#ex9 - Visualizing image in different color spaces
def ex9():
    img = c.imread('src_img/cat.jpg')  
  
    # We can alternatively convert 
    # image by using cv2color 
    img = c.cvtColor(img, c.COLOR_BGR2GRAY) 
    
    # Shows the image 
    c.imshow('image', img)  
    
    c.waitKey(0)          
    c.destroyAllWindows() 
if __name__ == "__main__":
    # ex1()
    # ex2()
    # ex3()
    # ex4('src_img/cat.jpg')
    # ex5()
    # ex6()
    # ex7('src_img/cat.jpg')
    # ex8()
    ex9()