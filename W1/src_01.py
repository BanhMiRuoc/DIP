import cv2 as c

#ex1 - reading and display an image
def ex1(path):
    img = c.imread(path, c.IMREAD_ANYCOLOR)

    c.imshow("cute cat", img)
    c.waitKey(0)

    c.destroyAllWindows() 
# ex1('src_img\cat_1.jpg')

#ex2 - writing an image
def ex2(path):
    img = c.imread(path)
    file_name = 'src_img/new_cat.jpg'
    
    c.imwrite(file_name, img)
# ex2('src_img/cat.jpg')

#ex3 - draw a text string
'''
putText(img, text, org, font, fontScale, color, thickness, linetype)
    draw a text string on any image
    img: img path
    text: string
    org: coordinates of text in the image (x, y)
    font: FONT_
    fontscale: weight of fontsize
    color: RGB()
    thickness: bold
'''
def ex3(path):
    img = c.imread(path)
    img = c.putText(img, 'bmr', (50,50), c.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    c.imshow('cat_name', img)
    c.waitKey(0)
    c.destroyAllWindows() 
# ex3('src_img/new_cat.jpg')

#ex4 - draw a line
'''
line(img, start_point, end_point, color, thickness)
    start: (x, y)
    end: (x, y)
    thickness: px
'''
def ex4(path):
    img = c.imread(path)
    img = c.line(img, (0, 0), (255, 255), (255, 0, 0), 2)
    
    c.imshow('cat_line', img)
    c.waitKey(0)
    c.destroyAllWindows()
# ex4('src_img/new_cat.jpg')

#ex5 - draw arrow segment
def ex5(path):
    img = c.imread(path)
    
    img = c.arrowedLine(img, (0,0), (255,255), (255,0,0), 2)
    
    c.imshow('cat_arrow', img)
    c.waitKey(0)
    c.destroyAllWindows()
# ex5('src_img/cat.jpg')
#ex6 - draw an ellipse
'''
cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])
    anh
    toa do tam
    toa do 2 truc 
    goc quay
    
'''
def ex6(path):
    img = c.imread(path)
    
    img = c.ellipse(img, (150, 50), (100, 30),0, 0, 360, (255, 0, 0), 4)
    
    c.imshow('cat_arrow', img)
    c.waitKey(0)
    c.destroyAllWindows()
# ex6('src_img/cat.jpg')
#ex7- draw a circle
def ex7(path):
    img = c.imread(path)

    img = c.circle(img, (50,50), 30, (255, 0, 0), 2)  
    
    c.imshow('cat_arrow', img)
    c.waitKey(0)
    c.destroyAllWindows()
# ex7('src_img/cat.jpg')
#ex8 - draw a rectangle
def ex8(path):
    img = c.imread(path)
    
    img = c.rectangle(img, (0,0), (255, 255), (255, 0, 0), 4)
    
    c.imshow('cat_arrow', img)
    c.waitKey(0)
    c.destroyAllWindows()
# ex8('src_img/cat.jpg')
#ex9 - color spaces
def ex9(path):
    img = c.imread(path)
    
    R, G, B = c.split(img)
    
    c.imshow('cat_arrow', img)
    c.waitKey(0)
    c.imshow('blue', B)
    c.waitKey(0)
    c.imshow('blue', G)
    c.waitKey(0)
    c.imshow('blue', R)
    c.waitKey(0)
    c.destroyAllWindows()
# ex9('src_img/cat.jpg')



