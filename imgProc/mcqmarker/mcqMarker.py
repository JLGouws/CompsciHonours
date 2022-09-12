import numpy as np

import cv2

from pdf2image import convert_from_path
pages = convert_from_path('2018.pdf', 500)

print(cv2.__version__)

page = cv2.cvtColor(np.array(pages[1]), cv2.COLOR_RGB2GRAY)

#th3 = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, -2)
blur = cv2.GaussianBlur(page,(15,15),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

width = int(th3.shape[1] / 5)
height = int(th3.shape[0] / 5)

re = cv2.resize(th3, (width, height))
dialated = re.copy()

col = cv2.cvtColor(re, cv2.COLOR_GRAY2RGB)

#
#contours, _ = cv2.findContours(re, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#
##circles = cv2.HoughCircles(re,cv2.HOUGH_GRADIENT,1,10, param1=80,param2=10,minRadius=6,maxRadius=10)
#circles = cv2.HoughCircles(re,cv2.HOUGH_GRADIENT_ALT,1.5,10, param1=300,param2=0.8,minRadius=6,maxRadius=10)
#
#circles = np.uint16(np.around(circles))
#
#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(col,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    #cv2.circle(col,(i[0],i[1]),2,(0,0,255),3)
#

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(re, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 200  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20  # minimum number of pixels making up a line
max_line_gap = 50  # maximum gap in pixels between connectable line segments
#
## Run Hough on edge detected image
## Output "lines" is an array containing endpoints of detected line segments
#lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                    min_line_length, max_line_gap)
#
#for line in lines:
#    for x1,y1,x2,y2 in line:
#        cv2.line(col,(x1,y1),(x2,y2),(255,0,0),5)
#

lines = cv2.HoughLines(edges, rho, theta, 200)

kernel = np.ones((5, 5), np.uint8)
 
dialated = cv2.erode(dialated, kernel, iterations=3)
dialated = cv2.dilate(dialated, kernel, iterations=3)
dialated = cv2.erode(dialated, kernel, iterations=3)
dialated = cv2.dilate(dialated, kernel, iterations=3)
dialated = cv2.erode(dialated, kernel, iterations=3)
 
# The below for loop runs till r and theta values
# are in the range of the 2d array
print(len(lines))
for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
 
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
 
    # x0 stores the value rcos(theta)
    x0 = a*r
 
    # y0 stores the value rsin(theta)
    y0 = b*r
 
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000*(-b))
 
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000*(a))

    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000*(-b))
 
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000*(a))

    state = np.array([x0,y0])
    lineStart = np.array([x0,y0])
    maxStart = np.array([x0,y0])
    maxEnd = np.array([x0,y0])
    shift = np.abs(np.array([b,a]))
    length = 0
    maxLength = 0
    online = False
    while(state[0] < re.shape[1] and state[1] < re.shape[0]):
        if(online):
            if(dialated[int(state[1]), int(state[0])] < 100):
                length += 1
            else:
                online = False
                if(length > maxLength):
                    maxLength = length;
                    maxStart[0] = lineStart[0]
                    maxStart[1] = lineStart[1]
                    maxEnd[0] = state[0]
                    maxEnd[1] = state[1]
        else:
            if(dialated[int(state[1]), int(state[0])] < 100):
                length = 1
                online = True
                lineStart[0] = state[0]
                lineStart[1] = state[1]
        state += shift;

    print(maxLength)
    print()
 
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    if(maxLength > height / 3):
        cv2.line(col, (round(maxStart[0]), round(maxStart[1])), (round(maxEnd[0]), round(maxEnd[1])), (0, 0, 255), 2)

cv2.imshow("sheet", col)

cv2.waitKey(15000)
