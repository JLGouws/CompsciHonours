import numpy as np

import cv2

import itertools

from pdf2image import convert_from_path
pages = convert_from_path('2018.pdf', 500)

def euDist(a, b):
    a = np.array(a[:2])
    b = np.array(b[:2])
    return np.sqrt(np.sum((a - b) * (a - b)))

def intersection(a, b):
    (s0, f0) = a
    (s1, f1) = b
    denom =  np.diff((f1 - s1) * ((f0 - s0) [::-1])) 
    num = (f1 - s1) * (np.diff(s0 * f0[::-1])) + (f0 - s0) * (np.diff(f1 * s1[::-1]))
    x, y = num / denom
    return (int(x), int(y))

def fixHoles(column, idx):
    l = 1;
    while l < len(column):
        t = column[l - 1]
        b = column[l]
        if(b[1] - t[1] > 180 and idx in [ 5, 6, 7, 8] and len(column) > 10):
            return l 
        if 60 < b[1] - t[1] and b[1] - t[1] < 100:
            column.insert(l, np.int64(np.around((t + b) / 2)))
        l += 1
    return -1

def taskSplitIdx(column):
    for i, c in enumerate(column):
        if c[1] > 770:
            return i 
    return -1

def checkFlip(circRows):
    return not(len(circRows[0]) <= 10 and len(circRows[1]) <= 10 and 10 < len(circRows[2]) and len(circRows[2]) <= 26 and len(circRows[3]) <= 10)

def flip(circRows, img):
    for g in circRows:
        for idx in range(len(g)):
            col = np.array(g[idx])
            flipped = np.zeros_like(col)
            flipped[:, 0] = 1300
            flipped[:, 1] = 1750
            g[idx] = list(np.abs(flipped - col)[::-1])
        g.reverse()
    circRows.reverse()
    return cv2.rotate(img, cv2.ROTATE_180)

def splitTask(admin):
    idx = 0
    runTop = -1
    tasks = []
    while idx < len(admin):
        colm = admin[idx]
        colm.sort(key = lambda x: x[1]) 
        runTop = colm[0][1] if runTop == -1 else (colm[0][1] + runTop) / 2
        if idx > 3 and len(colm) > 10 and abs(colm[-1][1] - colm[0][1]) > 200:
            sidx = taskSplitIdx(colm)
            if sidx != -1:
                tasks.append(colm[sidx:])
                admin[idx] = colm[:sidx]
        if colm[0][1] > 770:
            tasks.append(admin.pop(idx))
        idx += 1
    return tasks

def circEq(a, b):
    return np.all(a == b)

def fixMissing(allCirc):
    colAvg = np.zeros(5)
    rowAvg = np.zeros(30)
    colCount = np.zeros(5)
    rowCount = np.zeros(30)
    zero = np.array([0, 0, 0])
    for i in range(5):
        for j in range(30):
            if not(circEq(allCirc[i, j], zero)):
                colAvg[i] += allCirc[i, j, 0]
                colCount[i] += 1
                rowAvg[j] += allCirc[i, j, 1]
                rowCount[j] += 1

    colAvg /= colCount
    rowAvg /= rowCount

    for i in range(5):
        for j in range(30):
            if circEq(allCirc[i, j], zero):
                allCirc[i, j] = np.array([colAvg[i], rowAvg[j], 18], dtype = np.int64)

def fillNormal(allCirc, circRows, circCols):
    depth = [0, 0, 0, 0, 0]
    for i, row in enumerate(circRows):
        for a in row:
            for j, col in enumerate(circCols):
                if depth[j] == len(col):
                    continue
                b = col[depth[j]]
                if circEq(a, b):
                    depth[j] += 1
                    allCirc[j, i] = a

def findAllCircles(circles):
    allCirc = np.zeros((5, 30, 3), dtype = np.int64)
    circCols = circles.copy()
    circles = list(itertools.chain(*circles))
    circles.sort(key = lambda x : x[1])
    circRows = []
    idx = 1
    circRows.append([circles[0]])

    for col in circCols:
        col.sort(key = lambda x : x[1])

    while idx < len(circles):
        t, m = circles[idx - 1: idx + 1]
        if abs(t[1] - m[1]) < 20:
            circRows[-1].append(m)
        else:
            circRows[-1].sort(key = lambda x : x[0])
            circRows.append([m])
        idx += 1
    circRows[-1].sort(key = lambda x : x[0])

    if len(circRows) == 30 and len(circCols) == 5:
        fillNormal(allCirc, circRows, circCols)
    fixMissing(allCirc)
    return allCirc

def findAllCirclesSn(circles):
    avgY = 0
    topCount = 0
    for col in circles:
        col.sort(key = lambda x : x[1])
        if abs(col[0][1] - 280) < 40:
            avgY += col[0][1]
            topCount += 1
    if topCount != 0:
        avgY /= topCount
    else:
        avgY = 280

    for q, col in enumerate(circles):
        if abs(col[0][1] - 280) > 40:
            tmp = np.array(col)
            avgX = np.average(tmp, axis = 0)[0]
            col.insert(0, np.array([avgX, avgY, 19], dtype = np.int64))
        idx = 1
        while idx < len(col):
            t, b = col[idx - 1: idx + 1]
            if 70 < abs(b[1] - t[1]) and abs(b[1] - t[1]) < 100:
                col.insert(idx, np.int64((t + b) / 2))
                idx += 1
            elif (idx == 8 and len(col) == 9 and q != 2) or (idx == 24 and len(col) == 25 and q == 2):
                col.insert(idx, np.array([(t[0] + b[0]) / 2, 2 * b[1] - t[1], (t[2] + b[2]) / 2], dtype = np.int64))

            idx += 1
    return circles 

page = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2GRAY)

#th3 = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, -2)
blur = cv2.GaussianBlur(page,(15,15),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

width = int(th3.shape[1]/4)
height = int(th3.shape[0]/4)

re = cv2.resize(th3, (width, height))
dialated = re.copy()
th4 = re.copy()

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

kernel = np.ones((5, 5), np.uint8)
 
dialated = cv2.erode(dialated, kernel, iterations=3)
dialated = cv2.dilate(dialated, kernel, iterations=3)
dialated = cv2.erode(dialated, kernel, iterations=3)
dialated = cv2.dilate(dialated, kernel, iterations=3)
dialated = cv2.erode(dialated, kernel, iterations=3)

def findLines(re, width, height):

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

    boundaries = []
     
# The below for loop runs till r and theta values
# are in the range of the 2d array
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

        line = np.array([(x1, y1), (x2, y2)])

        tl = np.array([(0, 0), (width, 0)])

        ll = np.array([(0, 0), (0, height)])

        st = intersection(line, tl) if abs(theta - np.pi / 2) > 0.01 else (-1, -1)

        sl = intersection(line, ll) if abs(theta) > 0.01 and abs(theta - np.pi) > 0.01 else (-1, -1)

        s0 = st if 0 < st[0] and st[0] < width else sl

        bl = np.array([(0, height), (width, height)])

        rl = np.array([(width, 0), (width, height)])

        fb = intersection(line, bl) if abs(theta - np.pi / 2) > 0.01 else (-1, -1)

        fr = intersection(line, rl) if abs(theta) > 0.01 and abs(theta - np.pi) > 0.01 else (-1, -1)

        f0 = fb if 0 < fb[0] and fb[0] < width else fr

        shift = np.abs(np.array([b,a]))
#    state = np.array([x0,y0])
#    lineStart = np.array([x0,y0])
#    maxStart = np.array([x0,y0])
#    maxEnd = np.array([x0,y0])
#    shift = np.abs(np.array([b,a]))
#    length = 0
#    maxLength = 0
#    online = False
#    while(state[0] < re.shape[1] and state[1] < re.shape[0]):
#        if(online):
#            if(dialated[int(state[1]), int(state[0])] < 100):
#                length += np.sqrt(np.sum(shift * shift))
#            else:
#                online = False
#                if(length > maxLength):
#                    maxLength = length;
#                    maxStart[0] = lineStart[0]
#                    maxStart[1] = lineStart[1]
#                    maxEnd[0] = state[0]
#                    maxEnd[1] = state[1]
#        else:
#            if(dialated[int(state[1]), int(state[0])] < 100):
#                length = 0
#                online = True
#                lineStart[0] = state[0]
#                lineStart[1] = state[1]
#        state += shift;

     
        boundaries += [((s0, f0), theta)]
    return boundaries

def removeDupNonPerpLines(boundaries, height):
    maxDist = height / 20 
    reducedBound = []
    for i, b in enumerate(boundaries):
        ((s0, f0), th0) = b
        dupl = False
        for ((s1, f1), th1) in reducedBound:
            if ((abs(th0 - th1) < 0.1 or abs(np.pi - abs(th0 - th1)) < 0.1) and (min(euDist(s0, s1), euDist(f0, f1)) < maxDist)):
                dupl = True
                break
        if not dupl:
            reducedBound.append(b)
    i = 0
    while i < len(reducedBound):
        ((s0, f0), th0) = b = reducedBound.pop(0)
        print(b)
        for ((s1, f1), th1) in reducedBound:
            if abs(np.pi/2 - abs(th0 - th1)) < 0.001:
                reducedBound.append(b)
                break
        i += 1
    return reducedBound

boundaries = findLines(re, width, height)

boundaries = removeDupNonPerpLines(boundaries, max(width, height))

for ((x1, y1), (x2, y2)), shift in boundaries:
    cv2.line(col,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),5)

width = int(col.shape[1]/2)
height = int(col.shape[0]/2)


col = cv2.resize(col, (width, height))
dialated = cv2.resize(dialated, (width, height))

cv2.imshow("sheet", col)

cv2.waitKey(100000)

quit()


reducedBound = []#[boundaries.pop()]
while len(boundaries) > 0:
    reducedBound.append(boundaries.pop(0))
    (s0, f0), g0 = reducedBound[-1]
    for i in range(len(boundaries)):
        (s1, f1), g1 = b = boundaries.pop(0)
        if not ((np.sqrt(np.sum((s1 - s0) * (s1 - s0))) < height / 10 and np.sqrt(np.sum((f1 - f0) * (f1 - f0))) < height / 10) or np.sqrt(np.sum((s1 - f1) * (s1 - f1))) < height / 3):
            boundaries.append(b)


box = []
maxDist = height / 20 
for i in range(len(reducedBound)):
    ((s0, f0), sh0) = rb = reducedBound.pop(i)
    for ((s1, f1), sh1) in reducedBound:
        if (np.sum(np.round(1000 * np.abs(sh0 - sh1[::-1]))) == 0 and (min(np.sqrt(np.sum((s1 - f1) * (s1 - f1))), np.sqrt(np.sum((f0 - s0) * (f0 - s0))), np.sqrt(np.sum((s1 - s0) * (s1 - s0))), np.sqrt(np.sum((f1 - s0) * (f1 - s0)))) < maxDist)):
            box.append(rb)
    reducedBound.insert(i, rb)


corners = []
for i in range(len(box)):
    (a, sh0) = rb = box.pop(0)
    (s0, f0) = a
    for (b, sh1) in box:
        (s1, f1) = b
        if (np.sum(np.round(1000 * np.abs(sh0 - sh1[::-1]))) == 0 and (min(euDist(s0, s1), euDist(s0, f1), euDist(f0, s1), euDist(f0, f1)) < maxDist)):
            corners.append(intersection(a, b))

corners.sort()


th4 = th4[corners[0][1]: corners[1][1], corners[0][0]: corners[2][0]]
col = col[corners[0][1]: corners[1][1], corners[0][0]: corners[2][0]]

th4 = cv2.resize(th4, (1300, 1750))
col = cv2.resize(col, (1300, 1750))

circles = cv2.HoughCircles(th4,cv2.HOUGH_GRADIENT,1,8, param1=30,param2=30,minRadius=8,maxRadius=22)
#circles = cv2.HoughCircles(re,cv2.HOUGH_GRADIENT_ALT,1.5,10, param1=300,param2=0.8,minRadius=6,maxRadius=10)


circles = list(np.int64(np.around(circles))[0,:])

circles.sort(key = lambda x : x[0])


circRows = []

colidx = 0;
idx = 1
circRows.append([[circles[0]]])

while idx < len(circles):
    t, m = circles[idx - 1: idx + 1]
    if abs(t[0] - m[0]) > 180:
        circRows.append([[m]])
    elif abs(t[0] - m[0]) < 20:
        circRows[-1][-1].append(m)
    else:
        circRows[-1].append([m])
    idx += 1
    #cv2.circle(col,(i[0],i[1]),i[2],(0,255,0),4)

if checkFlip(circRows[0]):
    th4 = flip(circRows, th4)
    col = cv2.rotate(col, cv2.ROTATE_180)
    if checkFlip(circRows[0]):
        print("Make another scan")

tasks = splitTask(circRows[0])

studentNumberCols = findAllCirclesSn(circRows[0])

allCirc1 = np.transpose(findAllCircles(circRows[1]), axes = (1, 0, 2))
allCirc2 = np.transpose(findAllCircles(circRows[2]), axes = (1, 0, 2))



dialated = th4.copy()

kernel = np.ones((3, 3), np.uint8)
 
dialated = cv2.dilate(dialated, kernel, iterations=3)
dialated = cv2.erode(dialated, kernel, iterations=8)
dialated = cv2.dilate(dialated, kernel, iterations=4)

mask1 = np.zeros_like(th4)

sNo = []


for column in studentNumberCols:
    noLetters = 0
    for j, cir in enumerate(column):
        cv2.circle(col,(cir[0],cir[1]),cir[2],(0,255,0),4)
        mask1.fill(0)
        cv2.circle(mask1,(cir[0],cir[1]),cir[2],(255),-1)
        mean = cv2.mean(dialated, mask = mask1)[0]
        if mean < 127:
            sNo.append(j)
            noLetters += 1
    if noLetters != 1:
        print('Invalid Student Number')
    # draw the outer circle
    # print("circle: ", i[0], ", ", i[1] , "r = ", i[2])

sNo[2] = chr(97 + sNo[2])
print(sNo)

for column in tasks:
    for i in column:
        cv2.circle(col,(i[0],i[1]),i[2],(0,0,255),4)

for row in allCirc1:
    for i in row:
        cv2.circle(col,(i[0],i[1]),i[2],(255,0,255),4)
        mask1.fill(0)
        cv2.circle(mask1,(i[0],i[1]),i[2],(255),-1)
        mean = cv2.mean(dialated, mask = mask1)

for row in allCirc2:
    for i in row:
        cv2.circle(col,(i[0],i[1]),i[2],(255,0,255),4)

width = int(col.shape[1]/2)
height = int(col.shape[0]/2)


col = cv2.resize(col, (width, height))
dialated = cv2.resize(dialated, (width, height))

cv2.imshow("sheet", col)

cv2.waitKey(100000)
