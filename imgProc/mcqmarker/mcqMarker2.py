import numpy as np

import cv2

import itertools

from pdf2image import convert_from_path


def euDist(a, b):
    a = np.array(a[:2])
    b = np.array(b[:2])
    return np.sqrt(np.sum((a - b) * (a - b)))

def intersection(a, b):
    (s0, f0) = np.array(a)
    (s1, f1) = np.array(b)
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

def fixMissing(allCirc, brth, dpth):
    colAvg = np.zeros(brth)
    rowAvg = np.zeros(dpth)
    colCount = np.zeros(brth)
    rowCount = np.zeros(dpth)
    zero = np.array([0, 0, 0])
    for i in range(brth):
        for j in range(dpth):
            if not(circEq(allCirc[i, j], zero)):
                colAvg[i] += allCirc[i, j, 0]
                colCount[i] += 1
                rowAvg[j] += allCirc[i, j, 1]
                rowCount[j] += 1

    colAvg /= colCount
    rowAvg /= rowCount

    for i in range(brth):
        for j in range(dpth):
            if circEq(allCirc[i, j], zero):
                allCirc[i, j] = np.array([colAvg[i], rowAvg[j], 18], dtype = np.int64)

def fillNormal(allCirc, circRows, circCols, brth, dpth):
    depth = [0 for x in range(brth)]
    for i, row in enumerate(circRows):
        for a in row:
            for j, col in enumerate(circCols):
                if depth[j] == len(col):
                    continue
                b = col[depth[j]]
                if circEq(a, b):
                    depth[j] += 1
                    allCirc[j, i] = a

def findAllCircles(circles, brth, dpth):
    allCirc = np.zeros((brth, dpth, 3), dtype = np.int64)
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

    if len(circRows) == dpth and len(circCols) == brth:
        fillNormal(allCirc, circRows, circCols, brth, dpth)
    fixMissing(allCirc, brth, dpth)
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
                col.insert(idx + 1, np.array([(t[0] + b[0]) / 2, 2 * b[1] - t[1], (t[2] + b[2]) / 2], dtype = np.int64))

            idx += 1
    return circles 

def markPage(page, outFile):
    blur = cv2.GaussianBlur(page,(15,15),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    width = int(th3.shape[1]/4)
    height = int(th3.shape[0]/4)

    re = cv2.resize(th3, (width, height))
    dialated = re.copy()
    th4 = re.copy()

    col = cv2.cvtColor(re, cv2.COLOR_GRAY2RGB)

    kernel = np.ones((5, 5), np.uint8)
     
    dialated = cv2.erode(dialated, kernel, iterations=3)
    dialated = cv2.dilate(dialated, kernel, iterations=3)
    dialated = cv2.erode(dialated, kernel, iterations=3)
    dialated = cv2.dilate(dialated, kernel, iterations=3)
    dialated = cv2.erode(dialated, kernel, iterations=3)

    def findLines(re, width, height):

        low_threshold = 50
        high_threshold = 150
        copied = re.copy()
        edges = cv2.Canny(copied, low_threshold, high_threshold)

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

        lines = cv2.HoughLines(edges, rho, theta, 300)

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

            bl = np.array([(0, height), (width, height)])

            rl = np.array([(width, 0), (width, height)])

            s0 = None

            f0 = None

            sl = intersection(line, ll) if abs(theta) > 0.01 and abs(theta - np.pi) > 0.01 else (-1, -1)

            st = intersection(line, tl) if abs(theta - np.pi / 2) > 0.01 else (-1, -1)

            fb = intersection(line, bl) if abs(theta - np.pi / 2) > 0.01 else (-1, -1)

            fr = intersection(line, rl) if abs(theta) > 0.01 and abs(theta - np.pi) > 0.01 else (-1, -1)

            if (theta < np.pi / 2):
                s0 = sl if 0 < sl[1] and sl[1] < height else fb
                f0 = fr if 0 < fr[1] and fr[1] < height else st
            else:
                s0 = sl if 0 < sl[1] and sl[1] < height else st
                f0 = fr if 0 < fr[1] and fr[1] < height else fb



            shift = np.abs(np.array([b,a]))
         
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
            perpCount = 0
            ((s0, f0), th0) = b = reducedBound.pop(0)
            for ((s1, f1), th1) in reducedBound:
                if abs(np.pi/2 - abs(th0 - th1)) < 0.001:
                    if perpCount == 0:
                        perpCount += 1
                    else:
                        reducedBound.append(b)
                        break
            i += 1
        return reducedBound

    def findLongestSolid(img, boundaries):

        newLines = []
        dialated = img.copy()

        height = max(img.shape[1], img.shape[0])

        kernel = np.ones((5, 5), np.uint8)

        dialated = cv2.erode(dialated, kernel, iterations = 4)

        mask1 = np.zeros_like(img)

        i = 0

        while i < len(boundaries):
            (l1, th0) = b = boundaries.pop(0)

            maxSolid = 0

            intersections = []

            for (l2, th1) in boundaries:
                if abs(np.pi/2 - abs(th0 - th1)) < 0.001:
                    intersections.append(np.int64(intersection(l1, l2)))

            intersections.sort(key = lambda x: euDist(x, l1[0]))

            k = 1
            left = intersections[0]
            maxLine = ()
            while k < len(intersections):
                mask1.fill(0)
                right = intersections[k]
                cv2.line(mask1, left, right, (255), 3)
                mean = cv2.mean(dialated, mask = mask1)[0]
                if mean < 15:
                    length = euDist(left, right)
                    if length > maxSolid:
                        maxLine = (left, right)
                        maxSolid = length
                else:
                    left = intersections[k]

                k += 1

            if maxSolid > height / 3:
                newLines.append((maxLine, th0))


            boundaries.append(b)
            i += 1
        return newLines

    def findCorners(box):
        corners = []
        maxLen = 0.
        maxLenTh = 0.
        for i in range(len(box)):
            (a, sh0) = rb = box.pop(0)
            (s0, f0) = a
            for ((s1, f1), sh1) in box:
                if  min(euDist(s0, s1), euDist(s0, f1)) < 0.0001:
                    corners.append(tuple(s0))
                    if euDist(s0, f0) > maxLen:
                        maxLen = euDist(s0, f0)
                        maxLenTh = sh0
                elif min(euDist(f0, s1), euDist(f0, f1)) < 0.0001:
                    corners.append(tuple(f0))
                    if euDist(s0, f0) > maxLen:
                        maxLen = euDist(s0, f0)
                        maxLenTh = sh0
        return corners, maxLenTh

    def findTaskNo(tasks, dialated):
        mask1 = np.zeros_like(dialated)
        tNo = []
        for column in tasks:
            noLetters = 0
            column.sort(key = lambda x : x[1])
            for j, cir in enumerate(column):
                cv2.circle(col,(cir[0],cir[1]),cir[2],(0,255,0),4)
                mask1.fill(0)
                cv2.circle(mask1,(cir[0],cir[1]),cir[2],(255),-1)
                mean = cv2.mean(dialated, mask = mask1)[0]
                if mean < 127:
                    tNo.append(j)
                    noLetters += 1
            if noLetters != 1:
                return
        if len(tNo) == 2:
            return tNo
        else:
            return

    def fillTasks(tasks):
        tasks[0].sort(key = lambda x : x[1])
        tasks[1].sort(key = lambda x : x[1])

        avgY = 0
        topCount = 0
        for col in tasks:
            col.sort(key = lambda x : x[1])
            if abs(col[0][1] - 840) < 40:
                avgY += col[0][1]
                topCount += 1
        if topCount != 0:
            avgY /= topCount
        else:
            avgY = 840

        for q, col in enumerate(tasks):
            if abs(col[0][1] - 840) > 40:
                tmp = np.array(col)
                avgX = np.average(tmp, axis = 0)[0]
                col.insert(0, np.array([avgX, avgY, 19], dtype = np.int64))
            idx = 1
            while idx < len(col):
                t, b = col[idx - 1: idx + 1]
                if 70 < abs(b[1] - t[1]) and abs(b[1] - t[1]) < 100:
                    col.insert(idx, np.int64((t + b) / 2))
                    idx += 1
                elif idx == 8 and len(col) == 9:
                    col.insert(idx + 1, np.array([(t[0] + b[0]) / 2, 2 * b[1] - t[1], (t[2] + b[2]) / 2], dtype = np.int64))

                idx += 1
        return tasks

    boundaries = findLines(re, width, height)

    boundaries = removeDupNonPerpLines(boundaries, max(width, height))

    boundaries = findLongestSolid(th4, boundaries)

    corners, theta = findCorners(boundaries)

    print(theta)

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
            return

    tasks = splitTask(circRows[0])

    tasks = fillTasks(tasks)


    studentNumberCols = findAllCirclesSn(circRows[0])


    dialated = th4.copy()

    kernel = np.ones((3, 3), np.uint8)
     
    dialated = cv2.dilate(dialated, kernel, iterations=3)
    dialated = cv2.erode(dialated, kernel, iterations=8)
    dialated = cv2.dilate(dialated, kernel, iterations=4)



    def mark(allCirc, dialated):
        mask1 = np.zeros_like(dialated)
        choices = []
        for row in allCirc:
            choices.append(np.array([False, False, False, False, False]))
            for i, circ in enumerate(row):
                mask1.fill(0)
                cv2.circle(mask1,(circ[0], circ[1]), circ[2],(255),-1)
                mean = cv2.mean(dialated, mask = mask1)[0]
                if mean < 60:
                    choices[-1][i] = True
        return choices

    choices = []
    for group in circRows[1:]:
        allCirc = np.transpose(findAllCircles(group, 5, 30), axes = (1, 0, 2))
        choices.append(mark(allCirc, dialated))

    def findSNo(studentNumberCols, dialated):
        mask1 = np.zeros_like(dialated)
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
                return
        if len(sNo) == 7:
            sNo[2] = chr(97 + sNo[2])
            return sNo
        else:
            return
        # draw the outer circle
        # print("circle: ", i[0], ", ", i[1] , "r = ", i[2])

    
    tNo = findTaskNo(tasks, dialated)

    if tNo == None:
        print('Task Number')
        return

    sNo = findSNo(studentNumberCols, dialated)

    if sNo == None:
        print('Invalid Student Number')
        return

    stNo = ""
    ssNo = ""

    for c in tNo:
        stNo += str(c)

    for c in sNo:
        ssNo += str(c)

    questionNo = 0

    symbols = np.array(['A', 'B', 'C', 'D', 'E'])

    for c in choices:
        for row in c:
            questionNo += 1
            outFile.write(ssNo + "," + stNo + "," + str(questionNo) + "," + "".join(symbols[row]) + "\n")


    for column in tasks:
        for i in column:
            cv2.circle(col,(i[0],i[1]),i[2],(0,0,255),4)

    width = int(col.shape[1]/2)
    height = int(col.shape[0]/2)


    col = cv2.resize(col, (width, height))
    dialated = cv2.resize(dialated, (width, height))

    cv2.imshow("sheet", col)

    cv2.waitKey(100000)

    return True

outFile = open("ouput.csv", "w")

pages = convert_from_path('2018.pdf', 500)

page = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2GRAY)

markPage(page, outFile)

outFile.close()
