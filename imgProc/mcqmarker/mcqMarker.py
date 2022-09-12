import numpy as np

import cv2

from pdf2image import convert_from_path
pages = convert_from_path('pg1.pdf', 500)

page = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2GRAY)

#ret3, thrs = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY+cv2.THRESH_OTSU, 11, 10)
blur = cv2.GaussianBlur(page,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contours, _ = cv2.findContours(th3, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(contours[0].shape)

cv2.drawContours(th3, contours, -1, (0,255,0), 3)

re = cv2.resize(th3, ((int) (th3.shape[0] / 4), (int) (th3.shape[1] / 4)))

cv2.imshow("sheet", re)

cv2.waitKey(100000)
