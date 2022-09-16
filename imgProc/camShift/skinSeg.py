#!/usr/bin/env python

'''
Camshift tracker
================

This is a demo that shows mean-shift based tracking
You select a color objects such as your face and it tracks it.
This reads from video camera (0 by default, or the camera number the user enters)

http://www.robinhewitt.com/research/track/camshift.html

Usage:
------
    camshift.py [<video source>]

    To initialize tracking, select the object with mouse

Keys:
-----
    ESC   - exit
    b     - toggle back-projected probability visualization
'''

# Python 2/3 compatibility
from __future__ import print_function
import argparse
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets


class App(object):
    def __init__(self, video_src, face_cascade_name, eyes_cascade_name):
        self.cam = video.create_capture(video_src, presets['cube'])
        self.face_cascade_name = face_cascade_name
        self.eyes_cascade_name = eyes_cascade_name
        _ret, self.frame = self.cam.read()
        cv.namedWindow('camshift')
        cv.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

        self.face_cascade = cv.CascadeClassifier()
        self.eyes_cascade = cv.CascadeClassifier()
        #-- 1. Load the cascades
        if not self.face_cascade.load(cv.samples.findFile(self.face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        if not self.eyes_cascade.load(cv.samples.findFile(self.eyes_cascade_name)):
            print('--(!)Error loading eyes cascade')
            exit(0)

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            vis = self.frame.copy()
            self.track_window = None
            frame_gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)
            frame_gray = cv.equalizeHist(frame_gray)
            #-- Detect faces
            faces = self.face_cascade.detectMultiScale(frame_gray)
            if len(faces) < 1:
                return
            (x,y,w,h) = faces[0]
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = self.eyes_cascade.detectMultiScale(faceROI)
            if len(eyes) != 2:
                print("problemDetecting eyes")
                return
            else:
                eyes = list(eyes)
                eyes.sort(key = lambda x: x[0])
            (x1,y1,w1,h1), (x2,y2,w2,h2) = eyes
            rx1 = x1 + w1
            self.drag_start = (x, y)
            self.selection = (x + rx1, y + y1, x + x2 , y + y1 + h1)

        if self.drag_start:
            vis = self.frame.copy()
            frame_gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)
            frame_gray = cv.equalizeHist(frame_gray)
            #-- Detect faces
            faces = self.face_cascade.detectMultiScale(frame_gray)
            if len(faces) < 1:
                return
            (x,y,w,h) = faces[0]
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = self.eyes_cascade.detectMultiScale(faceROI)
            if len(eyes) != 2:
                print("problemDetecting eyes")
                return
            else:
                eyes = list(eyes)
                eyes.sort(key = lambda x: x[0])
            (x1,y1,w1,h1), (x2,y2,w2,h2) = eyes
            rx1 = x1 + w1

            self.selection = (x + rx1, y + y1, x + x2 , y + y1 + h1)

        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = self.selection
#        if self.drag_start:
#            xmin = min(x, self.drag_start[0])
#            ymin = min(y, self.drag_start[1])
#            xmax = max(x, self.drag_start[0])
#            ymax = max(y, self.drag_start[1])
#            self.selection = (xmin, ymin, xmax, ymax)
#        if event == cv.EVENT_LBUTTONUP:
#            self.drag_start = None
#            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def run(self):
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
                cv.rectangle(vis, (x0, y0), (x1, y1), (255,0,0), 2)

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try:
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)
                except:
                    print(track_box)

            cv.imshow('camshift', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv.destroyAllWindows()


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    parser.add_argument('--video', help='Camera divide number.', type=str, default="")
    args = parser.parse_args()
    if args.video != "":
        video_src = sys.argv[1]
    else:
        video_src = args.camera
    print(__doc__)
    App(video_src, args.face_cascade, args.eyes_cascade).run()
