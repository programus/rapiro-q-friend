#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import os
import os.path
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import cv2


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


if __name__ == '__main__':
  size = (320, 240)
  with PiCamera() as camera:
    camera.resolution = size
    camera.framerate = 12
    with PiRGBArray(camera, size=size) as rawCapture:
      print('preparing detector...')
      cascade_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "haarcascade_frontalface_alt.xml")
      cascade = cv2.CascadeClassifier(cascade_fn)
      print('detector prepared.')

      prev = time.time()
      for frame in camera.capture_continuous(
        rawCapture, format='bgr', use_video_port=True
      ):
        image = frame.array
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        print('detect start')
        now = time.time()
        rects = detect(gray, cascade)
        print('detected %d faces %f' % (len(rects), time.time() - now))
        draw_rects(image, rects, (0, 255, 0))

        now = time.time()
        fps = (1 / (now - prev))
        print('fps: %f' % (fps))
        cv2.putText(image, 'fps: %f' % (fps), (20, 20), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 255, 0), lineType=cv2.LINE_AA)
        prev = now
        cv2.imshow('Frame', image)
        key = cv2.waitKey(1) & 0xFF

        rawCapture.truncate(0)

        if key == ord('q'):
          break
