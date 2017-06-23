#!/usr/bin/env python

from picamera import PiCamera
from picamera.array import PiRGBArray
from imutils import face_utils
import time
import cv2
import dlib

if __name__ == '__main__':
  size = (256, 192)
  with PiCamera() as camera:
    camera.resolution = size
    camera.framerate = 12
    with PiRGBArray(camera, size=size) as rawCapture:
      print 'preparing detector...'
      detector = dlib.get_frontal_face_detector()
      print 'detector prepared.'

      prev = time.time()
      for frame in camera.capture_continuous(
        rawCapture, format='bgr', use_video_port=True
      ):
        image = frame.array
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        print 'detect start'
        now = time.time()
        faces = detector(gray, 0)
        print 'detected %d faces %f' % (len(faces), time.time() - now)

        for rect in faces:
          (x, y, w, h) = face_utils.rect_to_bb(rect)
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time()
        fps = (1 / (now - prev))
        print 'fps: %f' % (fps)
        prev = now
        cv2.imshow('Frame', image)
        key = cv2.waitKey(1) & 0xFF

        rawCapture.truncate(0)

        if key == ord('q'):
          break
