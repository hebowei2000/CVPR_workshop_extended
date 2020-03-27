import cv2
import os
import glob

PATH = '../data/exp_9/test80'
paths= glob.glob (os.path.join(PATH,'*.png'))
paths.sort()

for path in paths:
    img = cv2.imread(path)
    ret,thresh = cv2.threshold(img,80,225,cv2.THRESH_BINARY)
    cv2.imwrite(path,thresh)
