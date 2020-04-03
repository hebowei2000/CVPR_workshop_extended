import cv2
import os
import glob

PATH = '../data/exp_17/test_331_110'
paths= glob.glob (os.path.join(PATH,'*.png'))
paths.sort()

for path in paths:
    img = cv2.imread(path)
    ret,thresh = cv2.threshold(img,110,225,cv2.THRESH_BINARY)
    cv2.imwrite(path,thresh)
