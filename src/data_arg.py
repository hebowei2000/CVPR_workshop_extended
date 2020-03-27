import numpy as np
import cv2
import glob
import argparse
import os

def main():
    1
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='../data/train_argu',help='data argument file')
    args = parser.parse_args()


    paths = glob.glob(os.path.join(args.path,'*.png'))
    paths.sort()

    for path in paths:
        1
        image = cv2.imread(path)
        (h,w) = image.shape[:2]
        center = (w//2, h//2)
        #rotate 45 degree
        Rotate_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated_Img_45 = cv2.warpAffine(image,Rotate_45,(w,h))
        cv2.imwrite(path+'_rotated45',rotated_Img_45)

        #rotate 90 degree
        Rotate_90 = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated_Img_90 = cv2.warpAffine(image,Rotate_90,(w,h))
        cv2.imwrite(path+'_rotated90',rotated_Img_90)

        #rotate 135 degree
        Rotate_135 = cv2.getRotationMatrix2D(center, 135, 1.0)
        rotated_Img_135 = cv2.warpAffine(image,Rotate_135,(w,h))
        cv2.imwrite(path+'_rotate135',rotated_Img_135)

        #rotate 180 degree
        Rotate_180 = cv2.getRotationMatrix2D(center, 180, 1.0)
        rotated_Img_180 = cv2.warpAffine(image,Rotate_180,(w,h))
        cv2.imwrite(path+'_rotate180',rotated_Img_180)

if __name__=='__main__':
    main()


