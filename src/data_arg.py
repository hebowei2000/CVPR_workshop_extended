import numpy as np
import cv2
import glob
import argparse
import os

def main():
    1
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='../data/gt5',help='data argument file')
    args = parser.parse_args()


    paths = glob.glob(os.path.join(args.path,'*.png'))
    paths.sort()

    for path in paths:
        
        image = cv2.imread(path)
        (h,w) = image.shape[:2]
        center = (w//2, h//2)
        #rotate 45 degree
       # Rotate_45 = cv2.getRotationMatrix2D(center, 45, 1.0)
       # rotated_Img_45 = cv2.warpAffine(image,Rotate_45,(w,h))
       # cv2.imwrite(path+'_rotated45',rotated_Img_45)
        
        #.............................................................................#
        ##rotate 0 degree 
        #scale = 0.5
        Rotate_0_scale_05 = cv2.getRotationMatrix2D(center,0,0.5)
        rotate_0_scale_05_Image = cv2.warpAffine(image, Rotate_0_scale_05,(w,h))
        cv2.imwrite(path+'_rotate0_scale05',rotate_0_scale_05_Image)
        #scale = 0.8
        Rotate_0_scale_08 = cv2.getRotationMatrix2D(center,0,0.8)
        rotate_0_scale_08_Image = cv2.warpAffine(image,Rotate_0_scale_08,(w,h))
        cv2.imwrite(path+'_rotate0_scale_08',rotate_0_scale_08_Image)
        #scale = 1.0
        Rotate_0_scale_10 = cv2.getRotationMatrix2D(center,0,1.0)
        rotate_0_scale_10_Image = cv2.warpAffine(image, Rotate_0_scale_10,(w,h))
        cv2.imwrite(path+'_rotate0_scale10',rotate_0_scale_10_Image)
        #scale = 1.5
        Rotate_0_scale_15 = cv2.getRotationMatrix2D(center,0,1.5)
        rotate_0_scale_15_Image = cv2.warpAffine(image,Rotate_0_scale_15,(w,h))
        cv2.imwrite(path+'_rotate0_scale_15',rotate_0_scale_15_Image)

        #..............................................................................#


        ##rotate 90 degree
        #scale = 0.5
        Rotate_90_scale_05 = cv2.getRotationMatrix2D(center, 90, 0.5)
        rotate_90_scale_05_Image = cv2.warpAffine(image,Rotate_90_scale_05,(w,h))
        cv2.imwrite(path+'_rotate90_scale05',rotate_90_scale_05_Image)
        #scale = 0.8
        Rotate_90_scale_08 = cv2.getRotationMatrix2D(center, 90, 0.8)
        rotate_90_scale_08_Image = cv2.warpAffine(image,Rotate_90_scale_08,(w,h))
        cv2.imwrite(path+'_rotate90_scale_08',rotate_90_scale_08_Image)
        #scale = 1.0
        Rotate_90_scale_10 = cv2.getRotationMatrix2D(center,90,1.0)
        rotate_90_scale_10_Image = cv2.warpAffine(image,Rotate_90_scale_10,(w,h))
        cv2.imwrite(path+'_rotate90_scale_10',rotate_90_scale_10_Image)
        #scale = 1.5
        Rotate_90_scale_15 = cv2.getRotationMatrix2D(center,90,1.5)
        rotate_90_scale_15_Image = cv2.warpAffine(image,Rotate_90_scale_15,(w,h))
        cv2.imwrite(path+'_rotate90_scale15',rotate_90_scale_15_Image)

        #.............................................................................#

        ##rotate 180 degree
        #scale = 0.5
        Rotate_180_scale_05 = cv2.getRotationMatrix2D(center, 180, 0.5)
        rotate_180_scale_05_Image = cv2.warpAffine(image,Rotate_180_scale_05,(w,h))
        cv2.imwrite(path+'_rotate180_scale05',rotate_180_scale_05_Image)
        #scale = 0.8
        Rotate_180_scale_08 = cv2.getRotationMatrix2D(center, 180, 0.8)
        rotate_180_scale_08_Image = cv2.warpAffine(image,Rotate_180_scale_08,(w,h))
        cv2.imwrite(path+'_rotate180_scale08',rotate_180_scale_08_Image)
        #scale = 1.0
        Rotate_180_scale_10 = cv2.getRotationMatrix2D(center,180,1.0)
        rotate_180_scale_10_Image = cv2.warpAffine(image,Rotate_180_scale_10,(w,h))
        cv2.imwrite(path+'_rotate180_scale10',rotate_180_scale_10_Image)
        #scale = 1.5
        Rotate_180_scale_15 = cv2.getRotationMatrix2D(center,180,1.5)
        rotate_180_scale_15_Image = cv2.warpAffine(image,Rotate_180_scale_15,(w,h))
        cv2.imwrite(path+'_rotate180_scale15',rotate_180_scale_15_Image)

        #.............................................................................#

        ##rotate 270 degree
        #scale = 0.5
        Rotate_270_scale_05 = cv2.getRotationMatrix2D(center, 270, 0.5)
        rotate_270_scale_05_Image = cv2.warpAffine(image,Rotate_270_scale_05,(w,h))
        cv2.imwrite(path+'_rotate270_scale05',rotate_270_scale_05_Image)
        Rotate_270_scale_08 = cv2.getRotationMatrix2D(center, 270, 0.8)
        rotate_270_scale_08_Image = cv2.warpAffine(image,Rotate_270_scale_08,(w,h))
        cv2.imwrite(path+'_rotate270_scale-08',rotate_270_scale_08_Image)
        Rotate_270_scale_10 = cv2.getRotationMatrix2D(center,270,1.0)
        rotate_270_scale_10_Image = cv2.warpAffine(image,Rotate_270_scale_10,(w,h))
        cv2.imwrite(path+'_rotate270_scale10',rotate_270_scale_10_Image)
        Rotate_270_scale_15 = cv2.getRotationMatrix2D(center,270,1.5)
        rotate_270_scale_15_Image = cv2.warpAffine(image,Rotate_270_scale_15,(w,h))
        cv2.imwrite(path+'_rotate270_scale15',rotate_270_scale_15_Image)

        #................................................................................#

if __name__=='__main__':
    main()


