import os
import cv2
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issou")

    parser.add_argument("input", type=str,default='datatest2', help="The directory containing the samples to convert.")
    parser.add_argument("output", type=str,default='output2', help="The prefix of the output directory where the converted images will be put.")
    parser.add_argument("Qcoef", type=int, default=100, help="The coefficient for the compression.")

    args = parser.parse_args()
    
    for filename in os.listdir(args.input):
        img=cv2.imread(args.input+'/'+filename)
        cv2.imwrite(args.output+'/'+filename[0:-4]+'.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, args.Qcoef])

