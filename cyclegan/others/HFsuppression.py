import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imread, imsave
import torchvision
import torch.utils.data as data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import shutil

def frequencyfilter(x,r):
    temp = np.zeros((416, 416), "float32")
    cw = 416 // 2
    ch = 416 // 2
    if 416 % 2 == 0:
        dw = r
    else:
        dw = r + 1
    if 416 % 2 == 0:
        dh = r
    else:
        dh = r + 1
    temp[cw - r:cw + dw, ch - r:ch + dh] = 1.0
    temp = np.roll(temp, -cw, axis=0)
    temp = np.roll(temp, -ch, axis=1)
    temp = torch.tensor(temp)
    temp = temp.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    
    x_hat = torch.rfft(x, 2, onesided=False)
    x_hat = x_hat * temp
    y = torch.irfft(x_hat, 2, onesided=False)
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issou")

    parser.add_argument("input", type=str,default='datatest2', help="The directory containing the samples to convert.")
    parser.add_argument("output", type=str,default='output2', help="The prefix of the output directory where the converted images will be put.")
    parser.add_argument("Rcoef", type=int, default=100, help="The coefficient for the high frequencies suppression.")
    args = parser.parse_args()


    for filename in os.listdir(args.input):
        img=imread(args.input+'/'+filename)
        inp=torch.from_numpy(img).float()
        inp=inp.permute(2,0,1)
        image=frequencyfilter(inp,args.Rcoef).numpy()
        image2=np.zeros((416,416,3))
        for k in range(416):
            for j in range(416):
                image2[k][j][0]=image[0][0][k][j]
                image2[k][j][1]=image[0][1][k][j]
                image2[k][j][2]=image[0][2][k][j]
        imsave(args.output+'/'+filename,image2)
