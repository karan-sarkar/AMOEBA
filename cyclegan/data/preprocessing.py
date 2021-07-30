import os
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
from scipy.spatial import distance
import shutil


def standard(hist,hist2):
    moyenne=0
    for i in range(len(hist2)):
	    moyenne+=hist2[i][0]*i
    moyenne=moyenne/np.sum(hist2)
    standard=0
    for i in range(len(hist)):
	    standard+=hist[i][0]*(i-moyenne)**2
    standard=standard/np.sum(hist)
    return standard

def dataengineering(dirpath,coef):
    dic={}
    counter=0
    colors = ("b", "g", "r")
    for filename in os.listdir(dirpath): 
        counter+=1
        image=cv2.imread(dirpath+'/'+filename)
        chans = cv2.split(image)
        for (chan, color) in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
                if color not in dic.keys():
                        dic[color]=hist
                else:
                        dic[color]+=hist

    dic['r']=dic['r']/counter
    dic['g']=dic['g']/counter
    dic['b']=dic['b']/counter

    dictionnaire={}
    liste=[]

    for filename in os.listdir(dirpath):
        counter+=1
        image=cv2.imread(dirpath+'/'+filename)
        chans = cv2.split(image)
        dist=0
        derivate=0
        for (chan, color) in zip(chans, colors):
                hist = cv2.calcHist([chan], [0], None, [64], [0, 256])
                difference=np.divide(np.absolute(np.transpose(dic[color])[0]-np.transpose(hist)[0]),np.power(np.transpose(dic[color])[0],1))
                ecart=standard(hist,dic[color])
                derivate+=ecart
                dist+=np.sum(difference)
        dist=derivate*0.5+dist
        liste.append(dist)
        dictionnaire[dist]=filename

    liste.sort()
    liste2=[]

    taille=int(len(os.listdir(dirpath))*coef)
    for i in range(1,taille):
        path=os.path.join(dirpath,dictionnaire[liste[-i]])
        liste2.append(path)
        
    return liste2

