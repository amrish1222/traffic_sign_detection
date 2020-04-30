# -*- coding: utf-8 -*-
"""
Created on Fri May 17 23:59:53 2019

@author: balam
"""

import glob
import cv2
import numpy as np
from scipy.misc.pilutil import imread
from PIL import Image
#from common import clock, mosaic
from sklearn.svm import SVC
import pickle

SZ = 128
CLASS_N = 10

def loadTrainCategories():
    sign, signImgsPath =  [], []
    types = ['00045', '00021', '00038', '00035', '00017', '00001', '00019', '00014']
    for ty in types:
        for file in glob.glob(f"TSR/Training/{ty}/*.ppm"):
            signImgsPath.append(file)
            sign.append(ty)
    return sign, signImgsPath

def loadTestCategories():
    sign, signImgsPath =  [], []
    types = ['00045', '00021', '00038', '00035', '00017', '00001', '00019', '00014']
    for ty in types:
        for file in glob.glob(f"TSR/Testing/{ty}/*.ppm"):
            signImgsPath.append(file)
            sign.append(ty)
    return sign, signImgsPath

def loadTsign(filePath):
    tSign = imread(filePath)
    tSign = cv2.cvtColor(tSign,cv2.COLOR_RGB2GRAY)
#    tSign = deskew(tSign)
    tSign = cv2.resize(tSign, (128,128))
    return tSign

def deskew(img):
    SZ = 128
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])    
    img = cv2.warpAffine(img, M, img.shape, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    img = cv2.resize(img, (128,128))
    return img

def get_hog() : 
    winSize = (128,128)
    cellSize = (8,8)
    blockSize = (16,16)
    blockStride = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = False
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    return hog

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 12.5, gamma = 0.50625):
#        self.model = cv2.ml.SVM_create()
#        self.model.setGamma(gamma)
#        self.model.setC(C)
#        self.model.setKernel(cv2.ml.SVM_RBF)
#        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model = SVC(C=C,
                         probability=True,
                         verbose=True,
                         decision_function_shape='ovo')

    def train(self, samples, responses):    
        self.model.fit(samples, responses)

    def predict(self, samples):
        return self.model.predict(samples)
    
    def predict_proba(self, samples):
        return self.model.predict_proba(samples)

def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    respProbs = model.predict_proba(samples)
    print("resp:", resp)
    print("lable:", labels)
    print("Pobability: ", respProbs)
    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1 - err)*100))

def chunksImg(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunksSign(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    #%%
    print('train model')
    model = SVM()
    batchSize = 16
    hog = get_hog()
    
    #%%
    sign, signImgsPath = loadTrainCategories()
    print(len(sign))
    
    genImgPath = chunksImg(signImgsPath,batchSize)
    genSign = chunksImg(sign,batchSize)
    
    count = 0
    hog_descriptors = []
    signBatchAll = []
    for imgPathBatch,signBatch in zip(genImgPath,genSign):
        count+=1
        print(count)
    #    hog_descriptors = [ hog.compute(loadTsign(file)) for file in imgPathBatch ]
    #    hog_descriptors = []
        for file in imgPathBatch:
            imgg = loadTsign(file)
    #        cv2.imshow("trainImages", imgg)
    #        cv2.waitKey(50)
            hog_descriptors.append(hog.compute(imgg))
        signBatchAll.extend(signBatch)
    hog_descriptors = np.squeeze(hog_descriptors)
    signBatchNumbers = np.array([int(string) for string in signBatchAll])
    model.train(hog_descriptors, signBatchNumbers)
    print("Train Mean Accuracy: ", model.model.score(hog_descriptors, signBatchNumbers))
    
    saveModel = open("svm_model_traffic.pkl","wb")
    pickle.dump(model, saveModel)
    saveModel.close()
    
    #%%
    
    loadModel = open("svm_model_traffic.pkl","rb")
    model = pickle.load(loadModel)
    loadModel.close()
    
    print('testing model ')
    sign, signImgsPath = loadTestCategories()
    print(len(sign))
    
    genImgPath = chunksImg(signImgsPath,batchSize)
    genSign = chunksImg(sign,batchSize)
    
    count = 0
    hog_descriptors = []
    signs = []
    for imgPathBatch,signBatch in zip(genImgPath,genSign):
        count+=1
        print(count)
        for file in imgPathBatch:
            imgg = loadTsign(file)
    #        cv2.imshow("testimg", imgg)
    #        cv2.waitKey(50)
            hog_descriptors.append(hog.compute(imgg))
        signs.extend([int(string) for string in signBatch])
    hog_descriptors = np.squeeze(hog_descriptors)
    signBatchNumbers = np.array(signs)
    #print(signBatchNumbers)
    #vis = evaluate_model(model, imgg, hog_descriptors, signBatchNumbers)
    print("Test Mean Accuracy: ", model.model.score(hog_descriptors, signBatchNumbers))