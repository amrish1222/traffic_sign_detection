import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2
import glob
import pickle

import Hog_Batch as hb
from Hog_Batch import SVM

def imadjust(x,gamma=1):
    a = x.min()
    b = x.max()
    c = 0
    d = 1
    den = b-a
    if den< 1e-8:
        den = 1e-8
    y = (((x - a) / den) ** gamma) * (d - c) + c
    return y

def resize(image, x,y):
    img = image.copy()
    img = cv2.resize(img,None,fx=x,fy=y)
    return img   

def getBlueMask(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 90, 0])
    upper_blue = np.array([130, 255, 250])
    mask = cv2.inRange(img, lower_blue, upper_blue)
    return mask

def getRedMask(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 66, 73])
    upper_red = np.array([8, 203, 160])
    mask = cv2.inRange(img, lower_red, upper_red)
    return mask

def insideContourColor(image, contour):
    mask = np.zeros(image.shape,dtype='uint8')
    cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
    ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    result= cv2.bitwise_and(image,image,mask=mask)
    area = cv2.contourArea(contour)
    if cv2.countNonZero(result) > 0.05* area:
        return True, mask
    else:
        return False, mask

def get1Cnt(image, cnts):
    mask = np.zeros(image.shape,dtype='uint8')
    for cnt in cnts:
        cv2.fillPoly(mask, pts =[cnt], color=(255,255,255))
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    return contours

def prep4Classfier(image,x,y,w,h):
    pad = 10
    isOut = True
    while isOut:
        if x - pad <=0  or x+w+pad >image.shape[1] or y-pad <= 0 or y+h+pad > image.shape[0]:
            pad -=1
            isOut = True
        else:
            isOut = False
    image = image[y-pad:y+h+pad, x-pad:x+w+pad]
    
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res = cv2.resize(grey,(128,128))
    return res

def getSignPrediction(prediction):
    signImg = sign1
    if prediction == 1 :
        signImg = sign1
    elif prediction == 14:
        signImg = sign14
    elif prediction == 17:
        signImg = sign17
    elif prediction == 19:
        signImg = sign19
    elif prediction == 21:
        signImg = sign21
    elif prediction == 35:
        signImg = sign35
    elif prediction == 38:
        signImg = sign38
    elif prediction == 45:
        signImg = sign45
    else:
        signImg = sign19
    return signImg

def placeSign(image, prediction, row, col, proba):
    signImg = getSignPrediction(prediction)
    if np.any(proba>0.6):
        pass
    else:
        return image
        signImg = cv2.cvtColor(signImg,cv2.COLOR_BGR2GRAY)
        signImg = cv2.cvtColor(signImg, cv2.COLOR_GRAY2BGR)
    
    image[row:row+128,col:col+128,:] = signImg[:,:,:]
    return image



sign1 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/1.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign14 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/14.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign17 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/17.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign19 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/19.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign21 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/21.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign35 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/35.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign38 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/38.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])
sign45 = cv2.copyMakeBorder(cv2.resize(cv2.imread(glob.glob(f"signs/45.png")[0]),(108,108)),10,10,10,10,cv2.BORDER_CONSTANT,value=[255,0,0])



model = hb.SVM()
hog = hb.get_hog()

loadModel = open("svm_model_traffic.pkl","rb")
model = pickle.load(loadModel)
loadModel.close()

delta = 3
maxVar = 0.5
minDiv = 0.2
minArea = 300
maxArea = 4500
pad = 10

mser = cv2.MSER_create(delta, minArea, maxArea, maxVar, minDiv )
imgList=glob.glob(f"TSR/input/*.jpg")
imgList.sort()
outImg = cv2.imread(imgList[0])
out = cv2.VideoWriter('signDetect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (outImg.shape[1],outImg.shape[0]))
for img in imgList[70:]:

    image=cv2.imread(img)
    cv2.imshow("frame", resize(image,0.7,0.7))
    frame = image[0:int(image.shape[0]/1),:]
    frame=cv2.fastNlMeansDenoisingColored(frame,None,10,48,7,5)
    
    B,G,R = cv2.split(frame)
    
    arrB = np.asarray(B)
    B_=imadjust(arrB,1)
    
    arrG = np.asarray(G)
    G_=imadjust(arrG,1)
    
    arrR = np.asarray(R)
    R_=imadjust(arrR,1)
    
    norm_image = cv2.merge((B_,G_,R_))
    
    B,G,R = cv2.split(norm_image)
    
    roi = []
    posList = []
# Red ==================================================================
    nred = np.maximum(0, np.minimum(R-B, R-G)/(R+G+B+1e-8)) * 255
    nred = nred.astype(np.uint8)
    indices = nred<np.max(nred)/5
    nred[indices] = 0
    vis = image.copy()
    
    regions, _ = mser.detectRegions(nred)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    
    bestR = []
    
    for hull in hulls:
        if len(hull) > 5:
            
            xr,yr,w,h = cv2.boundingRect(hull)
            
            ell = cv2.fitEllipse(hull)
            (x,y),(MA,ma),angle = ell
            
            if   0.8<= MA/ma <=1.2 and 0.8 <= w/h <=1.2:
                bestR.append(hull)
    if len(bestR)>0:
        cntsR = get1Cnt(image, bestR)
        for cntR in cntsR:
            hasColor, mask = insideContourColor(nred,cntR)
            if hasColor:
    #                cv2.drawContours(vis, [cnt], 0, (0,0,255), 3)
                x,y,w,h = cv2.boundingRect(cntR)
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)
                roi.append(prep4Classfier(frame,x,y,w,h))
                posList.append((x-128,y,x+w,y))
    nblue = np.maximum(0,(B-R)/(R+G+B+1e-8)) * 255
    nblue = nblue.astype(np.uint8)
    indices = nblue<np.max(nblue)/5
    nblue[indices] = 0
    
    
    regions, _ = mser.detectRegions(nblue)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    

    bestB = []
    for hull in hulls:
        if len(hull) > 5:
            
            xr,yr,w,h = cv2.boundingRect(hull)
            
            ell = cv2.fitEllipse(hull)
            (x,y),(MA,ma),angle = ell
            
            if 0.6<= MA/ma <=1.2 and 0.6 <= w/h <=1.2:
                bestB.append(hull)
    if len(bestB)>0:
        cntsB = get1Cnt(image, bestB)
        for cntB in cntsB:
            hasColor, mask = insideContourColor(nblue,cntB)
            if hasColor:
                x,y,w,h = cv2.boundingRect(cntB)
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)
                roi.append(prep4Classfier(frame,x,y,w,h))
                posList.append((x-128,y,x+w,y))
        
#    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    cv2.imshow("nblue", resize(nblue,0.5,0.5))
    cv2.imshow("nRed", resize(nred,0.5,0.5))
    
    
    if len(roi)>0:
        detects = np.zeros((128,1)).astype(np.uint8)
        for rs in roi:
            detects = np.concatenate((detects, rs), axis=1)
#        cv2.imshow("detect",detects)
    
        hog_descriptors = [hog.compute(imgg) for imgg in roi]
        hog_descriptors.extend(hog_descriptors)
        hog_descriptors = np.squeeze(hog_descriptors)
        prediction = model.predict(hog_descriptors)
        prediction_proba = model.predict_proba(hog_descriptors)
        prediction = prediction[:len(roi)]
        prediction_proba = prediction_proba[:len(roi)]
        
        prediction_corrected = [prediction[ndx] for ndx,val in enumerate(prediction_proba) if np.any(prediction_proba[ndx]>0.6)]
        print("pred = " , prediction)
        print(prediction_proba)
        print("corr = " , prediction_corrected)
    
        FinalImage = vis
        for p,pos, proba in zip(prediction,posList, prediction_proba):
            if pos[0] < frame.shape[1]/2:
                vis = placeSign(vis, p, pos[3],pos[2],proba)
            else:
                vis = placeSign(vis, p, pos[1],pos[0],proba)
    out.write(vis)  
    cv2.imshow("Final", resize(vis,0.5,0.5))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.waitKey(0)  
cv2.destroyAllWindows()