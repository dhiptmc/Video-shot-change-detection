from cProfile import label
from operator import index
from re import I
from tkinter import Label
from typing import Counter
import cv2
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

mode = argv[1]

if mode == 'news':
	dir = 'news_out_modified/'
	amount = 1380
elif mode == 'ngc':
	dir = 'ngc_out_modified/'
	amount = 1060	
elif mode == 'ftfm':
	dir = 'ftfm_out_modified/'
	amount = 770	
else:
	print("Error input!")
	quit()

####################### Algorithm #######################

def evaluate(similarity1,similarity2,similarity3,similarity4,similarity5,weightedsim):
    # Evaluating the performance
    plt.figure("similarity - frame")
    plt.plot(similarity1, color='purple', label='Gray Hist')
    plt.plot(similarity2, color='yellow', label='RGB Hist')
    plt.plot(similarity3, color='red',  label='aHash')
    plt.plot(similarity4, color='green',  label='dHash')
    plt.plot(similarity5, color='blue',  label='pHash')
    plt.plot(weightedsim, color='aqua',  label='Weighted algorithm')	
    plt.plot(thres,color='black', label='Threshold')	
    plt.xlabel("frame")
    plt.ylabel("similarity")
    plt.legend(loc='best')
    plt.show()
    return

def create_gray_hist(image):
    #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256],  [0, 256])
    return hist

def hist_compare(image1, image2):

    hist1 = create_gray_hist(image1)
    hist2 = create_gray_hist(image2)
    # 進行三種方式的直方圖比較
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    similarity = ((1-match1) + match2)/2
    return similarity

def classify_hist_with_split(image1, image2, size=(256, 256)):
    # RGB每個通道的直方圖相似度
    # 將圖像resize後，分離爲RGB三個通道，再計算每個通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += hist_compare(im1, im2)
    sub_data = sub_data / 3
    return sub_data

def aHash(img):
    # 均值哈希算法
    # 縮放爲8*8
    img = cv2.resize(img, (8, 8))
    # 轉換爲灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s爲像素和初值爲0，hash_str爲hash值初值爲''
    s = 0
    hash_str = ''
    # 遍歷累加求像素和
    for i in range(8):
        for j in range(8):
            s = s+gray[i, j]
    # 求平均灰度
    avg = s/64
    # 灰度大於平均值爲1相反爲0生成圖片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def dHash(img):
    # 差值哈希算法
    # 縮放8*8
    img = cv2.resize(img, (9, 8))
    # 轉換灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一個像素大於後一個像素爲1，相反爲0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

def pHash(img):
    # 感知哈希算法
    # 縮放32*32
    img = cv2.resize(img, (32, 32))   # , interpolation=cv2.INTER_CUBIC

    # 轉換爲灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 將灰度圖轉爲浮點型，再進行dct變換
    dct = cv2.dct(np.float32(gray))
    # opencv實現的掩碼操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    average = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    return hash

def cmpHash(hash1, hash2):
    # Hash值對比
    # 算法中1和0順序組合起來的即是圖片的指紋hash。順序不固定，但是比較的時候必須是相同的順序。
    # 對比兩幅圖的指紋，計算漢明距離，即兩個64位的hash值有多少是不一樣的，不同的位數越小，圖片越相似
    # 漢明距離：一組二進制數據變成另一組數據所需要的步驟，可以衡量兩圖的差異，漢明距離越小，則相似度越高。漢明距離爲0，即兩張圖片完全一樣
    n = 0
    # hash長度不同則返回-1代表傳參出錯
    if len(hash1) != len(hash2):
        return -1
    # 遍歷判斷
    for i in range(len(hash1)):
        # 不相等則n計數+1，n最終爲相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

####################### Performance Analysis #######################

def pfanaly(similarity):
    correcthit = 0
    falsepositive = 0
    hitprebound2 = 0
    hitprebound3 = 0

    for i in range(1,amount):
        hit=0
        if similarity[i,0] < threshold:
            
            if mode == 'news':
                f1 = open('news_ground.txt')
                for line in f1:
                    if(i == int(line)):
                        hit=1
                        correcthit += 1
                        break
                
                if (hit == 0):
                    falsepositive += 1

                f1.close()

            elif mode == 'ngc':
                f2 = open('ngc_ground.txt')
                for line in f2:
                    if '~' in line:
                        prebound2=line.split("~")[0]
                        postbound2=line.split("~")[1]    

                        if( (i <= int(postbound2)) & (i >= int(prebound2)) ):
                            hit=1
                            if(hitprebound2 != int(prebound2)):
                                hitprebound2 = int(prebound2)
                                correcthit += 1
                                break
                    else:
                        if(i == int(line)):
                            hit=1
                            correcthit += 1
                            break
                    
                if (hit == 0):
                    falsepositive += 1

                f2.close()

            elif mode == 'ftfm':
                f3 = open('ftfm_ground.txt')
                for line in f3:
                    if '~' in line:
                        prebound3=line.split("~")[0]
                        postbound3=line.split("~")[1]    
                        if( (i <= int(postbound3)) & (i >= int(prebound3)) ):
                            hit=1
                            if(hitprebound3 != int(prebound3)):
                                hitprebound3 = int(prebound3)
                                correcthit += 1
                                break
                    else:
                        if(i == int(line)):
                            hit=1
                            correcthit += 1
                            break
                    
                if (hit == 0):
                    falsepositive += 1

                f3.close()

    print("Correct hit amount is : ",correcthit)
    print("False positive amount is : ",falsepositive)

    if mode == 'news':
        recall = correcthit/7 #groundhit1 = 7 
        if(correcthit + falsepositive) == 0:
            precision = 0   
        else:
            precision = correcthit/(correcthit + falsepositive)
        return recall, precision
    elif mode == 'ngc':
        recall = correcthit/36 #groundhit2 = 36
        if(correcthit + falsepositive) == 0:
            precision = 0   
        else:
            precision = correcthit/(correcthit + falsepositive)
        return recall, precision
    elif mode == 'ftfm':
        recall = correcthit/27 #groundhit3 = 27
        if(correcthit + falsepositive) == 0:
            precision = 0   
        else:
            precision = correcthit/(correcthit + falsepositive)
        return recall, precision

#######################Main#######################

recall1 = np.zeros(25)
precision1 = np.zeros(25)
recall2 = np.zeros(25)
precision2 = np.zeros(25)
recall3 = np.zeros(25)
precision3 = np.zeros(25)
recall4 = np.zeros(25)
precision4 = np.zeros(25)
recall5 = np.zeros(25)
precision5 = np.zeros(25)
recall = np.zeros(25)
precision = np.zeros(25)
index = 0 

for m in range (0,25):
    threshold = 0.5 + m*0.02

    similarity1 = np.ones((amount,1))
    similarity2 = np.ones((amount,1))
    similarity3 = np.ones((amount,1))
    similarity4 = np.ones((amount,1))
    similarity5 = np.ones((amount,1))
    weightedsim = np.ones((amount,1))
    thres = np.full((amount,1),threshold)

    for i in range(1,amount):
        j=str(i-1)
        #print(j)

        k=str(i)
        #print(k)
        
        if mode == 'ngc':
            testpath = dir + j + '.jpeg'
            #print (testpath)
            test_img = cv2.imread(testpath)
            test_img = cv2.resize(test_img, (128,128), interpolation = cv2.INTER_AREA)

            datapath = dir + k + '.jpeg'
            #print (datapath)
            data_img = cv2.imread(datapath)
            data_img = cv2.resize(data_img, (128,128), interpolation = cv2.INTER_AREA)
        else:
            testpath = dir + j + '.jpg'
            #print (testpath)
            test_img = cv2.imread(testpath)
            test_img = cv2.resize(test_img, (128,128), interpolation = cv2.INTER_AREA)

            datapath = dir + k + '.jpg'
            #print (datapath)
            data_img = cv2.imread(datapath)
            data_img = cv2.resize(data_img, (128,128), interpolation = cv2.INTER_AREA)   
            
        score1 = hist_compare(test_img,data_img)
        similarity1[i,0] = score1      
        
        score2 = classify_hist_with_split(test_img,data_img)
        similarity2[i,0] = score2

        hash1 = aHash(data_img)
        hash2 = aHash(test_img)
        score3 = cmpHash(hash1, hash2)
        score3 = (1-float(score3/64))
        similarity3[i,0] = score3

        hash1 = dHash(data_img)
        hash2 = dHash(test_img)
        score4 = cmpHash(hash1, hash2)
        score4 = (1-float(score4/64))
        similarity4[i,0] = score4	

        hash1 = pHash(data_img)
        hash2 = pHash(test_img)
        score5 = cmpHash(hash1, hash2)
        score5 = (1-float(score5/64))
        similarity5[i,0] = score5	

        weightedsim[i,0] = (score1+score2+score3+score4+score5)/5

    print("Gray hist comparison detects shot change at : ")
    for i in range (1,amount):
        if similarity1[i,0] < threshold:
            print (i)

    print("\nRGB hist comparison detects shot change at : ")
    for i in range (1,amount):
        if similarity2[i,0] < threshold:
            print (i)

    print("\naHash comparison detects shot change at : ")
    for i in range (1,amount):
        if similarity3[i,0] < threshold:
            print (i)

    print("\ndHash comparison detects shot change at : ")
    for i in range (1,amount):
        if similarity4[i,0] < threshold:
            print (i)

    print("\npHash comparison detects shot change at : ")
    for i in range (1,amount):
        if similarity5[i,0] < threshold:
            print (i)

    print("\nWeighted algorithms detects shot change at : ")
    for i in range (1,amount):
        if weightedsim[i,0] < threshold:
            print (i)

    result1=pfanaly(similarity1)
    recall1[index] = result1[0]
    precision1[index] = result1[1]

    result2=pfanaly(similarity2)
    recall2[index] = result2[0]
    precision2[index]= result2[1]

    result3=pfanaly(similarity3)
    recall3[index] = result3[0]
    precision3[index]= result3[1]

    result4=pfanaly(similarity4)
    recall4[index] = result4[0]
    precision4[index] = result4[1]

    result5=pfanaly(similarity5)
    recall5[index] = result5[0]
    precision5[index] = result5[1]

    result=pfanaly(weightedsim)
    recall[index] = result[0]
    #print(recall[index])
    precision[index] = result[1]
    #print(precision[index])

    index += 1

#print("Precision for Gray hist comparison is : ",precision1)
#print("Recall for Gray hist comparison is : ",recall1)

#print("Precision for RGB hist comparison is : ",precision2)
#print("Recall for RGB hist comparison is : ",recall2)

#print("Precision for aHash comparison is : ",precision3)
#print("Recall for aHash comparison is : ",recall3)

#print("Precision for dHash comparison is : ",precision4)
#print("Recall for dHash comparison is : ",recall4)

#print("Precision for pHash comparison is : ",precision5)
#print("Recall for pHash comparison is : ",recall5)

#print("Precision for weighted algorithms is : ",precision)
#print("Recall for weighted algorithms is : ",recall)

plt.figure("PR Curve")
plt.plot(recall1,precision1,label='Gray hist')
plt.plot(recall2,precision2,label='RGB hist')
plt.plot(recall3,precision3,label='aHash')
plt.plot(recall4,precision4,label='dHash')
plt.plot(recall5,precision5,label='pHash')
plt.plot(recall,precision,label='weighted algorithms')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc='best')
plt.show()

#hist = evaluate(similarity1,similarity2,similarity3,similarity4,similarity5,weightedsim)