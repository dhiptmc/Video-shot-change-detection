from typing import Counter
import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, psnr, ssim, fsim, issm, sre, sam, uiq
import numpy as np
import matplotlib.pyplot as plt

#Root mean square error (RMSE),
#Peak signal-to-noise ratio (PSNR),
#Structural Similarity Index (SSIM),
#Feature-based similarity index (FSIM), #slow
###Information theoretic-based Statistic Similarity Measure (ISSM), #not use due to prob
#Signal to reconstruction error ratio (SRE),
###Spectral angle mapper (SAM) #not used due to prob
#Universal image quality index (UIQ)

mode = argv[1]
result1 = {}
result2 = {}
result3 = {}
result4 = {}
result5 = {}
result6 = {}
result7 = {}
result8 = {}

rmse_measures = {}
psnr_measures = {}
ssim_measures = {}
#fsim_measures = {}
#issm_measures = {}
sre_measures = {}
#sam_measures = {}
#uiq_measures = {}

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
	print("Error input argument! Please enter news, ngc or ftfm")
	quit()

def evaluate(similarity1,similarity2,similarity3,similarity6):
    # Evaluating the performance
	
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(similarity1, label='RMSE similarity to value')
    ax[0, 0].plot(thres1, label='threshold')
    ax[0, 0].legend(loc='best')		
    ax[0, 1].plot(similarity2, label='PSNR similarity to value')
    ax[0, 1].plot(thres2, label='threshold')
    ax[0, 1].legend(loc='best')
    ax[1, 0].plot(similarity3, label='SSIM similarity to value')
    ax[1, 0].plot(thres3, label='threshold')
    ax[1, 0].legend(loc='best')
    ax[1, 1].plot(similarity6, label='SRE similarity to value')
    ax[1, 1].plot(thres6, label='threshold')
    ax[1, 1].legend(loc='best')

    ax[0, 0].set_title("RMSE")
    ax[0, 1].set_title("PSNR")
    ax[1, 0].set_title("SSIM")
    ax[1, 1].set_title("SRE")

    plt.show()
    return

def calc1(dict):
	counter1 = 1
	print("Shot change according to RMSE at frame : ")
	for key, value in dict.items():
		similarity1[counter1,0] = value 
		#print(counter1)
		#print(value)
		if (value > 0.01):
			hit[counter1,0] += 1
			result2[key] = counter1
			print(counter1)
		counter1 += 1
	return result1

def calc2(dict):
	counter2 = 1
	print("Shot change according to PSNR at frame : ")
	for key, value in dict.items():
		similarity2[counter2,0] = value 
		#print(counter2)
		#print(value)
		if (value < 42):
			hit[counter2,0] += 1
			result2[key] = counter2
			print(counter2)
		counter2 += 1
	return result2

def calc3(dict): 
	counter3 = 1
	print("Shot change according to SSIM at frame : ")
	for key, value in dict.items():
		similarity3[counter3,0] = value 
		#print(counter3)
		#print(value)
		if (value < 0.93):
			hit[counter3,0] += 1	
			result3[key] = counter3
			print(counter3)
		counter3 += 1
	return result3

def calc4(dict):
	counter4 = 1
	print("Shot change according to FSIM at frame : ")
	for key, value in dict.items():
		similarity4[counter4,0] = value 
		#print(counter4)
		#print(value)
		if (value < 0.3):
			hit[counter4,0] += 1
			result4[key] = counter4
			print(counter4)
		counter4 += 1
	return result4

###def calc5(dict):
	#counter5 = 1
	#print("Shot change according to ISSM at frame : ")
	#for key, value in dict.items():
		#similarity5[counter5+1,0] = value 
		#print(counter5)
		#print(value)
		#if (value < 0.918):
			#result5[key] = counter5
			#print(counter5)
		#counter5 += 1
	#return result5

def calc6(dict):
	counter6 = 1
	print("Shot change according to SRE at frame : ")
	for key, value in dict.items():
		similarity6[counter6,0] = value 
		#print(counter6)
		#print(value)
		if (value < 46):
			hit[counter6,0] += 1
			result6[key] = counter6
			print(counter6)
		counter6 += 1	
	return result6

###def calc7(dict):
	#counter7 = 1
	#print("Shot change according to SAM at frame : ")
	#for key, value in dict.items():
		#similarity7[counter7+1,0] = value 
		#print(counter7)
		#print(value)
		#if (value > 80):
			#result7[key] = counter7
			#print(counter7)
		#counter7 += 1
	#return result7

def calc8(dict):
	counter8 = 1
	print("Shot change according to UIQ at frame : ")
	for key, value in dict.items():
		similarity8[counter8,0] = value 
		#print(counter8)
		#print(value)
		if (value < 0.918):
			hit[counter8,0] += 1
			result8[key] = counter8
			print(counter8)
		counter8 += 1
	return result8

thres1 = np.full((amount,1),0.01)
thres2 = np.full((amount,1),42)
thres3 = np.full((amount,1),0.93)
thres6 = np.full((amount,1),46)

for i in range(1,amount):
	similarity1 = np.zeros((amount,1))
	similarity2 = np.zeros((amount,1))
	similarity3 = np.zeros((amount,1))
	similarity4 = np.zeros((amount,1))
	#similarity5 = np.zeros((amount,1))
	similarity6 = np.zeros((amount,1))
	#similarity7 = np.zeros((amount,1))
	similarity8 = np.zeros((amount,1))

	hit = np.zeros((amount,1))	

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
		print (testpath)
		test_img = cv2.imread(testpath)
		test_img = cv2.resize(test_img, (128,128), interpolation = cv2.INTER_AREA)

		datapath = dir + k + '.jpg'
		print (datapath)
		data_img = cv2.imread(datapath)
		data_img = cv2.resize(data_img, (128,128), interpolation = cv2.INTER_AREA)
	

	rmse_measures[datapath]= rmse(test_img, data_img)

	psnr_measures[datapath]= psnr(test_img, data_img)

	ssim_measures[datapath]= ssim(test_img, data_img)

	#fsim_measures[datapath]= fsim(test_img, data_img)

	###issm_measures[datapath]= issm(test_img, data_img)

	sre_measures[datapath]= sre(test_img, data_img)

	###sam_measures[datapath]= sam(test_img, data_img)

	#uiq_measures[datapath]= uiq(test_img, data_img)


rmse = calc1(rmse_measures)
#rmseev = evaluate(similarity1)

psnr = calc2(psnr_measures)
#psnrev = evaluate(similarity2)

ssim = calc3(ssim_measures)
#ssimev = evaluate(similarity3)

#fsim = calc4(fsim_measures)
#fsimev = evaluate(similarity4)

###issm = calc5(issm_measures)
###issmev = evaluate(similarity5)

sre = calc6(sre_measures)
#sreev = evaluate(similarity6)

###sam = calc7(sam_measures)
###samev = evaluate(similarity7)

#uiq = calc8(uiq_measures)
#uiqev = evaluate(similarity8)

print("\nWeighted algorithms detects shot change at : ")
for i in range(1,amount):
	if (hit[i,0] >= 3):
		print(i)

ev = evaluate(similarity1,similarity2,similarity3,similarity6)