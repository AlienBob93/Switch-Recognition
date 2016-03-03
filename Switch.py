import numpy as np
import cv2

orb = cv2.ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
Templatepath = "PATH TO Training Image of the Switches/"

# calculate descriptors for the template set
kpTrain_LIST = []
desTrain_LIST = []
imgTrain_LIST = []
for i in range(0, NUMBER_OF_TRAINING_IMAGES):
	filename = "TEMPLATE_(%d).png"%(i+1)
	print filename
	imgTrainColor = cv2.imread(Templatepath+filename)
	imgTrain_LIST.append(imgTrainColor)
	imgTrainGray = cv2.cvtColor(imgTrainColor, cv2.COLOR_BGR2GRAY)

	kpTrain = orb.detect(imgTrainGray, None)
	kpTrain, desTrain = orb.compute(imgTrainGray, kpTrain)
	print "number of descriptors: " + str(len(desTrain)) + "\n"
	kpTrain_LIST.append(kpTrain)
	desTrain_LIST.append(desTrain)
print "desTrain size: " + str(len(desTrain_LIST))

# calculate scene descriptors
# load video or access camera
camera = cv2.VideoCapture(0)	# USE FOR CAMERA CAPTURE
cap = cv2.VideoCapture('LOAD VIDEO FILE TO DETECT SWITCHES IN')	# USE FOR PRE_RECORDED VIDEO
while(cap.isOpened()):
	ret, imgCamColor = cap.read() # ret, imgCamColor = camera.read()
	if (imgCamColor == None):
		break	
	imgCamGray = cv2.cvtColor(imgCamColor, cv2.COLOR_BGR2GRAY)
	
	kpCam = orb.detect(imgCamGray,None)
	kpCam, desCam = orb.compute(imgCamGray, kpCam)
	print "number of scene descriptors: " + str(len(desCam))

	# match the features
	matches_LIST = []
	for desTrain in desTrain_LIST:
		matches = bf.match(desCam, desTrain)
		print "number of potential matches: " + str(len(matches))
		dist = [m.distance for m in matches]
		thres_dist = (sum(dist) / len(dist)) * 0.5
		matches = [m for m in matches if m.distance < thres_dist]
		matches_LIST.append(matches)

	# create image for display of results
	h1, w1 = imgCamColor.shape[:2]
	result_LIST = []
	hdif_LIST = []
	for imgTrainColor in imgTrain_LIST:
		h2, w2 = imgTrainColor.shape[:2]
		nWidth = w1 + w2
		nHeight = max(h1, h2)
		hdif = (h1 - h2) / 2
		result = np.zeros((nHeight, nWidth, 3), np.uint8)
		result[hdif:hdif+h2, :w2] = imgTrainColor
		result[:h1, w2:w1+w2] = imgCamColor
		result_LIST.append(result)
		hdif_LIST.append(hdif)
		
	# display results
	for kpTrain, matches, result in zip(kpTrain_LIST, matches_LIST, result_LIST):
		#for i in range(len(matches)):
			#pt_a = (int(kpTrain[matches[i].trainIdx].pt[0]), int(kpTrain[matches[i].trainIdx].pt[1] + hdif))
			#pt_b = (int(kpCam[matches[i].queryIdx].pt[0] + w2), int(kpCam[matches[i].queryIdx].pt[1]))
			#cv2.line(result, pt_a, pt_b, (255, 0, 0))
			#print "pt_a: " + str(pt_a)

		print "number of matches " + str(len(matches))
		#cv2.imshow('matches', result)
		#cv2.waitKey(0)

	# get the best match
	Max = 0
	for m in matches_LIST:
		if len(m) > Max:
			Max = len(m)
			i = matches_LIST.index(m)
			matches = matches_LIST[i]
	print "max value of good matches: " + str(len(matches))
	if len(matches) >= 4:
		kpTrain = kpTrain_LIST[i]
		imgTrainColor = imgTrain_LIST[i]
		pt_b_LIST = []
		for k in range(len(matches)):
			pt_a = (int(kpTrain[matches[k].trainIdx].pt[0]), int(kpTrain[matches[k].trainIdx].pt[1] + hdif))
			pt_b = (int(kpCam[matches[k].queryIdx].pt[0] + w2), int(kpCam[matches[k].queryIdx].pt[1]))
			pt_b_LIST.append(pt_b)
			cv2.line(result, pt_a, pt_b, (255, 0, 0))
		sum_pt = (0, 0)
		for pt_b in pt_b_LIST:
			sum_pt = (sum_pt[0] + pt_b[0], sum_pt[1] + pt_b[1])
		center_pt = (sum_pt[0]/len(matches), sum_pt[1]/len(matches))
		print "center point of matches: " + str(center_pt)
		h2, w2 = imgTrainColor.shape[:2]
		X1, Y1 = (center_pt[0] - w2 + 90, center_pt[1] - h2 + 50)
		X2, Y2 = (center_pt[0] + 60, center_pt[1] + 70)
		cv2.rectangle(result, (X1, Y1), (X2, Y2), (0, 255, 0), 5)
	print "Found Switch " + filename
	cv2.imshow('Result', result)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

cap.release()	# IF VIDEO
camera.release()	# IF CAMERA
cv2.destroyAllWindows()
