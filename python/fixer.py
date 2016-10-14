import os.path 
import numpy as np

def lreg(x,y):
	from sklearn import linear_model
	model = linear_model.LinearRegression()
	model.fit(x, y)

	model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
	model_ransac.fit(x, y)
	
	return model_ransac

def repair_1(x,y):
	n = len(x)
	
	k = 0
	if n >=3:
		for i in xrange(2,n-1):
			if y[i-1] == y[i+1] and y[i-1] != y[i]:
				y[i] = y[i-1]
				k += 1
	return k


def fix_prediction(images_numbers, predictions):
	x = np.int32(images_numbers)
	y = np.int32(predictions)
	th = 6
    
	n = len(x)
	
    #	while True:
    #	k = repair_1(x,y)
    #	if k == 0: break

	X = x.reshape((n,1))
	model = lreg(X,y)

	pred = model.predict(X)

	pred = np.int32(np.clip(pred,th,54))

	#difference
	diff = np.abs(y-pred)
	#find higher
	indices = np.where(diff>= th)

	y2 = np.array(y)
	y2[indices] = pred[indices]

#k = repair_1(x,y2)
#	print k

	return y2
