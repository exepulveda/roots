import os.path 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu, threshold_isodata, threshold_li, threshold_yen
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, rectangle
from skimage.color import label2rgb, rgb2gray

from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu, threshold_adaptive

from bound import load_templates
from bound import predict
from bound import predict_all_window

from utils import expand_folder

#im_path = "/Users/exequiel/projects/roots/python/processing/1.25.AVI/accepted/"
im_path = "/Users/a1613915/repos/roots/python/processing/3.12.AVI/accepted" 

def lreg(x,y, th):
	from sklearn import linear_model
	model = linear_model.LinearRegression()
	model.fit(x, y)

	model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=th)
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
	
def repair_2(x,y):
	n = len(x)
	
	k = 0
	if n >=3:
		for i in xrange(2,n-1):
			if y[i-1] != y[i] and y[i+1] != y[i] and abs(y[i-1] - y[i+1]) >= 1:
				if (x[i] - x[i-1]) <= (x[i+1] - x[i]):
					y[i] = y[i-1]
				else:
					y[i] = y[i+1]
				k += 1
	return k

if not os.path.exists("x.npy"):
	image_list = []
	expand_folder(im_path,image_list)
	image_list.sort()

	image_ids = []
	for image_name in image_list:
		fname, extension = os.path.splitext( os.path.basename(image_name))
		print (fname, extension)
		image_ids += [(int(fname),image_name)]

	image_ids.sort()

	image_list = [e[1] for e in image_ids]
	x = [e[0] for e in image_ids]


	#templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")
	templates = load_templates("/Users/a1613915/repos/roots/python/models/templates")

	n = len(x)

	ret = np.empty((n,54-6+1))
	y = []
	for i,im in enumerate(image_list):
		#print i,im
		prediction = predict(im,templates,debug=False)
		y += [prediction if prediction else 0]

	x = np.int32(x)
	y = np.int32(y)

	np.save("x",x)
	np.save("y",y)
else:
	x = np.load("x.npy")
	y = np.load("y.npy")
	
n = len(x)
	
import matplotlib.pyplot as plt

#plt.subplot(111)

#plt.plot(x,y,"o")
#plt.show()


while  True:
	#plt.subplot(111)
	k = repair_1(x,y)
	print k
	#plt.plot(x,y,"o")
	#plt.show()
	if k == 0: break

print "ransac regression"
X = x.reshape((n,1))

th = 1

model = lreg(X,y, th)

pred = model.predict(X)

pred = np.int32(np.clip(np.round(pred),6,54))
line_X = np.arange(np.min(x),np.max(x))
line_y = model.predict(line_X[:, np.newaxis])

# find ransac outliers
diff = np.abs(y-pred)
outlrs = np.where(diff> th)[0]
inlrs =  np.where(diff<= th)[0]

# re asign outliers
y2 = np.array(y)
#y2[outlrs] = pred[outlrs]

k=0

inlrs.sort()
revinlrs = inlrs[::-1] # inliers in reversed order

for xout in outlrs:
        
    a =  len(revinlrs) - np.argmax(revinlrs < xout) - 1
    b = a + np.argmax(xout < inlrs[a:]);
            
    xa = x[inlrs[a]];
    xb = x[inlrs[b]];
        
    ya = y[inlrs[a]];
    yb = y[inlrs[b]];
    
    
    print a , b, xa, x[xout], xb
        
    if ya==yb:
        y2[xout] = ya
    else:
        y2[xout] = ya + (x[xout]-xa)*(yb-ya)/float(xb-xa)
                
    k+=1

y2 = np.int32(np.clip(np.round(y2),6,54))


while  True:
	#plt.subplot(111)
	k = repair_1(x,y2)
	print k
	#plt.plot(x,y,"o")
	#plt.show()
	if k == 0: break

while not True:
	#plt.subplot(111)
	k = repair_2(x,y2)
	print k
	#plt.plot(x,y,"o")
	#plt.show()
	if k == 0: break

# save assignment
image_list = []
expand_folder(im_path,image_list)
image_list.sort()

image_ids = []
for image_name in image_list:
	fname, extension = os.path.splitext( os.path.basename(image_name))
	#print (fname, extension)
	image_ids += [(int(fname),image_name)]

image_ids.sort()

image_list = [e[1] for e in image_ids]

import shutil
if  os.path.exists("rsc_result"):
    shutil.rmtree("rsc_result")
    
os.makedirs("rsc_result")
for l in np.unique(y2) :
    os.makedirs("rsc_result/{}".format(l))

for imgpath, imgnum, l in zip(image_list,x, y2) :
    shutil.copyfile(imgpath, "rsc_result/{}/{}.tiff".format(l,imgnum))


# plot

plt.subplot(111)
plt.plot(x[outlrs],y[outlrs],"x",color="r") # ransac outliers
plt.plot(x[inlrs],y[inlrs],"o",color="b") # ransac inliers
plt.plot(x[outlrs],y2[outlrs],"x",color="y") # new asignation

plt.plot(line_X, line_y, color='navy', linestyle='-', label='Linear regressor')
plt.grid()
plt.show()



