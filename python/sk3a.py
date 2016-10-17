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

from fixer import fix_prediction

im_path = "/home/esepulveda/projects/roots/python/processing/1.25.AVI/accepted/"

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


	templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")
	#templates = load_templates("/Users/exequiel/projects/roots/python/models/templates")

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

print n

assert n  == len(y)
	
import matplotlib.pyplot as plt

y2 = np.array(y)
y3 = fix_prediction(x,y2)

plt.subplot(111)
plt.plot(x,y,"x",color="r")
plt.plot(x,y3,"o",color="g")
plt.show()

print "id,initial,final"
for a,b,c in zip(x,y,y3):
	print a,b,c
