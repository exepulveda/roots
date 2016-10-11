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

im_path = "/Users/exequiel/projects/roots/python/processing/1.25.AVI/accepted/"

images = []
expand_folder(im_path,images)


#templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")
templates = load_templates("/Users/exequiel/projects/roots/python/models/templates")

n = len(images)

ret = np.empty((n,54-6+1))
for i,im in enumerate(images):
    print i,im
    prediction = predict_all_window(im,templates,debug=False)
    ret[i,:] = prediction

np.savetxt("1.25-predictions.csv",ret,fmt="%2.4f")
