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

im_path = "/home/esepulveda/Documents/projects/roots/python/processing/5.26.AVI/windows/frame-8/3105.tiff"
im_path = "/home/esepulveda/Documents/projects/roots/python/processing/2.26.AVI/windows/frame-6/42.tiff"
im_path = "/media/esepulveda/Elements/4-training/1.11/windows/frame-19/1.11-769.jpg"
im_path = "/media/esepulveda/Elements/4-training/1.11/windows/frame-8/1.11-129.jpg"
im_path = "/home/esepulveda/Documents/projects/roots/python/processing/1.25.AVI/accepted/1492.tiff"



templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")

prediction = predict(im_path,templates,debug=True)

print prediction
