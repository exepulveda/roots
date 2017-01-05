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
from skimage.measure import compare_ssim

from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu, threshold_adaptive

from bound import load_templates
from bound import predict
from bound import predict_all_window

from utils import expand_folder

im_path = "/home/esepulveda/Documents/projects/roots/python/processing/1.25.AVI/windows/frame-51"

images = []
expand_folder(im_path,images)
images.sort()


#templates = load_templates("/home/esepulveda/Documents/projects/roots/python/models/templates")
templates = load_templates("/Users/exequiel/projects/roots/python/models/templates")

n = len(images)

sim = np.zeros((n,n))

print images
images_data = [data.imread(x) for x in images]

for i in xrange(n):
    image_i = images_data[i]
    for j in xrange(i,n):
        image_j = images_data[j]
        sim[i,j] = compare_ssim(image_i,image_j,multichannel=True) #,gaussian_weights=True)
        sim[j,i] = sim[i,j]

print np.sum(sim,axis=0)


print np.min(sim,axis=0)
