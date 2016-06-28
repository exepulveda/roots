winframe.py: search for a windows frame that minimise the error w.r.t. its central frame
restore.py: agregate a set of images by register them and taking the median by pixel


Requires opencv

To install opencv I followed steps at
http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
and it worked stringth forward on OS X. 

Standard installations should be followed for a different OS


To test methods set input and output paths in test_restore.py before run it.  


Method:

To obtain one image by frame, we conduct the two following steps for each set of images
classified as "good images" for a frame. We refer to as cassified-images to the sorted
set of "good images" (sorted by order of appearence in the video). 

1. Searching for a windows-frame
A windows-frame is defined as a subset of the classified-images such that each contiguous 
pair of images in the windows-frame are contiguous in the classified-images. 
We fix the size of the windows-frame (number of images) and we search for the windows
frame with less error. Here, the error is defined as the sum of histogram distances of each
image in the windows-frame w.r.t. the central image in the windows-frame. 


2. Reduce a windows-frame into one image
After the best windows-frame is found, we reduce it into a single image following 2 steps:

  a) Registering all images w.r.t. the central image in the windows-frame.
  b) Each pixel of the single image is obtained as the median of the pixels
     of all registered images at the same position.

 

