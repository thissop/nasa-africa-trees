import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

ndvi_images = []
pan_images = [] 
annotations = []
boundaries = []

output_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/output/'
for file in os.listdir(output_dir):
    full_path = output_dir+file
    if '.png' in file: 
        if 'annotation' in file: 
            annotations.append(full_path)

        elif 'boundary' in file: 
            boundaries.append(full_path)

        elif 'ndvi' in file: 
            ndvi_images.append(full_path)

        elif 'extracted_pan' in file: 
            pan_images.append(full_path) 

for i in range(len(pan_images)):
    annotation = Image.open(annotations[i])
    boundary = Image.open(boundaries[i])
    pan = Image.open(pan_images[i])
    ndvi = Image.open(ndvi_images[i])
    
    files = [annotation, boundary, pan, ndvi]

    widths = [i.width for i in files]
    heights = [i.height for i in files]

    if len(np.unique(widths))==len(np.unique(heights)) and len(np.unique(widths))==1: 
        print('all match')
    else: 
        print(widths, heights)
    
# RESULT # 

'''
all match
all match
all match
all match
all match
all match
all match
all match
all match
all match
'''