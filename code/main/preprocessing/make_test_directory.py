import os 
import geopandas as gpd
import shutil 

top_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/input/'

ndvi = []
pan = []
backgrounds = []
annotations = []

for i in os.listdir(top_dir): 
    path = top_dir+i
    if 'ndvi' in i: 
        ndvi.append(path)

    elif 'pan' in i: 
        pan.append(path)

    elif 'vector' in i:
        backgrounds.append(path)

    elif 'annotations' in i: 
        annotations.append(path) 

for f in annotations:
    trainingPolygons = gpd.read_file(f)
    print(f.split('/')[-1], f'Read a total of {trainingPolygons.shape[0]} object polygons')

annot = annotations[8]
vect = backgrounds[8]
ndvi = ndvi[8]
pan = pan[8]

temp_dir = '/mnt/c/Users/Research/Documents/GitHub/nasa-africa-trees/data/first_mosaic/rebuilt_approach/temp/'

for i in range(0,24,1):
    for f in [annot, vect, ndvi, pan]:
        shutil.copyfile(f, temp_dir+f.split('/')[-1].replace('_8', '_'+str(i)))