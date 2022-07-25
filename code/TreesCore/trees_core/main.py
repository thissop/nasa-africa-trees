def preprocess(area_files:list, 
               annotation_files:list,
               raw_ndvi_images:list,
               raw_pan_images:list,
               output_path:str,
               bands=[0],
               show_boundaries_during_preprocessing:bool=False, verbose:bool=False):
    import geopandas as gps
    import os
    from tqdm import tqdm
    from trees_core.preprocessing_utilities import readInputImages, extractAreasThatOverlapWithTrainingData, dividePolygonsInTrainingAreas
    import warnings 
    
    warnings.filterwarnings("ignore")

    allAreasWithPolygons = []
    for training_area, training_annotations in tqdm(zip(area_files, annotation_files), disable=not(verbose)): 
        trainingArea = gps.read_file(training_area)
        trainingPolygon = gps.read_file(training_annotations)

        if verbose: 
            print(f'Read a total of {trainingPolygon.shape[0]} object polygons and {trainingArea.shape[0]} training areas.')
            print(f'Polygons will be assigned to training areas in the next steps.') 

        #Check if the training areas and the training polygons have the same crs
        if trainingArea.crs  != trainingPolygon.crs:
            print('Training area CRS does not match training_polygon CRS')
            targetCRS = trainingPolygon.crs #Areas are less in number so conversion should be faster
            trainingArea = trainingArea.to_crs(targetCRS)
        
        if verbose: 
            print(trainingPolygon.crs)
            print(trainingArea.crs)
        
        assert trainingPolygon.crs == trainingArea.crs

        trainingArea['id'] = range(trainingArea.shape[0])
        if verbose: 
            print(trainingArea)
        
        # areasWithPolygons contains the object polygons and weighted boundaries for each area!
        areasWithPolygons = dividePolygonsInTrainingAreas(trainingPolygon, trainingArea, show_boundaries_during_processing=show_boundaries_during_preprocessing, verbose=verbose)
        allAreasWithPolygons.append(areasWithPolygons)
        if verbose: 
            print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))
    if verbose:
        print(f'Found a total of {len(inputImages)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    for i in range(len(inputImages)):
        extractAreasThatOverlapWithTrainingData(inputImages[i], areasWithPolygons=allAreasWithPolygons[i], writePath=output_path, ndviFilename='extracted_ndvi',
                                                panFilename='extracted_pan', annotationFilename='extracted_annotation',
                                                boundaryFilename='extracted_boundary', bands=bands, writeCounter=i)

def train(): 
    pass 

def evaluate():
    pass 