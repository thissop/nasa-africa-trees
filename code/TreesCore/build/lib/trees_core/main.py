from concurrent.futures import process


def preprocess(area_files:list, 
               annotation_files:list,
               raw_ndvi_images:list,
               raw_pan_images:list,
               output_path:str,
               bands=[0],
               show_boundaries_during_preprocessing:bool=False, verbose:bool=False, 
               n_jobs:int=4):

    import geopandas as gps
    from trees_core.preprocessing_utilities import extractAreasThatOverlapWithTrainingData, dividePolygonsInTrainingAreas
    import warnings 
    from joblib import Parallel 
    import multiprocessing 
    from multiprocessing.pool import ThreadPool as Pool 
    from functools import partial 
    
    cpus = multiprocessing.cpu_count()
    if cpus<4: 
        n_jobs=cpus

    warnings.filterwarnings("ignore")

    def preprocess_single(index:int, area_files=area_files, training_annotations=annotation_files): 
        
        trainingArea = gps.read_file(area_files[index])
        trainingPolygon = gps.read_file(training_annotations[index])

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
        if verbose: 
            print(f'Assigned training polygons in {len(areasWithPolygons)} training areas and created weighted boundaries for polygons')

        return areasWithPolygons

    total_jobs = len(area_files)

    if total_jobs<n_jobs: 
        n_jobs = total_jobs

    print(n_jobs)

    pool = Pool(processes=n_jobs)
    allAreasWithPolygons = pool.map(preprocess_single, range(total_jobs))

    #Parallel(n_jobs=n_jobs)(preprocess_single(index) for index in range(total_jobs))

    inputImages = list(zip(raw_ndvi_images,raw_pan_images))
    if verbose:
        print(f'Found a total of {len(inputImages)} pair of raw image(s) to process!')

    # For each raw satellite image, determine if it overlaps with a training area. 
    # If a overlap if found, then extract + write the overlapping part of the raw image, create + write an image from training polygons and create + write an image from boundary weights in the that overlapping region.
        
    # Run the main function for extracting part of ndvi and pan images that overlap with training areas
    pool = Pool(processes=n_jobs)
    partial_func = partial(extractAreasThatOverlapWithTrainingData, inputImages=inputImages, allAreasWithPolygons=allAreasWithPolygons, writePath=output_path, bands=bands)
    pool.map(partial_func, range(total_jobs))
    #Parallel(n_jobs=n_jobs)(extractAreasThatOverlapWithTrainingData() for i in range(total_jobs))

def train(): 
    pass 

def evaluate():
    pass 