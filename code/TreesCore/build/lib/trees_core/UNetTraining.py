def load_train_test(ndvi_images:list,
                    pan_images:list, 
                    annotations:list,
                    boundaries:list,
                    output_path:str, 
                    normalize:float = 0.4, BATCH_SIZE = 8, NB_EPOCHS = 200, patch_size=(256,256,4), 
                    input_shape = (256,256,2), input_image_channel = [0,1], input_label_channel = [2], input_weight_channel = [3]):
    
    r'''
    
    Arguments 
    ---------

    area_files : list 
        List of the area files 

    annotations : list
        List of the full file paths to the extracted annotations that got outputed by the earlier preproccessing step. 

    ndvi_images : list 
        List of full file paths to the extracted ndvi images 

    pan_images : list 
        Same as ndvi_images except for pan 

    boundaries : list
        List of boundary files extracted by previous preprocessing step 
    
    '''

    import rasterio 
    import numpy as np
    from PIL import Image
    from trees_core.original_core.frame_utilities import FrameInfo, split_dataset
    from trees_core.original_core.dataset_generator import DataGenerator

    patch_dir = './patches{}'.format(patch_size[0])
    frames_json = os.path.join(patch_dir,'frames_list.json')


    # Read all images/frames into memory
    frames = []

    for i in range(len(ndvi_images)):
        ndvi_img = rasterio.open(ndvi_images[i])
        pan_img = rasterio.open(pan_images[i])
        read_ndvi_img = ndvi_img.read()
        read_pan_img = pan_img.read()
        comb_img = np.concatenate((read_ndvi_img, read_pan_img), axis=0)
        comb_img = np.transpose(comb_img, axes=(1,2,0)) #Channel at the end
        annotation_im = Image.open(annotations[i])
        annotation = np.array(annotation_im)
        weight_im = Image.open(boundaries[i])
        weight = np.array(weight_im)
        f = FrameInfo(comb_img, annotation, weight)
        frames.append(f)
    
    training_frames, validation_frames, testing_frames  = split_dataset(frames, frames_json, patch_dir)

    annotation_channels = input_label_channel + input_weight_channel
    train_generator = DataGenerator(input_image_channel, patch_size, training_frames, frames, annotation_channels, augmenter = 'iaa').random_generator(BATCH_SIZE, normalize = normalize)
    val_generator = DataGenerator(input_image_channel, patch_size, validation_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)
    test_generator = DataGenerator(input_image_channel, patch_size, testing_frames, frames, annotation_channels, augmenter= None).random_generator(BATCH_SIZE, normalize = normalize)

def train_model(): 
    from tree_core.original_core.losses import tversky, accuracy, dice_coef, dice_loss, specificity, sensitivity 
    from tree_core.original_core.optimizers import adaDelta, adagrad, adam, nadam 
    import time 

    OPTIMIZER = adaDelta
    LOSS = tversky 

    #Only for the name of the model in the very end
    OPTIMIZER_NAME = 'AdaDelta'
    LOSS_NAME = 'weightmap_tversky'

    # Declare the path to the final model
    # If you want to retrain an exising model then change the cell where model is declared. 
    # This path is for storing a model after training.

    timestr = time.strftime("%Y%m%d-%H%M")
    chf = config.input_image_channel + config.input_label_channel
    chs = reduce(lambda a,b: a+str(b), chf, '')

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    model_path = os.path.join(config.model_path,'trees_{}_{}_{}_{}_{}.h5'.format(timestr,OPTIMIZER_NAME,LOSS_NAME,chs,config.input_shape[0]))

    # The weights without the model architecture can also be saved. Just saving the weights is more efficent.

    # weight_path="./saved_weights/UNet/{}/".format(timestr)
    # if not os.path.exists(weight_path):
    #     os.makedirs(weight_path)
    # weight_path=weight_path + "{}_weights.best.hdf5".format('UNet_model')
    # print(weight_path)


import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration: 
    def __init__(self):
        # Initialize the data related variables used in the notebook
        # For reading the ndvi, pan and annotated images generated in the Preprocessing step.
        # In most cases, they will take the same value as in the config/Preprocessing.py
        self.base_dir = '/mnt/c/Users/Research/Documents/GitHub/africa-trees/data/first_mosaic/annotations/ready/output' #'/mnt/c/Users/Research/Documents/GitHub/africa-trees/SampleResults-Preprocessing'
        self.image_type = '.png' 
        self.ndvi_fn = 'ndvi'
        self.pan_fn = 'pan'
        self.annotation_fn = 'annotation'
        self.weight_fn = 'boundary' 
        
        # Patch generation; from the training areas (extracted in the last notebook), we generate fixed size patches.
        # random: a random training area is selected and a patch in extracted from a random location inside that training area. Uses a lazy stratergy i.e. batch of patches are extracted on demand.
        # sequential: training areas are selected in the given order and patches extracted from these areas sequential with a given step size. All the possible patches are returned in one call.
        self.patch_generation_stratergy = 'random' # 'random' or 'sequential'
        self.patch_size = (256,256,4) # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)
        
        # The training areas are divided into training, validation and testing set. Note that training area can have different sizes, so it doesn't guarantee that the final generated patches (when using sequential stratergy) will be in the same ratio. 
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        
        # Probability with which the generated patches should be normalized 0 -> don't normalize, 1 -> normalize all
        self.normalize = 0.4 

        
        # The split of training areas into training, validation and testing set, is cached in patch_dir.
        self.patch_dir = './patches{}'.format(self.patch_size[0])
        self.frames_json = os.path.join(self.patch_dir,'frames_list.json')


        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (256,256,2)
        self.input_image_channel = [0,1]
        self.input_label_channel = [2]
        self.input_weight_channel = [3]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 200

        # number of validation images to use
        self.VALID_IMG_COUNT = 200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 1000
        self.model_path = './saved_models/UNet/'

