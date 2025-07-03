import torch
import os

############################################################
##################### DIRECTORIES ##########################
############################################################
# Source dir
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Base path of the dataset
DSET_BASEPATH = os.path.join(SRC_PATH, "data/MICCAI_BraTS2020/train")

# Define the path to the images
DSET_CPATHS = {
    # "FLAIR" : os.path.join(DSET_BASEPATH, "flair"),
    "T1CE" : os.path.join(DSET_BASEPATH, "t1ce"),
    # "T1" : os.path.join(DSET_BASEPATH, "t1"),
    # "T2" : os.path.join(DSET_BASEPATH, "t2")
}

# Define the path to the base output directory
BASE_OUTPUT = os.path.join(SRC_PATH, "output")

# Define the path to the output serialized model, model training
# Plot, and testing image paths
GNet_PATH = os.path.join(BASE_OUTPUT, "gnet.pth")
DNet_PATH = os.path.join(BASE_OUTPUT, "dnet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

############################################################
####################### HARDWARE ###########################
############################################################

# Determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# Define worker pool for dataloading
NUM_WORKERS = 10

############################################################
################### Brain Tumor Class ######################
############################################################

# Define the class for which model is being trained
BT_CLASS = "T1"

## config information
LATENT_SZ = 300 # 300 is good
# Size of feature maps for generator
NGF = 32
# Number of channels in the output
NGC = 1

# Define learning rate, beta1
GLR = 2e-4
GBETA1 = 0.5
NDC = 1
NDF = 32
NUM_CLASSES = 1
NUM_LEVELS = 3

# Define learning rate, beta1
#DLR = 2e-4
DLR = 1e-4
DBETA1 = 0.5

# Define negative slope for LeakyReLU activation
NEG_SLOPE = 0.2

NUM_EPOCHS = 100 # 15 seems good!
BATCH_SIZE = 8

# Weigth initialization
WINIT = "ortho"

############################################################
################### IMAGE TRANSFORMS #######################
############################################################

# define the input image dimensions
INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64