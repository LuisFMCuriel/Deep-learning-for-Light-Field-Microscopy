import os, random
import shutil
import numpy as np
from tifffile import imread, imsave
import time
import pandas as pd
from csbdeep.models import Config, CARE
from csbdeep import data
from csbdeep.data import RawData, create_patches 
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict
import csv

def Save_NPZ(base, Training_source, Training_target):

	raw_data = data.RawData.from_folder(
    basepath=base,
    source_dirs=[Training_source], 
    target_dir=Training_target, 
    axes='CYX', 
    pattern='*.tif*')

	X, Y, XY_axes = data.create_patches(
	    raw_data, 
	    patch_filter=None, 
	    patch_size=(patch_size,patch_size), 
	    n_patches_per_image=number_of_patches)

	print ('Creating 2D training dataset')
	training_path = model_path+"/rawdata"
	rawdata1 = training_path+".npz"
	np.savez(training_path,X=X, Y=Y, axes=XY_axes)
	return rawdata1


#def Load_NPZ(model_path):
#	training_path = model_path+"/rawdata.npz"
#	return np.load(training_path)


Training_source = r"" 
InputFile = Training_source+"/*.tif"

Training_target = r""
OutputFile = Training_target+"/*.tif"

#Define where the patch file will be saved
base = ""


# model name and path
model_name = ""
model_path = r"" 

number_of_epochs = 1
patch_size =  256
number_of_patches = 25
batch_size =  16
number_of_steps =   0
percentage_validation =  10
initial_learning_rate = 0.0004
percentage = percentage_validation/100

Create_npz = False

random_choice = random.choice(os.listdir(Training_source))
x = imread(os.path.join(Training_source,random_choice))
Image_Y = x.shape[0]
Image_X = x.shape[1]


#Check patch size
if patch_size > min(Image_Y, Image_X):
	patch_size = min(Image_Y, Image_X)
if not patch_size % 8 == 0:
	patch_size = ((int(patch_size / 8)-1) * 8)


#Loading weights from a pre-trained network
Use_pretrained_model = False
Weights_choice = "best"
pretrained_model_path = r""

if Use_pretrained_model:
	h5_file_path = os.path.join(pretrained_model_path, "weights_"+Weights_choice+".h5")

	if os.path.exists(h5_file_path):
		#Here we check if the learning rate can be loaded from the quality control folder
		if os.path.exists(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv')):
			with open(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv'),'r') as csvfile:
				csvRead = pd.read_csv(csvfile, sep=',')
				if "learning rate" in csvRead.columns: #Here we check that the learning rate column exist (compatibility with model trained un ZeroCostDL4Mic bellow 1.4)
					print("pretrained network learning rate found")
					#find the last learning rate
					lastLearningRate = csvRead["learning rate"].iloc[-1]
					#Find the learning rate corresponding to the lowest validation loss
					min_val_loss = csvRead[csvRead['val_loss'] == min(csvRead['val_loss'])]
					#print(min_val_loss)
					bestLearningRate = min_val_loss['learning rate'].iloc[-1]
					if Weights_choice == "last":
						print('Last learning rate: '+str(lastLearningRate))
					if Weights_choice == "best":
						print('Learning rate of best validation loss: '+str(bestLearningRate))
				if not "learning rate" in csvRead.columns: #if the column does not exist, then initial learning rate is used instead
					bestLearningRate = initial_learning_rate
					lastLearningRate = initial_learning_rate
					print(bcolors.WARNING+'WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of '+str(bestLearningRate)+' will be used instead')

		#Compatibility with models trained outside ZeroCostDL4Mic but default learning rate will be used
		if not os.path.exists(os.path.join(pretrained_model_path, 'Quality Control', 'training_evaluation.csv')):
			print(bcolors.WARNING+'WARNING: The learning rate cannot be identified from the pretrained network. Default learning rate of '+str(initial_learning_rate)+' will be used instead')
			bestLearningRate = initial_learning_rate
			lastLearningRate = initial_learning_rate


if Create_npz:
	rawdata1 = Save_NPZ(base, Training_source, Training_target)
else:
	training_path = model_path+"/rawdata"
	rawdata1 = training_path+".npz"

# Load Training Data
(X,Y), (X_val,Y_val), axes = load_training_data(rawdata1, validation_split=percentage, verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

if number_of_steps == 0:
	number_of_steps = int(X.shape[0]/batch_size)+1

if Use_pretrained_model:
	if Weights_choice == "last":
		initial_learning_rate = lastLearningRate

	if Weights_choice == "best":            
		initial_learning_rate = bestLearningRate


config = Config(axes, n_channel_in, n_channel_out, probabilistic=True, train_steps_per_epoch=number_of_steps, train_epochs=number_of_epochs, unet_kern_size=5, unet_n_depth=3, train_batch_size=batch_size, train_learning_rate=initial_learning_rate)

model_training= CARE(config, model_name, basedir=model_path)
if Use_pretrained_model:
	model_training.load_weights(h5_file_path)


start = time.time()

history = model_training.train(X,Y, validation_data=(X_val,Y_val))
shutil.copyfile(model_path+'/rawdata.npz',model_path+'/'+model_name+'/rawdata.npz')

lossData = pd.DataFrame(history.history) 


if os.path.exists(model_path+"/"+model_name+"/Quality Control"):
	shutil.rmtree(model_path+"/"+model_name+"/Quality Control")

os.makedirs(model_path+"/"+model_name+"/Quality Control")

# The training evaluation.csv is saved (overwrites the Files if needed). 
lossDataCSVpath = model_path+'/'+model_name+'/Quality Control/training_evaluation.csv'
with open(lossDataCSVpath, 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['loss','val_loss', 'learning rate'])
	for i in range(len(history.history['loss'])):
		writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])
