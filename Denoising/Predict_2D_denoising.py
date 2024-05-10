import tensorflow as tf
import numpy as np
from tifffile import imread, imsave
from csbdeep.utils import download_and_extract_zip_file, plot_some, axes_dict, plot_history, Path, download_and_extract_zip_file
from csbdeep.data import RawData, create_patches 
from csbdeep.io import load_training_data, save_tiff_imagej_compatible
from csbdeep.models import Config, CARE
from csbdeep import data
from pathlib import Path
import os, random
import shutil
import pandas as pd
import csv
import subprocess
from csbdeep.utils import normalize
import scipy.stats as sc
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


if tf.test.gpu_device_name()=='':
	print('You do not have GPU access.') 

else:
	print('You have GPU access')
	subprocess.run(["nvidia-smi"])


Test_data_folder = r"" 
Result_folder_root = r"" 

models = ["Model02062020", "Model02062020v2_aug", "Tribolium_Denoise", "FlyWing", "LF_Denoisev2"] #

for model in models:
	inference_model_name = model #@param {type:"string"}
	inference_model_path = r""
	Result_folder = os.path.join(Result_folder_root, inference_model_name)

	if not os.path.exists(Result_folder): #Create directories for the registered images
		os.makedirs(Result_folder)
	else:
		shutil.rmtree(Result_folder)
		os.makedirs(Result_folder)


	#Activate the pretrained model. 
	model_training = CARE(config=None, name=inference_model_name, basedir=inference_model_path)

	STACK = "N"
	# creates a loop, creating filenames and saving them
	if STACK == "Y":
		for filename in os.listdir(Test_data_folder):
			img = imread(os.path.join(Test_data_folder,filename))
			print(model, filename)
			r = np.zeros_like(img)
			for s in range(img.shape[0]):
				restored = model_training.predict(img[s,:,:], axes='YX')
				#restored = ((restored-np.amax(restored))/(np.amax(restored) - np.amin(restored)))*65535
				r[s,:,:] = restored
			os.chdir(Result_folder)
			imsave(filename,r.astype("uint16"))
	else:

		for filename in os.listdir(Test_data_folder):
			img = imread(os.path.join(Test_data_folder,filename))
			print(model, filename)
			restored = model_training.predict(img, axes='YX')
			os.chdir(Result_folder)
			imsave(filename,restored)
    

	print("Images saved into folder:", Result_folder)