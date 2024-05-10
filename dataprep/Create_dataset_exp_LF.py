import os
from tifffile import imread, imsave
import numpy as np
from scipy.ndimage import shift
#from skimage import registration
import matlab.engine
from pystackreg import StackReg
from patchify import patchify
import shutil

def Get_images(path_r, day, flag_LF = True, flag_WF = True, Only_middle_img = False, both = True):
	print("Getting the images")
	path_s = os.path.join(path_r, "Dataset_"+day, "LF_raw")
	path_s_WF = os.path.join(path_r, "Dataset_"+day, "WF255_middle")
	path_stacks_WF = os.path.join(path_r, "Dataset_"+day, "WF_stacks")
	path_r = os.path.join(path_r, day)
	xoff = -2.7368772 
	yoff = 24.509293
	cont_LF = 0
	N_WF = 0
	cont_WF = 0



	for filename in os.listdir(path_r):
		idx_worm = filename.find("worm") 
		idx_ = filename[idx_worm:].find("_")
		idx_ += idx_worm
		N_worm_id = filename[idx_worm+4:idx_]
		N_worm = int(N_worm_id)
		print(N_worm)
		print(filename)
		if "LF" in filename and flag_LF == True:
			for sub_filename in os.listdir(os.path.join(path_r, filename)):
				if sub_filename.endswith("tif"):
					img = imread(os.path.join(path_r, filename, sub_filename))
					N_middle = int(img.shape[0]/2)
					Middle_img = img[N_middle,100:-100,100:-100]
					rotated = np.rot90(Middle_img, -1)
					

					rotated = shift(rotated, shift=(xoff, yoff), mode ="constant")
					rotated = ((rotated - np.amin(rotated))/(np.amax(rotated) - np.amin(rotated)))*255.0
					imsave(os.path.join(path_s, str(cont_LF)+".tif"), rotated.astype("uint8"))
					print(str(cont_LF)+".tif")
					if cont_LF == 0:
						ID = N_worm
					if ID == N_worm:
						N_WF += 1
					else:
						#print("Needed {}".format(N_WF))
						N_WF = 1
						ID = N_worm
					cont_LF += 1




		elif "WF" in filename and flag_WF == True:
			for sub_filename in os.listdir(os.path.join(path_r, filename)):
				if sub_filename.endswith("tif"):
					img = imread(os.path.join(path_r, filename, sub_filename))
					if both:
						N_middle = int(img.shape[0]/2)
						Middle_img = img[N_middle,100:-100,100:-100]
						Middle_img = ((Middle_img - np.amin(Middle_img))/(np.amax(Middle_img) - np.amin(Middle_img)))*255.0
						
						Stack_img = img[:,100:-100,100:-100]
						Stack_img = ((Stack_img - np.amin(Stack_img))/(np.amax(Stack_img) - np.amin(Stack_img)))*255.0

						print("Doing {} copies".format(N_WF))
						for i in range(N_WF):
							imsave(os.path.join(path_s_WF, str(cont_WF)+".tif"), Middle_img.astype("uint8"))
							imsave(os.path.join(path_stacks_WF, str(cont_WF)+".tif"), Stack_img.astype("uint8"))
							cont_WF += 1

					elif Only_middle_img:
						N_middle = int(img.shape[0]/2)
						Middle_img = img[N_middle,100:-100,100:-100]
						Middle_img = ((Middle_img - np.amin(Middle_img))/(np.amax(Middle_img) - np.amin(Middle_img)))*255.0
						print("Doing {} copies".format(N_WF))
						for i in range(N_WF):
							imsave(os.path.join(path_s_WF, str(cont_WF)+".tif"), Middle_img.astype("uint8"))
							cont_WF += 1
					else:
						Middle_img = img[:,100:-100,100:-100]
						path_s_WF = path_stacks_WF
						Middle_img = ((Middle_img - np.amin(Middle_img))/(np.amax(Middle_img) - np.amin(Middle_img)))*255.0
						print("Doing {} copies".format(N_WF))
						for i in range(N_WF):
							imsave(os.path.join(path_s_WF, str(cont_WF)+".tif"), Middle_img.astype("uint8"))
							cont_WF += 1


def Rectification_matlab(path_img_to_rectify, path_r_rectification, matlab_script_path, stack):
	print("Rectification")
	for filename in os.listdir(path_img_to_rectify):
		if filename.endswith(".tif"):
			shutil.copy(os.path.join(path_img_to_rectify, filename), path_r_rectification)
	positionOfPath = 1
	os.chdir(matlab_script_path)
	
	eng = matlab.engine.start_matlab()
	if stack:
		eng.Register_3d_multiple_stacks(nargout=0)
	else:
		eng.Register_3d_multiple(nargout=0)
	eng.quit()

def Register(path_root):
	print("Registering")
	print(path_root)
	path_r_WF = os.path.join(path_root, "WF255_middle_rectified")
	path_r_LF = os.path.join(path_root, "LF_rectified")
	path_r_pred = os.path.join(path_root, "Reconstructions_middle")
	path_s_LF = os.path.join(path_root, "LF_rectified_registered")
	path_s_pred = os.path.join(path_root,"Reconstruction_middle_registered")

	sr = StackReg(StackReg.SCALED_ROTATION)

	cont = 0
	for i, filename in enumerate(os.listdir(path_r_WF)):
		if filename.endswith(".tif"):
			print(i)
			GT = imread(os.path.join(path_r_WF, filename))
			prediction = imread(os.path.join(path_r_pred, filename))
			LF = imread(os.path.join(path_r_LF, filename))

			if i == 0:
				matrix = sr.register(GT, prediction)

			prediction = sr.register_transform(GT, prediction)
			prediction = prediction.clip(min=0)
			


			LF = sr.transform(LF, tmat = matrix)


			LF = ((LF - np.amin(LF))/(np.amax(LF)- np.amin(LF)))*255.0
			prediction = (((prediction - np.amin(LF))/(np.amax(prediction) - np.amin(prediction))))*255.0
			imsave(os.path.join(path_s_LF, filename), LF.astype("uint8"))
			imsave(os.path.join(path_s_pred, filename), prediction.astype("uint8"))



def crop_imgs(path, path_s):
	print("Cropping images in {}".format(path))
	if "WF" in path:
		for filename in os.listdir(path):
			print(filename)
			if filename.endswith(".tif"):
				img = imread(os.path.join(path, filename))
				img = img[:,35:-20, 55:-25]
				img = ((img - np.amin(img))/(np.amax(img)- np.amin(img)))*255.0
				imsave(os.path.join(path_s, filename), img.astype("uint8"))
	else:
		for filename in os.listdir(path):
			print(filename)
			if filename.endswith(".tif"):
				img = imread(os.path.join(path, filename))
				img = img[35:-20, 55:-25]
				img = ((img - np.amin(img))/(np.amax(img)- np.amin(img)))*255.0
				imsave(os.path.join(path_s, filename), img.astype("uint8"))


def func_patchify(path_root, patch_size = 264, WF = True, LF = True, clean = True):
	print("Patchifying")
	path_LF = os.path.join(path_root, "LF_rectified_registered_cropped")
	path_WF = os.path.join(path_root, "WF_stacks_rectified_cropped")
	path_WF_patches = os.path.join(path_root, "patches_WF_no_augmentation_size={}".format(patch_size))
	path_LF_patches = os.path.join(path_root, "patches_LF_no_augmentation_size={}".format(patch_size))

	if WF == True:
		print("Making patches for WF")
		patch_wf = np.zeros((31,patch_size,patch_size), dtype="uint8")
		#patches_img = np.zeros((31,3,3,patch_size,patch_size))
		cont = 0
		for img_WF in os.listdir(path_WF):
			if img_WF.endswith(".tif"):
				img = imread(os.path.join(path_WF, img_WF))
				Size = patchify(img[0,:,:], (patch_size,patch_size), step=patch_size)
				patches_img = np.zeros((31,Size.shape[0],Size.shape[1],patch_size,patch_size))
				for i in range(img.shape[0]):
					patches_img[i,:,:,:,:] = patchify(img[i,:,:], (patch_size,patch_size), step=patch_size)
				#print(patches_img.shape)
				for i in range(patches_img.shape[1]):
					for j in range(patches_img.shape[1]):
						patch_wf = patches_img[:,i,j,:,:]

						imsave(os.path.join(path_WF_patches, str(cont) + ".tif"), patch_wf.astype("uint8"))
						cont += 1

	if LF == True:
		print("Making patches for LF")
		cont = 0
		for img_LF in os.listdir(path_LF):
			if img_LF.endswith(".tif"):
				img = imread(os.path.join(path_LF, img_LF))
				patches_img = patchify(img[:,:], (patch_size,patch_size), step=patch_size)
				#print(patches_img.shape)
				for j in range(patches_img.shape[0]):
					for k in range(patches_img.shape[1]):
						imsave(os.path.join(path_LF_patches, str(cont) + ".tif"), patches_img[j,k,:,:].astype("uint8"))
						cont += 1



	if clean == True:
		print("Cleaning")
		for filename in os.listdir(path_WF_patches):
			if filename.endswith(".tif"):
				img = imread(os.path.join(path_WF_patches, filename))
				if np.amax(img) < 10:
					os.remove(os.path.join(path_WF_patches, filename))
					os.remove(os.path.join(path_LF_patches, filename))

def Mkdirs(path_root, patch_size):
	print("Creating directories")
	dirs = ["LF_raw", "LF_rectified", "LF_rectified_registered", "LF_rectified_registered_cropped", "patches_LF_no_augmentation_size={}".format(patch_size), "patches_WF_no_augmentation_size={}".format(patch_size), "Reconstruction_middle_registered", "Reconstructions_middle", "Reconstructions_stacks", "WF_stacks", "WF_stacks_rectified", "WF_stacks_rectified_cropped", "WF255_middle", "WF255_middle_rectified"]
	for dir_name in dirs:
		if not os.path.isdir(os.path.join(path_root, dir_name)):
			os.mkdir(os.path.join(path_root, dir_name))

def Reconstruction(path_vcdnet = r"X:\Drive_F\VCD-Net-main\vcdnet", use_cpu = 0):
	print("Reconstructing stacks")
	warnings.filterwarnings('ignore')

	print("Parameters defined in config.py:")
	print("PSF related: ")
	for par, val in config.PSF.items():
	    print('    {:<30}   {:<30}'.format(par,val))
	        
	print("Prediction related: ")
	for par, val in config.VALID.items():
	    print('    {:<30}   {:<30}'.format(par,val))
	    
	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	ckpt = 0
	batch_size = 1
	#use_cpu = 0                                             
	infer(ckpt, batch_size=batch_size, use_cpu=use_cpu)
	# restore np.load for future normal usage
	np.load = np_load_old

def copy_imgs(path, dest):
	for filename in os.listdir(path):
		if filename.endswith(".tif"):
			shutil.copy(os.path.join(path, filename), dest)


def clean_dir(folder):
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))


def Get_middle_img(path, path_s):

	for cont, filename in enumerate(os.listdir(path)):
		img = imread(os.path.join(path, filename))
		N = int(img.shape[0]/2)
		Img = img[N,:,:]
		Img = ((Img - np.amin(Img))/(np.amax(Img) - np.amin(Img)))*255.0
		#Img[Img < 3.1] = 0.0

		imsave(os.path.join(path_s, str(cont) + ".tif"), Img.astype("uint8"))

day = "20220413"
mkdirs = True
Get_img = False
rectification_ = False
patch_size = 256

reconstruction_ = False
use_cpu = 0
register_ = False
patchify_ = True

path_r = r"Z:\LuisFel\0 FromMicroscopes\1 Leica\MSB577 LightField Dataset"
path_root = r"Z:\LuisFel\0 FromMicroscopes\1 Leica\MSB577 LightField Dataset\Dataset_{}".format(day)
path_r_matlab_rect = r"Z:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Data\01_Raw"
path_s_matlab_rect = r"Z:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Data\02_Rectified"
path_matlab_script = r"Z:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Code"

if mkdirs == True:
	Mkdirs(path_root, patch_size)

if Get_img == True:
	Get_images(path_r, day, flag_LF = True, flag_WF = True, Only_middle_img = False, both = True)

if rectification_ == True:
	dirs_to_rectify = ["LF_raw", "WF255_middle", "WF_stacks"]
	dirs_rectified = ["LF_rectified", "WF255_middle_rectified", "WF_stacks_rectified"]
	for i in range(len(dirs_to_rectify)):
		if "WF_stacks" in dirs_to_rectify[i]:
			stack = True
		elif "LF" in dirs_to_rectify[i] or "middle" in dirs_to_rectify[i]:
			stack = False
		else:
			print("There is something weird with the folder name, no LF or WF found in the name")
		path_r = os.path.join(path_root, os.path.join(path_root, dirs_to_rectify[i]))
		clean_dir(path_r_matlab_rect)
		clean_dir(path_s_matlab_rect)
		#copy_imgs(path, r"X:\LuisFel\Simultaneous whole-animal\41592_2014_BFnmeth2964_MOESM189_ESM\SIsoftware\Data\01_Raw")
		Rectification_matlab(path_r, path_r_matlab_rect, path_matlab_script, stack)

		copy_imgs(path_s_matlab_rect, os.path.join(path_root, dirs_rectified[i]))
		clean_dir(path_r_matlab_rect)
		clean_dir(path_s_matlab_rect)

if reconstruction_ == True:
	print("Performing Reconstruction")
	path_vcdnet = r"X:\Drive_F\VCD-Net-main\vcdnet"
	os.chdir(path_vcdnet)
	files = [filename for filename in os.listdir(os.getcwd())]
	print(files)
	from eval import *
	import warnings
	path_r_pred = r"X:\Drive_F\VCD-Net-main\vcdnet\data\to_predict"
	model_name = "VCD_MuscleNucleus40xdof30z1_dx29_experimentalNoAugm_TransferLearning_MixedDatasets_patchSize264_Dataset1192patches"
	path_s_pred = os.path.join(r"X:\Drive_F\VCD-Net-main\vcdnet\results", model_name)
	clean_dir(path_r_pred)
	clean_dir(path_s_pred)
	copy_imgs(os.path.join(path_root, "LF_rectified"), path_r_pred)
	Reconstruction(path_vcdnet = path_vcdnet, use_cpu = use_cpu)
	copy_imgs(path_s_pred, os.path.join(path_root, "Reconstructions_stacks"))
	clean_dir(path_r_pred)
	clean_dir(path_s_pred)

	Get_middle_img(os.path.join(path_root, "Reconstructions_stacks"), os.path.join(path_root, "Reconstructions_middle"))


if register_ == True:
	Register(path_root)
	crop_imgs(os.path.join(path_root, "LF_rectified_registered"), os.path.join(path_root, "LF_rectified_registered_cropped"))
	crop_imgs(os.path.join(path_root, "WF_stacks_rectified"), os.path.join(path_root, "WF_stacks_rectified_cropped"))

if patchify_ == True:
	func_patchify(path_root, patch_size = patch_size, WF = True, LF = True, clean = True)

