#Modification from the CARE2D notebook: https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/StarDist_2D_ZeroCostDL4Mic.ipynb

from csbdeep.models import Config, CARE
import csv
from tifffile import imread, imsave
from scipy import signal
from scipy import ndimage
from skimage import io
from sklearn.linear_model import LinearRegression
from skimage.util import img_as_uint
import matplotlib as mpl
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import numpy as np
import os, random
import shutil 
from glob import glob

def ssim(img1, img2):
  return structural_similarity(img1,img2,data_range=1.,full=True, gaussian_weights=True, use_sample_covariance=False, sigma=1.5)


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def norm_minmse(gt, x, normalize_gt=True):
    """This function is adapted from Martin Weigert"""

    """
    normalizes and affinely scales an image pair such that the MSE is minimized  
     
    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)
    Returns
    -------
    gt_scaled, x_scaled 
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy = False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    #x = x - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    #gt = gt - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x

Source_QC_folder = r""
Target_QC_folder = r"" 
Result_folder = r""
QC_model_path = r""

for QC_model_name in models:
    print("Running model" + QC_model_name)
    Result_folder_model = os.path.join(Result_folder, QC_model_name)
    if not os.path.exists(Result_folder_model):
        os.mkdir(Result_folder_model)
    if not os.path.exists(os.path.join(Result_folder_model, "Quality Control")):
        os.mkdir(os.path.join(Result_folder_model, "Quality Control"))
        os.mkdir(os.path.join(Result_folder_model, "Quality Control", "Prediction"))
    # Activate the pretrained model. 
    model_training = CARE(config=None, name=QC_model_name, basedir=QC_model_path)

    # List Tif images in Source_QC_folder
    Source_QC_folder_tif = Source_QC_folder+"/*.tif"
    Z = sorted(glob(Source_QC_folder_tif))
    Z = list(map(imread,Z))
    print('Number of test dataset found in the folder: '+str(len(Z)))


    # Perform prediction on all datasets in the Source_QC folder
    for filename in os.listdir(Source_QC_folder):
      img = imread(os.path.join(Source_QC_folder, filename))
      predicted = model_training.predict(img, axes='YX')
      os.chdir(Result_folder_model)
      imsave(filename, predicted)
      imsave(os.path.join(Target_QC_folder, filename), predicted)

    # Open and create the csv file that will contain all the QC metrics
    with open(os.path.join(Result_folder_model, "Quality Control", "QC_model_name.csv"), "w", newline='') as file:
        writer = csv.writer(file)

        # Write the header in the csv file
        writer.writerow(["image #","Prediction v. GT mSSIM","Input v. GT mSSIM", "Prediction v. GT NRMSE", "Input v. GT NRMSE", "Prediction v. GT PSNR", "Input v. GT PSNR"])  

        # Let's loop through the provided dataset in the QC folders


        for i in os.listdir(Source_QC_folder):
          if not os.path.isdir(os.path.join(Source_QC_folder,i)):
            print('Running QC on: '+i)
          # -------------------------------- Target test data (Ground truth) --------------------------------
            test_GT = io.imread(os.path.join(Target_QC_folder, i))

          # -------------------------------- Source test data --------------------------------
            test_source = io.imread(os.path.join(Source_QC_folder,i))

          # Normalize the images wrt each other by minimizing the MSE between GT and Source image
            test_GT_norm,test_source_norm = norm_minmse(test_GT, test_source, normalize_gt=True)

          # -------------------------------- Prediction --------------------------------
            test_prediction = io.imread(os.path.join(Result_folder_model, i))

          # Normalize the images wrt each other by minimizing the MSE between GT and prediction
            test_GT_norm,test_prediction_norm = norm_minmse(test_GT, test_prediction, normalize_gt=True)        


          # -------------------------------- Calculate the metric maps and save them --------------------------------

          # Calculate the SSIM maps
            index_SSIM_GTvsPrediction, img_SSIM_GTvsPrediction = ssim(test_GT_norm, test_prediction_norm)
            index_SSIM_GTvsSource, img_SSIM_GTvsSource = ssim(test_GT_norm, test_source_norm)

          #Save ssim_maps
            img_SSIM_GTvsPrediction_32bit = np.float32(img_SSIM_GTvsPrediction)
            io.imsave(os.path.join(Result_folder_model, "Quality Control", "SSIM_GTvsPrediction_"+i),img_SSIM_GTvsPrediction_32bit)
            img_SSIM_GTvsSource_32bit = np.float32(img_SSIM_GTvsSource)
            io.imsave(os.path.join(Result_folder_model, "Quality Control", "SSIM_GTvsSource_"+i),img_SSIM_GTvsSource_32bit)

          # Calculate the Root Squared Error (RSE) maps
            img_RSE_GTvsPrediction = np.sqrt(np.square(test_GT_norm - test_prediction_norm))
            img_RSE_GTvsSource = np.sqrt(np.square(test_GT_norm - test_source_norm))

          # Save SE maps
            img_RSE_GTvsPrediction_32bit = np.float32(img_RSE_GTvsPrediction)
            img_RSE_GTvsSource_32bit = np.float32(img_RSE_GTvsSource)
            io.imsave(os.path.join(Result_folder_model, "Quality Control", "RSE_GTvsPrediction_"+i),img_RSE_GTvsPrediction_32bit)
            io.imsave(os.path.join(Result_folder_model, "Quality Control", "RSE_GTvsSource_"+i),img_RSE_GTvsSource_32bit)


          # -------------------------------- Calculate the RSE metrics and save them --------------------------------

          # Normalised Root Mean Squared Error (here it's valid to take the mean of the image)
            NRMSE_GTvsPrediction = np.sqrt(np.mean(img_RSE_GTvsPrediction))
            NRMSE_GTvsSource = np.sqrt(np.mean(img_RSE_GTvsSource))

          # We can also measure the peak signal to noise ratio between the images
            PSNR_GTvsPrediction = psnr(test_GT_norm,test_prediction_norm,data_range=1.0)
            PSNR_GTvsSource = psnr(test_GT_norm,test_source_norm,data_range=1.0)

            writer.writerow([i,str(index_SSIM_GTvsPrediction),str(index_SSIM_GTvsSource),str(NRMSE_GTvsPrediction),str(NRMSE_GTvsSource),str(PSNR_GTvsPrediction),str(PSNR_GTvsSource)])


    # All data is now processed saved
    Test_FileList = os.listdir(Source_QC_folder) # this assumes, as it should, that both source and target are named the same