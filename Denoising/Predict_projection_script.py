from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread, imsave
from csbdeep.utils import Path, download_and_extract_zip_file, plot_some
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import ProjectionCARE

model_dir = r"X:\LuisFel\CARE\Models"
axes = "ZYX"
model = ProjectionCARE(config=None, name='FlyWing', basedir=model_dir)


path_r = r"X:\LuisFel\CARE\Data\Raw"
path_s = r"X:\LuisFel\CARE\Data\Results\FlyWing"
for filename in os.listdir(path_r):
	x = imread(os.path.join(path_r, filename))
	restored = model.predict(x, axes)
	imsave(os.path.join(path_s, filename), restored.astype("uint16"))