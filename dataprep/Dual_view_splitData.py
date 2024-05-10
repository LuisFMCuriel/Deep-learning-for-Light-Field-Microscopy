from tifffile import imread, imsave
import os

directories = [r"X:\LuisFel\CARE\Muscle_Data\15062020", r"X:\LuisFel\CARE\Muscle_Data\16062020", r"X:\LuisFel\CARE\Muscle_Data\17062020", r"X:\LuisFel\CARE\Muscle_Data\msb343"]
path_s_h = r"X:\LuisFel\CARE\Test_dataset\high"
path_s_l = r"X:\LuisFel\CARE\Test_dataset\low"

for path in directories:
	for filename in os.listdir(path):
		img = imread(os.path.join(path, filename))
		n,h,w = img.shape
		for i in range(img.shape[0]):
			high = img[i,100:int(h/2),:]
			low = img[i, int(h/2)+100:,:]
			imsave(os.path.join(path_s_h, str(i) + filename), high)
			imsave(os.path.join(path_s_l, str(i) + filename), low)