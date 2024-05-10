import os
from tifffile import imread, imsave


root_dir = r""
dir_s_GT = r""
dir_s_Noise = r""

cont = 0

for dire in os.listdir(root_dir): #dire will be the N stacks
	#print(dire)
	read_GT = True
	for subdir in os.listdir(os.path.join(root_dir, dire)): #subdir will be the exposure time
		#Use 10sec as your GT
		if "10_1" in os.listdir(os.path.join(root_dir, dire)) or "10" in os.listdir(os.path.join(root_dir, dire)):

			if read_GT == True:

				try:
					for filename in os.listdir(os.path.join(root_dir, dire, "10_1")):
						if filename.endswith("tif"):
							GT = imread(os.path.join(root_dir, dire, "10_1", filename))
							#print("GT is {}".format(os.path.join(root_dir, dire, "10_1", filename)))
				except:
					for filename in os.listdir(os.path.join(root_dir, dire, "10")):
						if filename.endswith("tif"):
							GT = imread(os.path.join(root_dir, dire, "10", filename))
							#print("GT is {}".format(os.path.join(root_dir, dire, "10", filename)))
				read_GT = False

			for filename in os.listdir(os.path.join(root_dir, dire, subdir)):
				if filename.endswith("tif") and subdir != "10" and subdir != "10_1":
					#·print("Coupled to {}".format(os.path.join(root_dir, dire, subdir, filename)))
					img = imread(os.path.join(root_dir, dire, subdir, filename))

					for i in range(img.shape[0]):
						imsave(os.path.join(dir_s_GT, str(cont+1) + ".tif"), GT[i,696:1672,1716:2650])
						imsave(os.path.join(dir_s_Noise, str(cont+1) + ".tif"), img[i,696:1672,1716:2650])
						cont += 1


		#Use 5sec as your GT
		else:
			if read_GT == True:
				try:
					for filename in os.listdir(os.path.join(root_dir, dire, "5_1")):
						if filename.endswith("tif"):
							GT = imread(os.path.join(root_dir, dire, "5_1", filename))
				except:
					for filename in os.listdir(os.path.join(root_dir, dire, "5")):
						if filename.endswith("tif"):
							GT = imread(os.path.join(root_dir, dire, "5", filename))
				read_GT = False

			for filename in os.listdir(os.path.join(root_dir, dire, subdir)):
				if filename.endswith("tif") and subdir != "5" and subdir != "5_1":
					img = imread(os.path.join(root_dir, dire, subdir, filename))

					for i in range(img.shape[0]):
						imsave(os.path.join(dir_s_GT, str(cont+1) + ".tif"), GT[i,696:1672,1716:2650])
						imsave(os.path.join(dir_s_Noise, str(cont+1) + ".tif"), img[i,696:1672,1716:2650])
						cont += 1
