"""
This script includes a multiprocess routine that decreases runtime by half by running the 
blob detection in parallel (2 parallel processes, 1 for each core in the macbook). 

Unfortunately the python CPU Pool.map function only allows passing one argument to each 
process, so the program had to be divided up in an awkward manner. It will be an issue if 
there are an odd number of list elements in each mapped function call.

It is possible to get around this using partial functions but I'm still learning how that 
works.

Running this on the macbook there is a decreases for single process to two processes of 
990.4 seconds to 509.5 seconds.
"""


# Load modules

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import warnings
import multiprocessing as mp

from scipy.ndimage import gaussian_filter
from scipy.stats import mode
from math import sqrt

from skimage import data, io, filters, measure, segmentation, morphology
from skimage import img_as_ubyte, img_as_float

from skimage.color import rgb2gray
from skimage.morphology import reconstruction
from skimage.feature import blob_dog, blob_log, blob_doh



#--------------------------------------------------------------------------------------

def temblob(image,ind):

	"""
	Laplacian of gaussian blob detection for TEM images
	"""

	org = image[4:-256,4:-4]


	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		warnings.warn("user", UserWarning)

		igray = img_as_ubyte(rgb2gray(org))
		iinv = np.invert(igray)


	igaus = img_as_float(iinv)
	igaus = gaussian_filter(igaus, 1)
	h = 0.5
	sd = igaus - h
	msk = igaus
	dilat = reconstruction(sd, msk, method='dilation')
	hdome = igaus - dilat

	if ind == 'AgNP':
		kwargs = {}
		kwargs['threshold'] = 0.01
		kwargs['overlap'] = 0.4
		kwargs['min_sigma'] = 25
		kwargs['max_sigma'] = 50
		kwargs['num_sigma'] = 25
		calib = 500/(969-26)
		
	elif ind == 'AuNP':
		kwargs = {}
		kwargs['threshold'] = 0.01
		kwargs['overlap'] = 0.4
		kwargs['min_sigma'] = 18
		kwargs['max_sigma'] = 30
		kwargs['num_sigma'] = 12
		calib = 200/(777-23)
		
	else:
		warnmsg='Unable to identify keyword: {:}'.format(ind)
		warnings.warn(warnmsg, UserWarning)

	blobs = blob_log(hdome, **kwargs)
	diam = 2*sqrt(2)*blobs[:,-1]
	npdiam = [ind]
	npdiam.extend(calib*diam)

	return(npdiam)
	


#--------------------------------------------------------------------------------------

def plotresults(agdia, audia):

	plt.ion()

	plt.rc('font', size='18')

	bins = range(10,80,2)

	fig, axes = plt.subplots(1,2, figsize=(16,6))
	for a,title,sample,c in zip(axes, ('AuNP','AgNP'),
									(audia, agdia), ['CornflowerBlue','IndianRed']):
		a.set_xlabel('Diameter (nm)')
		a.set_ylabel('Number of particles')
		a.set_title(title)
		_, _, _ = a.hist(sample, bins=bins, color=c)


#--------------------------------------------------------------------------------------

def statresults(agdia, audia):

	# Statistical report
	dfinal = pd.DataFrame([[len(audia)],[len(agdia)]], 
	columns=['Total particles counted'], index=['AuNP','AgNP'])
	dfinal['Mean Diameter (nm)'] = np.round([np.mean(audia),np.mean(agdia)],1)
	dfinal['Median Diameter (nm)'] = np.round([np.median(audia),np.median(agdia)],1)
	dfinal['Mode Diameter (nm)'] = np.round([mode(audia.tolist())[0],
	mode(agdia.tolist())[0]],1)
	
	return dfinal

#--------------------------------------------------------------------------------------
    
if __name__ == '__main__':

	# Select directory where files are stored
	rdir = '/Volumes/TRANSFER/TEM/2015-10-15 - AuNP and AgNP/'
	files = os.listdir(rdir)


	# Sort out non images '*.tif'
	srtf = [f for f in files if '.tif' in f]


	# Not all files are useful due to shifting magnification scales, must manually input
	manf = ['AgNP-01.tif','AgNP-02.tif','AgNP-03.tif','AgNP-04.tif','AgNP-05.tif']
	manf.extend(['AgNP-06.tif','AgNP-07.tif','AgNP-11.tif','AgNP-12.tif'])
	manf.extend(['AuNP-02.tif','AuNP-03.tif','AuNP-05.tif','AuNP-06.tif','AuNP-07.tif'])
	manf.extend(['AuNP-08.tif','AuNP-09.tif','AuNP-10.tif'])


	# Check all files are input correctly to prevent errors during runtime
# 	presnt = all(f in srtf for f in manf)
# 	print('Any typos in manual input file names:', str(not presnt))
# 	typo = all('Au' in s or 'Ag' in s for s in manf)
# 	print('Any typos in original file names:', str(not typo))


# Start timer to monitor program run time
	startTime = time.time()


# List of images(numpy arrays)
	images = [io.imread(rdir+f) for f in manf]
# List of indicators, if 'Au' or 'Ag'
	inds = [s[:4] for s in manf]
# Need to pass the arguments as tuples
	iterdata = [(i,j) for i,j in zip(images,inds)]


# Two process version, pass filename and corresponding NP indicator
	with mp.Pool(processes=2) as pool:
		reslist = pool.starmap(temblob, iterdata, 1)
	pool.close()
	pool.join()

# Serial or single process version
# 	agres = list(map(agtemblob, agimages))
# 	aures = list(map(autemblob, auimages))

# Print elapsed time
	print('')			
	print('Elapsed time:', round(time.time()-startTime,1), 'seconds')
	print('')


	agdia = np.concatenate([res[1:] for res in reslist if 'AgNP' in res])
	audia = np.concatenate([res[1:] for res in reslist if 'AuNP' in res])
# 	cols = list(set([res[0] for res in reslist]))
	

	plotresults(agdia, audia)
	
	print(statresults(agdia, audia))
	