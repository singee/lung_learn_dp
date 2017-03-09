#matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom, cv2
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
INPUT_FOLDER = './stage1/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
    
    
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    #import pdb; pdb.set_trace()
    
    new_shape =  np.array([300.0,  300.0,  300.0]) #np.round(new_real_shape)
    #new_shape =  np.array([64.0,  64.0,  64.0]) #np.round(new_real_shape) 
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing
    
    

def rotate_3d_slices(image, angle, axis=0):

    #input z_x_y 
    if axis == 2: # z_x_y => y_z_x
		image = np.transpose(image, (2, 0, 1))
    elif axis == 1: #z_x_y => x_y_z
		image = np.transpose(image, (1, 2, 0))
    else:           #z_x_y
		image = image
		
    rotate = []

    

    for i in range(len(image)):
		img = image[i]
		num_rows, num_cols = img.shape[:2]		
		rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), angle, 1)
		img1 = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
		rotate.append(img1)
     
    
    
    #if axis == 2: #y_z_x => z_x_y
	#	rotate = np.transpose(rotate, (1, 2, 0))
    #elif axis == 1: #x_y_z => z_x_y
	#	rotate = np.transpose(rotate, (2, 0, 1))
    #else:           #z_x_y
	#	rotate = rotate
		
		
    #for i in range(0,len(image),10):
    #    plt.subplot(2,1,1), plt.imshow(image[i],cmap='gray')
    #    plt.subplot(2,1,2), plt.imshow(rotate[i], cmap='gray')
    #    plt.show()
		
    return np.array(rotate)
    
    
    
def batch_save_rotated_scan(rotate_axis=0):
	for kk in range(len(patients)):
		#import pdb; pdb.set_trace() 
		current_patient = load_scan(INPUT_FOLDER + patients[kk])
		current_patient_pixels = get_pixels_hu(current_patient)
		pix_resampled, spacing = resample(current_patient_pixels, current_patient, [1,1,1])
		print("processing patient: ", patients[kk])
		print("Shape before resampling: ", current_patient_pixels.shape)
		print("Shape after resampling: ", pix_resampled.shape)
		print("spacing: ", spacing)
		
		for angle in range(0, 359, 30):
		    rotate = rotate_3d_slices(pix_resampled, angle, rotate_axis)
		    np.savez(patients[kk]+'_'+str(angle)+'_degree', rotate)
		    print("Rotated and Saved Patient: ", patients[kk]+'_'+str(angle)+'_degree')

    
    
def batch_save_scan():
	for kk in range(len(patients)):
		#import pdb; pdb.set_trace() 
		current_patient = load_scan(INPUT_FOLDER + patients[kk])
		current_patient_pixels = get_pixels_hu(current_patient)
		pix_resampled, spacing = resample(current_patient_pixels, current_patient, [1,1,1])
		#rotate_axis = 2
		#rotate_3d_slices(pix_resampled, 45, rotate_axis)
		
		np.savez(patients[kk], pix_resampled)
		print("processing patient: ", patients[kk])
		print("Shape before resampling: ", current_patient_pixels.shape)
		print("Shape after resampling: ", pix_resampled.shape)
		print("spacing: ", spacing)
        

        

def read_slices_by_range(scan_name='', start_slice = 0, end_slice = 300):
	npzfile = np.load(scan_name+'.npz')
	data = npzfile['arr_0']
	return data[start_slice:end_slice]
	

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image



def visualize_scan():
	for kk in range(len(patients)):
		#import pdb; pdb.set_trace() 
		current_patient = load_scan(INPUT_FOLDER + patients[kk])
		current_patient_pixels = get_pixels_hu(current_patient)
		pix_resampled, spacing = resample(current_patient_pixels, current_patient, [1,1,1])
		#plot_3d(pix_resampled,400)
		#rotate_axis = 0
		#rotate = rotate_3d_slices(pix_resampled, 45, rotate_axis)
		#plot_3d(rotate)
		#segmented_lungs = segment_lung_mask(pix_resampled, False)
		segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
        
		#for i in range(0,len(segmented_lungs),10):
		#	plt.imshow(segmented_lungs[i],cmap='gray')
		#	plt.show()
        
		#plot_3d(segmented_lungs, 0)
		plot_3d(segmented_lungs_fill, 0)
		#plot_3d(segmented_lungs_fill - segmented_lungs, 0)



#batch_save_scan()   
rotate_axis = 0 #rotate along z 
batch_save_rotated_scan(rotate_axis)  #angle interval fixed at 30 degree

''' 
#demo for load and display slices 
arr_slice = read_slices_by_range('0015ceb851d7251b8f399e39779d1e7d', 0, 390)

for i in range(len(arr_slice)):
	if (i%10 == 0):
		plt.imshow(arr_slice[i], cmap=plt.cm.gray)
		plt.show()
'''
