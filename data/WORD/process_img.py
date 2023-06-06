import nibabel as nib
import glob
import numpy as np
from scipy.ndimage import binary_fill_holes
import os
from scipy.ndimage import zoom
from tqdm import tqdm

def calculate_space(space_x, space_y, space_z, size_x, size_y, size_z):
    space_x, space_y, space_z = np.array(space_x), np.array(space_y), np.array(space_z)
    size_x, size_y, size_z = np.array(size_x), np.array(size_y), np.array(size_z)
    median_x_space, median_y_space, median_z_space = np.median(space_x), np.median(space_y), np.median(space_z)
    median_x_size, median_y_size, median_z_size = np.median(size_x), np.median(size_y), np.median(size_z)

    choose_median_space = (max(median_x_space, median_y_space, median_z_space) / min(median_x_space, median_y_space, median_z_space) <= 3)
    choose_median_size = (max(median_x_size, median_y_size, median_z_size) / min(median_x_size, median_y_size, median_z_size) <= 3)
    if choose_median_space or choose_median_size:
        print('not anisotropic, choose median')
        final_space = (median_x_space, median_y_space, median_z_space)
    else:
        print('anisotropic, choose 10% quantile')
        target = (median_x_space, median_y_space, median_z_space)
        worst_spacing_axis = np.argmax(target)

        if worst_spacing_axis == 0 or worst_spacing_axis == 1:
            target_spacing_of_that_axis = space_x
            target_spacing_of_that_axis = np.percentile(target_spacing_of_that_axis, 10)
            # don't let the spacing of that axis get lower than the other axes
            if target_spacing_of_that_axis < median_z_space:
                target_spacing_of_that_axis = max(median_z_space, target_spacing_of_that_axis) + 1e-5
            final_space = (target_spacing_of_that_axis, target_spacing_of_that_axis, median_z_space)
        else:
            target_spacing_of_that_axis = space_z
            target_spacing_of_that_axis = np.percentile(target_spacing_of_that_axis, 10)
            # don't let the spacing of that axis get lower than other axes
            if target_spacing_of_that_axis < min(median_x_space, median_y_space):
                target_spacing_of_that_axis = max(min(median_x_space, median_y_space), target_spacing_of_that_axis) + 1e-5
            final_space = (median_x_space, median_y_space, target_spacing_of_that_axis)
        
    return final_space

def resample(data, header, spacing):
    source_spacing = (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])
    target_spacing = spacing
    scale = np.array(source_spacing) / np.array(target_spacing)
    data_resampled = zoom(data, scale, order=3)
    return data_resampled

def resample_mask(npy_mask, header, spacing, num_label, order=0):
    source_spacing = (header['pixdim'][1], header['pixdim'][2], header['pixdim'][3])
    target_spacing = spacing
    scale = np.array(source_spacing) / np.array(target_spacing)
    target_npy_mask = np.zeros_like(npy_mask)
    target_npy_mask = zoom(target_npy_mask, scale, order=order)
    for i in range(0, num_label+1):
        current_mask = npy_mask.copy()

        current_mask[current_mask != i] = 0
        current_mask[current_mask == i] = 1

        current_mask = zoom(current_mask, scale, order=order)
        current_mask = (current_mask > 0.5).astype(np.uint8)
        current_mask = binary_fill_holes(current_mask)
        target_npy_mask[current_mask != 0] = i

    return target_npy_mask

def crop(data, label, background_value):
    # mark non-background region in images
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data > background_value
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)

    # determine coordinates of bounding_box
    mask_voxel_coords = np.where(nonzero_mask != False)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    # crop image and mask according to bounding_box
    if abs(bbox[2][0] - bbox[2][1]>=64):
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    else:
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]))
    data_cropped = data[resizer]
    label_cropped = label[resizer]

    return data_cropped, label_cropped

def window_transform(ct_array, windowWidth, windowCenter):
	"""
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
	minWindow = float(windowCenter) - 0.5*float(windowWidth)
	newimg = (ct_array - minWindow) / float(windowWidth)
	newimg[newimg < 0] = 0
	newimg[newimg > 1] = 1
	return newimg

def read_data(path, subjectid):
    nibimg_oral = nib.load(path[subjectid])
    nibimg = nibimg_oral.get_fdata()
    niblabel_oral = nib.load(path[subjectid].replace('/img/', '/label/'))
    niblabel = niblabel_oral.get_fdata()
    niblabel = niblabel.astype(np.int8)

    return nibimg_oral, nibimg, niblabel_oral, niblabel

def get_attributes(nibimg_oral, niblabel_oral):
    header = nibimg_oral.header
    header_label = niblabel_oral.header

    return header, header_label

def resamplig(nibimg, header, niblabel, header_label):
    final_space = (1, 1, 1.5)
    data_resampled = resample(nibimg, header, final_space)
    label_resampled = resample_mask(niblabel, header_label, final_space, 16)

    return data_resampled, label_resampled

def crop_all(nibimg, niblabel):
    data_cropped, label_cropped = crop(nibimg, niblabel, background_value=0)
    label_cropped = np.round(label_cropped).astype(np.int8)
    data_cropped[data_cropped<-500] = -500
    data_cropped[data_cropped>500] = 500

    return data_cropped, label_cropped

def save(path, subjectid, data_cropped, label_cropped):
    file_name = path[subjectid].split('/')[-1]
    new_path = path[subjectid].replace('WORD-V0.1.0', 'WORD-propressed').replace(file_name, '')
    os.makedirs(new_path, exist_ok=True)
    new_path_label = path[subjectid].replace('WORD-V0.1.0', 'WORD-propressed').replace('/img/', '/label/').replace(file_name, '')
    os.makedirs(new_path_label, exist_ok=True)
    np.save(new_path + file_name.replace('nii.gz', 'npy'), data_cropped.astype(np.float16)) # preprocess to float16 to accelerate data reading
    np.save(new_path_label + file_name.replace('nii.gz', 'npy'), label_cropped.astype(np.int8))

def pipeline(path, subjectid):

    # read data
    nibimg_oral, nibimg, niblabel_oral, niblabel = read_data(path, subjectid)

    # # not suitable for images with large difference for spacing (performance drop and training instability)
    # # get attributes
    # header, header_label = get_attributes(nibimg_oral, niblabel_oral)
    # # resamplig
    # data_resampled, label_resampled = resamplig(nibimg, header, niblabel, header_label)
    # # crop
    # data_cropped, label_cropped = crop_all(data_resampled, label_resampled)
    
    # crop
    data_cropped, label_cropped = crop_all(nibimg, niblabel)
    
    # save
    save(path, subjectid, data_cropped, label_cropped)

if __name__ == '__main__':

    train_path = './WORD-V0.1.0/Client1/train/img'
    train_file_path= glob.glob(train_path + '/*')
    for subjectid in tqdm(range(len(train_file_path))):
        pipeline(train_file_path, subjectid)

    val_path = './WORD-V0.1.0/Client1/val/img'
    val_file_path= glob.glob(val_path + '/*')
    for subjectid in tqdm(range(len(val_file_path))):
        pipeline(val_file_path, subjectid)

    test_path = './WORD-V0.1.0/Client1/test/img'
    test_file_path= glob.glob(test_path + '/*')
    for subjectid in tqdm(range(len(test_file_path))):
        pipeline(test_file_path, subjectid)