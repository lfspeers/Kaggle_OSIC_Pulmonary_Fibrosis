import pydicom
import pandas as pd
import numpy as np
import os
import glob
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import copy


TRAIN_INPUT_PATH = 'C:/Users/mezzo/PycharmProjects/Kaggle OSIC Pulmonary Fibrosis/Data/train/'
TEST_INPUT_PATH = 'C:/Users/mezzo/PycharmProjects/Kaggle OSIC Pulmonary Fibrosis/Data/test/'

train_patients = os.listdir(TRAIN_INPUT_PATH)
test_patients = os.listdir(TEST_INPUT_PATH)


def load_all_scans(path):
    """Args: path, returns: dict of files by patient
    Gets the filenames for all of the dicom images and puts them in a dict
    corresponding to the patient ID that the scan belongs to.
    """
    patients = os.listdir(path)
    sorted_files = {}
    for patient in patients:
        patient_path = path + patient + '/'
        patient_files = os.listdir(patient_path)
        patient_files = [patient_path + s for s in patient_files]
        sorted_files[patient] = patient_files
    return sorted_files


all_scans = load_all_scans(TRAIN_INPUT_PATH)

test_patient_ID = 'ID00426637202313170790466'
test_patient_path = TRAIN_INPUT_PATH + test_patient_ID + '/'


def get_patient_dicom(patient_id):
    """Args: patient id, returns: the dicom image"""
    dicom_path = TRAIN_INPUT_PATH + patient_id + '/'
    patient_files = os.listdir(dicom_path)
    image = pydicom.read_file(os.path.join(dicom_path, patient_files[0]))
    # pixel_dimensions = (int(image.Rows), int(image.Columns), len(patient_files))
    # pixel_spacing = (float(image.PixelSpacing[0]), float(image.PixelSpacing[1]), float(image.SliceThickness))
    return image,  # pixel_dimensions, pixel_spacing


dicom = get_patient_dicom(test_patient_ID)

"""x = np.arange(0.0, (pixel_dimensions[0] + 1) * pixel_spacing[0], pixel_spacing[0])
y = np.arange(0.0, (pixel_dimensions[1] + 1) * pixel_spacing[1], pixel_spacing[1])
z = np.arange(0.0, (pixel_dimensions[2] + 1) * pixel_spacing[2], pixel_spacing[2])"""


def load_slices(patient_path):
    slices = [pydicom.dcmread(patient_path + s) for s in
              os.listdir(patient_path)]
    slices = [s for s in slices if 'SliceLocation' in s]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except():
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)  # Set outside-of-scan pixels to 0
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


scans = load_slices(test_patient_path)
patient_pixels = get_pixels_hu(scans)
plt.imshow(patient_pixels[0], cmap='bone')
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

    binary_image = np.array(image >= -700, dtype=np.int8) + 1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the patient
    # More resistant to “trays” on which the patient lays cutting the
    # air around the person in half

    background_label = labels[1, 1, 1]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image


"""# get masks
segmented_lungs = segment_lung_mask(patient_pixels, fill_lung_structures=False)
segmented_lungs_fill = segment_lung_mask(patient_pixels, fill_lung_structures=True)
internal_structures = segmented_lungs_fill - segmented_lungs  # isolate lung from chest
copied_pixels = copy.deepcopy(patient_pixels)
for i, mask in enumerate(segmented_lungs_fill):
    get_high_vals = mask == 0
    copied_pixels[i][get_high_vals] = 0
seg_lung_pixels = copied_pixels  # sanity check
plt.imshow(seg_lung_pixels[30], cmap='bone')
plt.show()"""

