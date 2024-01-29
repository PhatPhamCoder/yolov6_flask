from PIL import Image
import pydicom
import os
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def convert_dicom_to_png(dicom_file_list, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dicom_filename in dicom_file_list:
        dicom_path = os.path.join('static', dicom_filename)

        # Read the DICOM file
        image = read_xray(dicom_path, voi_lut=True, fix_monochrome=True)

        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

        # Save the image as PNG
        png_filename = os.path.splitext(dicom_filename)[0] + '.jpg'
        png_path = os.path.join(output_dir, png_filename)
        image.save(png_path)

    return os.listdir(output_dir)  # Return a list of saved PNG file names
