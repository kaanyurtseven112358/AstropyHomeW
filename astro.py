from astropy.io import fits
import tarfile
import os
import numpy as np

base_dir = os.getcwd()  

masterbias_path = os.path.join(base_dir, 'masterbias.fits')
masterflat_path = os.path.join(base_dir, 'masterflat.fits')
tar_path = os.path.join(base_dir, 'science_long.tar')
extract_dir = os.path.join(base_dir, 'science_images')
os.makedirs(extract_dir, exist_ok=True)

with tarfile.open(tar_path, 'r') as tar:
    tar.extractall(path=extract_dir)

masterbias = fits.getdata(masterbias_path)
masterflat = fits.getdata(masterflat_path)

normalized_images = []
sky_medians = []
reduced_data = []
headers = []
fnames = []



# reduce and normalize
for fname in sorted(os.listdir(extract_dir)):
    if fname.endswith('.fits') and not fname.startswith('._'):
        sci_path = os.path.join(extract_dir, fname)
        try:
            data, header = fits.getdata(sci_path, header=True)
            reduced = (data - masterbias) / masterflat


            reduced_path = os.path.join(extract_dir, 'reduced_' + fname)
            fits.writeto(reduced_path, reduced, header, overwrite=True)

            # fringe construction
            sky_median = np.median(reduced)
            normalized = reduced / sky_median

            normalized_images.append(normalized)
            sky_medians.append(sky_median)
            reduced_data.append(reduced)
            headers.append(header)
            fnames.append(fname)

        except Exception as e:
            print(f"Skipping {fname}: {e}")

#Compute fringe pattern from median of normalized images
stack = np.array(normalized_images)
fringe_pattern = np.median(stack, axis=0)

# save 
fringe_path = os.path.join(extract_dir, 'fringe_pattern.fits')
fits.writeto(fringe_path, fringe_pattern, overwrite=True)

#subtract fringe from reduced images
for i in range(len(fnames)):
    rescaled_fringe = fringe_pattern * sky_medians[i]
    cleaned = reduced_data[i] - rescaled_fringe
    output_path = os.path.join(extract_dir, 'fringe_cleaned_' + fnames[i])
    fits.writeto(output_path, cleaned, headers[i], overwrite=True)


from scipy.ndimage import gaussian_filter


image_dir = '/home/kyurtsev24/Downloads/science_images'

# FWHM in arcseconds
fwhm_arcsec = 1.5  
sigma_pixels = fwhm_arcsec / 2.35  #gaussian sigma kernel in pixels

# process each fringe-cleaned image
for fname in sorted(os.listdir(image_dir)):
    if fname.startswith('fringe_cleaned_') and fname.endswith('.fits'):
        input_path = os.path.join(image_dir, fname)
        output_path = os.path.join(image_dir, 'unsharp_masked_' + fname.replace('fringe_cleaned_', ''))

        try:
            
            data, header = fits.getdata(input_path, header=True)

            # add gaussian filter to smooth the image
            smoothed = gaussian_filter(data, sigma=sigma_pixels)

            # unsharp mask (original / smoothed)
            with np.errstate(divide='ignore', invalid='ignore'):
                unsharp = np.where(smoothed > 0, data / smoothed, 0)


            fits.writeto(output_path, unsharp, header, overwrite=True)
            print(f"Unsharp masked image saved: {output_path}")

        except Exception as e:
            print(f"Failed to process {fname}: {e}")
fits.writeto(output_path, unsharp, header, overwrite=True)


#Pure fringe pattern extraction 

with tarfile.open(tar_path, 'r') as tar:
    tar.extractall(path=extract_dir)

#Load calibration frames
masterbias = fits.getdata(masterbias_path)
masterflat = fits.getdata(masterflat_path)

#Reduce and normalize images
normalized_images = []

for fname in sorted(os.listdir(extract_dir)):
    if fname.endswith('.fits') and not fname.startswith('._'):
        sci_path = os.path.join(extract_dir, fname)
        try:
            data, _ = fits.getdata(sci_path, header=True)
            reduced = (data - masterbias) / masterflat
            sky_median = np.median(reduced)
            normalized = reduced / sky_median
            normalized_images.append(normalized)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

#Median combine to create fringe pattern
stack = np.array(normalized_images)
fringe_pattern = np.median(stack, axis=0)


fringe_path = os.path.join(extract_dir, 'fringe_pattern.fits')
fits.writeto(fringe_path, fringe_pattern, overwrite=True)
print(f"Fringe pattern saved to: {fringe_path}")
