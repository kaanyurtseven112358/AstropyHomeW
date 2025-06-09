



#QUESTION A1:

import os
from astropy.io import fits
import tarfile
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



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

#From previous HW : Reduction, Variance Estimation, Fringe Construction
for fname in sorted(os.listdir(extract_dir)):
    if fname.endswith('.fits') and not fname.startswith('._'):
        sci_path = os.path.join(extract_dir, fname)
        try:
            data, header = fits.getdata(sci_path, header=True)
            reduced = (data - masterbias) / masterflat

            # === A1: Variance estimation ===
            # Variance from science image after reduction:
            
            variance = data / (masterflat ** 2)
            var_path = os.path.join(extract_dir, 'variance_' + fname)
            fits.writeto(var_path, variance, header, overwrite=True)

            
            reduced_path = os.path.join(extract_dir, 'reduced_' + fname)
            fits.writeto(reduced_path, reduced, header, overwrite=True)

            
            sky_median = np.median(reduced)
            normalized = reduced / sky_median
            normalized_images.append(normalized)
            sky_medians.append(sky_median)
            reduced_data.append(reduced)
            headers.append(header)
            fnames.append(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")

# Fringe Pattern Creation 
stack = np.array(normalized_images)
fringe_pattern = np.median(stack, axis=0)
fringe_path = os.path.join(extract_dir, 'fringe_pattern.fits')
fits.writeto(fringe_path, fringe_pattern, overwrite=True)

# === Fringe Subtraction + Variance propagation ===
for i in range(len(fnames)):
    rescaled_fringe = fringe_pattern * sky_medians[i]
    cleaned = reduced_data[i] - rescaled_fringe

    output_path = os.path.join(extract_dir, 'fringe_cleaned_' + fnames[i])
    fits.writeto(output_path, cleaned, headers[i], overwrite=True)

    # Variance unchanged by subtracting deterministic fringe pattern

fwhm_arcsec = 1.5
sigma_pixels = fwhm_arcsec / 2.35

for fname in sorted(os.listdir(extract_dir)):
    if fname.startswith('fringe_cleaned_') and fname.endswith('.fits'):
        input_path = os.path.join(extract_dir, fname)
        output_path = os.path.join(extract_dir, 'unsharp_masked_' + fname.replace('fringe_cleaned_', ''))

        try:
            data, header = fits.getdata(input_path, header=True)
            smoothed = gaussian_filter(data, sigma=sigma_pixels)

            with np.errstate(divide='ignore', invalid='ignore'):
                unsharp = np.where(smoothed > 0, data / smoothed, 0)

            fits.writeto(output_path, unsharp, header, overwrite=True)
            print(f"Unsharp masked image saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

#QUESTION A2:

# Parameters
gain = 1.5  # electrons/ADU
read_noise = 4.5  # electrons

bias = fits.getdata('masterbias.fits')
flat = fits.getdata('masterflat.fits')

sci_data, sci_header = fits.getdata('science_images/science4.fits', header=True)

#bias subtraction and flat-fielding
reduced = (sci_data - bias) / flat

#compute variance using full CCD equation
#signal  in ADU, convert to electrons -> divide by flat, multiply by gain
signal_electrons = (sci_data - bias) * gain
variance = signal_electrons / gain**2 + (read_noise / gain)**2
variance = variance / flat**2

fits.writeto('science_images/variance_science4.fits', variance, sci_header, overwrite=True)

plt.imshow(variance, cmap='inferno', origin='lower', vmin=np.percentile(variance, 5), vmax=np.percentile(variance, 95))
plt.colorbar(label='Variance (ADU²)')
plt.title('Variance Image of science4.fits')
plt.savefig('variance_science4.png', dpi=300)
plt.show()


# QUESTION B:
fits_galaxy= os.path.join(base_dir, 'ngc4696.fits')
data, header = fits.getdata(fits_galaxy, header=True)

# Gaussian smoothing with sigma between 10 and 50 pixels
sigma = 15 
smoothed = gaussian_filter(data, sigma=sigma)

#unsharp masking
with np.errstate(divide='ignore', invalid='ignore'):
    unsharp = np.where(smoothed > 0, data - smoothed, 0)

# Save FITS
fits.writeto('unsharp_ngc4696.fits', unsharp, header, overwrite=True)


plt.imshow(unsharp, cmap='gray', origin='lower',
           vmin=np.percentile(unsharp, 5),
           vmax=np.percentile(unsharp, 95))
plt.colorbar(label='Unsharp Mask Value')
plt.title(f'Unsharp Mask of NGC 4696 (σ={sigma})')
plt.savefig('unsharp_ngc4696.png', dpi=300)  # or use .pdf if required
plt.show()