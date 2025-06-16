#HOMEWORK 3:
# Load the FITS file
base_dir = os.getcwd()
fits_file = os.path.join(base_dir, 'arc_lamp_wav.fits')
if not os.path.exists(fits_file):
    raise FileNotFoundError(f"FITS file not found: {fits_file}")
hdul = fits.open(fits_file)
data = hdul[0].data
header = hdul[0].header

# Extract wavelength calibration
crval1 = header['CRVAL1']  # starting wavelength
cdelt1 = header['CDELT1']  # wavelength increment per pixel
crpix1 = header['CRPIX1']  # reference pixel
cdelt2 = header.get('CDELT2', 1.0)  # default to 1.0 if not present

# Build wavelength array
n_pixels = len(data)
wavelength = (np.arange(n_pixels) - (crpix1 - 1)) * cdelt1 + crval1
flux = data
print(f"Number of pixels: {n_pixels}")
print(f"Wavelength range: {wavelength[0]} Å to {wavelength[-1]} Å")
print(cdelt1, crval1, crpix1, cdelt2)

plt.plot(wavelength, flux)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.title('Arc Lamp Spectrum')
plt.grid()
plt.show()




#Question 4:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from lmfit.models import GaussianModel, ConstantModel

data = fits.getdata("ngc3263_wav.fits")
header = fits.getheader("ngc3263_wav.fits")

# Wavelength calibration
wavelength = np.array([header["CRVAL2"] + i * header["CDELT2"] for i in range(data.shape[0])])

# Spatial axis in arcsec (pixel scale: 0.332 arcsec/pixel)
x_center = 258
pixel_scale = 0.332
position = np.linspace(-x_center * pixel_scale, (data.shape[1] - x_center) * pixel_scale, data.shape[1])

def fit_halpha(wave, flux):
    model = GaussianModel(prefix='g_') + ConstantModel(prefix='c_')
    params = model.make_params()
    params['g_center'].set(value=6563, min=6550, max=6575)
    params['g_sigma'].set(value=1, min=0.5, max=5)
    params['g_amplitude'].set(value=np.max(flux)*10)
    params['c_c'].set(value=np.median(flux))

    result = model.fit(flux, params, x=wave)
    return result


line_indices = list(range(55, 356, 20))  # 55, 75, ..., 355
rest_wavelength = 6562.82  # Hα in air

d_arcsec = []
radial_velocities = []

for i in line_indices:
    spectrum = data[:, i]
    
    # Restrict to region around Hα (±20 Å)
    mask = (wavelength > 6540) & (wavelength < 6585)
    wave_cut = wavelength[mask]
    flux_cut = spectrum[mask]

    # Fit the emission line
    result = fit_halpha(wave_cut, flux_cut)
    center = result.params['g_center'].value

    # Calculate radial velocity (in km/s)
    v = 3e5 * (center - rest_wavelength) / rest_wavelength

    # Distance from galaxy center (in arcsec)
    d = (i - x_center) * pixel_scale

    # Only append if distance is greater than zero
    if d > 0:
        d_arcsec.append(d)
        radial_velocities.append(v)

plt.figure(figsize=(8, 5))
plt.plot(d_arcsec, radial_velocities, 'o-', label='Rotation Curve')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance from center (arcsec)')
plt.ylabel('Radial Velocity (km/s)')
plt.title('Rotation Curve of NGC 3263')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def compute_radial_velocity(lambda_obs, lambda_rest=6562.82):
    c = 3e5  # km/s
    return c * (lambda_obs - lambda_rest) / lambda_rest

# Example: Compute radial velocity for a measured centroid wavelength
measured_centroid = 6565.1  # Replace with your measured value
radial_velocity = compute_radial_velocity(measured_centroid)
print(f"Measured centroid: {measured_centroid:.2f} Å")
print(f"Radial velocity: {radial_velocity:.2f} km/s")
