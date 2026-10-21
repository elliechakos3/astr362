# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:51:36 2026

@author: EChak
"""

# Observational Assignment 13

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D

fits_filename = r"C:\Users\EChak\Downloads\image_ZTF-sci-20180331471019-zg-size-600.0arcsec.fits"

with fits.open(fits_filename) as hdul:
    image = hdul[0].data.astype(float)
    
print("Image Shape:", image.shape)

plt.figure(figsize=(8,8))
plt.imshow(image, origin="lower", cmap="gray", vmin=np.nanpercentile(image, 5), vmax=np.nanpercentile(image, 99))
plt.colorbar(label="Counts")
plt.xlabel("X (Pixels)")
plt.ylabel("Y (Pixels)")
plt.title("Science Image")
plt.show()

x_star = 300
y_star = 400

sigma_x = 3.0
sigma_y = 3.0

def gauss2d(x, y, x_std, y_std, x_mean, y_mean):
    gaussian = Gaussian2D.evaluate(x, y, amplitude=1.0, x_mean=x_mean, y_mean=y_mean, x_stddev=x_std, y_stddev=y_std, theta=0.0)
    gaussian_sum = np.sum(gaussian)
    if gaussian_sum == 0:
        raise ValueError("Gaussian Sum is Zero")
    return gaussian / gaussian_sum
    
half_size = 20

# Arrays
star_cutout = image[
    y_star - half_size : y_star + half_size,
    x_star - half_size : x_star + half_size]

xvals, yvals = np.meshgrid(
    np.arange(x_star - half_size, x_star + half_size),
    np.arange(y_star - half_size, y_star + half_size))

star_psf = gauss2d(
    xvals, yvals,
    sigma_x, sigma_y,
    x_star, y_star)

print("Cutout shape:", star_cutout.shape)
print("PSF shape:", star_psf.shape)

plt.figure(figsize=(8, 8))
plt.imshow(image, origin="lower", cmap="gray",
           vmin=np.nanpercentile(image, 5),
           vmax=np.nanpercentile(image, 99))
plt.contour(xvals, yvals, star_psf, colors="red")
plt.xlim(x_star - half_size, x_star + half_size)
plt.ylim(y_star - half_size, y_star + half_size)
plt.xlabel("x (pixels)")
plt.ylabel("y (pixels)")
plt.title("Gaussian PSF over the star")
plt.show()

star_weighted_image = star_cutout * star_psf
N_star = np.nansum(star_weighted_image)

print("Gaussian-weighted star counts:", N_star)

x_sky = 250  # replace with the x coordinate of blank sky
y_sky = 400  # replace with the y coordinate of blank sky

sky_cutout = image[
    y_sky - half_size : y_sky + half_size,
    x_sky - half_size : x_sky + half_size]

sky_xvals, sky_yvals = np.meshgrid(
    np.arange(x_sky - half_size, x_sky + half_size),
    np.arange(y_sky - half_size, y_sky + half_size))

sky_psf = gauss2d(
    sky_xvals, sky_yvals,
    sigma_x, sigma_y,
    x_sky, y_sky)

sky_weighted_image = sky_cutout * sky_psf
N_sky_med = np.nanmedian(sky_weighted_image)

print("Median weighted sky value:", N_sky_med)

plt.figure(figsize=(6, 6))
plt.imshow(sky_cutout, origin="lower", cmap="gray")
plt.title("Selected blank-sky cutout")
plt.colorbar(label="Counts")
plt.show()

n_aperture = star_cutout.size

N_net = N_star - n_aperture * N_sky_med
print("Net star counts:", N_net)

exposure_time_s = 30.0

if N_net <= 0:
    raise ValueError("Net counts must be positive to calculate the logarithmic magnitude.")

m_inst = -2.5 * np.log10(N_net / exposure_time_s)

print("Instrumental magnitude:", m_inst)
