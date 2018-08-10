# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:50:22 2017

@author: Joshua Torrance
"""

# Imports
from numpy import zeros, fromfunction, linspace, sqrt
from scipy.signal import fftconvolve
from scipy.constants import e
from matplotlib import pyplot as plt

from ElectronLens import ElectronBunch
from Gaussians import gaussian2d
from Fitting import fitCurve


# Constants
m_per_pixel = 0.04/(490*2)
single_e_spread = 4.795 * m_per_pixel


# Functions
def detectDistribution(bunch, spread_m=single_e_spread,
                       pixel_size_m=m_per_pixel):
    x_pos = bunch.getXs()
    y_pos = bunch.getYs()

    image_length_pix = int(max((x_pos.max() - x_pos.min()),
                           (y_pos.max() - y_pos.min())) / pixel_size_m) + 1

    image = zeros((image_length_pix, image_length_pix))

    for x, y in zip(x_pos, y_pos):
        i = int((x-x_pos.min())/pixel_size_m)
        j = int((y-y_pos.min())/pixel_size_m)

        image[i, j] = 1

    spread_length_pix = int(spread_m * 3 / pixel_size_m)

    def spread_func(i, j):
        return gaussian2d(1, spread_length_pix//2, spread_length_pix//2,
                          spread_length_pix, spread_length_pix, 0)(i, j)
    spread_image = fromfunction(spread_func,
                                (spread_length_pix, spread_length_pix))

    convolved_image = fftconvolve(image, spread_image, mode='same')

    return convolved_image


def calculate_emittance(zs, widths):
    zs -= zs[0]

    sigma_11s = widths**2
    sigma_0_11 = widths[0]**2

    def sigma_11_func(z, sigma_0_11, sigma_0_12, sigma_0_22):
        return sigma_0_11 + 2*z*sigma_0_12 + z**2*sigma_0_22

    def fit_func(z, sigma_0_12, sigma_0_22):
        return sigma_11_func(z, sigma_0_11, sigma_0_12, sigma_0_22)

    guess = 1, 1

    fit = fitCurve(zs, sigma_11s, fit_func, guess)
    fit_sigma_0_12, fit_sigma_0_22 = fit

    plt.plot(zs, sigma_11s, 'x')

    fit_zs = linspace(zs[0], zs[-1], 1000)
    fit_sigma_11s = sigma_11_func(fit_zs,
                                  sigma_0_11, fit_sigma_0_12, fit_sigma_0_22)

    plt.plot(fit_zs, fit_sigma_11s)

    emittance = sqrt(sigma_0_11*fit[-1]-fit[1]**2)

    return emittance

# Sciprt
if __name__ == '__main__':
    bunch_edge_x = 0.005
    bunch_edge_y = 0.005
    initial_emittance = 0.00001
    trons = ElectronBunch(10000, 17.1*e/2, bunch_edge_x, bunch_edge_y,
                          emittance=initial_emittance)

    zs = linspace(0, 2, 20)
    widths = zeros(zs.shape)
    widths[0] = trons.getXs().std()
    for i in range(zs.size-1):
        dz = zs[i+1] - zs[i]

        print(i, 'E =', trons.getEmittance())
        trons.propagate(dz, 10)

        widths[i+1] = trons.getXs().std()

    emittance = calculate_emittance(zs, widths)

    print('Set Emittance:', initial_emittance)
    print('True Emittance:', trons.getEmittance())
    print('Measured emittance:', emittance)

    plt.show()
