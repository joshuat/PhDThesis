##### Imports #####
from numpy import histogram, sqrt, arange, array, linspace, isnan, zeros, flipud, logspace, isnan
from matplotlib import rcParams, pyplot as plt
from peakutils import indexes, interpolate
from scipy.constants import eV, m_e, k as kB, c
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm
from h5py import File

from Bunch1D import Bunch
from Mask1D import maskBunch, maskBunch2, pinholeMask, pepperpotMask, pepperpotMaskFs
from Emittance import emittance_from_line, excess_energy_from_wavelength, expected_emittance, normal_prop
from Fitting import fitCurve

##### Constants #####
beam_energy = 8e3*eV
beam_size_rms = 570.68e-6
beam_size_rms_cherry = 340e-6

##### Functions #####
def find_peaks(x, line, thres=0.001, min_dist=200, smooth=False, smooth_size=5, diag=False):
    if smooth:
        l = gaussian_filter(line, smooth_size)
    else:
        l = line
    
    peaks = indexes(l, thres=thres, min_dist=min_dist)
    
    if diag:
        plt.figure('Peaks')
        plt.title('Peak Finding')
        plt.plot(line, 'b')
        if smooth:
            plt.plot(l, 'r')
        for peak in peaks:
            plt.axvline(peak, color='k', ls=':')
            
    return peaks
    
def simple_lens(bunch, strength=1):
    bunch.electrons[:, 1] += bunch.getXs() * strength
    
def spherical_aberration(bunch, strength=1):
    bunch.electrons[:, 1] += bunch.getXs()**2 * strength
    
def normal_cumulative_thingy(x):
    return 1-(1-norm.cdf(x))*2
    
def normal_prop(x_lo, x_hi):
    return norm.cdf(x_hi) - norm.cdf(x_lo)
    
    
##### Script #####
# Excess energy of steaks
if False:
    long_streak_wavelength = 487.2e-9
    short_streak_wavelength = 475.9e-9
    
    long_streak_excess = excess_energy_from_wavelength(long_streak_wavelength, field_ionisation=False)
    short_streak_excess = excess_energy_from_wavelength(short_streak_wavelength, field_ionisation=False)
    
    print('Long duration streak:')
    print('\tWavelength: {:.2f}nm'.format(long_streak_wavelength*1e9))
    print('\tExcess Energy: {:.2f}meV'.format(long_streak_excess*1e3))
    
    print()
    
    print('Short duration streak:')
    print('\tWavelength: {:.2f}nm'.format(short_streak_wavelength*1e9))
    print('\tExcess Energy: {:.2f}meV'.format(short_streak_excess*1e3))
    
    # For thesis figure from arbitrary shaping paper.
    wavelengths = [481.729, 481.185, 479.920, 479.152, \
                   483.300, 482.200, 481.987, 481.670, \
                   480.570, 479.865, 478.799, 477.658, 476.694, 475.438]
          
    print('\nFigure Energies:')
    for wav in wavelengths:
        e = excess_energy_from_wavelength(wav*1e-9, field_ionisation=True, electric_field=40e3)
                                          
        print('Wavelength: {:.3f}nm, Excess Energy: {:.4f}meV'.format(wav, e*1e3))
        
    exit()

# First Run.
if False:
    num_particles = int(1e6)
    
    expected_temperature = 5
    expected_emittance = beam_size_rms * sqrt(kB*expected_temperature/(m_e*c**2))
    
    # Pepperpots
    number_of_holes = 20
    pepperpot_pitch = 200e-6

    propagation_distance = 0.2

    
    print('Expected Emittance: {:.2f} nm rad'.format(expected_emittance*1e9))
    bunch = Bunch(n=num_particles, energy=beam_energy, rms_width=beam_size_rms, \
                  normalised_rms_emittance=expected_emittance, mass=m_e)
    print('Initial emittance: {:.2f} nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))

    hist, bin_edges = histogram(bunch.getXs(), bins=10000)
    plt.figure('hist')
    plt.plot(bin_edges[:-1], hist)

    print('Pre-Pepperpot emittance: {:.2f} nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
    pinhole = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes)
    bunch = maskBunch(bunch, pinhole)
    print('Post-Pepperpot emittance:', bunch.getNormalisedRMSEmittance()*1e9)

    hist, bin_edges = histogram(bunch.getXs(), bins=bin_edges)
    plt.figure('hist')
    plt.plot(bin_edges[:-1], hist)

    dz = propagation_distance
    bunch.propagate(dz)
    print('Final emittance:', bunch.getNormalisedRMSEmittance()*1e9)

    hist, bin_edges = histogram(bunch.getXs(), bins=bin_edges)
    plt.figure('hist')
    plt.plot(bin_edges[:-1], hist)

    m_per_pixel = bin_edges[1]-bin_edges[0]
    peaks = find_peaks(bin_edges[:-1], hist, diag=False)
    hist = array(hist, dtype='f')
    plt.figure()
    emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                        m_per_pixel=m_per_pixel, number_holes=number_of_holes,
                        pitch=pepperpot_pitch, propagation_distance=propagation_distance)
    emittance *= bunch.getBeta()
                        
    print('Measured Emittance: {:.2f} nm rad'.format(emittance*1e9))

# Wavelength Sweep
if False:
    if False:
        beam_size = beam_size_rms
        filename = 'wavelength_sweep - long2.h5'
    else:
        beam_size = beam_size_rms_cherry
        filename = 'wavelength_sweep - long2 min.h5'
            
    if False:
        # Simulate and save
        wavelengths = linspace(465e-9, 487e-9, 100)
            
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size)
        
        # Pepperpot
        number_of_holes = 15
        pepperpot_pitch = 200e-6
        aperture_size = 50e-6
        pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, pinhole_diameter=aperture_size)
        
        # Bins for histogram 'detector'
        if False:
            bins = 1000
        else:
            bins = linspace(-.0017, 0.0017, 1000)
        
        N = 10000000
        propagation_distance = 0.015
        # For each +'ve excess energy simulate a bunch.
        measured_emittances = zeros(wavelengths.size)
        measured_emittances_beam_size_corrected = zeros(wavelengths.size)
        measured_emittances_aperture_size_corrected = zeros(wavelengths.size)
        for i, expected_e in enumerate(expected_emittances):
            if isnan(expected_e):
                if False:
                    # Skip
                    measured_emittances[i] = float('nan')
                    measured_emittances_aperture_size_corrected[i] = float('nan')
                    continue
                else:
                    # Use zero emittance beam.
                    expected_e = 0
                    expected_emittances[i] = expected_e
            
            print(i)
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
            
            bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size, \
                          normalised_rms_emittance=expected_e, mass=m_e)
                      
            print('\tBunch RMS Size: {:.2f}um'.format(bunch.getWidth()*1e6))
            
            bunch = maskBunch(bunch, pepperpot)
                          
            bunch.propagate(propagation_distance)
            
            hist, bin_edges = histogram(bunch.getXs(), bins=bins)
            m_per_pixel = bin_edges[1]-bin_edges[0]
            hist = array(hist, dtype='f')
            
            peaks = find_peaks(bin_edges[:-1], hist, thres=0.005, min_dist=50, diag=False, smooth=True)
            
            
            plt.figure()
            
            # Don't correct for anything.
            emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=False, adjust_for_aperture_size=False, refine_parameters=True)
            emittance *= bunch.getBeta()
            
            measured_emittances[i] = emittance
            
            # Correct for beam size.
            emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=False,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=True, adjust_for_aperture_size=False, refine_parameters=False)
            emittance *= bunch.getBeta()
            
            measured_emittances_beam_size_corrected[i] = emittance
            
            # Now with aperture size correction.
            emittance, _rms_size, _total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=False,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=True, adjust_for_aperture_size=True, refine_parameters=True)
            emittance *= bunch.getBeta()
            measured_emittances_aperture_size_corrected[i] = emittance
            
            print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))

            if False:
                plt.figure('hist')
                plt.plot(bin_edges[:-1], hist)
                
                plt.show()
                exit()
                
            
        # Save data
        if True:
            with File(filename) as hdf:
                wavelengths_key = 'wavelengths'
                if wavelengths_key in hdf:
                    del hdf[wavelengths_key]
                    
                hdf.create_dataset(wavelengths_key, data=wavelengths)
                
                excess_energy_key = 'excess energy'
                if excess_energy_key in hdf:
                    del hdf[excess_energy_key]
                    
                hdf.create_dataset(excess_energy_key, data=excess_energys)
                
                expected_emittance_key = 'expected emittance'
                if expected_emittance_key in hdf:
                    del hdf[expected_emittance_key]
                    
                hdf.create_dataset(expected_emittance_key, data=expected_emittances)
                
                measured_emittance_key = 'measured emittance'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances)
                
                measured_emittance_key = 'measured emittance beam size correction'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances_beam_size_corrected)
                
                measured_emittance_key = 'measured emittance aperture size correction'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances_aperture_size_corrected)
                
                hdf.attrs['number of holes'] = number_of_holes
                hdf.attrs['aperture size'] = aperture_size
                hdf.attrs['pitch'] = pepperpot_pitch
    else:
        with File(filename) as hdf:
            wavelengths = array(hdf['wavelengths'])
            excess_energys = array(hdf['excess energy'])
            expected_emittances = array(hdf['expected emittance'])
            measured_emittances = array(hdf['measured emittance'])
            measured_emittances_beam_size_corrected = array(hdf['measured emittance beam size correction'])
            measured_emittances_aperture_size_corrected = array(hdf['measured emittance aperture size correction'])
            
            number_of_holes = hdf.attrs['number of holes']
            aperture_size = hdf.attrs['aperture size']
            pepperpot_pitch = hdf.attrs['pitch']
            
        
    # Plotting
    plt.figure('Wavelength vs. Excess Energy')
    plt.title('Wavelength vs. Excess Energy')
    plt.plot(wavelengths*1e9, excess_energys*1e3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Excess Energy (meV)')
    plt.axhline(0, color='k', ls=':')
    plt.xlim((wavelengths.min()*1e9, wavelengths.max()*1e9))
    
    
    plt.figure('Excess Emittance vs. Emittance')
    plt.title('Excess Emittance vs. Emittance')
    plt.plot(excess_energys*1e3, expected_emittances*1e9, 'r')
    plt.plot(excess_energys*1e3, measured_emittances*1e9, 'bx')
    plt.plot(excess_energys*1e3, measured_emittances_beam_size_corrected*1e9, 'gx')
    plt.plot(excess_energys*1e3, measured_emittances_aperture_size_corrected*1e9, 'kx')
    plt.xlabel('Excess Energy (meV)')
    plt.ylabel('Emittance (nm rad)')
    plt.xlim((0, excess_energys.max()*1e3))
    
    if True:
        # Plot for thesis.
        rcParams.update({'font.size': 10})
        rcParams.update({'pgf.rcfonts': False})
        rcParams.update({'pgf.texsystem': 'pdflatex'})
        
        colours = [(79/255,122/255,174/255),(255/255,102/255,51/255),(245/255,174/255,32/255),(77/255,155/255,77/255),(102/255,102/255,102/255)]

        # Font should match document
        rcParams['font.family'] = 'serif'

        rcParams['axes.unicode_minus'] = False
            # Minus sign from matplot lib is too long for my taste.
            
            
        linewidth = 5.71 # inches

        figwidth = linewidth
        figheight = figwidth/3
        figwidth *= 0.75
        figsize = (figwidth, figheight)
        
        # Higher res theory
        wavelengths_fine = linspace(465e-9, 487e-9, 10000)
            
        excess_energys_fine = excess_energy_from_wavelength(wavelengths_fine, field_ionisation=False)
        
        expected_emittances_fine = expected_emittance(excess_energys_fine, beam_size)
        for i in range(expected_emittances_fine.size):
            if isnan(expected_emittances_fine[i]):
                expected_emittances_fine[i] = 0
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(excess_energys*1e3, measured_emittances*1e9, '-', markersize=5, color=colours[0])
        plt.plot(excess_energys*1e3, measured_emittances_aperture_size_corrected*1e9, '-', markersize=5, color=colours[3])
        plt.plot(excess_energys_fine*1e3, expected_emittances_fine*1e9, ':', color=colours[1])
        plt.xlim((-5, excess_energys.max()*1e3))
        plt.xlabel('Excess Energy (meV)')
        plt.ylabel('Emittance (nm rad)')
        
        plt.tight_layout()
        
        plt.savefig('wavelength_sweep_sim.pgf')
  
# Attempting to show resolution limit.
if False:
    file_name = 'resolution_limit_demo.h5'
    
    N = 1000000
    d_source_to_lens = .25+.450
    d_lens_to_sample = .100
    d_sample_to_detector = .45+.430

    lens_strength = -1.0e9
        
    pepperpot_pitch = 200e-6
    number_of_holes = 21
    
    if False:
        # Simulate and Save.
        wavelengths = linspace(465e-9, 479e-9, 100)
        
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
            
            
        aperture_sizes = array([10e-6, 50e-6, 100e-6])
        
        measured_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_corrected_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_blurred_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_beam_size = zeros((expected_emittances.size, aperture_sizes.size))
        for j, expected_emittance in enumerate(expected_emittances):
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_emittance))
            base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                          normalised_rms_emittance=expected_emittance, mass=m_e)
            print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            
            base_bunch.propagate(d_source_to_lens)
            simple_lens(base_bunch, strength=lens_strength)
            print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            base_bunch.propagate(d_lens_to_sample)    
            print('Beam size at sample: {:.2f}mm'.format(base_bunch.getWidth()*1e3))
            rms_beam_size_sample = base_bunch.getWidth()
            
                
            print()
            
            for k, aperture_size in enumerate(aperture_sizes):
                
                print('Pepperpot aperture size: {:.2f}um'.format(aperture_size*1e6))
                # Pepperpot
                pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                          location=0, pinhole_diameter=aperture_size)
                
                bunch = base_bunch.copy()
                
                bunch = maskBunch(bunch, pepperpot)
                print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
                print('\tAfter sample bunch count: {:d}'.format(bunch.getSize()))
                
                bunch.propagate(d_sample_to_detector)
                
                hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                m_per_pixel = bin_edges[1]-bin_edges[0]
                
                hist = array(hist, dtype='f')
                
                #plt.figure()
                peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=False, smooth=True)
                
                
                #plt.figure()
                emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True,
                                diag=False, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                                adjust_for_size=True, adjust_for_aperture_size=False, refine_parameters=True)
                emittance *= bunch.getBeta()
                
                corrected_emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, \
                                diag=False, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                                aperture_size=aperture_size, adjust_for_size=True, \
                                adjust_for_aperture_size=True, refine_parameters=True)
                corrected_emittance *= bunch.getBeta()
                
                if True:
                    # Blur the detected data to replicate the mcp psf
                    psf_width_pixels = 35e-6 / m_per_pixel
                    hist = gaussian_filter(hist, psf_width_pixels)
                    
                    blurred_emittance, rms_size, total_count = emittance_from_line(hist, peaks, \
                                sum_peaks=True, diag=False, m_per_pixel=m_per_pixel, \
                                number_holes=len(peaks), pitch=pepperpot_pitch, \
                                propagation_distance=d_sample_to_detector,
                                aperture_size=aperture_size, adjust_for_size=True, \
                                adjust_for_aperture_size=True, refine_parameters=True)
                    blurred_emittance *= bunch.getBeta()
                    
                    measured_blurred_emittances[j, k] = blurred_emittance
                    
                measured_emittances[j, k] = emittance
                measured_corrected_emittances[j, k] = corrected_emittance
                measured_beam_size[j, k] = rms_size
                
                print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
                print('\tMeasured Emittance: {:.2f}nm rad'.format(corrected_emittance*1e9))
                print('\tMeasured Blurred Emittance: {:.2f}nm rad'.format(blurred_emittance*1e9))
              
            
        # Save data
        with File(file_name) as hdf:
            aperture_size_key = 'aperture size'
            if aperture_size_key in hdf:
                del hdf[aperture_size_key]
                
            hdf.create_dataset(aperture_size_key, data=aperture_sizes)
            
            expected_emittance_key = 'expected emittance'
            if expected_emittance_key in hdf:
                del hdf[expected_emittance_key]
                
            hdf.create_dataset(expected_emittance_key, data=expected_emittances)
                        
            excess_energy_key = 'excess energy'
            if excess_energy_key in hdf:
                del hdf[excess_energy_key]
                
            hdf.create_dataset(excess_energy_key, data=excess_energys)
            
            measured_emittance_key = 'measured emittance'
            if measured_emittance_key in hdf:
                del hdf[measured_emittance_key]
                
            hdf.create_dataset(measured_emittance_key, data=measured_emittances)
            
            
            measured_corrected_emittance_key = 'measured corrected emittance'
            if measured_corrected_emittance_key in hdf:
                del hdf[measured_corrected_emittance_key]
                
            hdf.create_dataset(measured_corrected_emittance_key, data=measured_corrected_emittances)
            
            measured__blurred_corrected_emittance_key = 'measured blurred corrected emittance'
            if measured__blurred_corrected_emittance_key in hdf:
                del hdf[measured__blurred_corrected_emittance_key]
                
            hdf.create_dataset(measured__blurred_corrected_emittance_key, data=measured_blurred_emittances)
            
            measured_beam_size_key = 'measured beam size'
            if measured_beam_size_key in hdf:
                del hdf[measured_beam_size_key]
                
            hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
            
            hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
    else:
        # Load data
        with File(file_name) as hdf:
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            excess_energys = array(hdf['excess energy'])
            expected_emittances = array(hdf['expected emittance'])
            aperture_sizes = array(hdf['aperture size'])
            measured_emittances = array(hdf['measured emittance'])
            measured_corrected_emittances = array(hdf['measured corrected emittance'])
            measured_blurred_emittances = array(hdf['measured blurred corrected emittance'])
    
    if True:
        # Plot Stuff
        for i in range(3):
            plt.figure()
        
            plt.title('Aperture Size: {:.2f}um'.format(aperture_sizes[i]*1e6))
            plt.plot(excess_energys*1e3, expected_emittances*1e9, 'r:')
            plt.plot(excess_energys*1e3, measured_emittances[:, i]*1e9)
            plt.plot(excess_energys*1e3, measured_corrected_emittances[:, i]*1e9)
            plt.plot(excess_energys*1e3, measured_blurred_emittances[:, i]*1e9)
            plt.xlabel('Excess Energy (meV)')
            plt.ylabel('Measured Emittance (nm rad)')
        
        
        
        
        if False:
            # Plot for thesis.
            rcParams.update({'font.size': 10})
            rcParams.update({'pgf.rcfonts': False})
            rcParams.update({'pgf.texsystem': 'pdflatex'})

            # Font should match document
            rcParams['font.family'] = 'serif'

            rcParams['axes.unicode_minus'] = False
                # Minus sign from matplot lib is too long for my taste.
                
                
            linewidth = 5.71 # inches

            figwidth = 0.95*linewidth
            figheight = figwidth/3*1.44
            figsize = (figwidth, figheight)
            
            plt.figure(figsize=figsize)
            plt.plot(aperture_sizes*1e6, measured_emittances*1e9)
            plt.plot(aperture_sizes*1e6, measured_corrected_emittances*1e9)
            plt.xlabel('Aperture Size ($\mu$m)')
            plt.ylabel('Measured Emittance (nm rad)')
            plt.axhline(expected_e*1e9, ls=':', color='k')
            
            plt.tight_layout()
            
            plt.savefig('resolution_limit_sim.pgf')

# Attempting to show resolution limit - taking larger MCP psf into account.
if False:
    file_name = 'resolution_limit_psf_demo.h5'
    
    N = 1000000
    d_source_to_lens = .25+.450
    d_lens_to_sample = .100
    d_sample_to_detector = .45+.430

    lens_strength = -1.0e9
        
    pepperpot_pitch = 200e-6
    number_of_holes = 21
    
    if False:
        # Simulate and Save.
        #wavelengths = linspace(465e-9, 479e-9, 100)
        wavelengths = linspace(475e-9, 479e-9, 50)
        
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
            
            
        aperture_sizes = array([10e-6, 50e-6, 100e-6])
        aperture_sizes = array([50e-6])
        
        measured_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_corrected_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_blurred_emittances = zeros((expected_emittances.size, aperture_sizes.size))
        measured_beam_size = zeros((expected_emittances.size, aperture_sizes.size))
        for j, expected_emittance in enumerate(expected_emittances):
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_emittance))
            base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                          normalised_rms_emittance=expected_emittance, mass=m_e)
            print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            
            base_bunch.propagate(d_source_to_lens)
            simple_lens(base_bunch, strength=lens_strength)
            print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            base_bunch.propagate(d_lens_to_sample)    
            print('Beam size at sample: {:.2f}mm'.format(base_bunch.getWidth()*1e3))
            rms_beam_size_sample = base_bunch.getWidth()
            
                
            print()
            
            for k, aperture_size in enumerate(aperture_sizes):
                
                print('Pepperpot aperture size: {:.2f}um'.format(aperture_size*1e6))
                # Pepperpot
                pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                          location=0, pinhole_diameter=aperture_size)
                
                bunch = base_bunch.copy()
                
                bunch = maskBunch(bunch, pepperpot)
                print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
                print('\tAfter sample bunch count: {:d}'.format(bunch.getSize()))
                
                bunch.propagate(d_sample_to_detector)
                
                hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                m_per_pixel = bin_edges[1]-bin_edges[0]
                
                hist = array(hist, dtype='f')
                
                #plt.figure()
                peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=False, smooth=True)
                
                
                #plt.figure()
                emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True,
                                diag=False, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                                adjust_for_size=True, adjust_for_aperture_size=False, refine_parameters=True)
                emittance *= bunch.getBeta()
                
                corrected_emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, \
                                diag=False, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                                aperture_size=aperture_size, adjust_for_size=True, \
                                adjust_for_aperture_size=True, refine_parameters=True)
                corrected_emittance *= bunch.getBeta()
                
                if True:
                    # Blur the detected data to replicate the mcp psf
                    psf_width_pixels = 4*35e-6 / m_per_pixel
                    hist = gaussian_filter(hist, psf_width_pixels)
                    
                    blurred_emittance, rms_size, total_count = emittance_from_line(hist, peaks, \
                                sum_peaks=True, diag=False, m_per_pixel=m_per_pixel, \
                                number_holes=len(peaks), pitch=pepperpot_pitch, \
                                propagation_distance=d_sample_to_detector,
                                aperture_size=aperture_size, adjust_for_size=True, \
                                adjust_for_aperture_size=True, refine_parameters=True)
                    blurred_emittance *= bunch.getBeta()
                    
                    measured_blurred_emittances[j, k] = blurred_emittance
                    
                measured_emittances[j, k] = emittance
                measured_corrected_emittances[j, k] = corrected_emittance
                measured_beam_size[j, k] = rms_size
                
                print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
                print('\tMeasured Emittance: {:.2f}nm rad'.format(corrected_emittance*1e9))
                print('\tMeasured Blurred Emittance: {:.2f}nm rad'.format(blurred_emittance*1e9))
              
            
        # Save data
        with File(file_name) as hdf:
            aperture_size_key = 'aperture size'
            if aperture_size_key in hdf:
                del hdf[aperture_size_key]
                
            hdf.create_dataset(aperture_size_key, data=aperture_sizes)
            
            expected_emittance_key = 'expected emittance'
            if expected_emittance_key in hdf:
                del hdf[expected_emittance_key]
                
            hdf.create_dataset(expected_emittance_key, data=expected_emittances)
                        
            excess_energy_key = 'excess energy'
            if excess_energy_key in hdf:
                del hdf[excess_energy_key]
                
            hdf.create_dataset(excess_energy_key, data=excess_energys)
            
            measured_emittance_key = 'measured emittance'
            if measured_emittance_key in hdf:
                del hdf[measured_emittance_key]
                
            hdf.create_dataset(measured_emittance_key, data=measured_emittances)
            
            
            measured_corrected_emittance_key = 'measured corrected emittance'
            if measured_corrected_emittance_key in hdf:
                del hdf[measured_corrected_emittance_key]
                
            hdf.create_dataset(measured_corrected_emittance_key, data=measured_corrected_emittances)
            
            measured__blurred_corrected_emittance_key = 'measured blurred corrected emittance'
            if measured__blurred_corrected_emittance_key in hdf:
                del hdf[measured__blurred_corrected_emittance_key]
                
            hdf.create_dataset(measured__blurred_corrected_emittance_key, data=measured_blurred_emittances)
            
            measured_beam_size_key = 'measured beam size'
            if measured_beam_size_key in hdf:
                del hdf[measured_beam_size_key]
                
            hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
            
            hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
    else:
        # Load data
        with File(file_name) as hdf:
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            excess_energys = array(hdf['excess energy'])
            expected_emittances = array(hdf['expected emittance'])
            aperture_sizes = array(hdf['aperture size'])
            measured_emittances = array(hdf['measured emittance'])
            measured_corrected_emittances = array(hdf['measured corrected emittance'])
            measured_blurred_emittances = array(hdf['measured blurred corrected emittance'])
    
    if True:
        # Plot Stuff
        for i in range(aperture_sizes.size):
            plt.figure()
        
            plt.title('Aperture Size: {:.2f}um'.format(aperture_sizes[i]*1e6))
            plt.plot(excess_energys*1e3, expected_emittances*1e9, 'r:')
            plt.plot(excess_energys*1e3, measured_emittances[:, i]*1e9)
            plt.plot(excess_energys*1e3, measured_corrected_emittances[:, i]*1e9)
            plt.plot(excess_energys*1e3, measured_blurred_emittances[:, i]*1e9)
            plt.xlabel('Excess Energy (meV)')
            plt.ylabel('Measured Emittance (nm rad)')
        
        
        
        
        if False:
            # Plot for thesis.
            rcParams.update({'font.size': 10})
            rcParams.update({'pgf.rcfonts': False})
            rcParams.update({'pgf.texsystem': 'pdflatex'})

            # Font should match document
            rcParams['font.family'] = 'serif'

            rcParams['axes.unicode_minus'] = False
                # Minus sign from matplot lib is too long for my taste.
                
                
            linewidth = 5.71 # inches

            figwidth = 0.95*linewidth
            figheight = figwidth/3*1.44
            figsize = (figwidth, figheight)
            
            plt.figure(figsize=figsize)
            plt.plot(aperture_sizes*1e6, measured_emittances*1e9)
            plt.plot(aperture_sizes*1e6, measured_corrected_emittances*1e9)
            plt.xlabel('Aperture Size ($\mu$m)')
            plt.ylabel('Measured Emittance (nm rad)')
            plt.axhline(expected_e*1e9, ls=':', color='k')
            
            plt.tight_layout()
            
            plt.savefig('resolution_limit_psf_sim.pgf')
  
            
# Realistic Parameters - Looking at beam coverage by pepperpots and centring.
if False:
    N = 1000000
    d_source_to_lens = .25+.450
    d_lens_to_sample = .100
    d_sample_to_detector = .45+.430

    lens_strength = -1.0e9
        
    pepperpot_pitch = 300e-6
    
    if False:
        # Simulate and Save.
        wavelengths = array([475e-9])
        
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        expected_e = expected_emittances[0]
        
        # To save time, propagate a bunch to the pepperpots and reuse.
        print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
        base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                      normalised_rms_emittance=expected_e, mass=m_e)
        print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        
        base_bunch.propagate(d_source_to_lens)
        simple_lens(base_bunch, strength=lens_strength)
        print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        base_bunch.propagate(d_lens_to_sample)    
        print('Beam size at sample: {:.2f}mm'.format(base_bunch.getWidth()*1e3))
        rms_beam_size_sample = base_bunch.getWidth()

        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
            
            
        centres = linspace(0, 4*rms_beam_size_sample, 20)
        num_holess = array([25])#flipud(arange(2, 40, 1))
        measured_emittances = zeros((centres.size, num_holess.size))
        measured_beam_size = zeros((centres.size, num_holess.size))
        for k, pepperpot_center in enumerate(centres):
            for j, num_holes in enumerate(num_holess):
                # Pepperpot
                number_of_holes = num_holes
                pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                          location=pepperpot_center)
            
                print('Pepperpot Centre: {:.2f}um'.format(pepperpot_center*1e6))
                print('Number of Holes:', num_holes)
                
                bunch = base_bunch.copy()
                
                bunch = maskBunch(bunch, pepperpot)
                print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
                
                bunch.propagate(d_sample_to_detector)
                
                hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                m_per_pixel = bin_edges[1]-bin_edges[0]
                
                peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=False, smooth=True)
                
                hist = array(hist, dtype='f')
                plt.figure()
                emittance, rms_size, total_count = emittance_from_line(hist, peaks, \
                                sum_peaks=True, diag=False,
                                m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector, \
                                adjust_for_aperture_size=True, refine_parameters=True, adjust_for_size=False)
                emittance *= bunch.getBeta()
                
                measured_emittances[k, j] = emittance
                measured_beam_size[k, j]
                
                print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
                
            
        # Save data
        with File('pepperpot_extent.h5') as hdf:
            number_of_holes_key = 'number of holes'
            if number_of_holes_key in hdf:
                del hdf[number_of_holes_key]
                
            hdf.create_dataset(number_of_holes_key, data=num_holess)
            
            centres_key = 'pepperpot centre'
            if centres_key in hdf:
                del hdf[centres_key]
                
            hdf.create_dataset(centres_key, data=centres)
            
            measured_emittance_key = 'measured emittance'
            if measured_emittance_key in hdf:
                del hdf[measured_emittance_key]
                
            hdf.create_dataset(measured_emittance_key, data=measured_emittances)
            
            measured_beam_size_key = 'measured beam size'
            if measured_beam_size_key in hdf:
                del hdf[measured_beam_size_key]
                
            hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
            
            hdf.attrs['expected emittance'] = expected_e
            hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
    else:
        # Load data
        with File('pepperpot_extent.h5') as hdf:
            expected_e = hdf.attrs['expected emittance']
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            num_holess = array(hdf['number of holes'])
            measured_emittances = array(hdf['measured emittance'])
            centres = array(hdf['pepperpot centre'])
    
    if False:    
        # Plotting
        xs = num_holess*pepperpot_pitch/rms_beam_size_sample
        exs = linspace(xs.min(), xs.max(), 1000)
        wys = normal_cumulative_thingy(exs/2.66)
        correction = normal_cumulative_thingy(xs/2.66)
        
        fit_func = lambda x, k: normal_cumulative_thingy(x/k)
        guess = [3]
        fit = fitCurve(xs, measured_emittances[0, :]/expected_e, fit_func, guess)
        print(fit)
        
        # Different number of holes
        plt.figure('Number of holes vs. Measured Emittance')
        plt.plot(num_holess, measured_emittances[0, :]*1e9, label='Measured')
        plt.plot(num_holess, measured_emittances[0, :]*1e9/correction, label='Corrected')
        plt.axhline(expected_e*1e9, ls=':', color='r')
        plt.xlabel('Number of Pepperpot Holes')
        plt.ylabel('Emittance (nm rad)')
        plt.legend(loc='lower right')
        
        
        plt.figure('Normalised Number of Holes vs. Measured E')
        plt.plot(xs, measured_emittances[0, :]/expected_e, label='Measured')
        plt.plot(exs, wys, label='Correction')
        plt.axhline(1, color='k', ls=':')
        plt.xlabel('Pepperpot Extent/Beam Size')
        plt.ylabel('Proportion of Expected Emittance')
        plt.legend(loc='lower right')
        
        # Off Centre 'pots
        corrections = zeros(measured_emittances.shape)
        for i, c in enumerate(centres):
            for j, n in enumerate(num_holess):
                x_lo = (c - n*pepperpot_pitch/2.66) / rms_beam_size_sample
                x_hi = (c + n*pepperpot_pitch/2.66) / rms_beam_size_sample
                
                corrections[i, j] = 1/normal_prop(x_lo, x_hi)

        plt.figure('Off Centre Pepperpot - Attempted Correction')
        plt.title('Off Centre Pepperpot - Attempted Correction')
        corrected_emittance = corrections * measured_emittances
        plt.plot(centres*1e3, 1e9*measured_emittances[:, 0], 'b:')
        plt.plot(centres*1e3, 1e9*corrected_emittance[:, 0], 'b', label='{:d} holes'.format(num_holess[0]))
        plt.plot(centres*1e3, 1e9*measured_emittances[:, 10], 'g:')
        plt.plot(centres*1e3, 1e9*corrected_emittance[:, 10], 'g', label='{:d} holes'.format(num_holess[10]))
        #plt.plot(centres*1e3, 1e9*measured_emittances[:, 20], 'r:')
        #plt.plot(centres*1e3, 1e9*corrected_emittance[:, 20], 'r', label='{:d} holes'.format(num_holess[20]))
        plt.plot(centres*1e3, 1e9*measured_emittances[:, -6], 'c:')
        plt.plot(centres*1e3, 1e9*corrected_emittance[:, -6], 'c', label='{:d} holes'.format(num_holess[-6]))
        plt.axhline(expected_e*1e9, ls='--', color='r')
        plt.xlabel('Centre (mm)')
        plt.ylabel('Emittance (nm rad)')
        plt.legend(loc='upper left')
        
        plt.figure('Normalised Off Centre Pepperpot')
        plt.title('Normalised Off Centre Pepperpot')
        cs = linspace(0, 2, 100)
        n_corrs = zeros((cs.size, num_holess.size))
        for i, c in enumerate(cs):
            for j, n in enumerate(num_holess):
                en = (n*pepperpot_pitch/2)/rms_beam_size_sample

                x_lo = (c - en)
                x_hi = (c + en)
                
                n_corrs[i, j] = normal_prop(x_lo, x_hi)
        
        corrected_emittance = corrections * measured_emittances
        plt.plot(centres/rms_beam_size_sample, measured_emittances[:, 0]/expected_e, 'b:')
        plt.plot(cs, n_corrs[:, 0], 'b', label='{:d} holes'.format(num_holess[0]))
        plt.plot(centres/rms_beam_size_sample, measured_emittances[:, 10]/expected_e, 'g:')
        plt.plot(cs, n_corrs[:, 10], 'g', label='{:d} holes'.format(num_holess[10]))
        #plt.plot(centres/rms_beam_size_sample, measured_emittances[:, 20]/expected_e, 'r:')
        #plt.plot(cs, n_corrs[:, 20], 'r', label='{:d} holes'.format(num_holess[20]))
        plt.plot(centres/rms_beam_size_sample, measured_emittances[:, -6]/expected_e, 'c:')
        plt.plot(cs, n_corrs[:, -6], 'c', label='{:d} holes'.format(num_holess[-6]))
        
        plt.axhline(1, ls='--', color='r')
        plt.xlabel('Centre (mm)')
        plt.ylabel('Emittance (nm rad)')
        plt.legend(loc='upper left')
    elif True:
        # Plotting for single num_holes, many centres.
        correction_f = lambda centre, magic_number: measured_emittances[:, 0]/normal_prop( \
                (centre - num_holess[0]*pepperpot_pitch/2)*magic_number / rms_beam_size_sample, \
                (centre + num_holess[0]*pepperpot_pitch/2)*magic_number / rms_beam_size_sample)
        
        guess = 3/4
        
        fit = fitCurve(centres, zeros(centres.size)+expected_e, correction_f, guess)
        
        print(fit)
        fitted_magic = 0.75
        
        corrections = zeros(measured_emittances.shape)
        for i, c in enumerate(centres):
            for j, n in enumerate(num_holess):
                x_lo = (c - n*pepperpot_pitch/2)*(3/4) / rms_beam_size_sample
                x_hi = (c + n*pepperpot_pitch/2)*(3/4) / rms_beam_size_sample
                
                corrections[i, j] = 1/normal_prop(x_lo, x_hi)
                
        print(1e9*measured_emittances[:, 0]*corrections[:, 0])
        print(1e9*correction_f(centres, fitted_magic))
                
        plt.figure()
        plt.plot(centres*1e3, 1e9*measured_emittances[:, 0], ':', label='Raw')
        plt.plot(centres*1e3, 1e9*measured_emittances[:, 0]*corrections[:, 0], label='Corrected')
        plt.plot(centres*1e3, 1e9*correction_f(centres, fitted_magic), 'x', label='Fitted')
        plt.axhline(expected_e*1e9, ls='--', color='r')
        plt.legend(loc='upper left')
        plt.xlabel('Pepperpot Centre Offset (mm)')
        plt.ylabel('Emittance (nm rad)')
        
# Looking at coverage at a focus.
if False:
    filename = 'coverage focus.h5'
    
    if False:
        wavelengths = array([475e-9])#linspace(465e-9, 487e-9, 100)
        
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        # Pepperpot
        number_of_holess = arange(2, 15, 1)
        centres = arange(0, 3, 0.5)*beam_size_rms
        number_of_holes = 10
        pepperpot_pitch = 400e-6
        aperture_size = 50e-6
        
        bins = arange(-2.5e-3, 2.5e-3, 5e-6)
        
        N = 1000000
        propagation_distance = 0.02
        expected_e = expected_emittances[0]
        
        base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                          normalised_rms_emittance=expected_e, mass=m_e)
                          
        # For each +'ve excess energy simulate a bunch.
        measured_emittances = zeros((centres.size, number_of_holess.size))
        measured_emittances_beam_size_corrected = zeros((centres.size, number_of_holess.size))
        for j, centre in enumerate(centres):
            for i, number_of_holes in enumerate(number_of_holess):
                if isnan(expected_e):
                    measured_emittances[i] = float('nan')
                    measured_emittances_beam_size_corrected[i] = float('nan')
                    continue
                
                print()
                print('\tEmittance: {:.2f}nm rad'.format(1e9*expected_e))
                print('\tCentre: {:.2f}um'.format(1e6*centre))
                print('\tNumber of holes:', number_of_holes)
                
                #pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, pinhole_diameter=aperture_size)
                
                pepperpotFs = pepperpotMaskFs(pitch=pepperpot_pitch, number_holes=number_of_holes, pinhole_diameter=aperture_size, location=centre)
                
                bunch = base_bunch.copy()
                
                print('\tBunch RMS Size: {:.2f}um'.format(bunch.getWidth()*1e6))
                
                bunch = maskBunch2(bunch, pepperpotFs)
                              
                bunch.propagate(propagation_distance)
                
                hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                m_per_pixel = bin_edges[1]-bin_edges[0]
                hist = array(hist, dtype='f')
                
                peaks = find_peaks(bin_edges[:-1], hist, min_dist=60, diag=False)
                
                
                plt.figure()
                # Don't correct for beam size.
                emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                                m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                                adjust_for_size=False, adjust_for_aperture_size=True, refine_parameters=True)
                emittance *= bunch.getBeta()
                
                measured_emittances[j, i] = emittance
                
                # Correct for beam size.
                emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=False,
                                m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                                adjust_for_size=True, adjust_for_aperture_size=True, refine_parameters=True)
                emittance *= bunch.getBeta()
                
                measured_emittances_beam_size_corrected[j, i] = emittance
                
                print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))

                if False:
                    plt.figure('hist')
                    plt.plot(bin_edges[:-1], hist)
                    
                    plt.show()
                    exit()
                    
            
        # Save data
        if True:
            with File(filename) as hdf:
                number_of_holes_key = 'number of holes'
                if number_of_holes_key in hdf:
                    del hdf[number_of_holes_key]
                    
                hdf.create_dataset(number_of_holes_key, data=number_of_holess)
                
                centre_key = 'centres'
                if centre_key in hdf:
                    del hdf[centre_key]
                    
                hdf.create_dataset(centre_key, data=centres)
                
                
                measured_emittance_key = 'measured emittance'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances)
                
                measured_emittance_key = 'measured emittance beam size correction'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances_beam_size_corrected)
                
                hdf.attrs['wavelength'] = wavelengths[0]
                hdf.attrs['excess energy'] = excess_energys[0]
                hdf.attrs['expected emittance'] = expected_e
                hdf.attrs['aperture size'] = aperture_size
                hdf.attrs['pitch'] = pepperpot_pitch
    else:
        with File(filename) as hdf:
            number_of_holess = array(hdf['number of holes'])
            centres = array(hdf['centres'])
            measured_emittances = array(hdf['measured emittance'])
            measured_emittances_beam_size_corrected = array(hdf['measured emittance beam size correction'])
            
            expected_e = hdf.attrs['expected emittance']
            aperture_size = hdf.attrs['aperture size']
            pepperpot_pitch = hdf.attrs['pitch']
            
        
    # Plotting
    for i, c in enumerate(centres):
        if False:
            plt.figure('Number of Holes vs. Emittance ' + str(c))
            plt.title('Number of Holes vs. Emittance ' + str(c))
            plt.plot(number_of_holess, measured_emittances[i]*1e9)
            #plt.plot(number_of_holess, measured_emittances_beam_size_corrected[i]*1e9)
            plt.axhline(expected_e*1e9, color='r', ls=':')
            plt.xlabel('Number of Holes')
            plt.ylabel('Emittance (nm rad)')
        
        plt.figure('Normalised ' + str(c))
        plt.title('Normalised ' + str(c))
        norm_xs = number_of_holess*pepperpot_pitch/beam_size_rms
        norm_ys = measured_emittances[i]/expected_e
        plt.plot(norm_xs, norm_ys)
        
        xs = arange(0.1, 15, 0.1)*pepperpot_pitch / beam_size_rms
        normalised_centre = c / beam_size_rms
        fit_f = lambda x, magic: normal_prop(-magic*(x-normalised_centre)/2, \
                                                      magic*(x-normalised_centre)/2)
        guess = [0.75]
        fit = fitCurve(norm_xs, norm_ys, fit_f, guess)
        
        
        MAGIC = fit[0]
        ys = array([normal_prop(-MAGIC*(x-normalised_centre)/2, \
                                 MAGIC*(x-normalised_centre)/2) for x in xs])         
        plt.plot(xs, ys)
    
    if False:
        # Plot for thesis.
        rcParams.update({'font.size': 10})
        rcParams.update({'pgf.rcfonts': False})
        rcParams.update({'pgf.texsystem': 'pdflatex'})

        # Font should match document
        rcParams['font.family'] = 'serif'

        rcParams['axes.unicode_minus'] = False
            # Minus sign from matplot lib is too long for my taste.
            
            
        linewidth = 5.71 # inches

        figwidth = 0.95*linewidth
        figheight = figwidth/3*1.44
        figsize = (figwidth, figheight)
        
        plt.figure(figsize=figsize)
        plt.plot(excess_energys*1e3, measured_emittances*1e9, '-', markersize=5)
        plt.plot(excess_energys*1e3, measured_emittances_aperture_size_corrected*1e9, '-', markersize=5)
        plt.plot(excess_energys*1e3, expected_emittances*1e9, ':r')
        plt.xlabel('Excess Energy (meV)')
        plt.ylabel('Emittance (nm rad)')
        
        plt.tight_layout()
        
        plt.savefig('coverage at a focus.pgf')
  
  
# Experimenting with aperture size.
if True:
    file_name = 'pepperpot_aperture_size.h5'
    
    n = 10000000
    d_source_to_lens = .25+.450
    d_lens_to_sample = .100
    d_sample_to_detector = .45+.430

    lens_strength = -1.0e9
        
    pepperpot_pitch = 250e-6
    number_of_holes = 21
    
    if False:
        # Simulate and Save.
        wavelengths = array([475e-9])
        
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        expected_e = expected_emittances[0]
        
        # To save time, propagate a bunch to the pepperpots and reuse.
        save_time = True
        if save_time:
            # Big bunch
            N = n*10
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
            big_base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                          normalised_rms_emittance=expected_e, mass=m_e)
            print('Initial bunch emittance: {:.2f}nm rad'.format(big_base_bunch.getNormalisedRMSEmittance()*1e9))
            
            big_base_bunch.propagate(d_source_to_lens)
            simple_lens(big_base_bunch, strength=lens_strength)
            print('After lens bunch emittance: {:.2f}nm rad'.format(big_base_bunch.getNormalisedRMSEmittance()*1e9))
            big_base_bunch.propagate(d_lens_to_sample)    
            print('Beam size at sample: {:.2f}mm'.format(big_base_bunch.getWidth()*1e3))
            rms_beam_size_sample = big_base_bunch.getWidth()
            
            # 'Small' bunch
            N = n
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
            small_base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                          normalised_rms_emittance=expected_e, mass=m_e)
            print('Initial bunch emittance: {:.2f}nm rad'.format(small_base_bunch.getNormalisedRMSEmittance()*1e9))
            
            small_base_bunch.propagate(d_source_to_lens)
            simple_lens(small_base_bunch, strength=lens_strength)
            print('After lens bunch emittance: {:.2f}nm rad'.format(small_base_bunch.getNormalisedRMSEmittance()*1e9))
            small_base_bunch.propagate(d_lens_to_sample)    
            print('Beam size at sample: {:.2f}mm'.format(small_base_bunch.getWidth()*1e3))
            rms_beam_size_sample = small_base_bunch.getWidth()

        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
            
            
        aperture_sizes = linspace(1e-6, 100e-6, 10)
        aperture_sizes = array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 10e-6, \
                                20e-6, 30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6, 90e-6, 100e-6])
        measured_emittances = zeros(aperture_sizes.size)
        measured_corrected_emittances = zeros(aperture_sizes.size)
        measured_beam_size = zeros(aperture_sizes.size)
        for k, aperture_size in enumerate(aperture_sizes):
            if not save_time:
                if aperture_size<10e-6:
                    N = n*10
                else:
                    N = n
                print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
                base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                              normalised_rms_emittance=expected_e, mass=m_e)
                print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
                
                base_bunch.propagate(d_source_to_lens)
                simple_lens(base_bunch, strength=lens_strength)
                print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
                base_bunch.propagate(d_lens_to_sample)    
                print('Beam size at sample: {:.2f}mm'.format(base_bunch.getWidth()*1e3))
                rms_beam_size_sample = base_bunch.getWidth()
            else:
                if aperture_size<10e-6:
                    base_bunch = big_base_bunch
                else:
                    base_bunch = small_base_bunch
                
            print()
            print('Pepperpot aperture size: {:.2f}um'.format(aperture_size*1e6))
            # Pepperpot
            pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                      location=0, pinhole_diameter=aperture_size)
            
            bunch = base_bunch.copy()
            
            bunch = maskBunch(bunch, pepperpot)
            print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
            print('\tAfter sample bunch count: {:d}'.format(bunch.getSize()))
            
            bunch.propagate(d_sample_to_detector)
            
            hist, bin_edges = histogram(bunch.getXs(), bins=bins)
            m_per_pixel = bin_edges[1]-bin_edges[0]
            
            hist = array(hist, dtype='f')
            
            plt.figure()
            peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=True, smooth=True)
            
            
            plt.figure()
            emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                            adjust_for_size=True, adjust_for_aperture_size=False, refine_parameters=True)
            emittance *= bunch.getBeta()
            
            corrected_emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, \
                            diag=True, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                            aperture_size=aperture_size, adjust_for_size=True, \
                            adjust_for_aperture_size=True, refine_parameters=True)
            corrected_emittance *= bunch.getBeta()
            
            measured_emittances[k] = emittance
            measured_corrected_emittances[k] = corrected_emittance
            measured_beam_size[k] = rms_size
            
            print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
              
            
        # Save data
        with File(file_name) as hdf:
            aperture_size_key = 'aperture size'
            if aperture_size_key in hdf:
                del hdf[aperture_size_key]
                
            hdf.create_dataset(aperture_size_key, data=aperture_sizes)
            
            measured_emittance_key = 'measured emittance'
            if measured_emittance_key in hdf:
                del hdf[measured_emittance_key]
                
            hdf.create_dataset(measured_emittance_key, data=measured_emittances)
            
            
            measured_corrected_emittance_key = 'measured corrected emittance'
            if measured_corrected_emittance_key in hdf:
                del hdf[measured_corrected_emittance_key]
                
            hdf.create_dataset(measured_corrected_emittance_key, data=measured_corrected_emittances)
            
            measured_beam_size_key = 'measured beam size'
            if measured_beam_size_key in hdf:
                del hdf[measured_beam_size_key]
                
            hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
            
            hdf.attrs['expected emittance'] = expected_e
            hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
    else:
        # Load data
        with File(file_name) as hdf:
            expected_e = hdf.attrs['expected emittance']
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            aperture_sizes = array(hdf['aperture size'])
            measured_emittances = array(hdf['measured emittance'])
            measured_corrected_emittances = array(hdf['measured corrected emittance'])
    
    # Plot Stuff
    plt.figure('Aperture Sizes vs Measured Emittance')
    
    plt.subplot(2, 1, 1)
    plt.title('Raw')
    plt.plot(aperture_sizes*1e6, measured_emittances*1e9)
    plt.plot(aperture_sizes*1e6, measured_corrected_emittances*1e9)
    plt.xlabel('Aperture Size (um)')
    plt.ylabel('Measured Emittance (nm rad)')
    plt.axhline(expected_e*1e9, ls=':', color='k')
    
    plt.subplot(2, 1, 2)
    plt.title('Normalised')
    plt.plot(aperture_sizes*1e6, measured_emittances/expected_e)
    plt.plot(aperture_sizes*1e6, measured_corrected_emittances/expected_e)
    plt.xlabel('Aperture Size (um)')
    plt.ylabel('Measured Emittance/Expected Emittance')
    plt.axhline(1, ls=':', color='k')
    
    if True:
        # Plot for thesis.
        rcParams.update({'font.size': 10})
        rcParams.update({'pgf.rcfonts': False})
        rcParams.update({'pgf.texsystem': 'pdflatex'})

        # Font should match document
        rcParams['font.family'] = 'serif'

        rcParams['axes.unicode_minus'] = False
            # Minus sign from matplot lib is too long for my taste.
            
            
        colours = [(79/255,122/255,174/255),(255/255,102/255,51/255),(245/255,174/255,32/255),(77/255,155/255,77/255),(102/255,102/255,102/255)]

            
        linewidth = 5.71 # inches

        figwidth = linewidth
        figheight = figwidth/3
        figwidth *= 0.75
        figsize = (figwidth, figheight)
        
        plt.figure(figsize=figsize)
        plt.plot(aperture_sizes*1e6, measured_emittances*1e9, color=colours[0])
        plt.plot(aperture_sizes*1e6, measured_corrected_emittances*1e9, color=colours[3])
        plt.xlabel('Aperture Size ($\mu$m)')
        plt.ylabel('Emittance (nm rad)')
        plt.axhline(expected_e*1e9, ls=':', color='k')
        
        plt.tight_layout()
        
        plt.savefig('aperture_size_sim.pgf')
        

# Looking at overlapping beamlets.
if False:
    file_name = 'pepperpot_overlap.h5'
    
    N = 10000000
    d_source_to_lens = .25 + .450
    d_lens_to_sample = .100
    d_sample_to_detector = .45 + .430

    lens_strength = -1.0e9
        
    pepperpot_pitch = 200e-6
    number_of_holes = 30
    aperture_size = 50e-6
    
    if True:
        # Simulate and Save.
        excess_energys = linspace(-5e-3, 100e-3, 20)
        
        excess_energys = excess_energys[-3:-1]
        
        expected_emittances = expected_emittance(excess_energys, beam_size_rms)
        
        mn = -0.035
        mx = 0.035
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 2000)
            
        measured_emittances = zeros(expected_emittances.size)
        measured_emittances_not_corrected = zeros(expected_emittances.size)
        measured_beam_size = zeros(expected_emittances.size)
        for k, expected_e in enumerate(expected_emittances):
            if isnan(expected_e):
                expected_e = 0
        
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
            base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms_cherry, \
                          normalised_rms_emittance=expected_e, mass=m_e)
            print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            
            base_bunch.propagate(d_source_to_lens)
            simple_lens(base_bunch, strength=lens_strength)
            print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
            base_bunch.propagate(d_lens_to_sample)    
            rms_beam_size_sample = base_bunch.getWidth()
            print('Beam size at sample: {:.2f}um'.format(rms_beam_size_sample*1e6))
            
            print()
            
            # Pepperpot
            pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                      location=0, pinhole_diameter=aperture_size)
            
            bunch = base_bunch.copy()
            
            bunch = maskBunch(bunch, pepperpot)
            print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
            print('\tAfter sample bunch count: {:d}'.format(bunch.getSize()))
            
            bunch.propagate(d_sample_to_detector)
            
            hist, bin_edges = histogram(bunch.getXs(), bins=bins)
            m_per_pixel = bin_edges[1]-bin_edges[0]
            
            peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=True, smooth=True)
            
            hist = array(hist, dtype='f')
            
            plt.figure()
            emittance, rms_size, _total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                            adjust_for_size=True, refine_parameters=True, adjust_for_aperture_size=True)
            emittance *= bunch.getBeta()
            
            plt.figure()
            emittance2, _rms_size, _total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector,
                            adjust_for_size=True, refine_parameters=False, adjust_for_aperture_size=True)
            emittance2 *= bunch.getBeta()
            
            measured_emittances[k] = emittance
            measured_emittances_not_corrected[k] = emittance2
            measured_beam_size[k] = rms_size
            
            print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
            print('\tMeasured Emittance (not corrected): {:.2f}nm rad'.format(emittance2*1e9))
            print()
            print()
            
            if False:
                plt.show()
                exit()
        
        # Save data
        if True:
            with File(file_name) as hdf:
                measured_emittance_key = 'measured emittance'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances)
                
                measured_emittance_nt_corrected_key = 'measured emittance nor corrected'
                if measured_emittance_nt_corrected_key in hdf:
                    del hdf[measured_emittance_nt_corrected_key]
                    
                hdf.create_dataset(measured_emittance_nt_corrected_key, data=measured_emittances_not_corrected)
                
                measured_beam_size_key = 'measured beam size'
                if measured_beam_size_key in hdf:
                    del hdf[measured_beam_size_key]
                    
                hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
                
                expected_emittance_key = 'expected emittance'
                if expected_emittance_key in hdf:
                    del hdf[expected_emittance_key]
                    
                hdf.create_dataset(expected_emittance_key, data=expected_emittances)
                
                excess_energy_key = 'excess energy'
                if excess_energy_key in hdf:
                    del hdf[excess_energy_key]
                    
                hdf.create_dataset(excess_energy_key, data=excess_energys)
                
    else:
        # Load data
        with File(file_name) as hdf:
            excess_energys = array(hdf['excess energy'])
            expected_emittances = array(hdf['expected emittance'])
            
            measured_emittances = array(hdf['measured emittance'])
            measured_emittances_not_corrected = array(hdf['measured emittance nor corrected'])
            
            
    if True:
        # Plot
        plt.figure('Beamlet overlap')
        
        plt.plot(excess_energys*1e3, measured_emittances*1e9)
        plt.plot(excess_energys*1e3, measured_emittances_not_corrected*1e9)
        plt.plot(excess_energys*1e3, expected_emittances*1e9)

# Looking at overlapping beamlets. Take 2.
if False:
    beam_size = beam_size_rms_cherry
    filename = 'overlap.h5'
            
    if True:
        # Simulate and save
        wavelengths = linspace(445e-9, 487e-9, 100)[11:12]
            
        excess_energys = excess_energy_from_wavelength(wavelengths, field_ionisation=False)
        
        expected_emittances = expected_emittance(excess_energys, beam_size)
        
        # Pepperpot
        number_of_holes = 9
        pepperpot_pitch = 150e-6
        aperture_size = 50e-6
        pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, pinhole_diameter=aperture_size)
        
        # Bins for histogram 'detector'
        if False:
            bins = 1000
        else:
            bins = linspace(-.001, 0.001, 2000)
        
        N = 10000000
        propagation_distance = 0.015
        # For each +'ve excess energy simulate a bunch.
        measured_emittances = zeros(wavelengths.size)
        measured_emittances_beam_size_corrected = zeros(wavelengths.size)
        measured_emittances_aperture_size_corrected = zeros(wavelengths.size)
        for i, expected_e in enumerate(expected_emittances):
            if isnan(expected_e):
                if False:
                    # Skip
                    measured_emittances[i] = float('nan')
                    measured_emittances_aperture_size_corrected[i] = float('nan')
                    continue
                else:
                    # Use zero emittance beam.
                    expected_e = 0
                    expected_emittances[i] = expected_e
            
            print(i)
            print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
            
            bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size, \
                          normalised_rms_emittance=expected_e, mass=m_e)
                      
            print('\tBunch RMS Size: {:.2f}um'.format(bunch.getWidth()*1e6))
            
            bunch = maskBunch(bunch, pepperpot)
                          
            bunch.propagate(propagation_distance)
            
            hist, bin_edges = histogram(bunch.getXs(), bins=bins)
            m_per_pixel = bin_edges[1]-bin_edges[0]
            hist = array(hist, dtype='f')
            
            peaks = find_peaks(bin_edges[:-1], hist, thres=0.005, min_dist=100, diag=True, smooth=True, smooth_size=10)
            
            
            if True:
                # Try smoothing the histogram.
                hist = gaussian_filter(hist, 5)
            
            # Don't correct for anything.
            emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=False,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=False, adjust_for_aperture_size=False, refine_parameters=False)
            emittance *= bunch.getBeta()
            
            measured_emittances[i] = emittance
            
            print('\tMeasured Emittance (no corrections):        {:.2f}nm rad'.format(emittance*1e9))
            
            # Don't correct for overlap.
            emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=False,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=True, adjust_for_aperture_size=True, refine_parameters=False)
            emittance *= bunch.getBeta()
            
            measured_emittances_beam_size_corrected[i] = emittance
            
            print('\tMeasured Emittance (no overlap correction): {:.2f}nm rad'.format(emittance*1e9))
            
            # Correct for everything.
            plt.figure()
            emittance, _rms_size, _total_count = emittance_from_line(hist, peaks, sum_peaks=True, diag=True,
                            m_per_pixel=m_per_pixel, number_holes=len(peaks),
                            pitch=pepperpot_pitch, propagation_distance=propagation_distance,
                            adjust_for_size=True, adjust_for_aperture_size=True, refine_parameters=True)
            emittance *= bunch.getBeta()
            measured_emittances_aperture_size_corrected[i] = emittance
            
            print('\tMeasured Emittance (all corrections):       {:.2f}nm rad'.format(emittance*1e9))

            with File('overlap_example.h5') as hdf:
                histogram_key = 'histogram'
                hdf.create_dataset(histogram_key, data=hist)
                
                hdf.attrs['expected emittance'] = expected_e
                hdf.attrs['excess energy'] = excess_energys[0]
                hdf.attrs['wavelength'] = wavelengths[0]
                hdf.attrs['source size'] = beam_size
                hdf.attrs['no correction emittance'] = measured_emittances[i]
                hdf.attrs['corrected unrefined emittance'] = measured_emittances_beam_size_corrected[i]
                hdf.attrs['corrected refined emittance'] = measured_emittances_aperture_size_corrected[i]
                hdf.attrs['m_per_pixel'] = m_per_pixel
                
            if True:
                plt.figure('hist')
                plt.plot(bin_edges[:-1], hist)
                
                plt.show()
                exit()
                
            
        # Save data
        if True:
            with File(filename) as hdf:
                wavelengths_key = 'wavelengths'
                if wavelengths_key in hdf:
                    del hdf[wavelengths_key]
                    
                hdf.create_dataset(wavelengths_key, data=wavelengths)
                
                excess_energy_key = 'excess energy'
                if excess_energy_key in hdf:
                    del hdf[excess_energy_key]
                    
                hdf.create_dataset(excess_energy_key, data=excess_energys)
                
                expected_emittance_key = 'expected emittance'
                if expected_emittance_key in hdf:
                    del hdf[expected_emittance_key]
                    
                hdf.create_dataset(expected_emittance_key, data=expected_emittances)
                
                measured_emittance_key = 'measured emittance'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances)
                
                measured_emittance_key = 'measured emittance beam size correction'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances_beam_size_corrected)
                
                measured_emittance_key = 'measured emittance aperture size correction'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances_aperture_size_corrected)
                
                hdf.attrs['number of holes'] = number_of_holes
                hdf.attrs['aperture size'] = aperture_size
                hdf.attrs['pitch'] = pepperpot_pitch
    else:
        with File(filename) as hdf:
            wavelengths = array(hdf['wavelengths'])
            excess_energys = array(hdf['excess energy'])
            expected_emittances = array(hdf['expected emittance'])
            measured_emittances = array(hdf['measured emittance'])
            measured_emittances_beam_size_corrected = array(hdf['measured emittance beam size correction'])
            measured_emittances_aperture_size_corrected = array(hdf['measured emittance aperture size correction'])
            
            number_of_holes = hdf.attrs['number of holes']
            aperture_size = hdf.attrs['aperture size']
            pepperpot_pitch = hdf.attrs['pitch']
            
        
    # Plotting
    plt.figure('Wavelength vs. Excess Energy')
    plt.title('Wavelength vs. Excess Energy')
    plt.plot(wavelengths*1e9, excess_energys*1e3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Excess Energy (meV)')
    plt.axhline(0, color='k', ls=':')
    plt.xlim((wavelengths.min()*1e9, wavelengths.max()*1e9))
    
    
    plt.figure('Excess Emittance vs. Emittance')
    plt.title('Excess Emittance vs. Emittance')
    plt.plot(excess_energys*1e3, expected_emittances*1e9, 'r')
    plt.plot(excess_energys*1e3, measured_emittances*1e9, 'bx')
    plt.plot(excess_energys*1e3, measured_emittances_beam_size_corrected*1e9, 'gx')
    plt.plot(excess_energys*1e3, measured_emittances_aperture_size_corrected*1e9, 'kx')
    plt.xlabel('Excess Energy (meV)')
    plt.ylabel('Emittance (nm rad)')
    plt.xlim((0, excess_energys.max()*1e3))
    
    if False:
        # Plot for thesis.
        rcParams.update({'font.size': 10})
        rcParams.update({'pgf.rcfonts': False})
        rcParams.update({'pgf.texsystem': 'pdflatex'})
        
        colours = [(79/255,122/255,174/255),(255/255,102/255,51/255),(245/255,174/255,32/255),(77/255,155/255,77/255),(102/255,102/255,102/255)]

        # Font should match document
        rcParams['font.family'] = 'serif'

        rcParams['axes.unicode_minus'] = False
            # Minus sign from matplot lib is too long for my taste.
            
            
        linewidth = 5.71 # inches

        figwidth = linewidth
        figheight = figwidth/3
        figwidth *= 0.75
        figsize = (figwidth, figheight)
        
        # Higher res theory
        wavelengths_fine = linspace(465e-9, 487e-9, 10000)
            
        excess_energys_fine = excess_energy_from_wavelength(wavelengths_fine, field_ionisation=False)
        
        expected_emittances_fine = expected_emittance(excess_energys_fine, beam_size)
        for i in range(expected_emittances_fine.size):
            if isnan(expected_emittances_fine[i]):
                expected_emittances_fine[i] = 0
        
        # Plot
        plt.figure(figsize=figsize)
        plt.plot(excess_energys*1e3, measured_emittances*1e9, '-', markersize=5, color=colours[0])
        plt.plot(excess_energys*1e3, measured_emittances_aperture_size_corrected*1e9, '-', markersize=5, color=colours[3])
        plt.plot(excess_energys_fine*1e3, expected_emittances_fine*1e9, ':', color=colours[1])
        plt.xlim((-5, excess_energys.max()*1e3))
        plt.xlabel('Excess Energy (meV)')
        plt.ylabel('Emittance (nm rad)')
        
        plt.tight_layout()
        
        plt.savefig('wavelength_sweep_sim.pgf')
  
  
# Realistic Parameters - Extensive look at corrective factors.
if False:
    hdf_name = 'pepperpot_correction.h5'
    
    N = 10000000
    d_source_to_lens = .25+.450
    d_lens_to_sample = .100
    d_sample_to_detector = .45+.430

    lens_strength = -1.0e9
        
    
    
    if False:
        # Simulate and Save.
        wavelength = 475e-9
        excess_energy = excess_energy_from_wavelength(wavelength, field_ionisation=False)
        expected_emittance = expected_emittance(excess_energy, beam_size_rms)
        
        # To save time, propagate a bunch to the pepperpots and reuse.
        print('Emittance: {:.2f}nm rad'.format(1e9*expected_emittance))
        base_bunch = Bunch(n=N, energy=beam_energy, rms_width=beam_size_rms, \
                      normalised_rms_emittance=expected_emittance, mass=m_e)
        print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        
        base_bunch.propagate(d_source_to_lens)
        simple_lens(base_bunch, strength=lens_strength)
        print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        base_bunch.propagate(d_lens_to_sample)    
        print('Beam size at sample: {:.2f}mm'.format(base_bunch.getWidth()*1e3))
        rms_beam_size_sample = base_bunch.getWidth()

        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
        
        # Pepperpot parameters
        pepperpot_pitchs = linspace(200e-6, 500e-6, 10)
        centres = linspace(0, 2*rms_beam_size_sample, 10)
        num_holess = array(flipud(arange(5, 20, 3)))
        
        
        measured_emittances = zeros((pepperpot_pitchs.size, centres.size, num_holess.size))
        measured_beam_size = zeros((pepperpot_pitchs.size, centres.size, num_holess.size))
        for i, pepperpot_pitch in enumerate(pepperpot_pitchs):
            for k, pepperpot_center in enumerate(centres):
                for j, num_holes in enumerate(num_holess):
                    print('{:d} of {:d}, {:d} of {:d}, {:d} of {:d}'.format(i, pepperpot_pitchs.size, \
                                                                            k, centres.size, \
                                                                            j, num_holess.size))
                    # Pepperpot
                    number_of_holes = num_holes
                    pepperpot = pepperpotMask(pitch=pepperpot_pitch, number_holes=number_of_holes, \
                                              location=pepperpot_center)
                
                    print('\tPepperpot Pitch: {:.2f}um'.format(pepperpot_pitch*1e6))
                    print('\tPepperpot Centre: {:.2f}um'.format(pepperpot_center*1e6))
                    print('\tNumber of Holes:', num_holes)
                    
                    bunch = base_bunch.copy()
                    
                    bunch = maskBunch(bunch, pepperpot)
                    
                    bunch.propagate(d_sample_to_detector)
                    
                    hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                    m_per_pixel = bin_edges[1]-bin_edges[0]
                    
                    peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=False, smooth=True)
                    
                    hist = array(hist, dtype='f')
                    
                    if False:
                        plt.figure()
                        plt.plot(hist)
                        plt.show()
                    
                    emittance, rms_size, total_count = emittance_from_line(hist, peaks, \
                                    sum_peaks=True, diag=False,
                                    m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                    pitch=pepperpot_pitch, propagation_distance=d_sample_to_detector, \
                                    adjust_for_aperture_size=True, refine_parameters=True, \
                                    adjust_for_size=False)
                    emittance *= bunch.getBeta()
                    
                    measured_emittances[i, k, j] = emittance
                    measured_beam_size[i, k, j] = rms_size
                    
                    print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
                
            
        # Save data
        with File(hdf_name) as hdf:
            pepperpot_pitch_key = 'pepperpot pitch'
            if pepperpot_pitch_key in hdf:
                del hdf[pepperpot_pitch_key]
                
            hdf.create_dataset(pepperpot_pitch_key, data=pepperpot_pitchs)
            
            number_of_holes_key = 'number of holes'
            if number_of_holes_key in hdf:
                del hdf[number_of_holes_key]
                
            hdf.create_dataset(number_of_holes_key, data=num_holess)
            
            centres_key = 'pepperpot centre'
            if centres_key in hdf:
                del hdf[centres_key]
                
            hdf.create_dataset(centres_key, data=centres)
            
            measured_emittance_key = 'measured emittance'
            if measured_emittance_key in hdf:
                del hdf[measured_emittance_key]
                
            hdf.create_dataset(measured_emittance_key, data=measured_emittances)
            
            measured_beam_size_key = 'measured beam size'
            if measured_beam_size_key in hdf:
                del hdf[measured_beam_size_key]
                
            hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
            
            hdf.attrs['expected emittance'] = expected_emittance
            hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
            hdf.attrs['number of electrons'] = N
    else:
        # Load data
        with File(hdf_name) as hdf:
            expected_emittance = hdf.attrs['expected emittance']
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            
            num_holess = array(hdf['number of holes'])
            centres = array(hdf['pepperpot centre'])
            pepperpot_pitchs = array(hdf['pepperpot pitch'])
            
            measured_emittances = array(hdf['measured emittance'])
            measured_beam_size = array(hdf['measured beam size'])
            
    if True:
        # Plot
        
        corrections = zeros(measured_emittances.shape)
        for i, pitch in enumerate(pepperpot_pitchs):
            for k, c in enumerate(centres):
                for j, n in enumerate(num_holess):
                    x_lo = (c - n*pitch/2)*1 / rms_beam_size_sample
                    x_hi = (c + n*pitch/2)*1 / rms_beam_size_sample
                    
                    corrections[i, k, j] = 1/normal_prop(x_lo, x_hi)
        
        
        # Pepperpot centre vs. emittance
        plt.figure()
        
        plt.axhline(expected_emittance*1e9, ls=':', color='r')
        
        for i, pitch in enumerate(pepperpot_pitchs):
            plt.plot(centres*1e3, 1e9*measured_emittances[i, :, 0])
            #plt.plot(centres*1e3, 1e9*measured_emittances[i, :, 0]*corrections[i, :, 0], 'x')
            
            correction_f = lambda centre, magic_number: measured_emittances[i, :, 0]/normal_prop( \
                (centre - num_holess[0]*pitch/2)*magic_number / rms_beam_size_sample, \
                (centre + num_holess[0]*pitch/2)*magic_number / rms_beam_size_sample)
        
            guess = 3/4
        
            fit = fitCurve(centres, zeros(centres.size)+expected_emittance, correction_f, guess)
        
            print(fit)
            
            plt.plot(centres*1e3, 1e9*correction_f(centres, 0.75), 'x')
            
        plt.xlabel('Pepperpot Centre (mm)')
        plt.ylabel('Measured Emittance (nm rad)')
            
    
#  Looking at beam coverage
if True:
    file_name = 'pepperpot_coverage.h5'
    
    N = 10000000
    d_source_to_lens = .25 + .450
    d_lens_to_sample = .100
    d_sample_to_detector = .45 + .430

    lens_strength = -1.0e9
        
    centres = array([0])#, 1, 2]) # Real beam RMS widths
    pepperpot_pitchs = array([200e-6, 250e-6, 300e-6])
    number_of_holess = arange(3, 31, 1)
    aperture_size = 50e-6
    
    if False:
        # Simulate and Save.
        source_size = beam_size_rms
        
        excess_energys = array([30e-3])
        
        expected_emittances = expected_emittance(excess_energys, source_size)
        
        expected_e = expected_emittances[0]
        
        mn = -0.025
        mx = 0.025
        rn = mx - mn
        bins = linspace(mn - rn, mx + rn, 1000)
            
        
        print('Emittance: {:.2f}nm rad'.format(1e9*expected_e))
        base_bunch = Bunch(n=N, energy=beam_energy, rms_width=source_size, \
                      normalised_rms_emittance=expected_e, mass=m_e)
        print('Initial bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        
        base_bunch.propagate(d_source_to_lens)
        simple_lens(base_bunch, strength=lens_strength)
        print('After lens bunch emittance: {:.2f}nm rad'.format(base_bunch.getNormalisedRMSEmittance()*1e9))
        base_bunch.propagate(d_lens_to_sample)    
        rms_beam_size_sample = base_bunch.getWidth()
        print('Beam size at sample: {:.2f}um'.format(rms_beam_size_sample*1e6))
        
        print()
        
        measured_emittances = zeros((centres.size, pepperpot_pitchs.size, number_of_holess.size))
        measured_emittances_corrected = zeros((centres.size, pepperpot_pitchs.size, number_of_holess.size))
        measured_beam_size = zeros((centres.size, pepperpot_pitchs.size, number_of_holess.size))
        for m, centre in enumerate(centres):
            for j, pitch in enumerate(pepperpot_pitchs):
                for k, number_of_holes in enumerate(number_of_holess):
                    print('Centre:', centre)
                    print('Pitch: {:.2f}um'.format(pitch*1e6))
                    print('Number of holes:', number_of_holes)
                    # Pepperpot
                    pepperpot = pepperpotMask(pitch=pitch, number_holes=number_of_holes, \
                                              location=centre*rms_beam_size_sample, \
                                              pinhole_diameter=aperture_size)
                    
                    bunch = base_bunch.copy()
                    
                    bunch = maskBunch(bunch, pepperpot)
                    print('\tAfter sample bunch emittance: {:.2f}nm rad'.format(bunch.getNormalisedRMSEmittance()*1e9))
                    print('\tAfter sample bunch count: {:d}'.format(bunch.getSize()))
                    
                    bunch.propagate(d_sample_to_detector)
                    
                    hist, bin_edges = histogram(bunch.getXs(), bins=bins)
                    m_per_pixel = bin_edges[1]-bin_edges[0]
                    
                    peaks = find_peaks(bin_edges[:-1], hist, min_dist=25, diag=True, smooth=True)
                    
                    hist = array(hist, dtype='f')
                    
                    plt.figure()
                    emittance, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, \
                                    diag=True, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                    pitch=pitch, propagation_distance=d_sample_to_detector,
                                    adjust_for_size=False, refine_parameters=True, adjust_for_aperture_size=True)
                    emittance *= bunch.getBeta()
                    
                    emittance2, rms_size, total_count = emittance_from_line(hist, peaks, sum_peaks=True, \
                                    diag=True, m_per_pixel=m_per_pixel, number_holes=len(peaks),
                                    pitch=pitch, propagation_distance=d_sample_to_detector,
                                    adjust_for_size=True, refine_parameters=True, adjust_for_aperture_size=True)
                    emittance2 *= bunch.getBeta()
                    
                    measured_emittances_corrected[m, j, k] = emittance2
                    measured_emittances[m, j, k] = emittance
                    measured_beam_size[m, j, k] = rms_size
                    
                    print('\tMeasured Beam RMS Size: {:.2f}um'.format(rms_size*1e6))
                    print('\tMeasured Emittance: {:.2f}nm rad'.format(emittance*1e9))
                    print('\tMeasured Emittance (corrected): {:.2f}nm rad'.format(emittance2*1e9))
                    print()
        
        # Save data
        if True:
            with File(file_name) as hdf:
                number_of_holes_key = 'number of holes'
                if number_of_holes_key in hdf:
                    del hdf[number_of_holes_key]
                    
                hdf.create_dataset(number_of_holes_key, data=number_of_holess)
                
                measured_emittance_key = 'measured emittance'
                if measured_emittance_key in hdf:
                    del hdf[measured_emittance_key]
                    
                hdf.create_dataset(measured_emittance_key, data=measured_emittances)
                
                measured_emittance_corrected_key = 'measured emittance corrected'
                if measured_emittance_corrected_key in hdf:
                    del hdf[measured_emittance_corrected_key]
                    
                hdf.create_dataset(measured_emittance_corrected_key, data=measured_emittances_corrected)
                
                measured_beam_size_key = 'measured beam size'
                if measured_beam_size_key in hdf:
                    del hdf[measured_beam_size_key]
                    
                hdf.create_dataset(measured_beam_size_key, data=measured_beam_size)
                
                hdf.attrs['expected emittance'] = expected_e
                hdf.attrs['rms beam width at pepperpot'] = rms_beam_size_sample
    else:
        # Load data
        with File(file_name) as hdf:
            number_of_holess = array(hdf['number of holes'])
            expected_e = hdf.attrs['expected emittance']
            rms_beam_size_sample = hdf.attrs['rms beam width at pepperpot']
            measured_emittances = array(hdf['measured emittance'])
            measured_emittances_corrected = array(hdf['measured emittance corrected'])
            
    if True:
        # Plot
        for j in range(centres.size):
            for i in range(pepperpot_pitchs.size):
                plt.figure('Raw ' + str(j) + ' Centre: {:.2f}'.format(centres[j]))
                plt.plot(number_of_holess, measured_emittances[j, i]*1e9, label='{:.2f}um Pitch'.format(pepperpot_pitchs[i]*1e6))
            
                plt.figure('Normalised ' + str(j))
                norm_xs = number_of_holess*pepperpot_pitchs[i] / rms_beam_size_sample
                norm_ys = measured_emittances[j, i]/expected_e
                plt.plot(norm_xs, norm_ys)
                
                fit_f = lambda x, magic: normal_prop(-magic*(x-centres[j]*rms_beam_size_sample)/2, \
                                                      magic*(x-centres[j]*rms_beam_size_sample)/2)
                guess = [0.77]
                
                fit = fitCurve(norm_xs, norm_ys, fit_f, guess)
                print('Pitch: {:.2f}um, Magic Number: {:.2f}'.format(pepperpot_pitchs[i]*1e6, fit[0]))
            
        
        
        for j in range(centres.size):
            
            xs = arange(0.1, 30, 0.1)*pepperpot_pitchs[-1] / rms_beam_size_sample
            MAGIC = 0.75
            ys = array([normal_prop(-MAGIC*(x-centres[j]*rms_beam_size_sample-aperture_size/2)/2, \
                                     MAGIC*(x-centres[j]*rms_beam_size_sample+aperture_size/2)/2) \
                                     for x in xs])
                                     
            MAGIC = 1
            ys2 = array([normal_prop(-MAGIC*(x-centres[j]*rms_beam_size_sample-aperture_size/2)/2, \
                                     MAGIC*(x-centres[j]*rms_beam_size_sample+aperture_size/2)/2) for x in xs])
            
            plt.figure('Normalised ' + str(j))
            plt.plot(xs, ys, '--')
            plt.plot(xs, ys2, ':')
            
            plt.xlabel('RMS Widths')
            plt.ylabel('Emittance / Expected Emittance')
        
            plt.figure('Raw ' + str(j) + ' Centre: {:.2f}'.format(centres[j]))
            plt.axhline(expected_e*1e9, color='r', ls=':')
            plt.xlabel('Number of Holes')
            plt.ylabel('Emittance (nm rad)')
            plt.legend(loc='lower right')
            
        if True:
            # Thesis plot
            rcParams.update({'font.size': 10})
            rcParams.update({'pgf.rcfonts': False})
            rcParams.update({'pgf.texsystem': 'pdflatex'})
            
            colours = [(79/255,122/255,174/255),(255/255,102/255,51/255),(245/255,174/255,32/255),(77/255,155/255,77/255),(102/255,102/255,102/255)]

            # Font should match document
            rcParams['font.family'] = 'serif'

            rcParams['axes.unicode_minus'] = False
                # Minus sign from matplot lib is too long for my taste.
                
                
            linewidth = 5.71 # inches

            figwidth = linewidth
            figheight = figwidth/3
            figwidth *= 0.75
            figsize = (figwidth, figheight)
            
            coverages = zeros(measured_emittances.size)
            emittances = zeros(measured_emittances.size)
            emittances_corrected = zeros(measured_emittances.size)
            for i in range(pepperpot_pitchs.size):
                coverage = pepperpot_pitchs[i]*number_of_holess / rms_beam_size_sample
                e = measured_emittances[0, i] / expected_e
                e_c = measured_emittances_corrected[0, i] / expected_e
                
                start = i*number_of_holess.size
                stop = (i+1)*number_of_holess.size
                
                coverages[start:stop] = coverage
                emittances[start:stop] = e
                emittances_corrected[start:stop] = e_c
                
            plt.figure('thesis', figsize=figsize)
            plt.plot(coverages, emittances, color=colours[0])
            plt.plot(coverages, emittances_corrected, color=colours[3])
            
            plt.xlim((0, coverages.max()))
            plt.xlabel('Pepperpot Extent ($\sigma$)')
            plt.ylabel('Accuracy')
            
            plt.figure('thesis 2', figsize=figsize)
            xs = pepperpot_pitchs[0]*number_of_holess
            ys = measured_emittances[0,0]
            ys2 = measured_emittances_corrected[0,0]
            plt.plot(xs*1e3, ys*1e9, color=colours[0])
            plt.plot(xs*1e3, ys2*1e9, color=colours[3])
            
            plt.axhline(expected_e*1e9, color=colours[1], ls=':')
            plt.axvline(rms_beam_size_sample*1e3, color='k', ls=':')
            
            plt.ylim((0, 150))
            
            plt.xlabel('Pepperpot Extent (mm)')
            plt.ylabel('Emittance (nm rad)')
            
            plt.tight_layout()
            
            plt.savefig('PepperpotExtent.pgf')
        
            
plt.show()
    
