##### Imports #####
from numpy import sqrt, pi, nanmean, histogram, zeros, average, arange, floor, \
                  errstate, linspace, exp, absolute, array
from matplotlib import pyplot as plt
from scipy.constants import physical_constants, speed_of_light, m_e, eV, k as kB
from scipy.stats import norm
from scipy.signal import fftconvolve

from TraceUtility import window_trace, normalise
from Fitting import fitCurve


##### Constants #####
# Physical Constants
Plank = physical_constants['Planck constant in eV s'][0]
Ry = physical_constants['Rydberg constant'][0] # Rydberg constant (per m)
electric_field_atomic_unit = physical_constants['atomic unit of electric field'][0]
                # Atomic unit of electric field (V/m)

# Experimental Constants
mcp_calibration = 34.27 # counts/electron
Rb_ionisation_energy=4.1771270
red_wavelength = 780.2414e-9
electric_field = 16e3/0.05 # (V / m)
m_per_pixel = 58.2e-6
mcp_calibration = 34.27 # counts/electron
propagation_distance = 0.475

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chartreuse']


##### Functions #####
def normal_cumulative_thingy(x):
    f = lambda x: 1-(1-norm.cdf(x))*2
    
    return f(x)
    
def normal_prop(x_lo, x_hi):
    return norm.cdf(x_hi) - norm.cdf(x_lo)

def tophat_f(width, centre=0):
    return lambda x: absolute(x-centre)<width/2    
    
def gaussian(x, height=1, centre=0, width=1, offset=0):
    return height * exp( -(x-centre)**2 / (2*width**2)) + offset
    
def gaussian_f(std):
    return lambda x: exp(-x**2/(2*std**2))
    
def n_gaussians(x, *args):
    y = zeros(x.size)
    for ey in arange(len(args)/3):
        # 3 arguments, ignore offset
        i = int(ey)*3
        
        height = args[i]
        centre = args[i+1]
        width = args[i+2]
        
        wy = gaussian(x, height, centre, width)
        
        y += wy
        
    return y
    
# Analysis
def emittance_from_line(line, peaks, sum_peaks=False, diag=False, m_per_pixel=m_per_pixel, \
                        number_holes=7, pitch=200e-6, propagation_distance=propagation_distance, adjust_for_size=True, return_beam_size=True, return_beam_count=True,
                        aperture_size=50e-6, adjust_for_aperture_size=True, refine_parameters=False, zero=True, index=-1, subplots=True):
    xs = arange(line.size, dtype='f')
    
    if diag and subplots:
        plt.subplot(2, 2, 1)
    ret = get_gaussians_from_line(xs, line, peaks, sum_peaks=sum_peaks, diag=diag, zero=zero)
    
    if sum_peaks:
        heights, centres, widths, sums = ret
    else:
        heights, centres, widths = ret
        sums = None
    
    if refine_parameters:
        try:
            if diag and subplots:
                plt.subplot(2, 2, 2)
            heights, centres, widths = refine_gaussian_parameters(xs, line, heights, centres, widths, diag=diag and subplots)
        except RuntimeError:
            # Leave them as they are.
            print('\tFailed to refine parameters.')
    
    try:
        if diag and subplots:
            plt.subplot(2, 2, 3)
        rms_size, total_beam_count, center = calc_beam_size(heights, centres, widths, \
                                        sums=sums, pepperpot_pitch=pitch, return_centre=True, \
                                        diag=diag and subplots)
    except RuntimeError as e:
        # Failed Fits. :(
        print('Failed to find beam size.')
        print(e)
        
        if False:
            plt.figure()
            if sums is not None:
                plt.title('Sums')
                plt.plot(sums)
            else:
                plt.title('Heights')
                plt.plot(heights)
            plt.figure()
            plt.plot(line)
        
        em = float('NaN')
        total_beam_count = float('NaN')
        rms_size = float('NaN')
        
        #rms_size, total_beam_count = -1, 0
    else:
        em = emittance(heights, centres, widths, num_particles=sums, m_per_pixel=m_per_pixel, \
                       number_holes=number_holes, pitch=pitch, propagation_distance=propagation_distance,
                       aperture_size=aperture_size, adjust_for_aperture_size=adjust_for_aperture_size)

        if adjust_for_size and rms_size!=-1:
            if False:
                # Centred correction
                correction = normal_cumulative_thingy(0.5*number_holes*pitch/rms_size)
            else:
                # Off centred correction
                x_lo = (3/4)*(center - number_holes*pitch/2) / rms_size
                x_hi = (3/4)*(center + number_holes*pitch/2) / rms_size
                    # Why 3/4? 1 rms to either side = /2
                
                correction = normal_prop(x_lo, x_hi)
            
            print('\tBeam rms size: {:.2f}um'.format(rms_size*1e6))
            print('\tCorrection: {:.2f}'.format(correction))
            em /= correction
            #print('\t{:.2f}'.format(correction))
        else:
            print('\tNot correcting.')
        
    if return_beam_size and return_beam_count:
        return em, rms_size, total_beam_count
    elif return_beam_size and not return_beam_count:
        return em, rms_size
    elif not return_beam_size and return_beam_count:
        return em, total_beam_count
    else:
        return em

def get_gaussians_from_line(xs, ys, peaks, sum_peaks=False, diag=False, zero=True):
    if zero:
        hist, bin_edges = histogram(ys, bins=int(ys.size))
        c = bin_edges[hist.argmax()]
    
        ys -= c
    
    if diag:
        #plt.figure()
        plt.plot(xs, ys)
        
    heights = zeros(len(peaks))
    widths = zeros(len(peaks))
    centres = zeros(len(peaks))
    sums = zeros(len(peaks))
    for i in range(peaks.size):
        if i==0:
            lo = peaks[i] - (peaks[i+1] - peaks[i])/2
        else:
            lo = hi
        
        if i==peaks.size-1:
            hi = peaks[i] + (peaks[i] - peaks[i-1])/2
        else:
            hi = (peaks[i] + peaks[i+1])/2
            
        exs, wys = window_trace(xs, ys, lo, hi)
        
        # Mebbe?
        #wys2 = wys - wys.min()
        wys2 = wys
        
        middle = average(exs, weights=wys2)
        
        # Occasionally we get -'ve numbers in ys due to subtracting c.
        # Set these to 0 so the sqrt doesn't fail.
        wys2[wys2<0] = zeros((wys2<0).sum())
        if wys.sum()==0:
            # With zero particles the std shouldn't matter.
            # Setting to very small to avoid divide by zero later.
            std = 1e-12
        else:
            std = sqrt(average((exs-middle)**2, weights=wys2))
        
        max = wys.max()
        sum = wys.sum()
        
        heights[i] = max
        widths[i] = std
        centres[i] = middle
        sums[i] = sum
        if diag:
            colour = colors[(i+1) % len(colors)]
            plt.plot(exs, wys, color=colour)
            plt.axvline(middle, color=colour)
            plt.axvspan(middle-std/2, middle+std/2, color=colour, alpha=0.25)
            plt.text(10, 0.9*ys.max()*(heights.size-i)/heights.size,
                     "x={:.2f}\ns={:.2f}".format(middle, std), color=colour, va='top')
            
            if i==3:
                to_plot_middle = middle
                to_plot_xs = exs-middle
                to_plot_ys = wys
    
    if sum_peaks:
        return heights, centres, widths, sums
    else:
        return heights, centres, widths
    
def emittance(heights, centres, widths, number_holes=7, pitch=200e-6, \
              propagation_distance=propagation_distance, m_per_pixel=m_per_pixel, \
              num_particles=None, aperture_size=50e-6, adjust_for_aperture_size=True):
    if num_particles is None:
        num_particles = sqrt(2*pi) * heights * widths * m_per_pixel
        # Not actually number of particles. Number of camera counts.
        
    # Pepperpot hole locations.
    slit_positions = arange(0, heights.size * pitch * 0.999, pitch)
        # Occasionally returns one too many elements.
        # Probably due to binary accuracy. So: * 0.999
        
    # Take the middle hole as the 'centre'
    slit_positions -= slit_positions[int(floor(number_holes)/2)]
    mean_slit_position = average(slit_positions, weights=num_particles)
        
    middle = centres[int(floor(number_holes)/2)]
    
    beamlet_positions = m_per_pixel * (centres - middle)
        # Beamlet locations
    beamlet_widths = widths * m_per_pixel
    
    if adjust_for_aperture_size:
        # Account for non-negligible aperture size.
        # What width gaussian do I need to convolve with the aperture to get the 
        # current demagnified gaussian width?
        
        # Magnification
        with errstate(invalid='ignore'):
            # Suppress the divide by zero warning.
            magnification = nanmean(beamlet_positions / slit_positions)
        
        demagnified_beamlet_positions = beamlet_positions / magnification
        demagnified_beamlet_widths = beamlet_widths / magnification
        
        aperture_f = tophat_f(aperture_size, 0)
        x_range = demagnified_beamlet_widths.max()*10
        xs = linspace(-x_range/2, x_range/2, 1000)
        aperture_y = aperture_f(xs)
        
        corrected_widths = zeros(demagnified_beamlet_widths.size)
        for i, width in enumerate(demagnified_beamlet_widths):
            result_f = gaussian_f(width)
            result_y = result_f(xs)
            
            fit_func = lambda x, std: normalise(fftconvolve(gaussian_f(std)(x), aperture_y, mode='same'))
            
            fit = fitCurve(xs, result_y, fit_func, 4*width)
            
            fit_ys = fit_func(xs, *fit)
            
            if False:
                plt.figure()
                plt.plot(xs, result_y, label='result')
                plt.plot(xs, fit_ys, label='fit')
                plt.plot(xs, aperture_y, label='aperture')
                
                plt.legend()
                
                print()
                print('Demagnified width:', width)
                print('Fitted width:', fit[0])
                print()
            
            corrected_widths[i] = absolute(fit[0]) * magnification
            
        beamlet_widths = corrected_widths
        
    mean_divergence = (beamlet_positions - slit_positions) / propagation_distance
    mean_mean_divergence = average(mean_divergence, weights=num_particles)
    
    rms_divergence = beamlet_widths / propagation_distance
    
    a = (num_particles*(slit_positions - mean_slit_position)**2).sum()
    b = (num_particles*rms_divergence**2 + num_particles*(mean_divergence - mean_mean_divergence)**2).sum()
    c = ((num_particles*slit_positions*mean_divergence).sum() -\
            num_particles.sum()*mean_slit_position*mean_mean_divergence)
    
    em = sqrt((a*b-c**2) / num_particles.sum()**2)
    
    return em

def calc_beam_size(heights, centres, widths, sums=None, pepperpot_pitch=200e-6, \
                   m_per_pixel=m_per_pixel, return_centre=False, diag=False):
    if False:
        print('HAAAAAAAAAX')
        heights = heights[1:]
        centres = centres[1:]
        widths = widths[1:]
        #sums = sums[1:]
                   
    centres_m = centres * m_per_pixel
    widths_m = widths * m_per_pixel
    heights_e = heights / mcp_calibration
    
    num_particles = heights * widths * sqrt(2*pi) if sums is None else sums
    num_particles /= mcp_calibration
    positions = arange(num_particles.size, dtype='f')
    positions -= positions[int(floor(positions.size/2))]
    positions *= pepperpot_pitch
    
    middle = average(positions, weights=num_particles)
    std = sqrt((num_particles*(positions-middle)**2).sum()/num_particles.sum())
    
    fit_func = lambda x, height, centre, width, c: gaussian(x, height, centre, width, c)
    guess = (num_particles.max(), middle, std, 0)
    
    #middle = positions[heights.argmax()]
    #fit_func = lambda x, height, width, c: gaussian(x, height, middle, width, c)
    #guess = (num_particles.max(), std, 0)
    
    
    # What if i add y=0 to x=+/- inf
    ex = zeros(positions.size+2)
    wy = zeros(positions.size+2)
    ex[1:-1] = positions
    ex[0] = positions.min()*100
    ex[-1] = positions.max()*100
    wy[1:-1] = num_particles
    
    fit = fitCurve(ex, wy, fit_func, guess)
    #fit = fit[0], middle, fit[1], fit[2]
    
    rms_size_m = fit[2]
    
    total_beam_count = fit[0]*(rms_size_m**2)*2*pi/(m_per_pixel**2)
    
    centre = fit[1]
    
    if diag:
        xs = linspace(positions[0], positions[-1], 1000)
        ys = gaussian(xs, *fit)
        
        plt.plot(positions, num_particles, '-x')
        plt.plot(xs, ys)
        plt.axvline(centre, color='r')
        plt.xlabel('x (um)')
        
    
    if return_centre:
        return rms_size_m, total_beam_count, centre
    else:
        return rms_size_m, total_beam_count

def refine_gaussian_parameters(xs, line, heights, centres, widths, diag=False):
    params_array = sum(zip(heights, centres, widths), ())
        # Creates a flattened array of parameters, [h, c, w, h, c, w, h, c, w....]
        
    guess = params_array
    fit = fitCurve(xs, line, n_gaussians, guess)
    
    refined_heights = fit[0::3]
    refined_centres = fit[1::3]
    refined_widths = fit[2::3]
    
    partially_successful = False
    for i in range(heights.size):
        # If the new width is much bigger or smaller than the orginal then it's probably wrong.
        if refined_widths[i]>widths[i]*2 or refined_widths[i]<widths[i]/2:
            refined_widths[i] = widths[i]
            partially_successful = True
            
        # If the new height is much taller or shorter than the original thatn it's probably wrong.
        if refined_heights[i]>heights[i]*2 or refined_heights[i]<heights[i]/2:
            refined_heights[i] = heights[i]
            partially_successful = True
    
        # If the new centre is many orginal width away from the original centre then it's probably wrong.
        if abs(refined_centres[i]-centres[i])>widths[i]*3:
            refined_centres[i] = centres[i]
            partially_successful = True
    
    if diag:
        plt.plot(xs, line)
    
        #plt.plot(xs, n_gaussians(xs, *params_array), '--k')
    
        smooth_xs = linspace(xs.min(), xs.max(), 5000)
        
        fit_ys = n_gaussians(smooth_xs, *fit)
        plt.plot(smooth_xs, fit_ys, 'r')
        
        individual_ys = []
        for h, c, w in zip(refined_heights, refined_centres, refined_widths):
            ys = gaussian(smooth_xs, height=h, centre=c, width=w, offset=0)
            
            individual_ys.append(ys)
        
        if partially_successful:
            fit_ys = n_gaussians(smooth_xs, *guess)
            plt.plot(smooth_xs, fit_ys, ':k')
        else:
            ys = refined_heights * refined_widths * sqrt(2*pi)
            ys *= line.max()/ys.max()
            plt.plot(refined_centres, ys, 'rx')
            
        if False:
            # Paper figure. Overlapping beamlets. Used with Analyse 8.py at column 134.
            # Prepare matplotlib
            from matplotlib import rcParams
            
            rcParams.update({'font.size': 10})
            rcParams.update({'pgf.rcfonts': False})
            rcParams.update({'pgf.texsystem': 'pdflatex'})
            rcParams['font.family'] = 'serif'
            rcParams['axes.unicode_minus'] = False
            
            linewidth = 5.71 # inches

            figwidth = 0.95*linewidth
            figheight = figwidth/3*1.44
            figsize = (figwidth, figheight)

            plt.figure(figsize=figsize)
            
            xs *= m_per_pixel
            smooth_xs *= m_per_pixel
            
            xs -= 0.0168
            smooth_xs -= 0.0168
            
            plt.plot(xs*1e3, line/mcp_calibration, 'b')
            
            for ys in individual_ys:
                plt.plot(smooth_xs*1e3, (ys - line.max()*2/3)/mcp_calibration, 'r')
                
            plt.plot(smooth_xs*1e3, fit_ys/mcp_calibration, 'r:')
            
            plt.yticks([0, 0.05, 0.1, 0.15])
            
            plt.xlim((-0.01e3, 0.01e3))
            plt.xlabel('Detector Position (mm)')
            plt.ylabel('Electron Count')
            
            plt.tight_layout()
            
            plt.savefig('overlapping_gaussians.pgf')
            
            plt.show()
            exit()
            
    if partially_successful:
        print('\tRefined parameter search partially unsucessful returning orginals.')
        return heights, centres, widths
    else:
        return refined_heights, refined_centres, refined_widths
    
# Theory
def expected_emittance(excess_energy, beam_radius):
    T = 2*excess_energy*eV / (2 * kB)
    return beam_radius * sqrt(kB*T/(m_e*speed_of_light**2))
    
def temperature_from_expected_emittance(emittance, beam_radius):
    T = (m_e*speed_of_light**2) * (emittance/beam_radius)**2 / kB
    return T
    
def excess_energy_from_wavelength(blue_wavelength, field_ionisation=False, \
                                  electric_field=electric_field):
    red_energy = Plank*speed_of_light/red_wavelength
    
    blue_energy = Plank*speed_of_light/blue_wavelength
    
    field_energy = Plank*speed_of_light*4*Ry*sqrt(electric_field/electric_field_atomic_unit)
    
    if field_ionisation:
        excess_energy = (red_energy + blue_energy) - Rb_ionisation_energy + field_energy
    else:
        excess_energy = (red_energy + blue_energy) - Rb_ionisation_energy
        
    return excess_energy