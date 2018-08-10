##### Import #####
from numpy import sqrt, linspace, zeros, nansum, errstate, \
                  isfinite, absolute, arange, array, meshgrid, \
                  arctan2, cos, sin
from numpy.random import randn
from matplotlib import pyplot as plt, cm
from scipy.constants import pi, h, m_e, e, epsilon_0
from warnings import catch_warnings, filterwarnings
from multiprocessing import cpu_count, Pool
from functools import partial
from h5py import File

from MagneticFields import quadrupole, octupole, quadrupole2

##### Constants #####
coulomb_factor = 4*pi*epsilon_0 / e**2

##### Objects #####
class ElectronBunch:
    def __init__(self, n, energy, radius_x=1, radius_y=1, emittance=None,
                 blank=False):
        if not blank:
            self.setEnergy(energy)

            self.electrons = randn(n, 4)

            self.electrons[:, 0] = self.electrons[:, 0] * radius_x
            self.electrons[:, 1] = self.electrons[:, 1] * radius_y

            if emittance is None:
                self.electrons[:, 2:] = zeros((n, 2))
            else:
                self._setEmittance(emittance)

    def copy(self):
        new_bunch = ElectronBunch(0, 0)

        new_bunch.electrons = self.electrons.copy()
        new_bunch.setEnergy(self.getEnergy())

        return new_bunch

    def __iter__(self):
        return self.electrons.__iter__()

    def next(self):
        self.electrons.next()

    def getXs(self):
        return self.electrons[:, 0]

    def getYs(self):
        return self.electrons[:, 1]

    def getVXs(self):
        return self.electrons[:, 2]

    def getVYs(self):
        return self.electrons[:, 3]

    def getXPrimes(self):
        return self.getVXs() / self.getSpeed()

    def getYPrimes(self):
        return self.getVYs() / self.getSpeed()

    def setEnergy(self, energy):
        self.energy = energy

    def getEnergy(self):
        return self.energy

    def getSpeed(self):
        return sqrt(2*self.energy/m_e)

    def getWavelength(self):
        return h/(m_e*self.getSpeed())

    def getCentre(self):
        return self.getXs().mean(), self.getYs().mean()

    def getBunchWidth(self):
        return 2 * self.getXs().std(), 2 * self.getYs().std()

    def getEmittance(self):
        """Get the RMS Emittance of the bunch."""
        x = self.getXs()
        y = self.getYs()

        xp = self.getXPrimes()
        yp = self.getYPrimes()

        rms_emittance_x = sqrt((x**2).mean()*(xp**2).mean() - (x*xp).mean()**2)
        rms_emittance_y = sqrt((y**2).mean()*(yp**2).mean() - (y*yp).mean()**2)

        return rms_emittance_x, rms_emittance_y

    def setEmittance(self, emittance):
        self.electrons[:, 2:] = randn(self.electrons.shape[0], 2)

        self._setEmittance(emittance)

    def _setEmittance(self, emittance):
        sigma_x, sigma_y = self.getXs().std(), self.getYs().std()

        # xp = x' = vx/vz
        # emittance = sqrt(det(sigma_matrix)) = sqrt(sig11*sig22-sig12**2)
        # sigma11=std_x**2, sigma22=std_y**2
        # sig12 = coupling between x and x'.
        # If we assume zero coupling then...
        sigma_xp = emittance/sigma_x
        sigma_yp = emittance/sigma_y

        # Assume that Vx, Vy are from the standard normal distribution.
        self.electrons[:, 2] *= sigma_xp * self.getSpeed()
        self.electrons[:, 3] *= sigma_yp * self.getSpeed()
        #   This assumes that emittance is geometric emittance and thus
        #   that sigma_p is from x' = dx/dz = v_x/v_z.

    def propagate(self, dz, slices, B_field_func=None, multithread=False, z=0):
        dt = (dz/slices)/self.getSpeed()
        for _ in range(slices):
            new_electrons = zeros(self.electrons.shape)
            if multithread:
                func = partial(_propagate, speed=self.getSpeed(),
                               Xs=self.getXs(), Ys=self.getYs(),
                               dt=dt, B_field_func=B_field_func, z=z)

                with Pool(processes=cpu_count()) as workers:
                    l = self.electrons.tolist()

                    new_electrons = \
                        array(workers.map(func, l))

            else:
                for i, tron in enumerate(self.electrons):
                    x, y, v_x, v_y = _propagate(tron, self.getSpeed(),
                                                self.getXs(), self.getYs(),
                                                dt, B_field_func, z)

                    new_electrons[i, 0] = x
                    new_electrons[i, 1] = y
                    new_electrons[i, 2] = v_x
                    new_electrons[i, 3] = v_y

            self.electrons[:, :] = new_electrons[:, :]

    def plot(self, figname=None, color=None, quiver=False):
        plt.figure(figname)

        if quiver:
            plt.quiver(self.getXs(), self.getYs(),
                       self.getVXs(), self.getVYs(),
                       color=color)
        else:
            plt.plot(self.getXs(), self.getYs(), 'x', c=color)

        xlims = plt.xlim()
        ylims = plt.ylim()

        square = max(abs(xlims[0]), abs(xlims[1]),
                     abs(ylims[0]), abs(ylims[1]))

        plt.xlim((-square, square))
        plt.ylim((-square, square))

    def plot_phasespace(self, figname=None, color=None):
        plt.figure(figname)

        # Plot X and Px
        plt.subplot(1, 2, 1)

        plt.title('X')
        plt.xlabel('X')
        plt.ylabel('Px')

        plt.plot(self.getXs(), self.getVXs()*m_e, 'x', c=color)

        # Plot Y and Py
        plt.subplot(1, 2, 2)

        plt.title('Y')
        plt.xlabel('Y')
        plt.ylabel('Py')

        plt.plot(self.getYs(), self.getVYs()*m_e, 'x', c=color)

    def plot_distr_phase(self, figname=None, color=None):
        plt.figure(figname)

        # Plot the particle distribution.
        plt.subplot(2, 2, 1)

        plt.title('Electron Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.plot(self.getXs(), self.getYs(), ',')

        # Plot the Y phase space.
        plt.subplot(2, 2, 2)

        plt.title('Y Phase Space')
        plt.xlabel('Y')
        plt.ylabel('Py')

        plt.plot(self.getYs(), self.getVYs()*m_e, ',', c=color)

        # Plot the X phase space
        plt.subplot(2, 2, 3)

        plt.title('X Phase Space')
        plt.xlabel('X')
        plt.ylabel('Px')

        plt.plot(self.getXs(), self.getVXs()*m_e, ',', c=color)

    def __str__(self):
        return str(self.electrons)

##### Functions #####
def _null_B_field(x, y, z):
    return 0, 0, 0

def _propagate(electron, speed, Xs, Ys, dt, B_field_func, z):
    if B_field_func==None:
        B_field_func = _null_B_field

    x, y = electron[0], electron[1]
    v_x, v_y, v_z = electron[2], electron[3], speed
    B_x, B_y, B_z = B_field_func(electron[0], electron[1], z)

    if False:
        # Self interaction
        dxs = x - Xs
        dys = y - Ys

        denominator = sqrt(dxs**2+dys**2)**3 * coulomb_factor

        # When we get to the electron of interest we'll have a divide by zero.
        with catch_warnings():
            filterwarnings("ignore", category=RuntimeWarning)

            F_x = dxs / denominator
            F_y = dys / denominator

        F_x = F_x[isfinite(F_x)].sum()
        F_y = F_y[isfinite(F_y)].sum()
    else:
        F_x = 0
        F_y = 0

    F_x += -e*(v_y*B_z - v_z*B_y)
    F_y += -e*(v_z*B_x - v_x*B_z)

    dV_x = dt * F_x / m_e
    dV_y = dt * F_y / m_e

    dx = v_x*dt
    dy = v_y*dt

    return x + dx, y + dy, v_x + dV_x, v_y + dV_y

def set_bunch_speeds(trons, s_x=7277.6, s_y=7928.6):
    # Give them speeds so that they'll focus
    # Speed should be relative to r
    width_x, width_y = trons.getBunchWidth()

    for i in range(trons.electrons.shape[0]):
        electron = trons.electrons[i]

        r = sqrt(electron[0]**2 + electron[1]**2)

        r_factor = r/edge_of_bunch

        # Needs to be directed towards 0,0 which happens to be opposite to
        # the electron coordinates.
        electron[2] = -s_x * abs(electron[0]/width_x) * electron[0] / r
        electron[3] = -s_y * abs(electron[1]/width_y) * electron[1] / r

def set_noisey_bunch_speeds(trons, std_speed=10000, astigmatism=None):
    if astigmatism is None:
        trons.electrons[:, 2:] = randn(trons.electrons.shape[0], 2) * std_speed
    else:
        trons.electrons[:, 2] = randn(trons.electrons.shape[0]) * std_speed
        trons.electrons[:, 3] = randn(trons.electrons.shape[0]) * std_speed * astigmatism

def plot_field(field_func, radius, z=0, figname=None):
    n = 30
    field_factor = 4
    xs = linspace(-radius*field_factor, radius*field_factor, n)
    ys = linspace(-radius*field_factor, radius*field_factor, n)

    zs = array([0])

    field_x, field_y, field_z = \
        zeros((zs.size, ys.size, xs.size)), \
        zeros((zs.size, ys.size, xs.size)), \
        zeros((zs.size, ys.size, xs.size))
    for z in range(zs.size):
        for y in range(ys.size):
            for x in range(xs.size):



                B_x, B_y, B_z = field_func(xs[x], ys[y], zs[z])

                field_x[z, y, x] = B_x
                field_y[z, y, x] = B_y
                field_z[z, y, x] = B_z

    total_magnitude = sqrt(field_x**2 + field_y**2, field_z**2)

    # Plot transverse cross-section
    z_zero = absolute(zs - 0).argmin()

    horizontal = field_x[z_zero,:,:]
    vertical = field_y[z_zero,:,:]

    if figname==None:
        plt.figure('z=' + str(z))
    else:
        plt.figure(figname)

    magnitude = sqrt(horizontal**2 + vertical**2)
    plt.quiver(xs, ys,
        horizontal/magnitude,
        vertical/magnitude,
        magnitude,
        cmap='inferno')
    plt.xlim((xs.min()*(n+1)/n, xs.max()*(n+1)/n))
    plt.ylim((ys.min()*(n+1)/n, ys.max()*(n+1)/n))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.gca().add_artist(plt.Circle((0, 0), radius, fill=False, color='r'))

    plt.axes().set_aspect('equal', 'box')

def simple_lens(bunch, strength, second_order=0, forth_order=0, sqrt_order=0):
    rs = sqrt(bunch.getXs()**2 + bunch.getYs()**2)
    angles = arctan2(bunch.getYs(), bunch.getXs())
    
    affects = strength * rs + second_order*rs**2 + forth_order*rs**4
    
    bunch.electrons[:, 2] += affects * cos(angles) 
    bunch.electrons[:, 3] += affects * sin(angles)
    
def set_dataset(hdf, key, data):
    if key in hdf:
        del hdf[key]
        
    return hdf.create_dataset(key, data=data)
    
##### Script #####
if __name__=='__main__':
    if False:
        # Looking at performance with different geometry.
        edge_of_bunch_x = 0.005
        edge_of_bunch_y = 0.005
        initial_trons = ElectronBunch(100, 17.1*e/2, edge_of_bunch_x, edge_of_bunch_y)

        set_noisey_bunch_speeds(initial_trons)

        min_stretch = 1
        max_stretch = 5
        n_stretches = 5
        stretches = array([0.1, 5])#linspace(min_stretch, max_stretch, n_stretches)

        min_width = 0.05
        max_width = 0.3
        n_widths = 1
        widths = array([0.015])#linspace(min_width, max_width, n_widths)

        min_current = 1
        max_current = 100
        n_currents = 5
        currents = linspace(min_current, max_current, n_currents)

        B_start = 0.2

        min_z = 0 - B_start
        max_z = 1 - B_start
        n_zs = 100
        zs = linspace(min_z, max_z, n_zs)

        data_stretch = []
        count = 1
        for m, stretch in enumerate(stretches):
            data = []
            for k, width in enumerate(widths):
                data_width = []

                current_mins = zeros(n_currents)
                current_mins_current = zeros(n_currents)
                for j, current in enumerate(currents):
                    trons = initial_trons.copy()

                    # Propagate through the octupole lens.
                    octupole_radius = 0.08


                    solenoid_transverse_radius = width*stretch
                    solenoid_longitudinal_radius = width
                    B_func = quadrupole2(current=current, inner_radius=octupole_radius,
                        solenoid_transverse_radius=solenoid_transverse_radius,
                        solenoid_longitudinal_radius=solenoid_longitudinal_radius,
                        solenoid_turns=1, solenoid_length=.01,
                        angle=pi/4)


                    B_f = lambda x, y, z: B_func(x, y, z) \
                        if abs(z)<width*4 else (0, 0, 0)

                    #plot_field(B_func, octupole_radius, z=B_start)

                    width_xs = zeros(n_zs)
                    width_ys = zeros(n_zs)
                    for i, z in enumerate(zs):
                        print('{:} of {:}'.format(
                            count, n_stretches*n_widths*n_currents*n_zs))
                        print('\t\t', solenoid_transverse_radius)
                        count += 1

                        trons.propagate(zs[1]-zs[0], 2, multithread=False,
                            B_field_func=B_f, z=z)

                        w_x, w_y = trons.getBunchWidth()

                        width_xs[i] = w_x
                        width_ys[i] = w_y

                    data_width.append((width_xs, width_ys))

                data.append(data_width)

            #plot_field(B_func, octupole_radius, z=B_start, figname=str(stretch))
            data_stretch.append(data)

        with File('electron lens datam mk2.h5') as hdf:
            for k, stretch in enumerate(stretches):
                stretch_group = hdf.require_group('s ' + str(stretch))
                for i, width in enumerate(widths):
                    width_group = stretch_group.require_group('w ' + str(width))
                    for j, current in enumerate(currents):
                        current_group = width_group.require_group('I ' + str(current))

                        name = 'W ' + str(width) + ' I ' + str(current)

                        z_group = current_group.require_group(name)

                        if 'x' in z_group:
                            del z_group['x']
                        if 'y' in z_group:
                            del z_group['y']
                        if 'z' in z_group:
                            del z_group['z']

                        z_group.create_dataset('x', data=data[i][j][0])
                        z_group.create_dataset('y', data=data[i][j][1])
                        z_group.create_dataset('z', data=zs)
    else:
        # Take 2
        zs_key = 'zs'
        currents_key = 'quadrupole currents'
        logitudinal_width_key = 'longitudinal width'
        stretch_key = 'stretch'
        widths_key = 'beam widths'
        
        #hdf_name =AR 'QuadrupoleSims.h5'
        #hdf_name = 'QuadrupoleSims-Long2.h5'
        #hdf_name = 'QuadrupoleSims-Long3.h5'
        #hdf_name = 'QuadrupoleSims-Long4.h5'
        hdf_name = 'QuadrupoleSims-Long5.h5'
        with File(hdf_name) as hdf:
            if False:
                # RUN THE SIM
                
                # Create bunch
                edge_of_bunch_x = 0.005
                edge_of_bunch_y = 0.005
                initial_trons = ElectronBunch(100, 17.1*e/2, edge_of_bunch_x, edge_of_bunch_y)

                # Make the bunch noisey and astigmatic. (1 = no astigmatism)
                set_noisey_bunch_speeds(initial_trons, astigmatism=2)
                
                # Z axis
                min_z = 0
                max_z = 1
                n_zs = 2000
                zs = linspace(min_z, max_z, n_zs)
                
                # Focusing lens
                lens_strength = -6e6
                focusing_lens_z = (min_z+max_z)/2
                focused = False
                
                # QUADRUPOLE LENS PARAMETERS
                quad_lens_z = (min_z + focusing_lens_z) /2
                
                quadrupole_radius = 0.045 # radius from centre of electron beam to solenoids
                
                # Current through solenoids.
                min_current = 10
                max_current = 30
                n_currents = 20
                quad_currents = linspace(min_current, max_current, n_currents)
                
                # Longitudinal width/depth of solenoids
                min_width = 0.010
                max_width = 0.010
                n_widths = 1
                solenoid_longitudinal_radiuss = array([0.015])#linspace(min_width, max_width, n_widths)
                
                # Ellipticity of solenoids. Transverse width = depth * stretch
                min_stretch = 1
                max_stretch = 7
                n_stretches = 14
                stretchs = linspace(min_stretch, max_stretch, n_stretches)
                
                # Widths of the beam
                widths = zeros((quad_currents.size, solenoid_longitudinal_radiuss.size, \
                                stretchs.size, zs.size, 2))
                
                # HDF stuff.
                set_dataset(hdf, zs_key, zs)
                set_dataset(hdf, currents_key, quad_currents)
                set_dataset(hdf, logitudinal_width_key, solenoid_longitudinal_radiuss)
                set_dataset(hdf, stretch_key, stretchs)
                widths = set_dataset(hdf, widths_key, widths)
                
                # Simulate
                number_of_calcs = quad_currents.size * solenoid_longitudinal_radiuss.size * \
                                  stretchs.size * zs.size
                count = 0
                for i, quad_current in enumerate(quad_currents):
                    for j, solenoid_longitudinal_radius in enumerate(solenoid_longitudinal_radiuss):
                        for k, stretch in enumerate(stretchs):
                            solenoid_transverse_radius = solenoid_longitudinal_radius * stretch
                            
                            # Quadrupole lens
                            B_func = quadrupole2(current=quad_current, inner_radius=quadrupole_radius,
                                            solenoid_transverse_radius=solenoid_transverse_radius,
                                            solenoid_longitudinal_radius=solenoid_longitudinal_radius,
                                            solenoid_turns=10, solenoid_length=.01,
                                            angle=pi/4)
                 
                            # Only have quad field near the quadrupole
                            B_f = lambda x, y, z: B_func(x, y, z) \
                                if abs(z-quad_lens_z)<solenoid_longitudinal_radius*2 \
                                        else (0, 0, 0)
                
                            # Propagate the bunch along z.
                            trons = initial_trons.copy()
                            
                            width_xs = zeros(n_zs)
                            width_ys = zeros(n_zs)
                            focused = False
                            for m, z in enumerate(zs):
                                print('{:} of {:}'.format(count, number_of_calcs))
                                count += 1

                                if not focused and z>focusing_lens_z:
                                    simple_lens(trons, lens_strength)
                                    focused = True
                                    
                                trons.propagate(zs[1]-zs[0], 2, multithread=False,
                                    B_field_func=B_f, z=z)

                                w_x, w_y = trons.getBunchWidth()

                                width_xs[m] = w_x
                                width_ys[m] = w_y
                                
                                widths[i, j, k, m, 0] = w_x
                                widths[i, j, k, m, 1] = w_y
                    
                            plt.figure('I: {:.1f}A, D: {:.1f}mm, W: {:.1f}mm'.format(quad_current, solenoid_longitudinal_radius*1e3, solenoid_transverse_radius*1e3))
                            
                            plt.plot(zs, width_xs)
                            plt.plot(zs, width_ys)
                            
                            plt.axvline(focusing_lens_z, color='k', ls=':')
                            plt.xlabel('z')
                            plt.ylabel('widths')
                            
            else:
                # PLOT STUFF
                currents = array(hdf[currents_key])
                logitudinal_widths = array(hdf[logitudinal_width_key])
                stretchs = array(hdf[stretch_key])
                zs = array(hdf[zs_key])
                
                widths = array(hdf[widths_key])
                dz = zs[1]-zs[0]
                
                # For each stretch factor, find the current with the lowest astigmatism.
                # Lowest astigmatism is closest minima.
                cmap = cm.get_cmap('inferno')
                best_currents = zeros(stretchs.size, dtype='int')
                dzs = zeros(stretchs.size, dtype='int')
                waist_xs = zeros(stretchs.size)
                waist_ys = zeros(stretchs.size)
                for k, stretch in enumerate(stretchs):
                    diffs = zeros(currents.size)
                    for i, current in enumerate(currents):
                        min_x_z = widths[i,0,k,:,0].argmin()
                        min_y_z = widths[i,0,k,:,1].argmin()
                        
                        diffs[i] = min_x_z - min_y_z
                        
                    best_currents[k] = absolute(diffs).argmin()
                    dzs[k] = absolute(diffs).min()
                    
                    #plt.figure('Stretch {:.1f}'.format(stretch))
                    xs = widths[best_currents[k], 0, k, :, 0]
                    ys = widths[best_currents[k], 0, k, :, 1]
                    colour = cmap(k/stretchs.size)
                    p = plt.plot(zs, xs, '-x', color=colour)
                    plt.plot(zs, ys, '-x', color=p[-1].get_color())
                    
                    plt.axvline(zs[xs.argmin()], color=p[-1].get_color())
                    plt.axvline(zs[ys.argmin()], color=p[-1].get_color())
                            
                    waist_x = xs.min()
                    waist_y = ys.min()
                    
                    waist_x_z = zs[xs.argmin()]
                    waist_y_z = zs[ys.argmin()]
                    
                    waist_xs[k] = waist_x
                    waist_ys[k] = waist_y
                    
                    print('Factor:', stretch)
                    print("Waist X: {:.2f}mm".format(1e3*waist_x))
                    print("Waist Y: {:.2f}mm".format(1e3*waist_y))
                    
                    print("Waist X, z: {:.2f}mm".format(1e3*waist_x_z))
                    print("Waist Y, z: {:.2f}mm".format(1e3*waist_y_z))
                    print()
                    
                p = plt.plot(zs, widths[0, 0, 0, :, 0], '-o')
                plt.plot(zs, widths[0, 0, 0, :, 1], '-o', color=p[-1].get_color())
                
                width = logitudinal_widths[0]
                
                from Thesis import thesis_init, linewidth, colours
                
                thesis_init()
                
                figwidth = linewidth
                figheight = linewidth/3
                
                plt.figure(figsize=(figwidth, figheight))
                num_rows = 1
                num_cols = 2
                
                plt.subplot(num_rows, num_cols, 1)
                #plt.title('Waist Separation')
                plt.plot(stretchs*width*1e3, 1e3*dzs*dz, 'x-', color=colours[0])
                plt.xlabel('Solenoid Transverse Radius (mm)')
                plt.ylabel('Waist Separation (mm)')
                
                plt.subplot(num_rows, num_cols, 2)
                #plt.title('Current Turns')
                plt.plot(stretchs*width*1e3, best_currents, 'x-', color=colours[0])
                plt.xlabel('Solenoid Transverse Radius (mm)')
                plt.ylabel('Current Turns (A turns)')
                
                if False:
                    plt.subplot(num_rows, num_cols, 3)
                    plt.title('Beam Waist')
                    plt.plot(stretchs*width*1e3, waist_xs*1e3, 'x-', label='x')
                    plt.plot(stretchs*width*1e3, waist_ys*1e3, 'x-', label='y')
                    
                    plt.xlabel('Solenoid Transverse Radius (mm)')
                    plt.ylabel('Beam Waist (mm)')
                    
                plt.tight_layout()
                
                plt.savefig('quad_sims.pgf')
                
                
    plt.show()
