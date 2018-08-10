##### Imports #####
from numpy import sqrt, sin, cos, arctan2, arange, meshgrid, linspace, degrees, zeros, absolute, array, matrix
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import mu_0, pi
from scipy.special import agm
from scipy.integrate import quad
from functools import partial

##### Functions #####
def wire(current, loc_x, loc_y, plot=False, color='r'):
    if plot:
        plt.plot(loc_x, loc_y, 'o' if current>0 else 'x', color=color)

    def func(x, y):
        B_tangential = mu_0*current / (2*pi*sqrt((x-loc_x)**2 + (y-loc_y)**2))
        
        B_x = -B_tangential*sin(arctan2(y-loc_y, x-loc_x))
        B_y = B_tangential*cos(arctan2(y-loc_y, x-loc_x))
        
        return B_x, B_y, 0

    return func

def solenoid(current, loc_x, loc_y, angle, width, length, turns, plot=False, color='r'):
    locs = zeros((turns*2, 2))
    
    x_positions = linspace(-length/2, length/2, turns) if turns>1 else [0]
    for i, ex in enumerate(x_positions):
        wy = width/2
        
        locs[i][0] = ex
        locs[i][1] = wy
        
        wy = -width/2
        
        locs[i+turns][0] = ex
        locs[i+turns][1] = wy
        
    
    for i, (ex, wy) in enumerate(locs):
        locs[i][0] = ex*cos(angle) - wy*sin(angle) + loc_x
        locs[i][1] = ex*sin(angle) + wy*cos(angle) + loc_y

    funcs = []
    for i in range(turns):
        funcs.append(wire(current, locs[i][0], locs[i][1], plot, color=color))
        funcs.append(wire(-current, locs[i+turns][0], locs[i+turns][1], plot, color=color))
        
        
    def func(x, y):
        return map(sum, zip(*[f(x, y) for f in funcs]))
        
    return func

def quadrupole(current, loc_x, loc_y, radius, turns, width=None, angle=0, astig=1, plot=False, color='r', octupole=False):
    # astig is x current:y current, so astig 2 give twice the current in x
    # x, y, angle, current
    
    # If octupole==True then we do not flip the current for half the coils since this
    # function is being used to construct an octupole.
    current2 = current if octupole else -current
    
    locs = [[radius*cos(angle+0*pi/2)+loc_x,          radius*sin(angle+0*pi/2)+loc_y,
                    0 + angle, current],
            [radius*cos(angle+1*pi/2)+loc_x,          radius*sin(angle+1*pi/2)+loc_y,
                    2*pi/4 + angle, current2],
            [radius*cos(angle+2*pi/2)+loc_x,          radius*sin(angle+2*pi/2)+loc_y,
                    4*pi/4 + angle, current],
            [radius*cos(angle+3*pi/2)+loc_x,          radius*sin(angle+3*pi/2)+loc_y,
                    6*pi/4 + angle, current2],]
             
    if width==None:
        width = 2*pi*radius/8

    funcs = []
    for x, y, angle, current in locs:
        funcs.append(solenoid(current, x, y, angle, width, radius/4, turns, plot=plot, color=color))

    def func(x, y, z):
        return map(sum, zip(*[f(x, y) for f in funcs]))
        
    return func

def octupole(current, loc_x, loc_y, radius, turns, width=None, angle=0, astig=1, plot=False, color='r'):
    funcs = [
            quadrupole(current, loc_x, loc_y, radius, turns,
                width=width, angle=angle, astig=astig, plot=plot,
                color=color, octupole=True),
            quadrupole(-current, loc_x, loc_y, radius, turns,
                width=width, angle=angle+pi/4, astig=astig, plot=plot,
                color=color, octupole=True)
            ]
                
    def func(x, y):
        return map(sum, zip(*[f(x, y) for f in funcs]))
        
    return func

def circular_current_loop(current, radius, x0=0, y0=0, z0=0,
                    thetaX=0, thetaY=0, thetaZ=0, rotation_matrix=None):
    return elliptical_current_loop(current, radius, radius,
                                    x0=x0, y0=y0, z0=z0,
                                    thetaX=thetaX, thetaY=thetaY, thetaZ=thetaZ,
                                    rotation_matrix=rotation_matrix)

def elliptical_current_loop(current, radius_1, radius_2, x0=0, y0=0, z0=0,
                            thetaX=0, thetaY=0, thetaZ=0, rotation_matrix=None):
    # Inspired by http://www.physicspages.com/2013/04/18/magnetic-field-of-current-loop-off-axis-field/
    if rotation_matrix is None:
        Rx = matrix([[1,           0,            0],
                     [0, cos(thetaX), -sin(thetaX)],
                     [0, sin(thetaX),  cos(thetaX)]])
        
        Ry = matrix([[ cos(thetaY), 0, sin(thetaY)],
                     [           0, 1,           0],
                     [-sin(thetaY), 0, cos(thetaY)]])
        
        Rz = matrix([[cos(thetaZ), -sin(thetaZ), 0],
                     [sin(thetaZ),  cos(thetaZ), 0],
                     [          0,            0, 1]])

        rotation_matrix = Rx * Ry * Rz
    
    def func(x, y, z):
        # Centred at x0, y0, z0
        ex = x - x0
        wy = y - y0
        zd = z - z0

        ex, wy, zd = (rotation_matrix * matrix([ex, wy, zd]).T).flat
        
        # Current loop located at 0, 0, 0 in the x, y, plane.
        
        denominator = lambda phi: ( ex**2 + wy**2 + zd**2  \
                                   +radius_1**2*cos(phi)**2\
                                   +radius_2**2*sin(phi)**2\
                                   -2*ex*radius_1*cos(phi) \
                                   -2*wy*radius_2*sin(phi) )**1.5

        f_x = lambda phi: zd*radius_2*cos(phi) / denominator(phi)
        f_y = lambda phi: zd*radius_1*sin(phi) / denominator(phi)
        f_z = lambda phi: ( radius_1*radius_2 \
                           -ex*radius_2*cos(phi) - wy*radius_1*sin(phi) ) \
                                                / denominator(phi)
        
        B_x = mu_0*current / (4*pi) * quad(f_x, 0, 2*pi)[0]
        B_y = mu_0*current / (4*pi) * quad(f_y, 0, 2*pi)[0]
        B_z = mu_0*current / (4*pi) * quad(f_z, 0, 2*pi)[0]
        
        # Undo rotation
        B_x, B_y, B_z = (rotation_matrix.I*matrix([B_x, B_y, B_z]).T).flat
            
        return B_x, B_y, B_z
        
    return func
    
def current_loop(current, radius,
                    x0=0, y0=0, z0=0,
                    thetaX=0, thetaY=0, thetaZ=0, rotation_matrix=None):
    # Inspired by http://www.physicspages.com/2013/04/18/magnetic-field-of-current-loop-off-axis-field/
    if rotation_matrix is None:
        Rx = matrix([[1,           0,            0],
                     [0, cos(thetaX), -sin(thetaX)],
                     [0, sin(thetaX),  cos(thetaX)]])
        
        Ry = matrix([[ cos(thetaY), 0, sin(thetaY)],
                     [           0, 1,           0],
                     [-sin(thetaY), 0, cos(thetaY)]])
        
        Rz = matrix([[cos(thetaZ), -sin(thetaZ), 0],
                     [sin(thetaZ),  cos(thetaZ), 0],
                     [          0,            0, 1]])

        rotation_matrix = Rx * Ry * Rz
    
    def func(x, y, z):
        # Centred at x0, y0, z0
        ex = x - x0
        wy = y - y0
        zd = z - z0

        ex, wy, zd = (rotation_matrix * matrix([ex, wy, zd]).T).flat
        
        # Current loop located at 0, 0, 0 in the x, y, plane.
        denominator = lambda phi: ( radius**2 + \
                                    ex**2 + wy**2 + zd**2 \
                                    - 2*wy*radius*sin(phi) \
                                    - 2*ex*radius*cos(phi) )**1.5

        f_x = lambda phi: zd*cos(phi) / denominator(phi)
        f_y = lambda phi: zd*sin(phi) / denominator(phi)
        f_z = lambda phi: (radius - ex*cos(phi) - wy*sin(phi)) / denominator(phi)
        
        B_x = mu_0*current*radius / (4*pi) * quad(f_x, 0, 2*pi)[0]
        B_y = mu_0*current*radius / (4*pi) * quad(f_y, 0, 2*pi)[0]
        B_z = mu_0*current*radius / (4*pi) * quad(f_z, 0, 2*pi)[0]
        
        # Undo rotation
        B_x, B_y, B_z = (rotation_matrix.I*matrix([B_x, B_y, B_z]).T).flat
            
        return B_x, B_y, B_z
        
    return func

def solenoid2(current, inner_radius_1, inner_radius_2, length, turns_per_layer,
                    x0=0, y0=0, z0=0,
                    thetaX=0, thetaY=0, thetaZ=0, rotation_matrix=None):
    if rotation_matrix is None:
        Rx = matrix([[1,           0,            0],
                     [0, cos(thetaX), -sin(thetaX)],
                     [0, sin(thetaX),  cos(thetaX)]])
        
        Ry = matrix([[ cos(thetaY), 0, sin(thetaY)],
                     [           0, 1,           0],
                     [-sin(thetaY), 0, cos(thetaY)]])
        
        Rz = matrix([[cos(thetaZ), -sin(thetaZ), 0],
                     [sin(thetaZ),  cos(thetaZ), 0],
                     [          0,            0, 1]])

        rotation_matrix = Rz * Rx * Ry
    
    wire_diameter = length/(turns_per_layer)
    funcs = []
    
    for turn in range(turns_per_layer):
        offset = ((turn+0.5)/turns_per_layer - 0.5)*length
        
        offset_x, offset_y, offset_z = (rotation_matrix * matrix([0, 0, offset]).T).flat
        
        ex0 = x0 + offset_x
        wy0 = y0 + offset_y
        zd0 = z0 + offset_z

        solenoid_transverse_radius = inner_radius_1
        solenoid_longitudinal_radius = inner_radius_2
        
        loop_func = elliptical_current_loop(current,
                    solenoid_longitudinal_radius, solenoid_transverse_radius,
                    x0=ex0, y0=wy0, z0=zd0,
                    thetaX=thetaX, thetaY=thetaY, thetaZ=thetaZ)
    
        funcs.append(loop_func)
    
    def func(x, y, z):
        return map(sum, zip(*[f(x, y, z) for f in funcs]))
    
    return func

def quadrupole2(current, inner_radius,
                solenoid_transverse_radius, solenoid_longitudinal_radius,
                solenoid_turns, solenoid_length,
                x0=0, y0=0, z0=0, angle=0):
    print('Ocutpole Radius:', inner_radius,
        '\nSolenoid (transverse, long):', solenoid_transverse_radius, solenoid_longitudinal_radius,
        '\nSolenoid (turns, length):', solenoid_turns, solenoid_length,
        '\nx, y, z:', x0, y0, z0)
    funcs = []
    
    if solenoid_turns==1:
        R = inner_radius
    else:
        R = inner_radius+solenoid_length/2
        
    for i in range(4):
        x = R*cos(i*pi/2 + angle) - x0
        y = R*sin(i*pi/2 + angle) - y0
        z = -z0
        
        I = current #if i%2==0 else -current
        funcs.append(solenoid2(I,
                        solenoid_transverse_radius, solenoid_longitudinal_radius,
                        solenoid_length, solenoid_turns,
                        x, y, z, 0, pi/2, i*pi/2 - angle))

    def func(x, y, z):
        return map(sum, zip(*[f(x, y, z) for f in funcs]))
    
    return func

##### Script #####
if __name__ == '__main__':
    n = 20
    xs = linspace(-10, 10, n)
    ys = linspace(-10, 10, n)
    #xs = linspace(-0.08*4, 0.08*4, n)
    #ys = linspace(-0.08*4, 0.08*4, n)
    zs = array([0.2])#linspace(-10, 10, n)
    
    xx, yy = meshgrid(xs, ys, sparse=True)
    
    if True:
        # 2D
        field_x, field_y = zeros((ys.size, xs.size)), zeros((ys.size, xs.size))
    
        if False:
            field_x, field_y = wire(2, 0, 0, True)(xx, yy)
            
        if False:
            for ex in range(-8, 9, 2):
                for wy in [-2, 2]:
                    f_x, f_y = wire(wy, ex, wy, True)(xx, yy)
                    
                    field_x += f_x
                    field_y += f_y
        
        if False:
            field_x, field_y = solenoid(3, 0, 0, 0, 3, 3, 10, True)(xx, yy)

        if False:
            f_x, f_y = solenoid(3, 5, 0, 0, 3, 3, 10, True)(xx, yy)
            field_x += f_x
            field_y += f_y
            
            f_x, f_y = solenoid(-3, 0, 5, pi/2, 3, 3, 10, True)(xx, yy)
            field_x += f_x
            field_y += f_y
            
            f_x, f_y = solenoid(3, -5, 0, pi, 3, 3, 10, True)(xx, yy)
            field_x += f_x
            field_y += f_y
            
            f_x, f_y = solenoid(-3, 0, -5, 3*pi/2, 3, 3, 10, True)(xx, yy)
            field_x += f_x
            field_y += f_y
        
        if False:
            field_x, field_y = octupole(1, 0, 0, 5, 1, width=2, angle=pi/8, astig=1, plot=True)(xx, yy)
        
        if True:
            # For Thesis
            green = (77/255, 155/255, 77/255)
            linewidth = 5.71
            figsize=(linewidth/2, linewidth/2)
            plt.figure(1, figsize=figsize)
            
            current = 10
            radius = 7.5
            n_turns = 3
            coil_width = 3.5
            
            field_x, field_y, field_z = quadrupole(current, 0, 0, radius, n_turns, width=coil_width, astig=0.8, angle=pi/4, plot=True, color=green)(xx, yy, 0)
            
            # And again to get solenoids on second subplot.
            plt.figure(2, figsize=figsize)
            quadrupole(current, 0, 0, radius, n_turns, width=coil_width, astig=0.8, angle=pi/4, plot=True, color=green)
            
        if False:
            field_x2, field_y2 = octupole(50, 0, 0, 8, 1, width = 0.01, astig=1, plot=True)(xx, yy)
        
            plt.quiver(xs, ys,
                field_x2/sqrt(field_x2**2 + field_y2**2),
                field_y2/sqrt(field_x2**2 + field_y2**2),
                sqrt(field_x2**2 + field_y2**2),
                cmap='inferno')
            plt.xlim((xs.min(), xs.max()))
            plt.ylim((ys.min(), ys.max()))
            
            plt.figure()
            
            field_x, field_y = octupole(50, 0, 0, 8, 1, width = None, astig=1, plot=True)(xx, yy)
        
        if False:
            field_x, field_y, field_z = \
                zeros((ys.size, xs.size)), zeros((ys.size, xs.size)),zeros((ys.size, xs.size))
                
            for i in range(ys.size):
                for j in range(xs.size):
                    for k in range(zs.size):
                        B_x, B_y, B_z =\
                            solenoid2(xs[j], ys[i], zs[k], radius=1, length=1, current=1, x_loc=0, y_loc=0, z_loc=0)
                        
                        field_x[i, j] = B_x
                        field_y[i, j] = B_y
                        field_z[i, j] = B_z
            

        # For Thesis Plot
        v_z = 1
        force_x = -v_z*field_y
        force_y = v_z*field_x
        
        plt.figure(1, figsize=figsize)
        plt.quiver(xs, ys,
            field_x/sqrt(field_x**2 + field_y**2),
            field_y/sqrt(field_x**2 + field_y**2),
            sqrt(field_x**2 + field_y**2),
            cmap='inferno')
        plt.xlim((xs.min(), xs.max()))
        plt.ylim((ys.min(), ys.max()))
        
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        
        plt.savefig('QuadrupoleField.pgf')
        
        plt.figure(2, figsize=figsize)
        plt.quiver(xs, ys,
            force_x/sqrt(force_x**2 + force_y**2),
            force_y/sqrt(force_x**2 + force_y**2),
            sqrt(field_x**2 + force_y**2),
            cmap='inferno')
            
        plt.xticks([])
        plt.yticks([])
        
        plt.xlim((xs.min(), xs.max()))
        plt.ylim((ys.min(), ys.max()))
        
        plt.tight_layout()
        
        plt.savefig('QuadrupoleForce.pgf')
        
        
    elif True:
        # 3D
        field_x, field_y, field_z = zeros((zs.size, ys.size, xs.size)),\
            zeros((zs.size, ys.size, xs.size)), zeros((zs.size, ys.size, xs.size))
        
        radius = 2.5
        length = 10
        current = 10
        if True:
            if False:
                func = current_loop(1, 5)
            elif False:
                func = solenoid2(-1, 5, 5, 3,
                                x0=0, y0=0, z0=0,
                                thetaX=0, thetaY=pi/2, thetaZ=pi/2)
            elif False:
                func = quadrupole2(current=1, inner_radius=5,
                        solenoid_transverse_radius=3, solenoid_longitudinal_radius=3,
                        solenoid_turns=1, solenoid_length=3,
                        angle=pi/4)
                        
            elif True:
                func = quadrupole2(current=1, inner_radius=0.08,
                        solenoid_transverse_radius=0.01, solenoid_longitudinal_radius=0.01,
                        solenoid_turns=1, solenoid_length=3,
                        z0=0.2,
                        angle=pi/4)

            elif False:
                func = solenoid2(1, 1, 5, 5,
                        5, 0, 0, 0, pi/2, 0)
            
            field_x, field_y, field_z = \
                zeros((zs.size, ys.size, xs.size)), \
                zeros((zs.size, ys.size, xs.size)), \
                zeros((zs.size, ys.size, xs.size))
            for z in range(zs.size):
                for y in range(ys.size):
                    for x in range(xs.size):
                        
                                            
                        
                        B_x, B_y, B_z = func(xs[x], ys[y], zs[z])
                        
                        field_x[z, y, x] = B_x
                        field_y[z, y, x] = B_y
                        field_z[z, y, x] = B_z
                   
            total_magnitude = sqrt(field_x**2 + field_y**2, field_z**2)
            
            # Plot transverse cross-section
            z_zero = absolute(zs - 0).argmin()
            
            horizontal = field_x[z_zero,:,:]
            vertical = field_y[z_zero,:,:]
            
            plt.figure('Transverse')
            magnitude = sqrt(horizontal**2 + vertical**2) + 1e-12
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
            
            if False:
                # Plot longitudinal cross-section
                # Bz vs By
                x_zero = absolute(xs - 0).argmin()
                
                horizontal = field_z[:,:,x_zero].T
                vertical = field_y[:,:,x_zero].T
                
                magnitude = sqrt(horizontal**2 + vertical**2)
                
                plt.figure('Longitudinal Y')
                plt.quiver(zs, ys,
                    horizontal/magnitude,
                    vertical/magnitude,
                    magnitude,
                    cmap='inferno')
                plt.xlim((zs.min()*(n+1)/n, zs.max()*(n+1)/n))
                plt.ylim((ys.min()*(n+1)/n, ys.max()*(n+1)/n))
                plt.xlabel('z')
                plt.ylabel('y')
                
                plt.gca().add_artist(plt.Rectangle((0-length/2, 0-radius),
                            length, radius*2, fill=False, color='r'))
                
                plt.axes().set_aspect('equal', 'box')
                
                # Bz vs Bx
                y_zero = absolute(ys - 0).argmin()
                
                horizontal = field_z[:,y_zero,:].T
                vertical = field_x[:,y_zero,:].T
                
                magnitude = sqrt(horizontal**2 + vertical**2)
                
                plt.figure('Longitudinal X')
                plt.quiver(zs, xs,
                    horizontal/magnitude,
                    vertical/magnitude,
                    magnitude,
                    cmap='inferno')
                plt.xlim((zs.min()*(n+1)/n, zs.max()*(n+1)/n))
                plt.ylim((xs.min()*(n+1)/n, xs.max()*(n+1)/n))
                plt.xlabel('z')
                plt.ylabel('x')
                
                plt.gca().add_artist(plt.Rectangle((0-length/2, 0-radius),
                            length, radius*2, fill=False, color='r'))
                
                plt.axes().set_aspect('equal', 'box')
    plt.show()