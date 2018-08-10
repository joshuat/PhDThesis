##### Imports #####
from numpy import array

from Bunch1D import Bunch


##### Functions #####
def maskBunch(bunch, mask_function):
    bunch.electrons = bunch.electrons[list(map(mask_function, bunch.getXs()))]
    
    return bunch
    
def pinholeMask(pinhole_diameter=50e-6, location=0):
    pinhole_function = lambda x: abs(x-location) < pinhole_diameter/2
    
    return pinhole_function
    
def pepperpotMaskFs(pinhole_diameter=50e-6, pitch=200e-6, number_holes=7, location=0):
    fs = []
    for i in range(number_holes):
        x = (i - number_holes/2 + 0.5)*pitch + location
        
        f = pinholeMask(pinhole_diameter, location=x)
        
        fs.append(f)
        
    return fs
        
def pepperpotMask(pinhole_diameter=50e-6, pitch=200e-6, number_holes=7, location=0):
    fs = pepperpotMaskFs(pinhole_diameter=pinhole_diameter, pitch=pitch, \
                         number_holes=number_holes, location=location)
        
    func = lambda x: any(list(map(lambda f: f(x), fs)))
    
    return func

def maskBunch2(bunch, pepperpotFs):
    bunch.electrons = bunch.electrons[list(map(lambda x: any(map(lambda f: f(x), \
                                                    pepperpotFs)), bunch.getXs()))]
    
    return bunch