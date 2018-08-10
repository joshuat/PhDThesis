from scipy.optimize import curve_fit, leastsq
from numpy import ravel, meshgrid, arange, array

def linearSearch(target, func, lower_bound, upper_bound, accuracy, iteration_limit=100):
    def _good_enough(val):
        return abs(target-val)<accuracy

    up_x = upper_bound
    lo_x = lower_bound
    
    lo_y = func(lo_x)
    up_y = func(up_x)
    
    print(up_x, up_y)
    print(lo_x, lo_y)
    
    print(target)
    
    if (lo_y<target and up_y>target) or (lo_y<target and up_y>target):
        if _good_enough(lo_y):
            return lo_x
            
        if _good_enough(up_y):
            return up_x
                
        i=0
        while(i<iteration_limit):
            # Interpolate
            grad = (up_y - lo_y) / (up_x - lo_x)
            
            new_x = lo_x + (target - lo_y) / grad
            new_y = func(new_x)
        
            print(i)
            print(lo_x, new_x, up_x)
            print(lo_y, new_y, up_y)
            print()
            
            if _good_enough(new_y):
                return new_x
            
            if grad>0:
                if new_y>target:
                    up_x = new_x
                    up_y = new_y
                else:
                    lo_x = new_x
                    lo_y = new_y
            else:
                if new_y>target:
                    lo_x = new_x
                    lo_y = new_y
                else:
                    up_x = new_x
                    up_y = new_y
            i += 1
    
    
    return None


def fitCurve(x, y, fitfunc, p_initial, y_err=None, errors=False):
    """
    Returns the parameters of a 2D distribution found by least squares given
    the fitting function (fitfunc), data (x,y) and initial parameters (p_initial).

    If error=True then the diagonals of the covariant matrix from the fit will
    be returned.

    I believe that the diagonals do not exactly represent the errors on the
    fit but they are related to it. More examination of the least squares
    procedure is required to figure this out.
    """

    p1, cov = curve_fit(fitfunc, x, y, p0=p_initial, sigma=y_err)

    if errors:
        return p1, [cov[i][i] for i in range(len(cov))]
    else:
        return p1

def fitPlane(x, y, z, fitfunc, p_initial, errors=False):
    """
    Returns the parrameters of a 3D distribution found by least squares given
    the fitting function (fitfunc), data ( z(x, y) ) and initial parameters (p_initial).

    If error=True then the diagonals of the covariant matrix from the fit will
    be returned.

    I believe that the diagonals do not exactly represent the errors on the
    fit but they are related to it. More examination of the least squares
    procedure is required to figure this out.
    """

    errorfunction = lambda p: ravel( (lambda x, y: fitfunc(x, y, *p)) (*meshgrid(x, y) ) -
                                 z)

    p, cov, infodict, mesg, ier = leastsq(errorfunction, p_initial, full_output=True)

    if errors:
        err = [cov[i][i] for i in range(len(cov))]

        return p, err
    else:
        return p

##############################################################################

def example_usage():
    # Normal 1D function
    function = lambda x: 3*x + 2

    x = arange(0, 10, 0.1)
    y = function(x)

    fit_func = lambda x, p1, p2: p1*x + p2
    p_guess = (1, 1)

    p = fitCurve(x, y, fit_func, p_guess)

    print()
    print("1D Function:")
    print("Actual p1 and p2:", 3, 2)
    print("Fitted p1 and p2:", p[0], p[1])

    # 2D function
    function = lambda x, y: 3*x + 2*y + 6

    x = arange(0, 10, 0.1)
    y = arange(0, 10, 0.1)
    z = []
    for yval in y:
        row = []
        for xval in x:
            row.append(function(xval, yval))

        z.append(row)
    z = array(z)

    fit_func = lambda x, y, p1, p2, p3: p1*x + p2*y + p3
    p_guess = (1, 1, 1)

    p = fitPlane(x, y, z, fit_func, p_guess)

    print()
    print("2D Function:")
    print("Actual p1, p2, p3:", 3, 2, 6)
    print("Fitted p1, p2, p3:", p[0], p[1], p[2])

    # Heavieside Step Function
    heavieside = lambda x, low, high, threshold: array([(low if ex<threshold else high) for ex in x])

    x = arange(-10, 10, 0.1)
    y = heavieside(x, -1, 3, 2)

    p = fitCurve(x, y, heavieside, (0, 0, 0))

    print()
    print("Heavieside Function")
    print("Actual low, high, threshold:", -1, 3, 2)
    print("Fitted low, high, threshold:", p[0], p[1], p[2])


if __name__ == "__main__":
    example_usage()
