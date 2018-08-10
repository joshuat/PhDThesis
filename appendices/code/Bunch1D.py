##### Imports #####
from numpy import array, zeros, sqrt
from numpy.random import randn
from scipy.constants import m_e, c
from matplotlib import pyplot as plt

##### Object #####
class Bunch:
    def __init__(self, n=1, energy=1, rms_width=1, \
                 rms_emittance=None, normalised_rms_emittance=None,\
                 mass=m_e, blank=False):
        if not blank:
            self.setMass(mass)
            self.setEnergy(energy)
            
            self._createBunch(n=n, rms_width=rms_width, \
                              rms_emittance=rms_emittance, \
                              normalised_rms_emittance=normalised_rms_emittance)
        
    def _createBunch(self, n=1, rms_width=1, rms_emittance=None, \
                     normalised_rms_emittance=None):
        self.electrons = randn(n, 2)
            # x, vx
        
        self.electrons[:, 0] = self.electrons[:, 0]*rms_width
        
        if normalised_rms_emittance!=None:
            self.setNormalisedRMSEmittance(normalised_rms_emittance)
        elif rms_emittance!=None:
            self.setRMSEmittance(rms_emittance)
        else:
            self.electrons[:, 1] = zeros(n)
        
    def copy(self):
        new_bunch = Bunch(blank=True)
        
        new_bunch.electrons = self.electrons.copy()
        
        new_bunch.speed = self.speed
        
        return new_bunch
        
    def __iter__(self):
        return self.electrons.__iter__()
        
    def next(self):
        return self.electrons.next()
        
    def propagate(self, dz):
        dt = dz / self.getSpeed()
        
        self.electrons[:, 0] += self.electrons[:, 1]*dt

    def getBeta(self):
        return self.getSpeed() / c
        
    def getRMSEmittance(self):
        xs = self.getXs()
        xprimes = self.getXPrimes()
        
        exs = xs-xs.mean()
        exs_p = xprimes-xprimes.mean()
        
        return sqrt((exs**2).mean() * (exs_p**2).mean() - (exs*exs_p).mean()**2)
        
    def setRMSEmittance(self, rms_emittance):
        if rms_emittance==0:
            self.electrons[:, 1] = zeros(self.electrons.shape[0])
        else:
            self.electrons[:, 1] = randn(self.electrons.shape[0])
            
            rms_position = self.getXs().std()
            
            rms_divergence = rms_emittance / rms_position
            
            self.electrons[:, 1] *= rms_divergence * self.getSpeed()
            
    def getNormalisedRMSEmittance(self):
        return self.getBeta() * self.getRMSEmittance()
        
    def setNormalisedRMSEmittance(self, norm_rms_emittance):
        self.setRMSEmittance(norm_rms_emittance/self.getBeta())
        
    def getSize(self):
        return self.electrons.shape[0]
        
    def getWidth(self):
        return self.getXs().std()
        
    def getMass(self):
        return self.mass
        
    def setMass(self, mass):
        self.mass = mass
        
    def getEnergy(self):
        return self.getMass()*self.getSpeed()**2 / 2

    def setEnergy(self, energy):
        self.setSpeed(sqrt(2*energy/self.mass))
        
    def getSpeed(self):
        return self.speed
        
    def setSpeed(self, speed):
        self.speed = speed
        
    def getXs(self):
        return self.electrons[:, 0]
        
    def setXs(self, Xs):
        self.electrons[:, 0] = Xs
        
    def getVXs(self):
        return self.electrons[:, 1]
        
    def setVXs(self, VXs):
        self.electrons[:, 1] = VXs
        
    def getXPrimes(self):
        return self.getVXs() / self.getSpeed()
        
    def plot_phasespace(self, figname=None, color=None):
        plt.figure(figname)

        # Plot X and Px
        plt.title('X')
        plt.xlabel('X')
        plt.ylabel('Px')

        plt.plot(self.getXs(), self.getVXs()*m_e, ',', c=color)