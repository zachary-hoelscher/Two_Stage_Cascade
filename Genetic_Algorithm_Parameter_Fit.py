import numpy as np
import math
from astropy.cosmology import Planck18 as cosmo
import os 
import scipy
from geneticalgorithm import geneticalgorithm as ga
os.chdir('/home/hoelsczj/Documents/Cosmology/Hubble_Tension')

"""
Sets the initial conditions (densities at recombination)
"""

def InitialConditions(h, FractionA):
    #Constants
    c = 299792458          #m/s
    G = 6.67430e-11       #m^3 kg^-1 s^-2
    MetersPerMpc = 3.086e22  #meters per Mpc
    z_rec = 1090          #Redshift at recombination

    H0 = 100*h           #km/s/Mpc
    H0_SI = H0*1000/MetersPerMpc  #s^-1
    rho_crit_0 = 3*(H0_SI**2)/(8*np.pi*G)  #Critical density today, kg/m^3
    Omega_r = 4.15e-5/(h**2)  #Radiation density parameter (photons + neutrinos), from Dodelson 
    Omega_matter_total = 0.14241/(h**2)
    Omega_EDM = Omega_matter_total*0.85*FractionA #We assume that dark matter is 85 percent of total matter density, and FractionA gives fraction of dark matter in cascade  
    Omega_m = Omega_matter_total - Omega_EDM #Matter that is not EDM
    Omega_Lambda = 1.0 - Omega_m - Omega_EDM - Omega_r #Dark energy density for flatness 

    #Density parameters at z=1090
    rho_EDM_initial = Omega_EDM*rho_crit_0*((1 + z_rec)**3)  #Total exotic dark matter density (the cascade fields A and B)
    rho_m_initial = Omega_m*rho_crit_0*((1 + z_rec)**3)  #Total (non-exotic dark matter) matter density
    rho_r_initial = Omega_r*rho_crit_0*((1 + z_rec)**4)  #Radiation density
    rho_Lambda_initial = Omega_Lambda*rho_crit_0  #Dark energy density, is constant with redshift 
    return rho_EDM_initial, rho_m_initial, rho_r_initial, rho_Lambda_initial

def InitialConditions_No_EDM(h):
    #Constants
    c = 299792458          #m/s
    G = 6.67430e-11       #m^3 kg^-1 s^-2
    MetersPerMpc = 3.086e22  #meters per Mpc
    z_rec = 1090          #Redshift at recombination

    H0 = 100*h           #km/s/Mpc
    H0_SI = H0*1000/MetersPerMpc  #s^-1
    rho_crit_0 = 3*(H0_SI**2)/(8*np.pi*G)  #Critical density today, kg/m^3
    Omega_r = 4.15e-5/(h**2)  #Radiation density parameter (photons + neutrinos), from Dodelson 
    Omega_m = 0.14241/(h**2)
    Omega_Lambda = 1.0 - Omega_m - Omega_r #Dark energy density for flatness 

    #Density parameters at z=1090
    rho_m_initial = Omega_m*rho_crit_0*((1 + z_rec)**3)  #Total matter density
    rho_r_initial = Omega_r*rho_crit_0*((1 + z_rec)**4)  #Radiation density
    rho_Lambda_initial = Omega_Lambda*rho_crit_0  #Dark energy density, is constant with redshift 
    return rho_m_initial, rho_r_initial, rho_Lambda_initial

"""
Here we compute H(z) for Lambda-CDM. 
"""

MetersPerParsec = 3.086*(10**(16))
MetersPerGpc = (10**9)*MetersPerParsec
MetersPerMpc = (10**6)*MetersPerParsec
SecondsperYear=365*24*60*60
c = 299792458 #m/s
G = 6.6743*(10**(-11)) # m^3 kg^−1 s^−2
H0UnitConversion = 1000/(MetersPerParsec*1000000)

h = 0.674
H0 = 100*h           #km/s/Mpc
H0_SI = H0*1000/MetersPerMpc  #s^-1
rho_crit_0 = 3*(H0_SI**2)/(8*np.pi*G)  #Critical density today, kg/m^3
Omega_r = (4.15e-5)/(h**2)  #Radiation density parameter (photons + neutrinos), from Dodelson 
Omega_m = 0.14241/(h**2)
Omega_Lambda = 1.0 - Omega_m - Omega_r #Dark energy density for flatness 

rho_m_initial, rho_r_initial, rho_Lambda_initial = InitialConditions_No_EDM(h)

#Friedman Equation 
def HLCDM(z, rho_m, rho_r, rho_Lambda):
    rho_total = rho_m + rho_r + rho_Lambda
    return math.sqrt((8*math.pi*G/3)*(rho_total))

def RhoCriticalLCDM(z, rho_m, rho_r, rho_Lambda):
    Hval=HLCDM(z, rho_m, rho_r, rho_Lambda)
    return 3*Hval*Hval/(8*math.pi*G) 

#Returns drho_r / dz 
def drhordzLCDM(z, rho_m, rho_r, rho_Lambda):
    Hval=HLCDM(z, rho_m, rho_r, rho_Lambda)
    return (-1/(Hval*(1+z)))*(-4*Hval*rho_r)

#Returns drho_m / dz 
def drhomdzLCDM(z, rho_m, rho_r, rho_Lambda):
    Hval=HLCDM(z, rho_m, rho_r, rho_Lambda)
    return (1/(Hval*(1+z)))*(3*Hval*rho_m)

#Setting initial conditions (at z = 1090)
rho_m = rho_m_initial
rho_r = rho_r_initial
rho_Lambda = rho_Lambda_initial
z = 1090 #Redshift at CMB formation 
dz = -0.005

HList_No_EDM=[]
zList_No_EDM=[]
#Euler's method, using it because it is simple 
while z>0:
    rho_r_old = rho_r
    rho_m_old = rho_m 
    rho_r = rho_r + dz*drhordzLCDM(z, rho_m_old, rho_r_old, rho_Lambda)
    rho_m = rho_m + dz*drhomdzLCDM(z, rho_m_old, rho_r_old, rho_Lambda)
    z = z + dz
    
    HList_No_EDM.append(HLCDM(z, rho_m, rho_r, rho_Lambda)/H0UnitConversion)
    zList_No_EDM.append(z)

H1pzList_No_EDM=[]
for index in range(len(HList_No_EDM)):
    H1pzList_No_EDM.append(HList_No_EDM[index]/(1+zList_No_EDM[index]))

def ExoticDarkMatter(DecayRate, zBreak, FractionA):
    #Constants
    c = 299792458  #m/s
    G = 6.67430e-11  #m^3 kg^-1 s^-2
    MetersPerParsec = 3.086e16  #meters per parsec
    MetersPerMpc = (1e6)*MetersPerParsec  #meters per Mpc
    H0UnitConversion = 1000/(MetersPerMpc)
    MetersPerGpc = (10**9)*MetersPerParsec
    SecondsperYear=365*24*60*60

    #Friedman Equation 
    def H(rho_A, rho_B, rho_m, rho_r, rho_Lambda):
        rho_total = rho_A + rho_B + rho_m + rho_r + rho_Lambda
        return math.sqrt((8*math.pi*G/3)*(rho_total))

    def RhoCritical(rho_A, rho_B, rho_m, rho_r, rho_Lambda):
        Hval=H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
        return 3*Hval*Hval/(8*math.pi*G) 
    
    #Returns drho_r / dz 
    def drhordz(z, rho_A, rho_B, rho_m, rho_r, rho_Lambda):
        Hval=H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
        return (-1/(Hval*(1+z)))*(-4*Hval*rho_r)
    
    #Returns drho_m / dz 
    def drhomdz(z, rho_A, rho_B, rho_m, rho_r, rho_Lambda):
        Hval=H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
        return (-1/(Hval*(1+z)))*(-3*Hval*rho_m)
    
    #Returns drho_A / dz 
    def drhoAdz(z, rho_A, rho_B, rho_m, rho_r, rho_Lambda, Gamma_A):
        w_A = 0 #Eqn of state parameter 
        Hval=H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
        return (-1/(Hval*(1+z)))*(-3*Hval*rho_A - 3*Hval*w_A*rho_A - Gamma_A*rho_A)
    
    #Returns drho_B / dz 
    def drhoBdz(z, rho_A, rho_B, rho_m, rho_r, rho_Lambda, Gamma_A):
        w_B = -1/3 #Eqn of state parameter 
        Hval=H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
        return (-1/(Hval*(1+z)))*(-3*Hval*rho_B - 3*Hval*w_B*rho_B + Gamma_A*rho_A)
    
    h_sample = 0.674
    
    rho_A, rho_m, rho_r, rho_Lambda = InitialConditions(h_sample, FractionA) #Obtaining densities at z = 1090 
    rho_B = 0 #Field B is not present until it is produced via decays
    z = 1090 #Redshift at CMB formation 
    dz = -0.005
    HList_EDM=[]
    zList_EDM=[]
    #Euler's method, using it because it is simple 
    while z>0:
        rho_r_old = rho_r
        rho_m_old = rho_m 
        rho_A_old = rho_A 
        rho_B_old = rho_B 

        if z >=zBreak:
            Gamma_A = 0
        if 0 <= z < zBreak: #Decays turn on 
            Gamma_A = DecayRate
        
        rho_r = rho_r_old + dz*drhordz(z, rho_A_old, rho_B_old, rho_m_old, rho_r_old, rho_Lambda)
        rho_m = rho_m_old + dz*drhomdz(z, rho_A_old, rho_B_old, rho_m_old, rho_r_old, rho_Lambda)

        #Here we have decays, where we prevent any negative densities. (Decay stops when decaying field is depleted totally.)
        if (rho_A_old + dz*drhoAdz(z, rho_A_old, rho_B_old, rho_m_old, rho_r_old, rho_Lambda, Gamma_A)) > 0:
            rho_A = rho_A_old + dz*drhoAdz(z, rho_A_old, rho_B_old, rho_m_old, rho_r_old, rho_Lambda, Gamma_A)
        else:
            rho_A = 0
            rho_A_old = 0 #Prevents sourcing B from A when this would result in negative rho_A

        rho_B = rho_B_old + dz*drhoBdz(z, rho_A_old, rho_B_old, rho_m_old, rho_r_old, rho_Lambda, Gamma_A)

        z = z + dz
        
        HList_EDM.append(H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)/H0UnitConversion)
        zList_EDM.append(z)

    RhoCrit=RhoCritical(rho_A, rho_B, rho_m, rho_r, rho_Lambda)
    Omega_EDM=(rho_A + rho_B)/RhoCrit
    Omega_m=rho_m/RhoCrit
    Omega_r=rho_r/RhoCrit 

    Omega_A=rho_A/RhoCrit
    Omega_B=rho_B/RhoCrit
    Omega_Lambda = rho_Lambda/RhoCrit

    H0New = H(rho_A, rho_B, rho_m, rho_r, rho_Lambda)/H0UnitConversion
   
    H1pzList_EDM=[] #Dividing H by 1+z, useful for plotting later. 
    for index in range(len(HList_EDM)):
        H1pzList_EDM.append(HList_EDM[index]/(1+zList_EDM[index]))

    #Here I apply a cubic spline interpolation, useful for plotting things.
    Interpolated_H = scipy.interpolate.CubicSpline(np.flip(zList_EDM), np.flip(HList_EDM), axis=0, extrapolate=None)

    return zList_EDM, H0New, H1pzList_EDM, Interpolated_H

#Here we have measured values for H(z), with upper and lower error bars. 
H_Measured = np.array([73.04, 81.2, 90.9, 99, 159, 224, 227.6]) #Hubble parameter, km / (sec Mpc)
H_Measured_Uncertainty_Upper = np.array([1.04, 2.42, 2.37, 2.51, 12, 8, 5.74])
H_Measured_Uncertainty_Lower = np.array([1.04, 2.42, 2.37, 2.51, 13, 8, 5.47])
H_Measured_Redshift = np.array([0, 0.38, 0.51, 0.61, 1.52, 2.33, 2.4]) #Redshifts for the values of H

#This is so we can plot things as H(z) / (1+z) :
H_Measured_1pz = []
H_Measured_Uncertainty_Upper_1pz = []
H_Measured_Uncertainty_Lower_1pz = []
for index in range(len(H_Measured)):
    H_Measured_1pz.append(H_Measured[index]/(1+H_Measured_Redshift[index]))
    H_Measured_Uncertainty_Upper_1pz.append(H_Measured_Uncertainty_Upper[index]/(1+H_Measured_Redshift[index]))
    H_Measured_Uncertainty_Lower_1pz.append(H_Measured_Uncertainty_Lower[index]/(1+H_Measured_Redshift[index]))

H_Measured_1pz = np.array(H_Measured_1pz) #Divided by (1+z)
H_Measured_Uncertainty_Upper_1pz = np.array(H_Measured_Uncertainty_Upper_1pz)
H_Measured_Uncertainty_Lower_1pz = np.array(H_Measured_Uncertainty_Lower_1pz)

#Computes chi^2, which we minimize in fits. 
def ChiSqr(x):
    DecayRate = x[0]
    zBreak = x[1]
    FractionA = x[2]
    Output = ExoticDarkMatter(DecayRate, zBreak, FractionA)[3]
    H_With_EDM = Output(H_Measured_Redshift) #H(z) from our cascade 
    #This code below is to handle the two values for H that have slightly different upper and lower error bars. 
    H_Measured_Uncertainty = H_Measured_Uncertainty_Upper.copy()
    if Output(1.52) > 159:
        H_Measured_Uncertainty[4] = H_Measured_Uncertainty_Upper[4]
    else:
        H_Measured_Uncertainty[4] = H_Measured_Uncertainty_Lower[4]
    if Output(2.4) > 227.6:
        H_Measured_Uncertainty[6] = H_Measured_Uncertainty_Upper[6]
    else:
        H_Measured_Uncertainty[6] = H_Measured_Uncertainty_Lower[6]
    Chisquare = np.sum(((H_With_EDM - H_Measured)**2) / H_Measured_Uncertainty**2)
    return Chisquare

print("Beginning parameter fit.")
print(" ")
varbound = np.array([[10**(-18), 10**(-15)], [0.0, 20.0], [0.0, 0.25]]) #Place bounds on parameters 
model=ga(function=ChiSqr,dimension=3,variable_type='real',variable_boundaries=varbound) 
model.run() #Run parameter fit with genetic algorithm 

