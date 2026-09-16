# -*- coding: iso-8859-15 -*-
# Build a Hyperion dust file which mimics the dust used by Ferrara et al. (1999; ApJ; 123; 437; http://adsabs.harvard.edu/abs/1999ApJS..123..437F).
# Andrew Benson (18-July-2018)
import sys
import numpy as np
import h5py
import h5py.highlevel
from hyperion.dust import SphericalDust
from hyperion.dust import HenyeyGreensteinDust

# Determine dust model to use.
if ( len(sys.argv) != 2 ):
    sys.exit("Usage: hyperionustFerrara.py <dustType>")
dustType = sys.argv[1]
if dustType != "MW" and dustType != "SMC":
    sys.exit("invalid dust type")

# Read Draine dust data and determine the opacity to extinction at 0.55µm (V band). We will use this to match the V-band opacity
# in our new dust file. This is arbitrary as we typically normalize the dust by optical depth anyway, but it is convenient.
dust  = SphericalDust()
dust.read('hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5')
kappaV = dust.optical_properties.interp_chi_wav(0.55)

# Construct Ferrara et al. (1999). They use dust with Henyey-Greenstein scattering, and with an albedo, scattering asymmetry, and
# extinction curve taken from Gordon et al. (1997; ApJ; 487; 625; http://adsabs.harvard.edu/abs/1997ApJ...487..625G).
dustWavelength = np.flipud(np.array([ 1250,
                                      1515,
                                      1775,
                                      1995,
                                      2215,
                                      2480,
                                      2895,
                                      3605,
                                      4413,
                                      5512,
                                      6594,
                                      8059,
                                      12369,
                                      16464,
                                      21578 ]))
dustNu = 2.998e18/dustWavelength
if dustType == "MW":
    dustAlbedo = np.flipud(np.array([  0.60,
                                       0.67,
                                       0.65,
                                       0.55,
                                       0.46,
                                       0.56,
                                       0.61,
                                       0.63,
                                       0.61,
                                       0.59,
                                       0.57,
                                       0.55,
                                       0.53,
                                       0.51,
                                       0.50 ]))
    dustAsymmetry = np.flipud(np.array([ 0.75,
                                         0.75,
                                         0.73,
                                         0.72,
                                         0.71,
                                         0.70,
                                         0.69,
                                         0.65,
                                         0.63,
                                         0.61,
                                         0.57,
                                         0.53,
                                         0.47,
                                         0.45,
                                         0.43 ]))
    dustChi = np.flipud(np.array([ 3.11,
                                   2.63,
                                   2.50,
                                   2.78,
                                   3.12,
                                   2.35,
                                   2.00,
                                   1.52,
                                   1.32,
                                   1.00,
                                   0.76,
                                   0.48,
                                   0.28,
                                   0.167,
                                   0.095 ]))
elif dustType == "SMC":
    dustAlbedo = np.flipud(np.array([ 0.40,  
                                      0.40,  
                                      0.58,  
                                      0.58,  
                                      0.55,  
                                      0.56,  
                                      0.53,  
                                      0.46,  
                                      0.43,  
                                      0.43,  
                                      0.41,  
                                      0.38,  
                                      0.33,  
                                      0.30,  
                                      0.29  ]))

    dustAsymmetry = np.flipud(np.array([ 0.53,
                                         0.53,
                                         0.54,
                                         0.51,
                                         0.46,
                                         0.37,
                                         0.35,
                                         0.34,
                                         0.32,
                                         0.29,
                                         0.26,
                                         0.23,
                                         0.21,
                                         0.23,
                                         0.22 ]))

    dustChi = np.flipud(np.array([ 5.00, 	
	                           4.36, 	
	                           3.51, 	
	                           3.20, 	
	                           2.90, 	
	                           2.40, 	
	                           2.13, 	
	                           1.58, 	
	                           1.35, 	
	                           1.00, 	
	                           0.74, 	
	                           0.52, 	
	                           0.28, 	
	                           0.17, 	
	                           0.11  ]))


## Scale opacity to extinction to match Draine dust model.
dustChi    *= kappaV
## Allow full linear polarization.
dustPLinMax = np.repeat(1.0,15)
## Build the dust object.
dustFerrara = HenyeyGreensteinDust(dustNu,dustAlbedo,dustChi,dustAsymmetry,dustPLinMax)
## Allow extrapolation of the optical properties across a full range of wavelenghts of interest.
dustFerrara.optical_properties.extrapolate_wav(0.005, 1000)
## Store the dust model to file.
dustFerrara.write('data/dustFerrara_'+dustType+'.hdf5')
