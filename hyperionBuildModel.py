# -*- coding: iso-8859-15 -*-
# Construct Hyperion galaxy and dust models for simple geometries.
# Andrew Benson (16-July-2018)
import sys
import re
import numpy as np
from hyperion.model import Model
from hyperion.dust import SphericalDust
from hyperion.dust import HenyeyGreensteinDust
from hyperion.util.constants import pc, lsun, pi

def HernquistMassCylindrical(R,z):
    # Indefinite integral of the Hernquist profile (with unit scale radius) in cylindrical coordinates.
    radialTerm   = np.sqrt(np.square(R)-1.0+0.0J)
    mass         = np.zeros_like(radialTerm)
    center       = np.logical_and(R == 0.0,np.absolute(z) != 1.0)
    centerZScale = np.logical_and(R == 0.0,np.absolute(z) == 1.0)
    scale        = np.logical_and(R == 1.0,z != 0.0)
    scalePlane   = np.logical_and(R == 1.0,z == 0.0)
    offCenter    = np.logical_and(R != 0.0,R != 1.0)
    mass[offCenter] = 0.5*(z[offCenter]*(np.sqrt(np.square(R[offCenter])+np.square(z[offCenter]))-1.0)/(np.square(R[offCenter])-1.0)/(np.square(R[offCenter])+np.square(z[offCenter])-1.0)+np.square(R[offCenter])*(np.arctan(z[offCenter]/np.sqrt(np.square(R[offCenter])+np.square(z[offCenter]))/radialTerm[offCenter])-np.arctan(z[offCenter]/radialTerm[offCenter]))/np.power(radialTerm[offCenter],3))
    mass[center] = -z[center]*(np.sqrt(np.square(z[center]))-1.0)/2.0/(np.square(z[center])-1.0)
    mass[centerZScale] = -0.25
    mass[scale]  = (-2.0-4.0*np.square(z[scale])-2.0*np.square(z[scale])*np.square(z[scale])+2.0*np.sqrt(1.0+np.square(z[scale]))+3.0*np.square(z[scale])*np.sqrt(1.0+np.square(z[scale])))/(6.0*np.power(z[scale],3)*np.sqrt(1.0+np.square(z[scale])))
    mass[scalePlane]  = 0.0
    return np.real(mass)

def JaffeMassCylindrical(R,z):
    # Indefinite integral of the Jaffe profile (with unit scale radius) in cylindrical coordinates.
    radialTerm   = np.sqrt(np.square(R)-1.0+0.0J)
    mass         = np.zeros_like(radialTerm)
    centerZZero  = np.logical_and(R == 0.0,np.absolute(z) == 0.0)
    centerZScale = np.logical_and(R == 0.0,np.absolute(z) == 1.0)
    center       = np.logical_and(R == 0.0,np.logical_and(np.absolute(z) != 0.0,np.absolute(z) != 1.0))
    scale        = np.logical_and(R == 1.0,z != 0.0)
    scalePlane   = np.logical_and(R == 1.0,z == 0.0)
    offCenter    = np.logical_and(R != 0.0,R != 1.0)
    mass[offCenter   ] = R[offCenter]*(np.arctan(z[offCenter]/R[offCenter])+R[offCenter]*(-np.arctan(z[offCenter]/radialTerm[offCenter])+np.arctan(z[offCenter]/radialTerm[offCenter]/np.sqrt(np.square(R[offCenter])+np.square(z[offCenter]))))/radialTerm[offCenter])+0.5*z[offCenter]*(np.log(np.square(R[offCenter])+np.square(z[offCenter]))-2.0*np.log(1.0+np.sqrt(np.square(R[offCenter])+np.square(z[offCenter]))))
    mass[center      ] = 0.5*z[center]*(np.log(np.square(z[center]))-2.0*np.log(1.0+np.sqrt(np.square(z[center]))))
    mass[centerZZero ] = 0.0
    mass[centerZScale] = -np.log(2.0)
    mass[scale       ] = np.arctan(z[scale])+0.5*z[scale]*np.log(1.0+np.square(z[scale]))-(-1.0+np.sqrt(1.0+np.square(z[scale]))+np.square(z[scale])*np.log(1.0+np.sqrt(1.0+np.square(z[scale]))))/z[scale]
    mass[scalePlane  ] = 0.0
    return np.real(mass)

# Get arguments.
if ( len(sys.argv) != 19 ):
    sys.exit("Usage: hyperionBuildModel.py <dustFile> <diskStructureVertical> <diskScaleRadial> <diskScaleVertical> <dustScaleVertical> <diskCutOff> <spheroidStructure> <spheroidScaleRadial> <spheroidCutOff> <stellarComponent> <inclinationCount> <wavelengthMinimum> <wavelengthMaximum> <wavelengthCount> <diskOpticalDepth> <spheroidOpticalDepth> <seed> <fileName>")
dustFile              =                          sys.argv[ 1]  # Dust file name.
diskStructureVertical =                          sys.argv[ 2]  # Vertical structure of the disk.
diskScaleRadial       = np.dtype('float32').type(sys.argv[ 3]) # Disk radial scale length in Mpc.
diskScaleVertical     = np.dtype('float32').type(sys.argv[ 4]) # Disk vertical scale height in Mpc.
dustScaleVertical     = np.dtype('float32').type(sys.argv[ 5]) # Dust vertical scale height in Mpc.
diskCutOff            = np.dtype('float32').type(sys.argv[ 6]) # Distance (in units of scale lengths) at which to truncate the disk.
spheroidStructure     =                          sys.argv[ 7]  # Structure of the spheroid.
spheroidScaleRadial   = np.dtype('float32').type(sys.argv[ 8]) # Disk radial scale length in Mpc.
spheroidCutOff        = np.dtype('float32').type(sys.argv[ 9]) # Distance (in units of scale lengths) at which to truncate the spheroid.
stellarComponent      =                          sys.argv[10]  # Which stellar component (disk or spheroid) to use.
inclinationCount      =                          sys.argv[11]  # Number of inclinations to use, or list of specific inclinations.
wavelengthMinimum     = np.dtype('float32').type(sys.argv[12]) # Minimum wavelength to compute in microns.
wavelengthMaximum     = np.dtype('float32').type(sys.argv[13]) # Maximum wavelength to compute in microns.
wavelengthCount       =                          sys.argv[14]  # Number of wavelengths at which to compute, or a list of specific wavelengths.
diskOpticalDepth      = np.dtype('float32').type(sys.argv[15]) # Optical depth of the disk from -infinity to +infinity through the center of the galaxy when viewed face-on.
spheroidOpticalDepth  = np.dtype('float32').type(sys.argv[16]) # Optical depth of the spheroid from -infinity to +infinity through the center of the galaxy.
seed                  =                      int(sys.argv[17]) # Seed for the random number generator.
fileName              =                          sys.argv[18]  # File name to which model should be written.

# Validate disk vertical structure.
if diskStructureVertical != "exponential" and diskStructureVertical != "sechSquared":
    sys.exit("invalid disk vertical structure")

# Validate spheroid structure.
if spheroidStructure != "hernquist" and spheroidStructure != "jaffe":
    sys.exit("invalid spheroid structure")

# Validate stellar component.
if stellarComponent != "disk" and stellarComponent != "spheroid":
    sys.exit("invalid stellar component")

# Convert to internal units.
diskScaleRadial     *= 1.0e6*pc;
diskScaleVertical   *= 1.0e6*pc;
dustScaleVertical   *= 1.0e6*pc;
spheroidScaleRadial *= 1.0e6*pc;

# Determine maximum scales.
scaleRadial   = diskScaleRadial
scaleVertical = dustScaleVertical
if spheroidScaleRadial > scaleRadial:
    scaleRadial   = spheroidScaleRadial
if diskScaleVertical   > scaleVertical:
    scaleVertical = diskScaleVertical
if spheroidScaleRadial > scaleVertical:
    scaleVertical = spheroidScaleRadial

# Generate the model grid.
m   = Model()
## Set the random seed.
m.set_seed(seed)
## Radial cell walls.
w   = np.logspace(np.log10(1.0e-2*scaleRadial),np.log10(diskCutOff*scaleRadial),100) # Logarithmically-distributed cells.
w   = np.hstack([0.0,w])                                                             # Add a cell wall at w=0.
## Vertical cell walls.
z   = np.linspace(-diskCutOff*scaleVertical,+diskCutOff*scaleVertical,101)
## Azimuthal cell walls (arbitrary since we have cylindrical symmetry in our models).
phi = np.linspace(0.0,2.0*pi,2)
## Set the grid.
m.set_cylindrical_polar_grid(w,z,phi)

# Build 3D arrays giving the cell walls on the grid.
walls            = {}
walls['gwLower'] =             np.zeros(m.grid.shape)
walls['gwUpper'] =             np.zeros(m.grid.shape)
walls['gzLower'] = np.swapaxes(np.zeros(m.grid.shape),1,2)
walls['gzUpper'] = np.swapaxes(np.zeros(m.grid.shape),1,2)
walls['gwLower'][:,:,:] = w[0:-1]
walls['gwUpper'][:,:,:] = w[1:  ]
walls['gzLower'][:,:,:] = z[0:-1]
walls['gzUpper'][:,:,:] = z[1:  ]
walls['gzLower'] = np.swapaxes(walls['gzLower']      ,1,2)
walls['gzUpper'] = np.swapaxes(walls['gzUpper']      ,1,2)

# Read dust data and determine the opacity to extinction at 0.55µm (V band).
dust  = SphericalDust()
dust.read(dustFile)
kappaV = dust.optical_properties.interp_chi_wav(0.55)

# Find the central dust density of the disk. Note dust file gives opacities per unit dust mass - so densities should be dust
# density. Note that the following relation holds for both exponential and sechSquared vertical profiles, but not necessarily for
# any other profile.
diskRhoCentral = diskOpticalDepth/kappaV/diskScaleVertical/2.0

# Find the central dust density of the spheroid. Note dust file gives opacities per unit dust mass - so densities should be dust
# density. Note that for both Hernquistand Jaffe spheroids the integral for the optical depth through the center is divergent (due
# to the lack of a core in the profile). Therefore, we instead define spheroidOpticalDepth in this case to be the optical depth
# through the spheroid (from -infinity to +infinity) at the scale radius.
if spheroidStructure == "hernquist":
    spheroidRhoCentral = spheroidOpticalDepth/kappaV/spheroidScaleRadial*15.0/4.0
elif spheroidStructure == "jaffe":
    spheroidRhoCentral = spheroidOpticalDepth/kappaV/spheroidScaleRadial/(pi-8.0/3.0)

# Construct the dust density on the grid.
density = np.zeros(m.grid.shape)
## Disk.
if diskStructureVertical == "exponential":
    density += diskRhoCentral*np.exp(-m.grid.gw/diskScaleRadial)*np.exp(-np.abs(m.grid.gz)/diskScaleVertical)
elif diskStructureVertical == "sechSquared":
    coshArgument = m.grid.gz/diskScaleVertical
    largeArgument = np.absolute(coshArgument) >= 50.0
    smallArgument = np.absolute(coshArgument) <  50.0
    density[smallArgument] += diskRhoCentral*np.exp(-m.grid.gw[smallArgument]/diskScaleRadial)/np.cosh(coshArgument[smallArgument])**2
    density[largeArgument] += 0.0
## Spheroid
if spheroidStructure == "hernquist":
    r = np.sqrt(m.grid.gw**2+m.grid.gz**2)/spheroidScaleRadial
    density += spheroidRhoCentral/r/(1.0+r)**3
elif spheroidStructure == "jaffe":
    r = np.sqrt(m.grid.gw**2+m.grid.gz**2)/spheroidScaleRadial
    density += spheroidRhoCentral/r**2/(1.0+r)**2

# Add dust to the model.
m.add_density_grid(density,dust)

# Build the distribution of sources.
source            = m.add_map_source()
## Specify luminosity. Use an arbitrary normalization here as we are only interested in ratios of emergent to emitted luminosity.
source.luminosity = lsun
## Load a simple, arbitrary, flat spectrum.
spectrum          = np.loadtxt('spectrum.txt', dtype=[('nu', float),('fnu', float)])
source.spectrum   = (spectrum['nu'], spectrum['fnu'])
# Construct source (stellar) distribution on our grid. Note that the value of each cell should be proportional to the probability
# of a photon being emitted from that cell - and therefore proportional to the mass of stars in the cell.
stars             = np.zeros(m.grid.shape)
if stellarComponent == "disk":
    ## Stellar disk.
    ### Radial part - always assumed to be an exponential profile.
    stars             = +(-np.exp(-walls['gwUpper']/diskScaleRadial)*(1.0+walls['gwUpper']/diskScaleRadial)+np.exp(-walls['gwLower']/diskScaleRadial)*(1.0+walls['gwLower']/diskScaleRadial))
    ### Vertical part
    if diskStructureVertical == "exponential":
        #### Exponential vertical structure.
        stars        *= np.abs(-np.exp(-np.abs(walls['gzUpper'])/diskScaleVertical)+np.exp(-np.abs(walls['gzLower'])/diskScaleVertical))
    elif diskStructureVertical == "sechSquared":
        #### Sech^2 vertical structure.
        stars        *= np.abs(np.tanh(np.abs(walls['gzUpper'])/diskScaleVertical)-np.tanh(np.abs(walls['gzLower'])/diskScaleVertical))
elif stellarComponent == "spheroid":
    ## Stellar spheroid.
    RLower = walls['gwLower']/spheroidScaleRadial
    RUpper = walls['gwUpper']/spheroidScaleRadial
    zLower = walls['gzLower']/spheroidScaleRadial
    zUpper = walls['gzUpper']/spheroidScaleRadial
    if spheroidStructure == "hernquist":
        stars = abs(+abs(+HernquistMassCylindrical(RUpper,zUpper)-HernquistMassCylindrical(RLower,zUpper))-abs(+HernquistMassCylindrical(RUpper,zLower)-HernquistMassCylindrical(RLower,zLower)))
    elif spheroidStructure == "jaffe":
        stars = abs(+abs(+    JaffeMassCylindrical(RUpper,zUpper)-    JaffeMassCylindrical(RLower,zUpper))-abs(+    JaffeMassCylindrical(RUpper,zLower)-    JaffeMassCylindrical(RLower,zLower)))

# Set the source map.
source.map        = stars

# Set monochromatic radiative transfer.
if re.search(":",wavelengthCount):
    wavelengths     = np.array(wavelengthCount.split(":"),dtype=float)
    wavelengthCount = wavelengths.size
else:
    wavelengths     = np.logspace(np.log10(wavelengthMinimum),np.log10(wavelengthMaximum),int(wavelengthCount))
m.set_monochromatic(True,wavelengths=wavelengths)

# Set to ignore dust emission.
## No iterations are required as we do not want to compute emission from the dust.
m.set_n_initial_iterations(0)
## Kill photons when first absorbed.
m.set_kill_on_absorb(True)
## Use raytracing.
m.set_raytracing(True)

# Specify viewing options.
## Create an "image" specifying that we want SEDs only.
image = m.add_peeled_images(image=False)
## Specify wavelength range.
image.set_wavelength_index_range(0,int(wavelengthCount)-1)
## Generate the viewing angles. These are uniformly spaced in inclination and at fixed (arbitrary) azimuthal angle.
if re.search(":",inclinationCount):
    theta            = np.array(inclinationCount.split(":"),dtype=float)
    inclinationCount = theta.size
else:
    theta            = np.linspace(  0.0,90.0,int(inclinationCount))
phi1 = np.repeat( 90.0,int(inclinationCount))
phi2 = np.repeat(270.0,int(inclinationCount))
image.set_viewing_angles(np.hstack([theta,theta]),np.hstack([phi1,phi2]))
## Compute uncertainties in model parameters.
image.set_uncertainties(True)
## Specify the number of photons to use.
m.set_n_photons(imaging_sources=100000,imaging_dust=0,raytracing_sources=100000,raytracing_dust=0)

# Write the model input file.
m.write(fileName)
