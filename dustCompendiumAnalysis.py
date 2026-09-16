#!/usr/bin/env python
import sys
import numpy as np
from galacticus.io import GalacticusHDF5
from galacticus.galaxies import Galaxies
from galacticus.datasets import Dataset
from galacticus.properties import Property

if ( len(sys.argv) != 1 ):
    sys.exit("Usage: test.py")

model          = GalacticusHDF5('data/galacticus_AlphaQv3LB_long_99_nbody_AlphaQ_z1.1_BH_SDSS_LSST_BV_ELG_SEDval44.hdf5','r')
galaxies       = Galaxies(model)
for redshift in model.availableRedshifts():
    redshiftLabel = "%.4f" % redshift
    propertyNames = \
                    [ 
                    "diskLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V",
                    "diskLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V",
                    "spheroidLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V",
                    "spheroidLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V",
                    "totalLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V",
                    "totalLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V",
                    "inclination",
                    "tauV0:dustCompendium",
                    "diskRadius",
                    "spheroidRadius"
                    ]
    properties     = {}
    for propertyName in propertyNames:
        properties[propertyName] = galaxies.retrieveProperty(propertyName,redshift)
    for i in range(0,properties["diskLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V"].data.size-1):
        print \
            str(properties["diskLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V"].data[i])+"\t"+\
            str(properties["diskLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V"].data[i])+"\t"+\
            str(properties["spheroidLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V"].data[i])+"\t"+\
            str(properties["spheroidLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V"].data[i])+"\t"+\
            str(properties["totalLuminositiesStellar:z"+redshiftLabel+":dustCompendium:A_V"].data[i])+"\t"+\
            str(properties["totalLuminositiesStellar:z"+redshiftLabel+":dustCompendium:R_V"].data[i])+"\t"+\
            str(properties["inclination"].data[i])+"\t"+\
            str(properties["tauV0:dustCompendium"].data[i])+"\t"+\
            str(properties["diskRadius"].data[i])+"\t"+\
            str(properties["spheroidRadius"].data[i])
