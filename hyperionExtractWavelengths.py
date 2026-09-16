import sys
from hyperion.model import ModelOutput

# Get arguments.
if ( len(sys.argv) != 2 ):
    sys.exit("Usage: hyperionExtractInclinations.py <fileName>")
fileName =sys.argv[1] # File name from which model should be read.

# Extract the SED object and write out wavelengths in a simple format.
model     = ModelOutput(fileName)
sed       = model.get_sed()
separator = " "
print(separator.join([str(v) for i,v in enumerate(sed.wav)]))
