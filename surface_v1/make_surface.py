# David R Thompson
from isofit.utils import surface_model

# Note - the following references 
surface_model('surface.json',
            wavelength_path='emit_wavelengths.txt',
            output_path='EMIT_surface_test.mat')
