# emit-sds-l2a

Welcome to the EMIT Level 2a science data system repository.  To understand how this repository is linked to the rest of the emit-sds repositories, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

The "surface_v2" subdirectory contains the data files used to generate the EMIT default surface model.  The data files mostly come from the USGS spectral library version 7 (Kokaly et al., 2017), with some adjustments to smooth out artifacts at the periphery.   There are some additional spectra from Tan et al (2016). The python script "make_surface.py" demonstrates how to call the surface model utility.  It must be created anew for each wavelength file.

The OE code is best called with the "apply_oe" script in the isofit/utils subdirectory of the isofit repository.  An example command using current best practices might look like:

>  python apply_oe --modtran_path [MODTRAN_INSTALL_DIRECTORY]\  
          --wavelength_path [WAVELENGTH FILE]  \  
          --surface_path [SURFACE_MODEL .mat FILE]  \  
          --atmosphere_type [ATM_SUBARC_SUMMER |_SUBATM_MIDLAT_SUMMER | ATM_TROPICAL]  \  
          --multiple_restarts  \  
          --presolve  \  
          --empirical_line 1  \  
          --ray_temp_dir [MY_USER_TEMPORARY_DIRECTORY]  \  
          [INPUT_RADIANCE]  \  
          [INPUT_LOC_FILE]  \  
          [INPUT_OBS_FILE]  \  
          [WORKING_DIRECTORY]  \  
        
The output reflectance file will appear in [WORKING_DIRECTORY]/output

References:
- Kokaly, R. F., R. N. Clark, G. A. Swayze, K. E. Livo, T. M. Hoefen, N. C. Pearson, R. A. Wise et al. USGS spectral library version 7. No. 1035. US Geological Survey (2017).
- J. Tan, K.A. Cherkauer, I. Chaubey Developing a comprehensive spectral-biogeochemical database of Midwestern rivers for water quality retrieval using remote sensing data: a case study of the Wabash River and its tributary, Indiana Remote Sens., 8 (2016)
