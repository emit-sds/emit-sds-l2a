<h1 align="center"> emit-sds-l2a </h1>

_NOTE - at this time the EMIT repositories are not supporting Pull Requests from members outside of the EMIT-SDS Team.  This is expected to change in March, and guidance on branches will be provided at that time. At present, public migration of this repository is a work-in-progress, and not all features are yet fully up to date.  See the **develop** branch - set as default - for the latest code._

Welcome to the EMIT Level 2a science data system repository.  To understand how this repository is linked to the rest of the emit-sds repositories, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

L2A consists of three components:
1) Construction of a surface model (one time)
2) Estimation of surface reflectance and uncertainy using an optimal-estimation based retrieval
3) Generation of masks (cloud / cirrus / water / spacecraft / etc.)


## Surface Model Construction
The  "surface_v2" subdirectory contains the data files used to generate the EMIT default surface model.  The data files mostly come from the USGS spectral library version 7 [(Kokaly et al., 2017)](https://dx.doi.org/10.5066/F7RR1WDJ), with some additional spectra from [Tan et al (2016)](https://doi.org/10.3390/rs8060517).  Some adjustments to smooth out artifacts at the periphery were included, and the finalized, adjusted spectra can be found at these locations: [snow / ice](https://doi.org/10.21232/xhgtM3A9), [vegetation](https://doi.org/10.21232/6sQDNjfv), [water](https://doi.org/10.21232/ZbyfMgxY), and [other](https://doi.org/10.21232/ezrQtdcw). To construct the surface model, utilize the "surface_model" utility in the isofit/utils subdirectory of the [isofit repository](https://github.com/isofit/isofit).  An example call may look like:

```
python -c "from isofit.utils import surface_model; surface_model('configs/surface.json')"
```

This will create an output .mat file, at the location specified by "output_model_file" in the json configuration file.  This surface model will be passed into subsequent calls to apply_oe (see below) and need only be generated once for general use (though specific investigations may require custom surface models).

## Estimating Surface Reflectance and Uncertainty

The core optimal estimation code is best called with the "apply_oe" script in the isofit/utils subdirectory of the [isofit repository](https://github.com/isofit/isofit).  The current implementation for the EMIT SDS looks like:

```
python apply_oe.py,
       --presolve=1, \
       --empirical_line=1, \
       --emulator_base=[sRTMnet_DIRECTORY], \
       --n_cores str(self.n_cores), \
       --surface_path [SURFACE_MODEL .mat FILE], \
       --ray_temp_dir [SCRATCH_PARALLEL_DIRECTORY], \
       --log_file [OUTPUT_LOG_FILE], \
       --logging_level "INFO", \
       [INPUT_RADIANCE_FILE], \
       [INPUT_LOC_FILE], \
       [INPUT_OBS_FILE], \
       [OUTPUT_DIRECTORY], \
       "emit"
``` 
        
The estimated surface reflectance and uncertainty files will appear in [WORKING_DIRECTORY]/output.  The isofit repository is the result of extensive scientific inquiry that extends an initial presentation by [Thompson et al](https://doi.org/10.1016/j.rse.2018.07.003), and a bibliography can be found [here](https://isofit.readthedocs.io/en/latest/custom/bibliography.html).  The sRTMnet dependency comes from [Brodrick et al](https://doi.org/10.1016/j.rse.2021.112476), and is available for [download](https://doi.org/10.5281/zenodo.4096627).

## Applying Masks

The EMIT mask can be generated using the following call, which is dependent on having completed the above surface reflectance estimation step:

```
python make_emit_masks.py \
       [INPUT_RADIANCE_FILE]
       [INPUT_LOC_FILE]
       [INPUT_LABEL_FILE]
       [INPUT_STATE_SUBS_FILE]
       [OUTPUT_RHO_FILE]
       [OUTPUT_MASK_FILE]
```

The [INPUT_LABEL_FILE] and [INPUT_STATE_SUBS_FILE] can be found in the [WORKING_DIRECTORY]/output file noted about with the extensions \_lbl \_subs_state, respectively.  The [OUTPUT_MASK_FILE] is a 7 band output, designating:

1) Cloud flag
2) Cirrus flag
3) Surface water flag
4) Spacecraft flag
5) Dilated cloud flag
6) Aerosol Optical Depth at 550 nm
7) Atmsopheric water vapor [g / cm^2]
8) Aggregate mask flag

Masks are denoted as ones.


