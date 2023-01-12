"""
This code contains support code for formatting L2A products for the LP DAAC.

Authors: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
from netCDF4 import Dataset
from emit_utils.daac_converter import add_variable, makeDims, makeGlobalAttr, add_loc, add_glt
from emit_utils.file_checks import netcdf_ext, envi_header
from spectral.io import envi
import logging
import numpy as np


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''This script \
    converts L2A PGE outputs to DAAC compatable formats, with supporting metadata''', add_help=True)

    parser.add_argument('rfl_output_filename', type=str, help="Output Reflectance netcdf filename")
    parser.add_argument('rfl_unc_output_filename', type=str, help="Output Reflectance Uncertainty netcdf filename")
    parser.add_argument('mask_output_filename', type=str, help="Output Mask netcdf filename")
    parser.add_argument('rfl_file', type=str, help="EMIT L2A reflectance ENVI file")
    parser.add_argument('rfl_unc_file', type=str, help="EMIT L2A reflectance uncertainty ENVI file")
    parser.add_argument('mask_file', type=str, help="EMIT L2A water/cloud mask ENVI file")
    parser.add_argument('band_mask_file', type=str, help="EMIT L1B band mask ENVI file")
    parser.add_argument('loc_file', type=str, help="EMIT L1B location data ENVI file")
    parser.add_argument('glt_file', type=str, help="EMIT L1B glt ENVI file")
    parser.add_argument('version', type=str, help="3 digit (with leading V) version number")
    parser.add_argument('--ummg_file', type=str, help="Output UMMG filename")
    parser.add_argument('--log_file', type=str, default=None, help="Logging file to write to")
    parser.add_argument('--log_level', type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=args.log_level, filename=args.log_file)

    rfl_ds = envi.open(envi_header(args.rfl_file))
    rfl_unc_ds = envi.open(envi_header(args.rfl_unc_file))
    mask_ds = envi.open(envi_header(args.mask_file))
    bandmask_ds = envi.open(envi_header(args.band_mask_file))

    # Start with Reflectance File

    # make the netCDF4 file
    logging.info(f'Creating netCDF4 file: {args.rfl_output_filename}')
    nc_ds = Dataset(args.rfl_output_filename, 'w', clobber=True, format='NETCDF4')

    # make global attributes
    logging.debug('Creating global attributes')
    makeGlobalAttr(nc_ds, args.rfl_file, args.glt_file)

    nc_ds.title = "EMIT L2A Estimated Surface Reflectance 60 m " + args.version
    nc_ds.summary = nc_ds.summary + \
        f"\\n\\nThis file contains L2A estimated surface reflectances \
and geolocation data. Reflectance estimates are created using an Optimal Estimation technique - see ATBD for \
details. Reflectance values are reported as fractions (relative to 1). \
Geolocation data (latitude, longitude, height) and a lookup table to project the data are also included."
    nc_ds.sync()

    logging.debug('Creating dimensions')
    makeDims(nc_ds, args.rfl_file, args.glt_file)

    logging.debug('Creating and writing reflectance metadata')
    add_variable(nc_ds, "sensor_band_parameters/wavelengths", "f4", "Wavelength Centers", "nm",
                 [float(d) for d in rfl_ds.metadata['wavelength']], {"dimensions": ("bands",)})
    add_variable(nc_ds, "sensor_band_parameters/fwhm", "f4", "Full Width at Half Max", "nm",
                 [float(d) for d in rfl_ds.metadata['fwhm']], {"dimensions": ("bands",)})

    # Handle data pre January, where bbl was not set in ENVI header
    if 'bbl' not in rfl_ds.metadata or rfl_ds.metadata['bbl'] == '{}':
        wl = np.array(nc_ds['sensor_band_parameters']['wavelengths'])
        bbl = np.ones(len(wl))
        bbl[np.logical_and(wl > 1325, wl < 1435)] = 0
        bbl[np.logical_and(wl > 1770, wl < 1962)] = 0
    else:
        bbl = [bool(d) for d in rfl_ds.metadata['bbl']]

    add_variable(nc_ds, "sensor_band_parameters/good_wavelengths", "u1", "Wavelengths where reflectance is useable: 1 = good data, 0 = bad data", "unitless",
                 bbl, {"dimensions": ("bands",)})

    logging.debug('Creating and writing location data')
    add_loc(nc_ds, args.loc_file)

    logging.debug('Creating and writing glt data')
    add_glt(nc_ds, args.glt_file)

    logging.debug('Write reflectance data')
    add_variable(nc_ds, 'reflectance', "f4", "Surface Reflectance", "unitless", rfl_ds.open_memmap(interleave='bip')[...].copy(),
                 {"dimensions":("downtrack", "crosstrack", "bands")})
    nc_ds.sync()
    nc_ds.close()
    del nc_ds
    logging.debug(f'Successfully created {args.rfl_output_filename}')



    # Start Reflectance Uncertainty File

    # make the netCDF4 file
    logging.info(f'Creating netCDF4 file: {args.rfl_unc_output_filename}')
    nc_ds = Dataset(args.rfl_unc_output_filename, 'w', clobber=True, format='NETCDF4')

    # make global attributes
    logging.debug('Creating global attributes')
    makeGlobalAttr(nc_ds, args.rfl_unc_file, args.glt_file)

    nc_ds.title = "EMIT L2A Estimated Surface Reflectance Uncertainty 60 m " + args.version
    nc_ds.summary = nc_ds.summary + \
        f"\\n\\nThis file contains L2A estimated surface reflectance uncertainties \
and geolocation data. Reflectance uncertainty estimates are created using an Optimal Estimation technique - see ATBD for \
details. Reflectance uncertainty values are reported as fractions (relative to 1). \
Geolocation data (latitude, longitude, height) and a lookup table to project the data are also included."
    nc_ds.sync()

    logging.debug('Creating dimensions')
    makeDims(nc_ds, args.rfl_unc_file, args.glt_file)

    logging.debug('Creating and writing reflectance metadata')
    add_variable(nc_ds, "sensor_band_parameters/wavelengths", "f4", "Wavelength Centers", "nm",
                 [float(d) for d in rfl_ds.metadata['wavelength']], {"dimensions": ("bands",)})
    add_variable(nc_ds, "sensor_band_parameters/fwhm", "f4", "Full Width at Half Max", "nm",
                 [float(d) for d in rfl_ds.metadata['fwhm']], {"dimensions": ("bands",)})
    add_variable(nc_ds, "sensor_band_parameters/good_wavelengths", "u1", "Wavelengths where reflectance is useable: 1 = good data, 0 = bad data", "unitless",
                 bbl, {"dimensions": ("bands",)})

    logging.debug('Creating and writing location data')
    add_loc(nc_ds, args.loc_file)
    logging.debug('Creating and writing glt data')
    add_glt(nc_ds, args.glt_file)

    add_variable(nc_ds, 'reflectance_uncertainty', "f4", "Surface Reflectance Uncertainty", "unitless",
                 rfl_unc_ds.open_memmap(interleave='bip')[...].copy(),
                 {"dimensions":("downtrack", "crosstrack", "bands")})

    nc_ds.sync()
    nc_ds.close()
    del nc_ds
    logging.debug(f'Successfully created {args.rfl_unc_output_filename}')


    # Start Mask File

    # make the netCDF4 file
    logging.info(f'Creating netCDF4 file: {args.mask_output_filename}')
    nc_ds = Dataset(args.mask_output_filename, 'w', clobber=True, format='NETCDF4')

    # make global attributes
    logging.debug('Creating global attributes')
    makeGlobalAttr(nc_ds, args.mask_file, args.glt_file)

    nc_ds.title = "EMIT L2A Masks 60 m " + args.version
    nc_ds.summary = nc_ds.summary + \
        f"\\n\\nThis file contains masks for L2A estimated surface reflectances \
and geolocation data. Masks account for clouds, cloud shadows (via buffering), spacecraft interference, and poor \
atmospheric conditions. \
Geolocation data (latitude, longitude, height) and a lookup table to project the data are also included."
    nc_ds.sync()

    logging.debug('Creating dimensions')
    makeDims(nc_ds, args.mask_file, args.glt_file)
    nc_ds.createDimension('packed_wavelength_bands', int(bandmask_ds.metadata['bands']))

    logging.debug('Creating and writing mask metadata')
    add_variable(nc_ds, "sensor_band_parameters/mask_bands", str, "Mask Band Names", None,
                 mask_ds.metadata['band names'], {"dimensions": ("bands",)})

    logging.debug('Creating and writing location data')
    add_loc(nc_ds, args.loc_file)

    logging.debug('Creating and writing glt data')
    add_glt(nc_ds, args.glt_file)

    logging.debug('Write mask data')
    add_variable(nc_ds, 'mask', "f4", "Masks", "unitless", mask_ds.open_memmap(interleave='bip')[...].copy(),
                 {"dimensions":("downtrack", "crosstrack", "bands"), "zlib": True, "complevel": 9})
    add_variable(nc_ds, 'band_mask', "u8", "Per-Wavelength Mask", "unitless", bandmask_ds.open_memmap(interleave='bip')[...].copy(),
                 {"dimensions":("downtrack", "crosstrack", "packed_wavelength_bands"), "zlib": True, "complevel": 9})
    nc_ds.sync()
    nc_ds.close()
    del nc_ds
    logging.debug(f'Successfully created {args.mask_output_filename}')

    return


if __name__ == '__main__':
    main()
