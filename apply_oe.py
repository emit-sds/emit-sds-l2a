#! /usr/bin/env python3
#
# Author: David R Thompson original, modified by Philip G. Brodrick
#

import argparse
import os
import sys
from os.path import join, exists, split, abspath
from shutil import copyfile
from datetime import datetime
from spectral.io import envi
import logging
import json
import gdal
import numpy as np

eps = 1e-6


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Representative subset")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('sensor', type=str, choices=['ang', 'avcl'])
    parser.add_argument('--copy_input_files', type=int, choices=[0,1], default=0)
    parser.add_argument('--h2o', action='store_true')
    parser.add_argument('--isofit_path', type=str)
    parser.add_argument('--modtran_path', type=str)
    parser.add_argument('--wavelength_path', type=str)
    parser.add_argument('--aerosol_climatology_path', type=str, default=None)
    parser.add_argument('--rdn_factors_path', type=str)
    parser.add_argument('--surface_path', type=str)
    parser.add_argument('--channelized_uncertainty_path', type=str)
    parser.add_argument('--level', type=str, default="INFO")
    parser.add_argument('--nodata_value', type=float, default=-9999)
    parser.add_argument('--log_file', type=str, default=None)
    args = parser.parse_args()

    if args.copy_input_files == 1:
        args.copy_input_files = True
    else:
        args.copy_input_files = False

    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.level)
    else:
        logging.basicConfig(format='%(message)s', level=args.level, filename=args.log_file)

    # TODO: Update this to a normal import, referencing the (likely ported) version of ISOFIT used for EMIT.
    if args.isofit_path:
        isofit_path = args.isofit_path
    else:
        isofit_path = os.getenv('ISOFIT_BASE')
    sys.path.append(isofit_path)
    from isofit.utils import segment, extractions, empirical_line
    from isofit.core import isofit

    if args.modtran_path:
        modtran_path = args.modtran_path
    else:
        modtran_path = os.getenv('MODTRAN_DIR')

    if args.surface_path:
        surface_path = args.surface_path
    else:
        surface_path = os.getenv('ISOFIT_SURFACE_MODEL')

    if args.channelized_uncertainty_path:
        channelized_uncertainty_path = args.channelized_uncertainty_path
    else:
        channelized_uncertainty_path = os.getenv('ISOFIT_CHANNELIZED_UNCERTAINTY')

    # Define paths based on input arguments
    aerosol_climatology = args.aerosol_climatology_path
    working_path = abspath(args.working_directory)
    input_radiance_file = args.input_radiance
    input_loc_file = args.input_loc
    input_obs_file = args.input_obs
    lut_h2o_directory = abspath(join(working_path, 'lut_h2o/'))
    lut_modtran_directory = abspath(join(working_path, 'lut_full/'))
    config_directory = abspath(join(working_path, 'config/'))
    data_directory = abspath(join(working_path, 'data/'))
    input_data_directory = abspath(join(working_path, 'input/'))
    output_directory = abspath(join(working_path, 'output/'))

    # Based on the sensor type, get appropriate year/month/day info
    if args.sensor == 'ang':
        # parse flightline ID (AVIRIS-NG assumptions)
        fid = split(input_radiance_file)[-1][:18]
        logging.info('Flightline ID: %s' % fid)
        month = int(fid[7:9])
        day = int(fid[9:11])

        dt = datetime.strptime(fid[3:], '%Y%m%dt%H%M%S')
        dayofyear = dt.timetuple().tm_yday
    elif args.sensor == 'avcl':
        # parse flightline ID (AVIRIS-Classic assumptions)
        fid = split(input_radiance_file)[-1][:16]
        logging.info('Flightline ID: %s' % fid)

        year = int(fid[1:3])
        month = int(fid[3:5])
        day = int(fid[5:7])

        # get day of year for hour 0 of our given day (haven't gotten
        # time yet, and we'll adjust for UTC day overrun later)
        dt = datetime.strptime('20{}t000000'.format(fid[1:7]), '%Y%m%dt%H%M%S')
        dayofyear = dt.timetuple().tm_yday

    h_m_s, day_increment = get_time_from_obs(input_obs_file)
    if day_increment:
        day += 1
        dayofyear += 1

    gmtime = float(h_m_s[0] + h_m_s[1] / 60.)

    # define staged file locations
    rdn_fname = fid + '_rdn'
    if args.copy_input_files is True:
        radiance_working_path = abspath(join(input_data_directory, rdn_fname))
        obs_working_path = abspath(join(input_data_directory, fid + '_obs'))
        loc_working_path = abspath(join(input_data_directory, fid + '_loc'))
    else:
        radiance_working_path = input_radiance_file
        obs_working_path = input_obs_file
        loc_working_path = input_loc_file

    rfl_working_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_rfl')))
    uncert_working_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_uncert')))
    lbl_working_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_lbl')))
    surface_working_path = abspath(join(data_directory, 'surface.mat'))
    chnunct_working_path = abspath(join(data_directory, 'channelized_uncertainty.txt'))

    rdn_subs_path = abspath(join(input_data_directory, rdn_fname.replace('_rdn', '_subs_rdn')))
    obs_subs_path = abspath(join(input_data_directory, os.path.basename(
        obs_working_path).replace('_obs', '_subs_obs')))
    loc_subs_path = abspath(join(input_data_directory, os.path.basename(
        loc_working_path).replace('_loc', '_subs_loc')))
    rfl_subs_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_subs_rfl')))
    state_subs_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_subs_state')))
    uncert_subs_path = abspath(join(output_directory, rdn_fname.replace('_rdn', '_subs_uncert')))
    h2o_subs_path = abspath(join(output_directory, os.path.basename(
        loc_working_path).replace('_loc', '_subs_h2o')))

    wavelength_path = abspath(join(data_directory, 'wavelengths.txt'))

    modtran_template_path = abspath(join(config_directory, fid + '_modtran_tpl.json'))
    h2o_template_path = abspath(join(config_directory, fid + '_h2o_tpl.json'))

    modtran_config_path = abspath(join(config_directory, fid + '_modtran.json'))
    h2o_config_path = abspath(join(config_directory, fid + '_h2o.json'))

    #esd_path = join(isofit_path, 'data', 'earth_sun_distance.txt')
    #irradiance_path = join(isofit_path, 'data', 'kurudz_0.1nm.dat')

    # TODO: AVCL
    if args.sensor == 'ang':
        noise_path = join(isofit_path, 'data', 'avirisng_noise.txt')
    elif args.sensor == 'avcl':
        noise_path = join(isofit_path, 'data', 'avirisc_noise.txt')
    else:
        logging.info('no noise path found, check sensor type')
        quit()

    aerosol_tpl_path = join(isofit_path, 'data', 'aerosol_template.json')

    # TODO: either make these globals or aruments
    chunksize = 256
    segmentation_size = 400
    num_integrations = 400

    num_elev_lut_elements = 3
    elev_lut_grid_margin = 0.1

    h2o_min = 0.2
    num_h2o_lut_elements = 10

    uncorrelated_radiometric_uncertainty = 0.02

    inversion_windows = [[400.0,1300.0],[1450,1780.0],[2050.0,2450.0]]

    # create missing directories
    for dpath in [working_path, lut_h2o_directory, lut_modtran_directory, config_directory,
                  data_directory, input_data_directory, output_directory]:
        if not exists(dpath):
            os.mkdir(dpath)

    # stage data files by copying into working directory
    for src, dst in [(input_radiance_file, radiance_working_path),
                     (input_obs_file, obs_working_path),
                     (input_loc_file, loc_working_path)]:
        if not exists(dst):
            logging.info('Staging %s to %s' % (src, dst))
            copyfile(src, dst)
            copyfile(src + '.hdr', dst + '.hdr')

    files_to_stage = [(surface_path, surface_working_path)]
    if (channelized_uncertainty_path is not None):
        files_to_stage.append((channelized_uncertainty_path, chnunct_working_path))
    else:
        chnunct_working_path=None
        logging.info('No valid channelized uncertainty file found, proceeding without uncertainty')

    # Staging files without headers
    for src, dst in files_to_stage:
        if not exists(dst):
            logging.info('Staging %s to %s' % (src, dst))
            copyfile(src, dst)

    # Superpixel segmentation
    if not exists(lbl_working_path) or not exists(radiance_working_path):
        logging.info('Segmenting...')
        segment(spectra=(radiance_working_path, lbl_working_path),
                flag=args.nodata_value, npca=5, segsize=segmentation_size, nchunk=chunksize)

    # Extract input data
    for inp, outp in [(radiance_working_path, rdn_subs_path),
                      (obs_working_path, obs_subs_path),
                      (loc_working_path, loc_subs_path)]:
        if not exists(outp):
            logging.info('Extracting ' + outp)
            extractions(inputfile=inp, labels=lbl_working_path,
                        output=outp, chunksize=chunksize, flag=args.nodata_value)

    # get radiance file, wavelengths
    radiance_dataset = envi.open(rdn_subs_path + '.hdr')
    if args.wavelength_path:
        chn, wl, fwhm = np.loadtxt(args.wavelength_path).T
    else:
        wl = np.array([float(w) for w in radiance_dataset.metadata['wavelength']])
        if 'fwhm' in radiance_dataset.metadata:
            fwhm = np.array([float(f) for f in radiance_dataset.metadata['fwhm']])
        else:
            fwhm = np.ones(wl.shape) * (wl[1] - wl[0])

    # Convert to microns if needed
    if wl[0] > 100:
        wl = wl / 1000.0
        fwhm = fwhm / 1000.0

    # TODO: re-write IO for BIL oriented datasets, so we're not taking inefficient slices through the data cube

    # Recognize bad data flags
    valid = np.abs(radiance_dataset.read_band(0) + 2*eps - args.nodata_value) > eps

    # Grab zensor position and orientation information
    loc_dataset = envi.open(loc_subs_path + '.hdr')
    mean_latitude = np.mean(loc_dataset.read_band(1)[valid])
    mean_longitude = -np.mean(loc_dataset.read_band(0)[valid])

    elevation_km = loc_dataset.read_band(2) / 1000.0
    mean_elevation_km = np.mean(elevation_km[valid])

    obs_dataset = envi.open(obs_subs_path + '.hdr')
    mean_path_km = np.mean(obs_dataset.read_band(0)[valid]) / 1000.

    to_sensor_azimuth = obs_dataset.read_band(1)
    to_sensor_zenith = 180.0 - obs_dataset.read_band(2)  # 180 reversal follows MODTRAN convention

    mean_to_sensor_azimuth = np.mean(to_sensor_azimuth[valid])

    # In radians to match convention - re-reverse for this calc
    mean_to_sensor_zenith_rad = (np.mean(180 - to_sensor_zenith[valid]) / 360.0 * 2.0 * np.pi)

    mean_altitude_km = mean_elevation_km + np.cos(mean_to_sensor_zenith_rad) * mean_path_km

    logging.info('Path (km): %f, To-sensor Zenith (rad): %f, Mean Altitude: %6.2f km' %
                 (mean_path_km, mean_to_sensor_zenith_rad, mean_altitude_km))

    # make view zenith and azimuth grids
    geom_margin = eps * 2.0
    to_sensor_zenith_lut_grid = np.array([max((to_sensor_zenith[valid].min() - geom_margin), 0),
                                          180.0])
    to_sensor_azimuth_lut_grid = np.array([(to_sensor_azimuth[valid].min() - geom_margin) % 360,
                                           (to_sensor_azimuth[valid].max() + geom_margin) % 360])

    # make elevation grid
    elevation_lut_grid = np.linspace(max(elevation_km[valid].min(), eps),
                                     elevation_km[valid].max() + elev_lut_grid_margin,
                                     num_elev_lut_elements)

    # write wavelength file
    wl_data = np.concatenate([np.arange(len(wl))[:, np.newaxis], wl[:, np.newaxis],
                              fwhm[:, np.newaxis]], axis=1)
    np.savetxt(wavelength_path, wl_data, delimiter=' ')

    if not exists(h2o_subs_path + '.hdr') or not exists(h2o_subs_path):

        atmosphere_type = 'ATM_MIDLAT_SUMMER'
        write_modtran_template(atmosphere_type='ATM_MIDLAT_SUMMER', fid=fid, altitude_km=mean_altitude_km,
                               dayofyear=dayofyear, latitude=mean_latitude, longitude=mean_longitude,
                               to_sensor_azimuth=mean_to_sensor_azimuth, gmtime=gmtime, elevation_km=mean_elevation_km,
                               output_file=h2o_template_path)

        h2o_configuration = {
            "wavelength_file": wavelength_path,
            "lut_path": lut_h2o_directory,
            "modtran_template_file": h2o_template_path,
            "modtran_directory": modtran_path,
            "statevector": {
                "H2OSTR": {
                    "bounds": [0.5, 5.0],
                    "scale": 0.01,
                    "init": 1.5,
                    "prior_sigma": 100.0,
                    "prior_mean": 1.5}
            },
            "lut_grid": {
                "H2OSTR": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            },
            "unknowns": {
                "H2O_ABSCO": 0.0
            },
            "domain": {"start": 340, "end": 2520, "step": 0.1}
        }

        # make isofit configuration
        isofit_config_h2o = {'ISOFIT_base': isofit_path,
                             'input': {'measured_radiance_file': rdn_subs_path,
                                       'loc_file': loc_subs_path,
                                       'obs_file': obs_subs_path},
                             'output': {'estimated_state_file': h2o_subs_path},
                             'forward_model': {
                                 'instrument': {'wavelength_file': wavelength_path,
                                                'parametric_noise_file': noise_path,
                                                'integrations': num_integrations,
                                                'unknowns': {
                                                    'uncorrelated_radiometric_uncertainty': uncorrelated_radiometric_uncertainty}},
                                 "multicomponent_surface": {"wavelength_file": wavelength_path,
                                                            "surface_file": surface_working_path,
                                                            "select_on_init": True},
                                 "modtran_radiative_transfer": h2o_configuration},
                             "inversion": {"windows": inversion_windows}}

        if chnunct_working_path is not None:
            isofit_config_h2o['forward_model']['unknowns'][
                'channelized_radiometric_uncertainty_file'] = chnunct_working_path

        if args.rdn_factors_path:
            isofit_config_h2o['input']['radiometry_correction_file'] = \
                args.rdn_factors_path

        # write modtran_template
        with open(h2o_config_path, 'w') as fout:
            fout.write(json.dumps(isofit_config_h2o, cls=SerialEncoder, indent=4, sort_keys=True))

        # Run modtran retrieval
        logging.info('H2O first guess')
        retrieval_h2o = isofit.Isofit(h2o_config_path, level='DEBUG')

        # clean up unneeded storage
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'rm ' + join(lut_h2o_directory, to_rm)
            logging.info(cmd)
            os.system(cmd)

    # Extract h2o grid avoiding the zero label (periphery, bad data)
    # and outliers
    h2o = envi.open(h2o_subs_path + '.hdr')
    h2o_est = h2o.read_band(-1)[:].flatten()

    h2o_grid = np.linspace(np.percentile(
        h2o_est[h2o_est > h2o_min], 5), np.percentile(h2o_est[h2o_est > h2o_min], 95), num_h2o_lut_elements)

    logging.info(state_subs_path)
    if not exists(state_subs_path) or \
            not exists(uncert_subs_path) or \
            not exists(rfl_subs_path):

        write_modtran_template(atmosphere_type='ATM_MIDLAT_SUMMER', fid=fid, altitude_km=mean_altitude_km,
                               dayofyear=dayofyear, latitude=mean_latitude, longitude=mean_longitude,
                               to_sensor_azimuth=mean_to_sensor_azimuth, gmtime=gmtime, elevation_km=mean_elevation_km,
                               output_file=modtran_template_path)

        modtran_configuration = {
            "wavelength_file": wavelength_path,
            "lut_path": lut_modtran_directory,
            "aerosol_template_file": aerosol_tpl_path,
            "modtran_template_file": modtran_template_path,
            "modtran_directory": modtran_path,
            "statevector": {
                "H2OSTR": {
                    "bounds": [h2o_grid[0], h2o_grid[-1]],
                    "scale": 0.01,
                    "init": (h2o_grid[1] + h2o_grid[-1]) / 2.0,
                    "prior_sigma": 100.0,
                    "prior_mean": (h2o_grid[1] + h2o_grid[-1]) / 2.0,
                }
            },
            "lut_grid": {
                "H2OSTR": [max(0.0, float(q)) for q in h2o_grid],
                "GNDALT": [max(0.0, float(q)) for q in elevation_lut_grid],
                "TRUEAZ": [float(q) for q in to_sensor_azimuth_lut_grid],
                "OBSZEN": [float(q) for q in to_sensor_zenith_lut_grid]
            },
            "unknowns": {
                "H2O_ABSCO": 0.0
            },
            "domain": {"start": 340, "end": 2520, "step": 0.1}
        }

        # add aerosol elements from climatology
        aerosol_state_vector, aerosol_lut_grid, aerosol_model_path = \
            load_climatology(aerosol_climatology, mean_latitude, mean_longitude, dt,
                             isofit_path)
        modtran_configuration['statevector'].update(aerosol_state_vector)
        modtran_configuration['lut_grid'].update(aerosol_lut_grid)
        modtran_configuration['aerosol_model_file'] = aerosol_model_path

        # make isofit configuration
        isofit_config_modtran = {'ISOFIT_base': isofit_path,
                                 'input': {'measured_radiance_file': rdn_subs_path,
                                           'loc_file': loc_subs_path,
                                           'obs_file': obs_subs_path},
                                 'output': {'estimated_state_file': state_subs_path,
                                            'posterior_uncertainty_file': uncert_subs_path,
                                            'estimated_reflectance_file': rfl_subs_path},
                                 'forward_model': {
                                     'instrument': {'wavelength_file': wavelength_path,
                                                    'parametric_noise_file': noise_path,
                                                    'integrations': num_integrations,
                                                    'unknowns': {'uncorrelated_radiometric_uncertainty': uncorrelated_radiometric_uncertainty}},
                                     "multicomponent_surface": {"wavelength_file": wavelength_path,
                                                                "surface_file": surface_working_path,
                                                                "select_on_init": True},
                                     "modtran_radiative_transfer": modtran_configuration},
                                 "inversion": {"windows": inversion_windows}}

        if chnunct_working_path is not None:
            isofit_config_modtran['forward_model']['unknowns'][
                'channelized_radiometric_uncertainty_file'] = chnunct_working_path

        if args.rdn_factors_path:
            isofit_config_modtran['input']['radiometry_correction_file'] = \
                args.rdn_factors_path

        # write modtran_template
        with open(modtran_config_path, 'w') as fout:
            fout.write(json.dumps(isofit_config_modtran, cls=SerialEncoder, indent=4, sort_keys=True))

        # Run modtran retrieval
        logging.info('Running ISOFIT with full LUT')
        retrieval_full = isofit.Isofit(modtran_config_path, level='DEBUG')

        # clean up unneeded storage
        for to_rm in ['*r_k', '*t_k', '*tp7', '*wrn', '*psc', '*plt', '*7sc', '*acd']:
            cmd = 'rm ' + join(lut_modtran_directory, to_rm)
            logging.INFO(cmd)
            os.system(cmd)

    if not exists(rfl_working_path) or not exists(uncert_working_path):
        # Empirical line
        logging.info('Empirical line inference')
        empirical_line(reference_radiance=rdn_subs_path,
                       reference_reflectance=rfl_subs_path,
                       reference_uncertainty=uncert_subs_path,
                       reference_locations=loc_subs_path,
                       hashfile=lbl_working_path,
                       input_radiance=radiance_working_path,
                       input_locations=loc_working_path,
                       output_reflectance=rfl_working_path,
                       output_uncertainty=uncert_working_path,
                       isofit_config=modtran_config_path)

    logging.info('Done.')


class SerialEncoder(json.JSONEncoder):
    """Encoder for json to help ensure json objects can be passed to the workflow manager.

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(SerialEncoder, self).default(obj)


def load_climatology(config_path: str, latitude: float, longitude: float, acquisition_datetime: datetime, isofit_path: str):
    """ Load climatology data, based on location and configuration
    Args:
        config_path: path to the base configuration directory for isofit
        latitude: latitude to set for the segment (mean of acquisition suggested)
        longitude: latitude to set for the segment (mean of acquisition suggested)
        acquisition_datetime: datetime to use for the segment( mean of acquisition suggested)
        isofit_path: base path to isofit installation (needed for data path references)

    :Returns
        aerosol_state_vector: A dictionary that defines the aerosol state vectors for isofit
        aerosol_lut_grid: A dictionary of the aerosol lookup table (lut) grid to be explored
        aerosol_model_path: A path to the location of the aerosol model to use with MODTRAN.
    """

    aerosol_model_path = join(isofit_path, 'data', 'aerosol_twopart_model.txt')
    aerosol_lut_grid = {"AERFRAC_0": [0.001, 0.2, 0.5],
                        "AERFRAC_1": [0.001, 0.2, 0.5]}
    aerosol_state_vector = {
        "AERFRAC_0": {
            "bounds": [0.001, 0.5],
            "scale": 1,
            "init": 0.02,
            "prior_sigma": 10.0,
            "prior_mean": 0.02},
        "AERFRAC_1": {
            "bounds": [0.001, 0.5],
            "scale": 1,
            "init": 0.25,
            "prior_sigma": 10.0,
            "prior_mean": 0.25}}

    logging.INFO('Loading Climatology')
    # If a configuration path has been provided, use it to get relevant info
    if config_path is not None:
        month = acquisition_datetime.timetuple().tm_mon
        year = acquisition_datetime.timetuple().tm_year
        with open(config_path, 'r') as fin:
            for case in json.load(fin)['cases']:
                match = True
                logging.INFO('matching', latitude, longitude, month, year)
                for criterion, interval in case['criteria'].items():
                    logging.INFO(criterion, interval, '...')
                    if criterion == 'latitude':
                        if latitude < interval[0] or latitude > interval[1]:
                            match = False
                    if criterion == 'longitude':
                        if longitude < interval[0] or longitude > interval[1]:
                            match = False
                    if criterion == 'month':
                        if month < interval[0] or month > interval[1]:
                            match = False
                    if criterion == 'year':
                        if year < interval[0] or year > interval[1]:
                            match = False

                if match:
                    aerosol_state_vector = case['aerosol_state_vector']
                    aerosol_lut_grid = case['aerosol_lut_grid']
                    aerosol_model_path = case['aerosol_mdl_path']
                    break

    logging.INFO('Climatology Loaded.  Aerosol State Vector:\n{}\nAerosol LUT Grid:\n{}\nAerosol model path:{}'.format(
        aerosol_state_vector, aerosol_lut_grid, aerosol_model_path))
    return aerosol_state_vector, aerosol_lut_grid, aerosol_model_path


def get_time_from_obs(obs_filename: str, time_band: int = 9, max_flight_duration_h: int = 8):
    """ Scan through the obs file and find mean flight time
    Args:
        obs_filename: observation file name
        time_band: time band inside of observation file (normally 9)
        max_flight_duration_h: assumed maximum length of a flight

    :Returns:
        h_m_s: list of the hour, minute, and second mean of the given data section
        increment_day: a boolean to indicate if the mean day is greater than the starting day
    """
    dataset = gdal.Open(obs_filename, gdal.GA_ReadOnly)
    min_time = 25
    max_time = -1
    mean_time = np.zeros(dataset.RasterYSize)
    mean_time_w = np.zeros(dataset.RasterYSize)
    for line in range(dataset.RasterYSize):
        local_time = dataset.ReadAsArray(0, line, dataset.RasterXSize, 1)[time_band, ...]
        local_time = local_time[local_time != -9999]
        min_time = min(min_time, np.min(local_time))
        max_time = max(max_time, np.max(local_time))
        mean_time[line] = np.mean(local_time)
        mean_time_w[line] = np.prod(local_time.shape)

    mean_time = np.average(mean_time, weights=mean_time_w)

    increment_day = False
    # UTC day crossover corner case
    if (max_time > 24 - max_flight_duration_h and
            min_time < max_flight_duration_h):
        mean_time[mean_time < max_flight_duration_h] += 24
        mean_time = np.average(mean_time, weights=mean_time_w)

        # This means the majority of the line was really in the next UTC day,
        # increment the line accordingly
        if (mean_time > 24):
            mean_time -= 24
            increment_day = True

    # Calculate hour, minute, second
    h_m_s = [np.floor(mean_time)]
    h_m_s.append(np.floor((mean_time - h_m_s[-1]) * 60))
    h_m_s.append(np.floor((mean_time - h_m_s[-2] - h_m_s[-1] / 60.) * 3600))

    return h_m_s, increment_day


def write_modtran_template(atmosphere_type: str, fid: str, altitude_km: float, dayofyear: int,
                           latitude: float, longitude: float, to_sensor_azimuth: float, gmtime: float,
                           elevation_km: float, output_file: str):
    """ Write a MODTRAN template file for use by isofit look up tables
    Args:
        atmosphere_type: label for the type of atmospheric profile to use in modtran
        fid: flight line id (name)
        altitude_km: altitude of the sensor in km
        dayofyear: the current day of the given year
        latitude: acquisition latitude
        longitude: acquisition longitude
        to_sensor_azimuth: azimuth view angle to the sensor, in degrees TODO - verify that this is/should be in degrees
        gmtime: greenwich mean time
        elevation_km: elevation of the land surface in km
        output_file: location to write the modtran template file to

    :Returns:
        None
    """
    # make modtran configuration
    h2o_template = {"MODTRAN": [{
        "MODTRANINPUT": {
            "NAME": fid,
            "DESCRIPTION": "",
            "CASE": 0,
            "RTOPTIONS": {
                "MODTRN": "RT_CORRK_FAST",
                "LYMOLC": False,
                "T_BEST": False,
                "IEMSCT": "RT_SOLAR_AND_THERMAL",
                "IMULT": "RT_DISORT",
                "DISALB": False,
                "NSTR": 8,
                "SOLCON": 0.0
            },
            "ATMOSPHERE": {
                "MODEL": atmosphere_type,
                "M1": atmosphere_type,
                "M2": atmosphere_type,
                "M3": atmosphere_type,
                "M4": atmosphere_type,
                "M5": atmosphere_type,
                "M6": atmosphere_type,
                "CO2MX": 410.0,
                "H2OSTR": 1.0,
                "H2OUNIT": "g",
                "O3STR": 0.3,
                "O3UNIT": "a"
            },
            "AEROSOLS": {"IHAZE": "AER_NONE"},
            "GEOMETRY": {
                "ITYPE": 3,
                "H1ALT": altitude_km,
                "IDAY": dayofyear,
                "IPARM": 11,
                "PARM1": latitude,
                "PARM2": longitude,
                "TRUEAZ": to_sensor_azimuth,
                "GMTIME": gmtime
            },
            "SURFACE": {
                "SURFTYPE": "REFL_LAMBER_MODEL",
                "GNDALT": elevation_km,
                "NSURF": 1,
                "SURFP": {"CSALB": "LAMB_CONST_0_PCT"}
            },
            "SPECTRAL": {
                "V1": 340.0,
                "V2": 2520.0,
                "DV": 0.1,
                "FWHM": 0.1,
                "YFLAG": "R",
                "XFLAG": "N",
                "FLAGS": "NT A   ",
                "BMNAME": "p1_2013"
            },
            "FILEOPTIONS": {
                "NOPRNT": 2,
                "CKPRNT": True
            }
        }
    }]}

    # write modtran_template
    with open(output_file, 'w') as fout:
        fout.write(json.dumps(h2o_template, cls=SerialEncoder, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
