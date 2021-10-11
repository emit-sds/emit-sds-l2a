#!/home/dthompson/src/anaconda/bin/python
# David R Thompson

import os
import argparse
import numpy as np
from spectral.io import envi
from isofit.core.sunposition import sunpos
from isofit.core.common import resample_spectrum
from datetime import datetime
from scipy.ndimage.morphology import distance_transform_edt
from emit_utils.file_checks import envi_header


def haversine_distance(lon1, lat1, lon2, lat2, radius=6335439):
    """ Approximate the great circle distance using Haversine formula

    :param lon1: point one longitude
    :param lat1: point one latitude
    :param lon2: point two longitude
    :param lat2: point two latitude
    :param radius: radius to use (default is approximate radius at equator)

    :return: great circle distance in radius units
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    d = 2 * radius * np.arcsin(np.sqrt(np.sin(delta_lat/2)**2 + np.cos(lat1)
                               * np.cos(lat2) * np.sin(delta_lon/2)**2))

    return d


def main():

    parser = argparse.ArgumentParser(description="Remove glint")
    parser.add_argument('rdnfile', type=str, metavar='RADIANCE')
    parser.add_argument('locfile', type=str, metavar='LOCATIONS')
    parser.add_argument('lblfile', type=str, metavar='SUBSET_LABELS')
    parser.add_argument('statefile', type=str, metavar='STATE_SUBSET')
    parser.add_argument('irrfile', type=str, metavar='SOLAR_IRRADIANCE')
    parser.add_argument('rhofile', type=str, metavar='OUTPUT_RHO')
    parser.add_argument('outfile', type=str, metavar='OUTPUT_MASKS')
    parser.add_argument('--wavelengths', type=str, default=None)
    args = parser.parse_args()

    dtypemap = {'4': np.float32, '5': np.float64, '2': np.float16}

    rdnhdrfile = envi_header(args.rdnfile)
    rdnhdr = envi.read_envi_header(rdnhdrfile)
    rdnlines = int(rdnhdr['lines'])
    rdnsamples = int(rdnhdr['samples'])
    rdnbands = int(rdnhdr['bands'])
    rdndtype = dtypemap[rdnhdr['data type']]
    rdnframe = rdnsamples * rdnbands

    lochdrfile = envi_header(args.locfile)
    lochdr = envi.read_envi_header(lochdrfile)
    loclines = int(lochdr['lines'])
    locsamples = int(lochdr['samples'])
    locbands = int(lochdr['bands'])
    locintlv = lochdr['interleave']
    locdtype = dtypemap[lochdr['data type']]
    locframe = locsamples * 3

    lblhdrfile = envi_header(args.lblfile)
    lblhdr = envi.read_envi_header(lblhdrfile)
    lbllines = int(lblhdr['lines'])
    lblsamples = int(lblhdr['samples'])
    lblbands = int(lblhdr['bands'])
    lbldtype = dtypemap[lblhdr['data type']]
    lblframe = lblsamples

    statehdrfile = envi_header(args.statefile)
    statehdr = envi.read_envi_header(statehdrfile)
    statelines = int(statehdr['lines'])
    statesamples = int(statehdr['samples'])
    statebands = int(statehdr['bands'])
    statedtype = dtypemap[statehdr['data type']]
    stateframe = statesamples * statebands

    # Check file size consistency
    if loclines != rdnlines or locsamples != rdnsamples:
        raise ValueError('LOC and input file dimensions do not match.')
    if lbllines != rdnlines or lblsamples != rdnsamples:
        raise ValueError('Label and input file dimensions do not match.')
    if locbands != 3:
        raise ValueError('LOC file should have three bands.')

    # Get wavelengths and bands
    if args.wavelengths is not None:
        c, wl, fwhm = np.loadtxt(args.wavelengths).T
    else:
        if not 'wavelength' in rdnhdr:
            raise IndexError('Could not find wavelength data anywhere')
        else:
            wl = np.array([float(f) for f in rdnhdr['wavelength']])
        if not 'fwhm' in rdnhdr:
            raise IndexError('Could not find fwhm data anywhere')
        else:
            fwhm = np.array([float(f) for f in rdnhdr['fwhm']])

    # Find H2O and AOD elements in state vector
    aod_bands, h2o_band = [], []
    for i, name in enumerate(statehdr['band names']):
        if 'H2O' in name:
            h2o_band.append(i)
        elif 'AER' in name:
            aod_bands.append(i)

    # find pixel size
    if 'map info' in rdnhdr.keys():
        pixel_size = float(rdnhdr['map info'][5].strip())
    else:
        loc_memmap = envi.open(lochdrfile).open_memmap()
        center_y = int(loclines/2)
        center_x = int(locsamples/2)
        center_pixels = loc_memmap[center_y-1:center_y+1, center_x, :2]
        pixel_size = haversine_distance(
            center_pixels[0, 1], center_pixels[0, 0], center_pixels[1, 1], center_pixels[1, 0])
        del loc_memmap, center_pixels

    # find solar zenith
    fid = os.path.split(args.rdnfile)[1].split('_')[0]
    for prefix in ['prm', 'ang', 'emit']:
        fid = fid.replace(prefix, '')
    dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')
    day_of_year = dt.timetuple().tm_yday
    print(day_of_year, dt)

    # convert from microns to nm
    if not any(wl > 100):
        wl = wl*1000.0

    # irradiance
    irr_wl, irr = np.loadtxt(args.irrfile, comments='#').T
    irr = irr / 10  # convert to uW cm-2 sr-1 nm-1
    irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
    irr_resamp = np.array(irr_resamp, dtype=np.float32)

    # determine glint bands having negligible water reflectance
    BLUE = np.logical_and(wl > 440, wl < 460)
    NIR = np.logical_and(wl > 950, wl < 1000)
    SWIRA = np.logical_and(wl > 1250, wl < 1270)
    SWIRB = np.logical_and(wl > 1640, wl < 1660)
    SWIRC = np.logical_and(wl > 2200, wl < 2500)
    b450 = np.argmin(abs(wl-450))
    b762 = np.argmin(abs(wl-762))
    b780 = np.argmin(abs(wl-780))
    b1000 = np.argmin(abs(wl-1000))
    b1250 = np.argmin(abs(wl-1250))
    b1380 = np.argmin(abs(wl-1380))
    b1650 = np.argmin(abs(wl-1650))

    maskbands = 8
    mask = np.zeros((rdnlines, maskbands, rdnsamples), dtype=np.float32)
    noise = []
    dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')

    with open(args.statefile, 'rb') as fstate:
        statesize = statelines * statesamples * statebands
        state = np.fromfile(fstate, dtype=statedtype, count=statesize)
        state = state.reshape((statelines, statebands, statesamples))

    with open(args.rdnfile, 'rb') as frdn:
        with open(args.locfile, 'rb') as floc:
            with open(args.lblfile, 'rb') as flbl:
                with open(args.rhofile, 'wb') as frho:
                    for line in range(rdnlines):

                        print('line %i/%i' % (line+1, rdnlines))
                        loc = np.fromfile(floc, dtype=locdtype, count=locframe)
                        if locintlv == 'bip':
                            loc = np.array(loc.reshape((locsamples, locbands)), dtype=np.float32)
                        else:
                            loc = np.array(loc.reshape((locbands, locsamples)).T, dtype=np.float32)
                        rdn = np.fromfile(frdn, dtype=rdndtype, count=rdnframe)
                        rdn = np.array(rdn.reshape((rdnbands, rdnsamples)).T, dtype=np.float32)
                        lbl = np.fromfile(flbl, dtype=lbldtype, count=lblframe)
                        lbl = np.array(lbl.reshape((1, lblsamples)).T, dtype=np.float32)
                        x = np.zeros((rdnsamples, statebands))

                        elevation_m = loc[:, 2]
                        latitude = loc[:, 1]
                        longitudeE = loc[:, 0]
                        az, zen, ra, dec, h = sunpos(dt, latitude, longitudeE,
                                                     elevation_m, radians=True).T

                        rho = (((rdn * np.pi) / (irr_resamp.T)).T / np.cos(zen)).T

                        rho[rho[:, 0] < -9990, :] = -9999.0
                        rho_bil = np.array(rho.T, dtype=np.float32)
                        frho.write(rho_bil.tobytes())
                        bad = (latitude < -9990).T

                        # Cloud threshold from Sandford et al.
                        total = np.array(rho[:, b450] > 0.28, dtype=int) + \
                            np.array(rho[:, b1250] > 0.46, dtype=int) + \
                            np.array(rho[:, b1650] > 0.22, dtype=int)
                        mask[line, 0, :] = total > 2

                        # Cirrus Threshold from Gao and Goetz, GRL 20:4, 1993
                        mask[line, 1, :] = np.array(rho[:, b1380] > 0.1, dtype=int)

                        # Water threshold as in CORAL
                        mask[line, 2, :] = np.array(rho[:, b1000] < 0.05, dtype=int)

                        # Threshold spacecraft parts using their lack of an O2 A Band
                        mask[line, 3, :] = np.array(rho[:, b762]/rho[:, b780] > 0.8, dtype=int)

                        for i, j in enumerate(lbl[:, 0]):
                            if j <= 0:
                                x[i, :] = -9999.0
                            else:
                                x[i, :] = state[int(j), :, 0]

                        max_cloud_height = 3000.0
                        mask[line, 4, :] = np.tan(zen) * max_cloud_height / pixel_size

                        # AOD 550
                        mask[line, 5, :] = x[:, aod_bands].sum(axis=1)
                        aerosol_threshold = 0.4

                        mask[line, 6, :] = x[:, h2o_band].T

                        mask[line, 7, :] = np.array((mask[line, 0, :] + mask[line, 2, :] +
                                                     (mask[line, 3, :] > aerosol_threshold)) > 0, dtype=int)
                        mask[line, :, bad] = -9999.0

    bad = np.squeeze(mask[:, 0, :]) < -9990
    good = np.squeeze(mask[:, 0, :]) > -9990

    cloudinv = np.logical_not(np.squeeze(mask[:, 0, :]))
    cloudinv[bad] = 1
    cloud_distance = distance_transform_edt(cloudinv)
    invalid = (np.squeeze(mask[:, 4, :]) >= cloud_distance)
    mask[:, 4, :] = invalid.copy()

    hdr = rdnhdr.copy()
    hdr['bands'] = str(maskbands)
    hdr['band names'] = ['Cloud flag', 'Cirrus flag', 'Water flag',
                         'Spacecraft Flag', 'Dilated Cloud Flag',
                         'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']
    hdr['interleave'] = 'bil'
    del hdr['wavelength']
    del hdr['fwhm']
    envi.write_envi_header(args.outfile+'.hdr', hdr)
    mask.astype(dtype=np.float32).tofile(args.outfile)


if __name__ == "__main__":
    main()
