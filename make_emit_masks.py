#!/home/dthompson/src/anaconda/bin/python
# David R Thompson

import os
import sys
import argparse
from scipy import logical_and as aand
import scipy as s
import numpy as np
import spectral
import spectral.io.envi as envi
from scipy.stats.stats import mode
from scipy.interpolate import interp1d
from scipy.ndimage.morphology import binary_dilation 
from scipy.ndimage.morphology import generate_binary_structure 
from isofit.core.sunposition import sunpos
from isofit.core.common import resample_spectrum
from datetime import datetime
from scipy.ndimage.morphology import binary_dilation, binary_opening
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import generate_binary_structure


# Return the header associated with an image file
def find_header(imgfile):
  if os.path.exists(imgfile+'.hdr'):
    return imgfile+'.hdr'
  ind = imgfile.rfind('.raw')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  ind = imgfile.rfind('.img')
  if ind >= 0:
    return imgfile[0:ind]+'.hdr'
  raise IOError('No header found for file {0}'.format(imgfile));


# parse the command line (perform the correction on all command line arguments)
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

  dtypemap = {'4':s.float32, '5':s.float64, '2':s.float16}

  rdnhdrfile = find_header(args.rdnfile)
  rdnhdr = envi.read_envi_header(rdnhdrfile)
  rdnlines   = int(rdnhdr['lines'])
  rdnsamples = int(rdnhdr['samples'])
  rdnbands   = int(rdnhdr['bands'])
  rdndtype   = dtypemap[rdnhdr['data type']]
  rdnframe   = rdnsamples * rdnbands
  
  lochdrfile = find_header(args.locfile)
  lochdr = envi.read_envi_header(lochdrfile)
  loclines   = int(lochdr['lines'])
  locsamples = int(lochdr['samples'])
  locbands   = int(lochdr['bands'])
  locintlv   = lochdr['interleave']
  locdtype   = dtypemap[lochdr['data type']]
  locframe   = locsamples * 3 
  
  lblhdrfile = find_header(args.lblfile)
  lblhdr = envi.read_envi_header(lblhdrfile)
  lbllines   = int(lblhdr['lines'])
  lblsamples = int(lblhdr['samples'])
  lblbands   = int(lblhdr['bands'])
  lbldtype   = dtypemap[lblhdr['data type']]
  lblframe   = lblsamples

  statehdrfile = find_header(args.statefile)
  statehdr = envi.read_envi_header(statehdrfile)
  statelines   = int(statehdr['lines'])
  statesamples = int(statehdr['samples'])
  statebands   = int(statehdr['bands'])
  statedtype   = dtypemap[statehdr['data type']]
  stateframe   = statesamples * statebands

  # Check file size consistency
  if loclines != rdnlines or locsamples != rdnsamples:
    raise ValueError('LOC and input file dimensions do not match.')
  if lbllines != rdnlines or lblsamples != rdnsamples:
    raise ValueError('Label and input file dimensions do not match.')
  if locbands != 3:
    raise ValueError('LOC file should have three bands.')
    
  # Get wavelengths and bands
  if args.wavelengths is not None: 
    c, wl, fwhm = s.loadtxt(args.wavelengths).T
  else:
    if not 'wavelength' in rdnhdr:
      raise IndexError('Could not find wavelength data anywhere')
    else:
      wl = s.array([float(f) for f in rdnhdr['wavelength']])
    if not 'fwhm' in rdnhdr:
      raise IndexError('Could not find fwhm data anywhere')
    else:
      fwhm = s.array([float(f) for f in rdnhdr['fwhm']])
   
  # Find H2O and AOD elements in state vector
  aod_bands, h2o_band = [],[]
  for i,name in enumerate(statehdr['band names']):
    if 'H2O' in name:
        h2o_band.append(i)
    elif 'AER' in name:
        aod_bands.append(i)

  # find pixel size
  pixel_size = float(rdnhdr['map info'][5].strip())
  
  # find solar zenith 
  fid = os.path.split(args.rdnfile)[1].split('_')[0]
  for prefix in ['prm','ang','emit']:
     fid = fid.replace(prefix,'')
  dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')
  day_of_year = dt.timetuple().tm_yday
  print(day_of_year,dt.tm_hour)

  # convert from microns to nm
  if not any(wl>100): 
    wl = wl*1000.0  
      
  # irradiance
  irr_wl, irr = s.loadtxt(args.irrfile, comments='#').T 
  irr = irr / 10 # convert to uW cm-2 sr-1 nm-1
  irr_resamp = resample_spectrum(irr, irr_wl, wl, fwhm)
  irr_resamp = s.array(irr_resamp, dtype=s.float32)

  # determine glint bands having negligible water reflectance
  BLUE  = s.logical_and(wl>440, wl<460)
  NIR   = s.logical_and(wl>950, wl<1000)
  SWIRA = s.logical_and(wl>1250, wl<1270)
  SWIRB = s.logical_and(wl>1640, wl<1660)
  SWIRC = s.logical_and(wl>2200, wl<2500)
  b450  = s.argmin(abs(wl-450))
  b762  = s.argmin(abs(wl-762))
  b780  = s.argmin(abs(wl-780))
  b1000 = s.argmin(abs(wl-1000))
  b1250 = s.argmin(abs(wl-1250))
  b1650 = s.argmin(abs(wl-1650))

  maskbands = 7
  mask = s.zeros((rdnlines,maskbands,rdnsamples),dtype=s.float32)
  noise = []
  dt = datetime.strptime(fid, '%Y%m%dt%H%M%S')

  with open(args.statefile,'rb') as fstate:
    statesize = statelines * statesamples * statebands
    state = s.fromfile(fstate, dtype=statedtype, count=statesize)
    state = state.reshape((statelines,statebands,statesamples))

  with open(args.rdnfile,'rb') as frdn:
   with open(args.locfile,'rb') as floc:
    with open(args.lblfile,'rb') as flbl:
     with open(args.rhofile,'wb') as frho:
       for line in range(rdnlines):
      
          print('line %i/%i'%(line+1,rdnlines))
          loc = s.fromfile(floc, dtype=locdtype, count=locframe)
          if locintlv == 'bip':
              loc = s.array(loc.reshape((locsamples,locbands)), dtype=s.float32)
          else:
              loc = s.array(loc.reshape((locbands,locsamples)).T, dtype=s.float32)
          rdn = s.fromfile(frdn, dtype=rdndtype, count=rdnframe)
          rdn = s.array(rdn.reshape((rdnbands,rdnsamples)).T, dtype=s.float32)
          lbl = s.fromfile(flbl, dtype=lbldtype, count=lblframe)
          lbl = s.array(lbl.reshape((1,lblsamples)).T, dtype=s.float32)
          x   = s.zeros((rdnsamples, statebands))
          
          elevation_m  = loc[:,2]
          latitude     = loc[:,1]
          longitudeE   = loc[:,0]
          print(np.median(latitude),np.median(longitudeE),np.median(elevation_m),dt)
          az, zen, ra, dec, h = sunpos(dt, latitude, longitudeE,
                         elevation_m, radians=True).T
         
          print('solar zenith:',np.median(zen))
          rho = (((rdn * s.pi) / (irr_resamp.T)).T / s.cos(zen)).T

          rho[rho[:,0]<-9990,:] = -9999.0
          rho_bil = s.array(rho.T, dtype=s.float32)
          frho.write(rho_bil.tobytes())
          bad = (latitude<-9990).T
          
          # Cloud threshold from Sandford et al.
          total = np.array(rho[:,b450]>0.28,dtype=int) + \
                  np.array(rho[:,b1250]>0.46,dtype=int) + \
                  np.array(rho[:,b1650]>0.22,dtype=int)
          mask[line,0,:] = total > 2

          # Water threshold as in CORAL
          mask[line,1,:] = np.array(rho[:,b1000]<0.05,dtype=int)

          # Threshold spacecraft parts using their lack of an O2 A Band
          mask[line,2,:] = np.array(rho[:,b762]/rho[:,b780] > 0.8,dtype=int)


          for i,j in enumerate(lbl[:,0]):
            if j==0: 
              x[i,:] = -9999.0
            else: 
              x[i,:] = state[int(j),:,0]
             
          max_cloud_height = 3000.0
          mask[line,3,:] = s.tan(zen) * max_cloud_height / pixel_size
          
          # AOD 550 
          mask[line,4,:] = x[:,aod_bands].sum(axis=1)
          aerosol_threshold = 0.4
          
          mask[line,5,:] = x[:,h2o_band].T

          mask[line,6,:] = np.array((mask[line,0,:] + mask[line,2,:] + \
                             (mask[line,3,:]>aerosol_threshold)) > 0, dtype=int)
          mask[line,:,bad] = -9999.0

  bad = s.squeeze(mask[:,0,:])<-9990
  good = s.squeeze(mask[:,0,:])>-9990

  cloudinv = s.logical_not(s.squeeze(mask[:,0,:]))
  cloudinv[bad] = 1
  cloud_distance = distance_transform_edt(cloudinv)
  invalid = (s.squeeze(mask[:,3,:]) >= cloud_distance)
  mask[:,3,:] = invalid.copy()
 
  hdr = rdnhdr.copy()
  hdr['bands'] = str(maskbands)
  hdr['band names'] = ['Cloud flag', 'Water flag', 'Spacecraft Flag',
          'Dilated Cloud Flag',
          'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']
  hdr['interleave'] = 'bil'
  del hdr['wavelength']
  del hdr['fwhm']
  envi.write_envi_header(args.outfile+'.hdr', hdr)
  mask.astype(dtype=s.float32).tofile(args.outfile)

if __name__ == "__main__":
  main()



