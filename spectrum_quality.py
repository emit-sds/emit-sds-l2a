"""
Spectral quality evaluation metrics.

Authors: David R. Thompson, david.r.thompson@jpl.nasa.gov
"""

import os
import argparse
import numpy as np
import spectral.io.envi as envi
import pylab as plt
import logging


def find_header(inputpath:str) -> str:
  """Return the header associated with an image file

  Args:
      inputpath (str): input pathname to search for header from

  Returns:
      str: envi header path
  """
  if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
      # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
      # does, if not return the latter (new file creation presumed).
      hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
      if os.path.isfile(hdrfile):
          return hdrfile
      elif os.path.isfile(inputpath + '.hdr'):
          return inputpath + '.hdr'
      return hdrfile
  elif os.path.splitext(inputpath)[-1] == '.hdr':
      return inputpath
  else:
      return inputpath + '.hdr'


def smooth(x:np.array, window_length:int = 3) -> np.array:
  """Moving average smoother
  Args:
      x (np.array): Input spectrum
      window_length (int, optional): Window size for smoothing. Defaults to 3.

  Returns:
      np.array: smoothed spectra
  """
  q=np.r_[x[window_length-1:0:-1],x,x[-1:-window_length:-1]]
  w=np.ones(window_length,'d')/float(window_length)
  y=np.convolve(w,q,mode='valid')
  y= y[int(window_length/2):-int(window_length/2)]
  return y


def wl2band(w: float, wl: np.array) -> int:
  """Translate wavelength to nearest channel index

  Args:
      w (float): input wavelength to match
      wl (np.array): reference wavelengths

  Returns:
      int: closest index of given wavelength
  """
  return np.argmin(abs(wl-w))


# parse the command line (perform the correction on all command line arguments)
def main():

  parser = argparse.ArgumentParser(description="Spectrum quality")
  parser.add_argument('rflfile', type=str, metavar='REFLECTANCE')
  parser.add_argument('outfile', type=str, metavar='OUTPUT')
  parser.add_argument('--sample', type=int, default=1)
  parser.add_argument('--wavelengths', type=str, metavar='WAVELENGTH', default=None) 
  parser.add_argument('--plot', action='store_true')
  parser.add_argument('--log_file', type=str, default=None)
  parser.add_argument('--log_level', type=str, default='INFO')
  args = parser.parse_args()

  if args.log_file is None:
      logging.basicConfig(format='%(message)s', level=args.log_level)
  else:
      logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)


  dtypemap = {'4':np.float32, '5':np.float64, '2':np.float16}

  # Get file dimensions
  rflhdrfile = find_header(args.rflfile)
  rflhdr = envi.read_envi_header(rflhdrfile)
  rfllines   = int(rflhdr['lines'])
  rflsamples = int(rflhdr['samples'])
  rflbands   = int(rflhdr['bands'])
  rflintlv   = rflhdr['interleave']
  rfldtype   = dtypemap[rflhdr['data type']]
  rflframe   = rflsamples * rflbands
    
  # Get wavelengths 
  if args.wavelengths is not None: 
    c, wl, fwhm = np.loadtxt(args.wavelengths).T
  else:
    if not 'wavelength' in rflhdr:
      raise IndexError('Could not find wavelength data anywhere')
    else:
      wl = np.array([float(f) for f in rflhdr['wavelength']])
    if not 'fwhm' in rflhdr:
      raise IndexError('Could not find fwhm data anywhere')
    else:
      fwhm = np.array([float(f) for f in rflhdr['fwhm']])
   
  # Convert from microns to nm if needed
  if not any(wl>100): 
    logging.info('Assuming wavelengths provided in microns, converting to nm')
    wl = wl*1000.0  
      
  # Define start and end channels for two water bands  
  # reference regions outside these features.  The reference
  # intervals serve to assess the channel-to-channel instrument
  # noise
  s940,e940 = wl2band(910,wl), wl2band(990,wl)
  s1140,e1140 = wl2band(1090,wl), wl2band(1180,wl) 
  srefA,erefA = wl2band(1010,wl), wl2band(1080,wl)
  srefB,erefB = wl2band(780,wl), wl2band(900,wl)
  
  samples = 0
  errors = []
  with open(args.rflfile,'rb') as fin:
       for line in range(rfllines):
      
          logging.debug('line %i/%i'%(line+1,rfllines))

          # Read reflectances and translate the frame to BIP
          # (a sequential list of spectra)
          rfl = np.fromfile(fin, dtype=rfldtype, count=rflframe)
          if rflintlv == 'bip':
              rfl = np.array(rfl.reshape((rflsamples,rflbands)), dtype=np.float32)
          else: 
              rfl = np.array(rfl.reshape((rflbands,rflsamples)).T, dtype=np.float32)

          # Loop through all spectra 
          for spectrum in rfl:

              if any(spectrum<-9990):
                 continue

              samples = samples + 1
              if samples % args.sample != 0:
                 continue

              # Get divergence of each spectral interval from the local 
              # smooth spectrum
              ctm = smooth(spectrum)
              errsA = spectrum[s940:e940] - ctm[s940:e940]
              errsB = spectrum[s1140:e1140] - ctm[s1140:e1140]
              referenceA = spectrum[srefA:erefA] - ctm[srefA:erefA]
              referenceB = spectrum[srefB:erefB] - ctm[srefB:erefB]

              # calcualte the root mean squared error of each interval
              errsA = np.sqrt(np.mean(pow(errsA,2)))
              errsB = np.sqrt(np.mean(pow(errsB,2)))
              referenceA = np.sqrt(np.mean(pow(referenceA,2)))
              referenceB = np.sqrt(np.mean(pow(referenceB,2)))

              # We use the better of two reference regions and two 
              # water regions for robustness
              errs = min(errsA,errsB)
              reference = min(referenceA,referenceB)
              excess_error = errs/reference

              # Running tally of errors
              errors.append(excess_error)

              if args.plot:
                 plt.plot(wl,spectrum)
                 plt.plot(wl,ctm)
                 plt.title(excess_error)
                 plt.show()
  
  # Write percentiles
  errors.sort()
  errors = np.array(errors)
  with open(args.outfile,'w') as fout:
      for pct in [50,95,99.9]:
          fout.write('%8.6f\n'%np.percentile(errors,pct))

if __name__ == "__main__":
  main()



