#!/home/dthompson/src/anaconda/bin/python
# David R Thompson

import os
import sys
import argparse
from scipy import logical_and as aand
import numpy as np
import spectral
import spectral.io.envi as envi
from scipy.interpolate import interp1d
from scipy.stats import chi2
import pylab as plt


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


# Moving average smoother
def smooth(x,window_length=3):
    q=np.r_[x[window_length-1:0:-1],x,x[-1:-window_length:-1]]
    w=np.ones(window_length,'d')/float(window_length)
    y=np.convolve(w,q,mode='valid')
    y= y[int(window_length/2):-int(window_length/2)]
    return y


# Translate wavelength to nearest channel index
def wl2band(w, wl):
    return np.argmin(abs(wl-w))



# parse the command line (perform the correction on all command line arguments)
def main():

  parser = argparse.ArgumentParser(description="Spectrum quality")
  parser.add_argument('rflfile', type=str, metavar='REFLECTANCE')
  parser.add_argument('outfile', type=str, metavar='OUTPUT')
  parser.add_argument('--sample', type=int, default=1)
  parser.add_argument('--wavelengths', type=str, metavar='WAVELENGTH', default=None) 
  parser.add_argument('--plot', action='store_true')
  args = parser.parse_args()

  dtypemap = {'4':np.float32, '5':np.float64, '2':np.float16}

  rflhdrfile = find_header(args.rflfile)
  rflhdr = envi.read_envi_header(rflhdrfile)
  rfllines   = int(rflhdr['lines'])
  rflsamples = int(rflhdr['samples'])
  rflbands   = int(rflhdr['bands'])
  rflintlv   = rflhdr['interleave']
  rfldtype   = dtypemap[rflhdr['data type']]
  rflframe   = rflsamples * rflbands
    
  # Get wavelengths and bands
  if args.wavelengths is not None: 
    c, wl, fwhm = np.loadtxt(argnp.wavelengths).T
  else:
    if not 'wavelength' in rflhdr:
      raise IndexError('Could not find wavelength data anywhere')
    else:
      wl = np.array([float(f) for f in rflhdr['wavelength']])
    if not 'fwhm' in rflhdr:
      raise IndexError('Could not find fwhm data anywhere')
    else:
      fwhm = np.array([float(f) for f in rflhdr['fwhm']])
   
  # convert from microns to nm
  if not any(wl>100): 
    wl = wl*1000.0  
      
  # start and end channels for two water bands and a reference region
  s940,e940 = wl2band(910,wl), wl2band(990,wl)
  s1140,e1140 = wl2band(1090,wl), wl2band(1180,wl) 
  srefA,erefA = wl2band(1010,wl), wl2band(1080,wl)
  srefB,erefB = wl2band(780,wl), wl2band(900,wl)
  
  samples = 0
  errors = []
  with open(args.rflfile,'rb') as fin:
       for line in range(rfllines):
      
          print('line %i/%i'%(line+1,rfllines))
          rfl = np.fromfile(fin, dtype=rfldtype, count=rflframe)
          if rflintlv == 'bip':
              rfl = np.array(rfl.reshape((rflsamples,rflbands)), dtype=np.float32)
          else: # bil
              rfl = np.array(rfl.reshape((rflbands,rflsamples)).T, dtype=np.float32)

   
          for spectrum in rfl:
              if any(spectrum<-9990):
                 continue
              samples = samples + 1
              if samples % args.sample != 0:
                 continue

              ctm = smooth(spectrum)
              errsA = spectrum[s940:e940] - ctm[s940:e940]
              errsB = spectrum[s1140:e1140] - ctm[s1140:e1140]
              referenceA = spectrum[srefA:erefA] - ctm[srefA:erefA]
              referenceB = spectrum[srefB:erefB] - ctm[srefB:erefB]

              errsA = np.sqrt(np.mean(pow(errsA,2)))
              errsB = np.sqrt(np.mean(pow(errsB,2)))
              referenceA = np.sqrt(np.mean(pow(referenceA,2)))
              referenceB = np.sqrt(np.mean(pow(referenceB,2)))

              errs = min(errsA,errsB)
              reference = min(referenceA,referenceB)
              excess_error = errs/reference

              errors.append(excess_error)

             #df = len(errsB)-1
             #p = 1.0 - chi2.cdf(sse, df)
              if args.plot:
                 plt.plot(wl,spectrum)
                 plt.plot(wl,ctm)
                 plt.title(excess_error)
                 plt.show()
  errors.sort()
  errors = np.array(errors)

  # Write percentiles
  with open(args.outfile,'w') as fout:
      for pct in [50,90,95,99,99.9]:
          fout.write('%8.6f\n'%np.percentile(errors,pct))

if __name__ == "__main__":
  main()



