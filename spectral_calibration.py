# David R Thompson
from datetime import datetime
from time import strptime
import argparse
import json
import sys, os
import numpy as np
from spectral.io import envi
from isofit.core.common import envi_header
from isofit.utils.surface_model import surface_model

batch_template='''#!/bin/sh
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1 
ray stop
ps -ef | grep ray | awk '{{print $1}}' | xargs kill
python {isofit_exe} {config_path}
'''



isofit_template='''{{
    "ISOFIT_base": "{isofit_base}",
    "forward_model": {{
        "instrument": {{
            "integrations": 1,
            "calibration_fixed": false,
            "statevector": {{
                "GROW_FWHM": {{
                    "bounds": [
                        -1,
                        1
                    ],
                    "init": 0,
                    "prior_mean": 0,
                    "prior_sigma": 100.0,
                    "scale": 1
                }},
                "WL_SHIFT": {{
                    "bounds": [
                        -1,
                        1
                    ],
                    "init": 0,
                    "prior_mean": 0,
                    "prior_sigma": 100.0,
                    "scale": 1
                }}
            }},
            "parametric_noise_file": "{noise_file}",
            "unknowns": {{
                "uncorrelated_radiometric_uncertainty": 0.01
            }},
            "wavelength_file": "{wavelength_file}"
        }},
        "radiative_transfer": {{
            "lut_grid": {{
                "GNDALT": {alt_grid},
                "OBSZEN": [160,170,179.9],
                "H2OSTR": {h2o_grid}
            }},
            "radiative_transfer_engines": {{
                "vswir": {{
                    "wavelength_file": "{wavelength_file_hires}",
                    "engine_base_dir": "/beegfs/store/shared/MODTRAN6/MODTRAN6.0.0/",
                    "engine_name": "modtran",
                    "lut_names": [
                        "H2OSTR",
                        "GNDALT",
                        "OBSZEN",
                        "AOT550"
                    ],
                    "lut_path": "{lut_directory}",
                    "statevector_names": [
                        "H2OSTR",
                        "GNDALT",
                        "AOT550"
                    ],
                    "template_file": "{modtran_template_file}"
                }}
            }},
            "statevector": {{
                "GNDALT": {{
                    "bounds": [
                        {alt_min},
                        {alt_max}
                    ],
                    "init": {alt_avg},
                    "prior_mean": {alt_avg},
                    "prior_sigma": 100.0,
                    "scale": 1
                }},
                "H2OSTR": {{
                    "bounds": [
                    {h2o_min},
                    {h2o_max}
                    ],
                    "init": {h2o_avg},
                    "prior_mean": {h2o_avg},
                    "prior_sigma": 100.0,
                    "scale": 1
                }}
            }},
            "unknowns": {{
                "H2O_ABSCO": 0.0
            }}
        }},
        "surface": {{
            "select_on_init": true,
            "surface_category": "multicomponent_surface",
            "surface_file": "{surface_path}",
        }}
    }},
    "implementation": {{
        "inversion": {{
            "windows": [
                [
                    380.0,
                    1340.0
                ],
                [
                    1450,
                    1800.0
                ],
                [
                    1970.0,
                    2500.0
                ]
            ]
        }},
        "n_cores": 40,
        "ray_temp_dir": "/tmp/ray",
        "rte_auto_rebuild": true
    }},
    "input": {{
        "measured_radiance_file": "{rdn_path}",
        "obs_file": "{obs_path}"
    }},
    "output": {{
        "estimated_reflectance_file": "{rfl_path}",
        "estimated_state_file": "{state_path}"
    }}
}}'''

modtran_template='''{{
    "MODTRAN": [
        {{
            "MODTRANINPUT": {{
                "AEROSOLS": {{
                    "IHAZE": "AER_NONE"
                }},
                "ATMOSPHERE": {{
                    "CO2MX": 410.0,
                    "H2OSTR": 1.0,
                    "H2OUNIT": "g",
                    "M1": "ATM_MIDLAT_SUMMER",
                    "M2": "ATM_MIDLAT_SUMMER",
                    "M3": "ATM_MIDLAT_SUMMER",
                    "M4": "ATM_MIDLAT_SUMMER",
                    "M5": "ATM_MIDLAT_SUMMER",
                    "M6": "ATM_MIDLAT_SUMMER",
                    "MODEL": "ATM_MIDLAT_SUMMER",
                    "O3STR": 0.3,
                    "O3UNIT": "a"
                }},
                "CASE": 0,
                "DESCRIPTION": "",
                "FILEOPTIONS": {{
                    "CKPRNT": true,
                    "NOPRNT": 2
                }},
                "GEOMETRY": {{
                    "GMTIME": {gmtime},
                    "H1ALT": 400,
                    "IDAY": {dayofyear},
                    "IPARM": 11,
                    "ITYPE": 3,
                    "PARM1": {lat},
                    "PARM2": {lon} 
                }},
                "NAME": "emit_wavelength_check",
                "RTOPTIONS": {{
                    "DISALB": false,
                    "IEMSCT": "RT_SOLAR_AND_THERMAL",
                    "IMULT": "RT_DISORT",
                    "LYMOLC": false,
                    "MODTRN": "RT_CORRK_FAST",
                    "NSTR": 8,
                    "SOLCON": 0.0,
                    "T_BEST": false
                }},
                "SPECTRAL": {{
                    "BMNAME": "p1_2013",
                    "DV": 0.1,
                    "FLAGS": "NT A   ",
                    "FWHM": 0.1,
                    "V1": 320.0,
                    "V2": 2600.0,
                    "XFLAG": "N",
                    "YFLAG": "R"
                }},
                "SURFACE": {{
                    "GNDALT": 0.0,
                    "NSURF": 1,
                    "SURFP": {{
                        "CSALB": "LAMB_CONST_0_PCT"
                    }},
                    "SURFTYPE": "REFL_LAMBER_MODEL"
                }}
            }}
        }}
    ]
}}'''


def main(rawargs=None):
    """ This is a helper script to assess EMIT wavelength calibration
    using atmospheric features.

    Args:
        input_radiance (str): radiance data cube [expected ENVI format]
        input_loc (str): location data cube, (Lon, Lat, Elevation) [expected ENVI format]
        input_obs (str): observation data cube, (path length, to-sensor azimuth, to-sensor zenith, to-sun azimuth,
            to-sun zenith, phase, slope, aspect, cosine i, UTC time) [expected ENVI format]
        working_directory (str): directory to stage multiple outputs, will contain subdirectories
        
    Returns:

    """

    # Parse arguments

    parser = argparse.ArgumentParser(description="Apply OE to a block of data.")
    parser.add_argument('input_radiance', type=str)
    parser.add_argument('input_loc', type=str)
    parser.add_argument('input_obs', type=str)
    parser.add_argument('working_directory', type=str)
    parser.add_argument('--isofit_base', default='/home/drt/src/isofit/')
    parser.add_argument('--flip_fpa', default=True)

    args = parser.parse_args(rawargs)
    isofit_base = args.isofit_base
    noise_file = isofit_base+'/data/emit_noise.txt'
    isofit_exe = isofit_base+'/bin/isofit'

    # Get file ID and GMT time

    fid = os.path.split(args.input_radiance)[-1].split('_')[0]
    dt = datetime.strptime(fid, 'emit%Y%m%dt%H%M%S')
    dayofyear = dt.timetuple().tm_yday
    hour = dt.timetuple().tm_hour
    minute = dt.timetuple().tm_min
    gmtime = float(hour + minute / 60.)
    print('FID:', fid, 'GMT Time:',gmtime)

    # Find geographic location

    loc = envi.open(envi_header(args.input_loc)).load()
    lon, lat, elevation = loc[:,:,0], loc[:,:,1], loc[:,:,2]
    lat = np.mean(lat)
    lon = np.mean(lon)

    # Make atmospheric grids

    alt_min, alt_max = np.min(elevation)/1000.0, np.max(elevation)/1000.0
    alt_avg = (alt_max-alt_min)/2+alt_min
    alt_grid = np.arange(alt_min, alt_max, 0.2)
    alt_grid = '['+','.join([str(a) for a in alt_grid])+']'
    h2o_min, h2o_max = 0.2, 3.0
    h2o_grid = np.arange(h2o_min, h2o_max, 0.2)
    h2o_grid = '['+','.join([str(h) for h in h2o_grid])+']'
    h2o_avg = 1.5
 
    # Create subdirectories

    working_directory = os.path.abspath(args.working_directory)
    input_directory = os.path.abspath(os.path.join(working_directory,'input'))
    config_directory = os.path.abspath(os.path.join(working_directory,'config'))
    output_directory = os.path.abspath(os.path.join(working_directory,'output'))
    data_directory = os.path.abspath(os.path.join(working_directory,'data'))
    lut_directory = os.path.abspath(os.path.join(working_directory,'lut_'+fid))
    for dir in [working_directory, input_directory, output_directory,
                config_directory, data_directory, lut_directory]:
       if not os.path.exists(dir):
          os.mkdir(dir)

    # Downtrack average of radiance and OBS data
      
    I = envi.open(envi_header(args.input_obs)).load()
    lines,rows,cols = I.shape
    I = I.mean(axis=0)
    obs_path = input_directory+'/'+fid+'_subs_obs'
    envi.save_image(obs_path+'.hdr',np.array(I.reshape(1,rows,cols), 
          dtype=np.float32), ext='', force=True)

    I = envi.open(envi_header(args.input_radiance)).load()
    lines,rows,cols = I.shape
    I = I.mean(axis=0)
    rdn_path = input_directory+'/'+fid+'_subs_obs'
    envi.save_image(rdn_path+'.hdr',np.array(I.reshape(1,rows,cols), 
          dtype=np.float32), ext='', force=True)

    # Make highres wavelengths file

    wavelength_file_hires = data_directory+'/wavelengths_hires.txt'
    x = np.arange(360,2511,0.025) / 1000.0
    w = np.ones(len(x))*0.025 / 1000.0
    n = np.ones(len(x))
    D = np.c_[n,x,w]
    np.savetxt(wavelength_file_hires, D, fmt='%8.6f')

    # Standard wavelength file

    wavelength_file = data_directory+'/wavelengths.txt'
    hdr = envi.open(envi_header(args.input_radiance)).metadata.copy()
    wl = np.array([float(f) for f in hdr['wavelength']])
    fwhm = np.array([float(f) for f in hdr['fwhm']])
    c = np.arange(len(wl))
    D = np.c_[c,wl,fwhm]
    np.savetxt(wavelength_file, D, fmt='%8.6f')

    # Surface model

    surface_dir = os.path.dirname(os.path.realpath(__file__))+'/surface/'
    config = surface_dir + 'surface_constrained.json'
    surface_path = data_directory+'/surface.mat'
    surface_model(config,output_path=surface_path,wavelength_path=wavelength_file)

    # Write MODTRAN file

    template = modtran_template.format(**locals())
    modtran_template_file = config_directory+'/'+fid+'_modtran.json'
    with open(modtran_template_file,'w') as fout:
        fout.write(template)

    # Write ISOFIT config file

    rfl_path = output_directory+'/'+fid+'_rfl'
    state_path = output_directory+'/'+fid+'_state'
    template = isofit_template.format(**locals())
    config_path = config_directory+'/'+fid+'.json'
    with open(config_path,'w') as fout:
        fout.write(template)

    # Write batch script

    script_path = config_directory+'/'+fid+'.sh'
    with open(script_path,'w') as fout:
        fout.write(batch_template.format(**locals()))

    # Run

    cmd = 'sbatch -N 1 -n 1 -c 40 --mem=180G --partition=emit '+script_path 
    print(cmd)
    os.system(cmd)

    # Extract output

    state = envi.open(envi_header(state_path)).load()
    wl_shift_by_position = np.squeeze(state[0,:,-1])
    if args.flip_fpa:
        wl_shift_by_position = np.flip(wl_shift_by_position)
    use = abs(wl_shift_by_position)>1e-6
    wl_shift_by_position = wl_shift_by_position[use]
    wl_shift = np.median(wl_shift_by_position)

    # Plot the wavelength shift by position

    x = np.where(use)[0]
    plt.plot(x, wl_shift_by_position, color+'.', markersize=2)
    poly = np.polyfit(x, wl_shift_by_position, 1)
    plt.plot(x, np.polyval(poly, x), 'k')

    # Write to file
   
    print(np.median(np.array(wl_shifts), axis=0))
    plt.xlabel('Position')
    plt.ylabel('Wavelength shift (nm)')
    plt.title(fid)
    plt.grid(True)
    plt.box(False)
    plt.ylim([-1.00,1.00])
    plt.savefig('output/'+fid+'_plot.pdf')
    np.savetxt('output/'+fid+'_shift_by_position.txt', 
               np.c_[x,wl_shift_by_position], fmt='%8.6f')


 
if __name__ == '__main__':
   main()
