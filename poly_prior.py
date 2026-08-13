# Imports

import argparse
import logging

import numpy as np
import scipy.io



def main():
    parser = argparse.ArgumentParser(description='Generate polynomial prior')
    parser.add_argument('input_file', type=str, help='Input file path')
    parser.add_argument('output_file', type=str, help='Output file path')
    parser.add_argument('--prior_index', type=int, nargs='+', default=[0], help='Index of the prior to use (default: 0)')
    parser.add_argument('--poly_degree', type=int, default=3, help='Degree of the polynomial (default: 3)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    args = parser.parse_args()

    if args.plot:
        from matplotlib import gridspec as grid_spec
        import matplotlib.pyplot as plt

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f'Processing {args.input_file}')

    reference_mat = scipy.io.loadmat(args.input_file)
    wl = reference_mat['wl'] 
    n_wavelengths = wl.shape[1]
    means = reference_mat['means'].copy() 

    windows = [
        [wl[0, :].min(), 1400],      # Bottom to 1400 nm
        [1400, 1800],                 # 1400 to 1800 nm
        [1800, wl[0, :].max()]       # 1800 to 2500 nm
    ]

    poly_means = means.copy()
    for class_idx in args.prior_index:
        for win_start, win_end in windows:
            # Find wavelengths in this window
            mask = np.logical_and(wl[0, :] >= win_start, wl[0, :] <= win_end)
            wl_window = wl[0, mask]
            mean_window = means[class_idx, mask]

            # Fit polynomial
            poly_coeffs = np.polyfit(wl_window, mean_window, args.poly_degree)

            # Insert polynomial
            poly_means[class_idx, mask] = np.polyval(poly_coeffs, wl_window)

    
    poly_orig_cov_mat = {
        'normalize': reference_mat['normalize'],
        'wl': wl,
        'means': poly_means,  # Using polynomial-smoothed mean for class 1, original for others
        'covs': reference_mat['covs'],  # Using ORIGINAL covariance matrices
        'attribute_means': reference_mat['attribute_means'],
        'attribute_covs': reference_mat['attribute_covs'],
        'attributes': reference_mat['attributes'],
        'refwl': reference_mat['refwl']
    }

    if 'fwhm' in reference_mat:
        poly_orig_cov_mat['fwhm'] = reference_mat['fwhm']
    if 'selection_metric' in reference_mat:
        poly_orig_cov_mat['selection_metric'] = reference_mat['selection_metric']
    if 'surface_categories' in reference_mat:
        poly_orig_cov_mat['surface_categories'] = reference_mat['surface_categories']

    scipy.io.savemat(args.output_file, poly_orig_cov_mat)
    logging.info(f'Output written to {args.output_file}')

    
    if args.plot:
        plt.figure(figsize=(10, 6))
        for class_idx in args.prior_index:
            plt.plot(wl[0, :], means[class_idx, :], label=f'Original Class {class_idx}')
            plt.plot(wl[0, :], poly_means[class_idx, :], label=f'Polynomial Class {class_idx}', linestyle='--')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Mean Reflectance')
        plt.title('Polynomial Prior Generation')
        plt.legend()
        plt.grid()
        plt.savefig('poly_prior_plot.png')


if __name__ == '__main__':
    main()