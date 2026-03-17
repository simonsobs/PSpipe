description = """
Compute masks from ivar (smooth+threshold on non-zeros percentile).
May also use xlink if available.
Optionally plots all ivar and maps
"""

from pspy import so_dict, so_map, so_window, pspy_utils
from pspipe_utils import log

from pixell import enmap, enplot      

import numpy as np
import yaml

import os
from os.path import join as opj
import sys
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--use-xlink', action='store_true',
                    help='If passed, also use coadd xlink files to help make mask')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

use_xlink = args.use_xlink

# log mask infos from mask yaml file
with open(d['xtra_mask_yaml'], "r") as f:
    mask_dict: dict = yaml.safe_load(f)
mask_infos = mask_dict['get_xtra_mask.py']

mask_dir = d['mask_dir']

save_plot_mask = mask_infos['save_plot_mask'] # should we make and save the plots of masks?
save_plot_maps_ivar = mask_infos['save_plot_maps_ivar'] # should we save the plotted maps and ivars?

if save_plot_mask or save_plot_maps_ivar:
    plot_dir_mask = opj(d['plots_dir'], 'mask')
    pspy_utils.create_directory(plot_dir_mask)

if save_plot_maps_ivar:
    plot_dir_map_ivar = opj(plot_dir_mask, 'maps_ivar')
    pspy_utils.create_directory(plot_dir_map_ivar)

# get reasonable ivar, top 90% of nonzero values seems to work decently
ivar_smooth_deg = mask_infos['ivar_smooth_deg']
ivar_quantile = mask_infos['ivar_quantile']
if use_xlink:
    xlink_smooth_deg = mask_infos['xlink_smooth_deg']
    xlink_quantile = mask_infos['xlink_quantile']

ivar_mask_intersect = True # intersection of every single mask
ivar_mask_union = False # union of every single mask
if use_xlink:
    xlink_mask_intersect = True # intersection of every single mask
    xlink_mask_union = False # union of every single mask

for sv in mask_infos['surveys_to_xtra_mask']:
    for m in d[f'arrays_{sv}']:
        ivar_mask = True # intersection of masks just for this map (over splits)
        
        map_fns = d[f'maps_{sv}_{m}']
        for i, map_fn in enumerate(map_fns):
            # mix-in 0's from the splits
            if d[f"src_free_maps_{sv}"] == True:
                map_fn = map_fn.replace('_srcfree', '')
            ivar_fn = map_fn.replace('_map', '_ivar')

            # mask is based on the smoothed ivar map
            # only the pixels where the original ivar were nonzero though
            ivar = enmap.read_map(ivar_fn)
            ivar = ivar.reshape(-1, *ivar.shape[-2:])[0] # the "first" ivar map

            ivar_smooth = enmap.smooth_gauss(ivar, np.deg2rad(ivar_smooth_deg))
            ivar_set_mask = ivar_smooth > np.quantile(ivar_smooth[ivar > 0], ivar_quantile)
            ivar_set_mask *= ivar > 0
            
            # check that inside of ivar_mask, there are no zero ivar
            assert np.all(ivar[ivar_set_mask] > 0), \
                f'{sv}, {m}, set{i} has zero ivar inside ivar_mask'
        
            # possibly plot maps and ivars
            if save_plot_maps_ivar:
                log.info(f'plot {sv}, {m}, set{i} map and ivar')

                map = enmap.read_map(map_fn)
                p = enplot.plot(map, downgrade=8, ticks=1, colorbar=True, range=[1000, 300, 300])
                map_plot_fn = os.path.splitext(os.path.basename(map_fn))[0]
                enplot.write(opj(plot_dir_map_ivar, map_plot_fn), p)

                p = enplot.plot(ivar, downgrade=8, ticks=1, colorbar=True)
                ivar_plot_fn = os.path.splitext(os.path.basename(ivar_fn))[0]
                enplot.write(opj(plot_dir_map_ivar, ivar_plot_fn), p)

            log.info(f'{sv}, {m}, set{i} ivar survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_set_mask)) / (4 * np.pi) * 41253:.5f}')

            # save mask
            ivar_set_mask_fn = opj(mask_dir, f'xtra_ivar_mask_{sv}_{m}_set{i}.fits')
            enmap.write_map(ivar_set_mask_fn, ivar_set_mask.astype(np.float32))

            # save plot of mask
            if save_plot_mask:
                p = enplot.plot(ivar_set_mask, downgrade=8, ticks=1, colorbar=True)
                ivar_set_mask_plot_fn = os.path.splitext(os.path.basename(ivar_set_mask_fn))[0]
                enplot.write(opj(plot_dir_mask, ivar_set_mask_plot_fn), p)

            ivar_mask = np.logical_and(ivar_mask, ivar_set_mask)

        log.info(f'{sv}, {m} ivar survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_mask)) / (4 * np.pi) * 41253:.5f}')

        # also make a mask based on the xlink
        if use_xlink:
            coadd_map_fn = map_fns[0].replace('_00_', '_coadd_')
            if d[f"src_free_maps_{sv}"] == True:
                coadd_map_fn = coadd_map_fn.replace('_srcfree', '')
            coadd_xlink_fn = coadd_map_fn.replace('_map', '_xlink')
        
            xlink_downgrade = d['xlink_downgrade']
            xlink = enmap.read_map(coadd_xlink_fn).upgrade(xlink_downgrade)
            xlink = np.divide(np.sqrt(xlink[1]**2 + xlink[2]**2), xlink[0], where=xlink[0]!=0) * (xlink[0]!=0)

            xlink_smooth = enmap.smooth_gauss(xlink, np.deg2rad(xlink_smooth_deg))
            xlink_mask = xlink_smooth < np.quantile(xlink_smooth[xlink > 0], xlink_quantile)
            xlink_mask *= xlink > 0

            log.info(f'{sv}, {m} xlink survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(xlink_mask)) / (4 * np.pi) * 41253:.5f}')

        # also build xtra masks that are the union and intersection
        # of all the xtra masks
        ivar_mask_intersect = np.logical_and(ivar_mask_intersect, ivar_mask)
        ivar_mask_union = np.logical_or(ivar_mask_union, ivar_mask)

        if use_xlink:
            xlink_mask_intersect = np.logical_and(xlink_mask_intersect, xlink_mask)
            xlink_mask_union = np.logical_or(xlink_mask_union, xlink_mask)

        # save mask
        ivar_mask_fn = opj(mask_dir, f'xtra_ivar_mask_{sv}_{m}.fits')
        enmap.write_map(ivar_mask_fn, ivar_mask.astype(np.float32))

        if use_xlink:
            xlink_mask_fn = opj(mask_dir, f'xtra_xlink_mask_{sv}_{m}.fits')
            enmap.write_map(xlink_mask_fn, xlink_mask.astype(np.float32))

        # save plot of mask
        if save_plot_mask:
            p = enplot.plot(ivar_mask, downgrade=8, ticks=1, colorbar=True)
            ivar_mask_plot_fn = os.path.splitext(os.path.basename(ivar_mask_fn))[0]
            enplot.write(opj(plot_dir_mask, ivar_mask_plot_fn), p)

            if use_xlink:
                p = enplot.plot(xlink_mask, downgrade=8, ticks=1, colorbar=True)
                xlink_mask_plot_fn = os.path.splitext(os.path.basename(xlink_mask_fn))[0]
                enplot.write(opj(plot_dir_mask, xlink_mask_plot_fn), p)
            
log.info(f'All ivar intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_mask_intersect)) / (4 * np.pi) * 41253:.5f}')
log.info(f'All ivar union survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(ivar_mask_union)) / (4 * np.pi) * 41253:.5f}')
if use_xlink:
    log.info(f'All xlink intersection survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(xlink_mask_intersect)) / (4 * np.pi) * 41253:.5f}')
    log.info(f'All xlink union survey solid angle : {so_window.get_survey_solid_angle(so_map.from_enmap(xlink_mask_union)) / (4 * np.pi) * 41253:.5f}')

# plot and save union and intersect masks
p = enplot.plot(ivar_mask_intersect, downgrade=8, ticks=1, colorbar=True)
enplot.write(opj(plot_dir_mask, f'xtra_ivar_mask_intersect'), p)
enmap.write_map(opj(mask_dir, f'xtra_ivar_mask_intersect.fits'), ivar_mask_intersect.astype(np.float32))

p = enplot.plot(ivar_mask_union, downgrade=8, ticks=1, colorbar=True)
enplot.write(opj(plot_dir_mask, f'xtra_ivar_mask_union'), p)
enmap.write_map(opj(mask_dir, f'xtra_ivar_mask_union.fits'), ivar_mask_union.astype(np.float32))

if use_xlink:
    p = enplot.plot(xlink_mask_intersect, downgrade=8, ticks=1, colorbar=True)
    enplot.write(opj(plot_dir_mask, f'xtra_xlink_mask_intersect'), p)
    enmap.write_map(opj(mask_dir, f'xtra_xlink_mask_intersect.fits'), xlink_mask_intersect.astype(np.float32))

    p = enplot.plot(xlink_mask_union, downgrade=8, ticks=1, colorbar=True)
    enplot.write(opj(plot_dir_mask, f'xtra_xlink_mask_union'), p)
    enmap.write_map(opj(mask_dir, f'xtra_xlink_mask_union.fits'), xlink_mask_union.astype(np.float32))
