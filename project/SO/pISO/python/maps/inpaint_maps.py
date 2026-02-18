from pixell import enmap, enplot 
import numpy as np
from os.path import join as opj

# TODO: change this to the right directories and filename templates
map_dir = '/home/zatkins/scratch/projects/lat-iso/piso/maps/lat/deep56/20260201_skn/raw'
inpaint_dir = '/home/zatkins/scratch/projects/lat-iso/piso/maps/lat/deep56/20260201_skn/inpainted'
map_fn_template = 'deep56_full_{tube}_4way_0{split}_{freq}_sky_{maptype}.fits'

# TODO: change these to what you need. the ivar for the bad splits also has to be
# inpainted based on the splits that are ok. if all the splits are bad for some
# map, we need to figure out a different way to get the ivar in the inpaint
# region
maps_to_inpaint = ['i6_f090', 'i6_f150']
maps_to_inpaint2splits_to_inpaint = {
    'i6_f090': [1],
    'i6_f150': [1]
}
nsplits = 4

# TODO: change this to whatever region you want to define as "the planet region",
# or use this, doesn't matter
center = (-1, 1)
thumb_width = 4

# see below for how these work
inpaint_dist = 0.25 # this is the distance in degrees from the edge of any holes that will be inpainted
cvar_fac = 6 # add up the ivar from the good splits, this is "cvar". divide cvar by cvar_fac to get the ivar for the inpainted splits.
             # 6 happened to work well for i6 baseline maps. probably need to play around with this, cause it might not work if more
             # than 1 split needs to be inpainted or a different tube etc.
mean_dist = 0.1 # this is how far from the edge of the inpainted region to use to determine the mean to
                # fill the inpainted region with (deg)
thumb_apod = 0.1 # this is the apod width on the thumbnail. we smooth the entire thumbnail after apod to
                 # bring in some large scale features on top of the mean (deg)
smooth_width = 0.1 # this is how wide of the smoothing we should use (deg). this seemed to work well

# silly seed
seed = 51255
rng = np.random.default_rng(seed=seed)

for m in maps_to_inpaint:
    tube, freq = m.split('_')

    splits_to_inpaint = maps_to_inpaint2splits_to_inpaint[m]

    cvar = 0
    for k in range(nsplits):
        if k not in splits_to_inpaint:
            cvar += enmap.read_map(opj(map_dir, map_fn_template.format(tube=tube, split=k, freq=freq, maptype='ivar')))
    
    for split in splits_to_inpaint:
        ivar = enmap.read_map(opj(map_dir, map_fn_template.format(tube=tube, split=split, freq=freq, maptype='ivar')))
        imap = enmap.read_map(opj(map_dir, map_fn_template.format(tube=tube, split=split, freq=freq, maptype='map0050')))

        # get thumbnail
        box = [[center[0] - thumb_width/2, center[1] + thumb_width/2],
            [center[0] + thumb_width/2, center[1] - thumb_width/2]]
        fr, to, _ = ivar.subinds(np.deg2rad(box))
        sel = np.s_[..., fr[0]:to[0], fr[1]:to[1]]
        
        ivar_thumb = ivar[sel]
        cvar_thumb = cvar[sel]
        imap_thumb = imap[sel]

        print(m)
        p = enplot.plot(ivar[sel], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_ivar_thumb_raw'), p)
        p = enplot.plot(imap[sel], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_imap_thumb_raw'), p)

        # get the pixels to be inpainted, the pixels from which we get the mean, and
        # the apod mask for the smoothing the thumbnail
        bad_pix = ivar_thumb == 0
        pix_to_inpaint = enmap.grow_mask(bad_pix, np.deg2rad(inpaint_dist))
        pix_for_mean = enmap.grow_mask(pix_to_inpaint, np.deg2rad(mean_dist))
        pix_for_mean *= np.logical_not(pix_to_inpaint)
        mask_apod = enmap.apod_mask(enmap.ones(*ivar_thumb.geometry, dtype=np.float32), np.deg2rad(thumb_apod))

        p = enplot.plot([pix_to_inpaint, pix_for_mean, mask_apod], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_pix_to_inpaint_and_pix_for_mean_and_mask_apod'), p)

        # inpaint the ivar
        ivar_thumb[pix_to_inpaint] = cvar_thumb[pix_to_inpaint] / cvar_fac
        
        p = enplot.plot(ivar[sel], colorbar=True, ticks=1, min=0)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_ivar_thumb_inpaint'), p)

        # inpaint the mean
        mean = imap_thumb[:, pix_for_mean].mean(axis=(1))
        for i in range(3):
            imap_thumb[i, pix_to_inpaint] = mean[i]

        p = enplot.plot(imap[sel], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_imap_thumb_mean_inpaint'), p)

        # add smoothed stuff 
        smooth_thumb = enmap.smooth_gauss(imap_thumb*mask_apod, np.deg2rad(smooth_width))
        for i in range(3):
            imap_thumb[i, pix_to_inpaint] = smooth_thumb[i, pix_to_inpaint]

        p = enplot.plot(imap[sel], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_imap_thumb_smooth_inpaint'), p)

        # add white noise
        facs = [1, 2, 2]
        for i, fac in enumerate(facs):
            sig_thumb = (np.divide(fac, ivar_thumb, where=ivar_thumb!=0) * (ivar_thumb!=0))**0.5
            imap_thumb[i, pix_to_inpaint] += (sig_thumb * rng.standard_normal(size=sig_thumb.shape, dtype=np.float32))[pix_to_inpaint]

        p = enplot.plot(imap[sel], colorbar=True, ticks=1)
        enplot.write(opj(inpaint_dir, f'{tube}_{freq}_set{split}_imap_thumb_white_inpaint'), p)

        # save
        enmap.write_map(opj(inpaint_dir, map_fn_template.format(tube=tube, split=split, freq=freq, maptype='ivar_inpaint')), ivar)
        enmap.write_map(opj(inpaint_dir, map_fn_template.format(tube=tube, split=split, freq=freq, maptype='map0050_inpaint')), imap)