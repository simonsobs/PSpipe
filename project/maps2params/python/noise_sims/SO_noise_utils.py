import numpy as np


def steve_effective_fsky(mask):

    """ Hivon formula for fsky which works for equal area projections """
    npix = len(mask.reshape(-1))
    non0 = np.where(mask != 0)[0]
    fs = len(non0) / float(npix)
    w2 = np.sum(mask[non0] ** 2) / (npix * fs)
    w4 = np.sum(mask[non0] ** 4) / (npix * fs)
    
    return fs * w2 **2  / w4



