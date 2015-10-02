import scipy
import numpy as NP
from math import pi

def angulardistance(ra1, dec1, ra2, dec2):
    degrad = pi/180
    ra1 = scipy.where(ra1-ra2>pi,ra1-2*pi,ra1)
    ra2 = scipy.where(ra2-ra1>pi,ra2-2*pi,ra2)
    div = NP.sin(dec2)*NP.sin(dec1)+NP.cos(dec2)*NP.cos(dec1)*NP.cos(ra1-ra2)
    del_a = NP.cos(dec2)*NP.sin(ra1-ra2)/(div*degrad)
    del_d = -(NP.sin(dec2)*NP.cos(dec1)-NP.cos(dec2)*NP.sin(dec1)*NP.cos(ra1-ra2)/div)/degrad
    del_r = NP.sqrt(del_a**2+del_d**2)
    phi = NP.arctan2(NP.abs(del_a), NP.abs(del_d))
    
    return del_r, phi

def getmask(masking_matrix, lower_limit, upper_limit):
    mask_low = masking_matrix >= lower_limit
    mask_high = masking_matrix < upper_limit
    mask = mask_low*mask_high
    
    return mask

