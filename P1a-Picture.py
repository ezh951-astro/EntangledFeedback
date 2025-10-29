import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import zHead as hd
from astropy.cosmology import FlatLambdaCDM

import TorreyLabTools.visualization.contour_makepic as makepic
import TorreyLabTools.util.calc_hsml as calc_hsml

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import imageio.v2 as imageio

import contextlib

arg_options = hd.handle_arguments()
ActiveRunIDs = arg_options[0]
print(ActiveRunIDs)
date = arg_options[1]

ActiveRunIDs = [5,4,3]
iSnaps = [500,490,800]
SimColors = ['C3','C2','C0']
SimNames = ['LYRA','RIGEL','SMUGGLE']

CALCULATE = True

RMAX = 1.5
NBINS = 200
SMOOTH_NGB = 16

LOGMASS_MIN = -1.5
LOGMASS_MAX = 1.5

LOGTEMP_MIN = 3.5
LOGTEMP_MAX = 5.5

ALPHA_MIN = 0.2
ALPHA_MAX = 1.0

cmap, norm = hd.create_cmap( 'viridis', lo=LOGTEMP_MIN, hi=LOGTEMP_MAX, Nbins=20 )

def fancy_pic( x,y,z,m, LOG_LO, LOG_HI, smooth_ngb=SMOOTH_NGB, func_scale=1.e0 ):

    gas_hsml = calc_hsml.get_particle_hsml( x,y,z, DesNgb=smooth_ngb )
    print( gas_hsml )

    massmap_0f,face_image_gas = makepic.contour_makepic( x,y,z, gas_hsml, func_scale*m/1.e10 , # must have mass in original sim units for some reason
        xlen = RMAX,
        pixels = NBINS, set_aspect_ratio = 1.0,
        set_maxden = func_scale * 10.0**LOG_HI/1.e4, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
        set_dynrng = 10.0**(LOG_HI-LOG_LO)  )

    face_image_gas = np.array(face_image_gas)
    csfac_gas = (LOG_HI-LOG_LO) / 256.

    return face_image_gas * csfac_gas + LOG_LO

def do_calculations(i,j):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    s = Snap.s

    origin, vcom, Ltot = hd.adjustdata( Snap, subhalo_id=0, rad=3.0 )
    print('Obtained Origin')

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### LOAD THINGS

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', verbose=True, **snap_kwargs )
    m0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )

    U0    = hd.extract( f,u'InternalEnergy',0, **snap_kwargs )
    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )

    xyz0 -= origin

    kkRad0 = hd.calc_distance(xyz0) < RMAX*np.sqrt(3)

    xyz0 = xyz0[ kkRad0 ]
    m0   = m0[ kkRad0 ]

    U0 = U0[ kkRad0 ]
    Nelec = Nelec[ kkRad0 ]
    temp = hd.GasTemp( U0, Nelec )

    ### RETURN ANSWER

    mass_hist_xy = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], m0, LOGMASS_MIN, LOGMASS_MAX )
    temp_xy      = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], m0*temp, LOGMASS_MIN+LOGTEMP_MIN, LOGMASS_MAX+LOGTEMP_MAX )

    temp_xy  /= mass_hist_xy

    return np.log10(mass_hist_xy)[:-1,:-1], np.log10(temp_xy)[:-1,:-1]

### ### ###

fig,axs = plt.subplots(1,3,figsize=(7,3))
plt.subplots_adjust(wspace=0.05)

for k,(j,i,SimColor,SimName) in enumerate(zip( ActiveRunIDs, iSnaps, SimColors, SimNames )):

    if CALCULATE:
        massIm, tempIm = do_calculations(i,j)
        np.save('data/massIm{}.npy'.format(j),massIm)
        np.save('data/tempIm{}.npy'.format(j),tempIm)
    else:
        massIm = np.load('data/massIm{}.npy'.format(j))
        tempIm = np.load('data/tempIm{}.npy'.format(j))

    m_alpha = (ALPHA_MAX-ALPHA_MIN) / (LOGMASS_MAX-LOGMASS_MIN)
    alpha_density = np.where( massIm>LOGMASS_MIN, m_alpha*(massIm-LOGMASS_MIN)+ALPHA_MIN , np.zeros_like(massIm) ) # min density is 0.3, max density is 1.0, below min density is 0.

    im = axs[k].imshow( tempIm, alpha=alpha_density, extent=(-RMAX,RMAX,-RMAX,RMAX), origin='lower', cmap=cmap, norm=norm )

    axs[k].set_xticks([])
    axs[k].set_yticks([])

    axs[k].tick_params(color=SimColor, labelcolor='green')
    for spine in axs[k].spines.values():
        spine.set_edgecolor(SimColor)
        spine.set_linewidth(4)

    axs[k].set_title(SimName,color=SimColor)

feet = fm.FontProperties(size=16)

scalebar0 = AnchoredSizeBar( axs[2].transData, 0.5, '500 pc', 'lower right', pad=0.15, color='white', frameon=False, size_vertical=0.025, fontproperties=feet )
axs[2].add_artist(scalebar0)

fig.savefig('OfficialPlots/P1a-Picture.png',dpi=360)
plt.show()