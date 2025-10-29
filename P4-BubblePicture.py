import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import zHead as hd
from astropy.cosmology import FlatLambdaCDM

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from cmcrameri import cm as cmsci

import imageio.v2 as imageio

import TorreyLabTools.visualization.contour_makepic as makepic
import TorreyLabTools.util.calc_hsml as calc_hsml

arg_options = hd.handle_arguments()
ActiveRunIDs = arg_options[0]

ActiveRunIDs = [5]

print(ActiveRunIDs)
date = arg_options[1]

###

CorotationDataAll = np.loadtxt('data/corotation_LYRA.txt')

#mode = 'FANCY'

RMAX = 0.5  # kpc
ZMAX = 2.0 # kpc

NBINS = 100

SMOOTH_NGB = 8

NBINS_RATIO = int( ZMAX / RMAX )

LOGMASS_MIN = -2.5
LOGMASS_MAX = 1.5

LOGTEMP_MIN = 3.0
LOGTEMP_MAX = 7.0

LOGVEL_MIN = 0.0
LOGVEL_MAX = 2.0

LOGMETALS_MIN = -1.5
LOGMETALS_MAX = 0.5

LOGSPECEGY_MIN = 1.75
LOGSPECEGY_MAX = 5.75
ENERGY_SHIFT = 43.25

LOGERATIO_MIN = -0.2
LOGERATIO_MAX = 1.0

E_RATIO_PLOT = True

AGE_THRESH = 50.

#ALPHA_MIN = 0.3
#ALPHA_MAX = 1.0

###

levels0 = MaxNLocator(nbins=16).tick_values(LOGMASS_MIN,LOGMASS_MAX)
cmap0 = plt.get_cmap('magma')
norm0 = BoundaryNorm(levels0, ncolors=cmap0.N, clip=True)

levels1 = MaxNLocator(nbins=10).tick_values(LOGVEL_MIN,LOGVEL_MAX)
cmap1 = plt.get_cmap('winter')
norm1 = BoundaryNorm(levels1, ncolors=cmap1.N, clip=True)

levels2 = MaxNLocator(nbins=16).tick_values(LOGTEMP_MIN,LOGTEMP_MAX)
cmap2 = plt.get_cmap('coolwarm')
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=True)

levels3 = MaxNLocator(nbins=8).tick_values(LOGMETALS_MIN,LOGMETALS_MAX)
cmap3 = plt.get_cmap('copper_r')
norm3 = BoundaryNorm(levels3, ncolors=cmap3.N, clip=True)

levels4 = MaxNLocator(nbins=10).tick_values(LOGVEL_MIN,LOGVEL_MAX)
cmap4 = plt.get_cmap('viridis')
norm4 = BoundaryNorm(levels4, ncolors=cmap4.N, clip=True)

if E_RATIO_PLOT:
    levels5 = MaxNLocator(nbins=12).tick_values( LOGERATIO_MIN, LOGERATIO_MAX )
    cmap5 = plt.get_cmap('RdYlGn') #cmsci.vanimo, no quotes
    norm5 = BoundaryNorm(levels5, ncolors=cmap5.N, clip=True)
else:
    levels5 = MaxNLocator(nbins=16).tick_values( LOGSPECEGY_MIN + ENERGY_SHIFT, LOGSPECEGY_MAX + ENERGY_SHIFT )
    cmap5 = plt.get_cmap('RdYlGn_r')
    norm5 = BoundaryNorm(levels5, ncolors=cmap5.N, clip=True)

cmaps = [cmap0, cmap1, cmap2, cmap3, cmap4, cmap5]
norms = [norm0, norm1, norm2, norm3, norm4, norm5]

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

def do_calculations(i,j,ROT_RATE,ROT_INTERCEPT,GALRAD):

    Snap = hd.Snapshot(i,j)
    f = Snap.f

    #SID = 0

    print('NOW CALCULATING LYRA (RUN NO. {}) SNAPSHOT NO. {} '.format(j,i))

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### LOAD THINGS

    # Load Coordinates
    xyz0raw = hd.extract( f,u'Coordinates',0, units='gal', **snap_kwargs ) - 5000.
    vel0raw = hd.extract( f,u'Velocities',0, units='gal', **snap_kwargs )

    xyz4raw = hd.extract( f,u'Coordinates',4, units='gal', **snap_kwargs ) - 5000.

    age4 = Snap.find_star_age( hd.extract( f,u'GFM_StellarFormationTime',4, **snap_kwargs ) )

    # rotate the whole galaxy so that its comoving. first grab the rotation rates
    theta = ( ROT_RATE * Snap.SimTime + ROT_INTERCEPT ) * np.pi / 180. # degrees converted into radians
    omega = ROT_RATE * ( 525600. * 60. * 1.e6 * 180. / 3.0857e16 / np.pi )**(-1.0) # converted into km/s / kpc

    print('Rotation Rate: {} km/s/kpc'.format( omega ))
    print('Galactocentric radius: {} kpc'.format( GALRAD ))
    print('Local Circular Velocity: {} km/s'.format( omega * GALRAD ))

    # change the coordinates x y z via rotation matrix
    xNew =  np.cos(theta)*xyz0raw[:,0] + np.sin(theta)*xyz0raw[:,1]
    yNew = -np.sin(theta)*xyz0raw[:,0] + np.cos(theta)*xyz0raw[:,1]
    xyz0 =  np.transpose(np.array([ xNew, yNew, xyz0raw[:,2] ]))

    x4New =  np.cos(theta)*xyz4raw[:,0] + np.sin(theta)*xyz4raw[:,1]
    y4New = -np.sin(theta)*xyz4raw[:,0] + np.cos(theta)*xyz4raw[:,1]
    xyz4 =  np.transpose(np.array([ x4New, y4New, xyz4raw[:,2] ]))

    # rotating the positions is just a change of coordinates. Now we need to actually convert the velocities by rotating AND adjusting for corotation
    vxNew =  np.cos(theta)*vel0raw[:,0] + np.sin(theta)*vel0raw[:,1] - ( -omega*yNew )
    vyNew = -np.sin(theta)*vel0raw[:,0] + np.cos(theta)*vel0raw[:,1] - (  omega*xNew )
    vel0  =  np.transpose(np.array([ vxNew, vyNew, vel0raw[:,2] ]))

    # set the origin to (GALRAD,0,0) or rather (5000.+GALRAD,5000,5000)
    origin = np.array([ GALRAD, 0.0, 0.0 ])
    xyz0 -= origin
    xyz4 -= origin

    # Load Everything Else
    mass  = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )
    U0    = hd.extract( f,u'InternalEnergy',0, **snap_kwargs )
    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )

    metals = hd.extract( f,u'GFM_Metals',0, **snap_kwargs ) / 0.012 # hydrogen, helium, 8 other elements, in absolute units, converted to solar

    ### Restrict Particles

    kk0 = ( np.sqrt(xyz0[:,0]**2.+xyz0[:,1]**2.) < RMAX*np.sqrt(2) ) * ( np.abs(xyz0[:,2])<ZMAX )

    xyz0  = xyz0[kk0]
    vel0  = vel0[kk0]
    mass  = mass[kk0]
    U0    = U0[kk0]
    Nelec = Nelec[kk0]

    metals = np.sum( metals[kk0,2:], axis=1 )

    kkAge = age4 < AGE_THRESH

    ### CALCULATE THINGS

    # Temperature
    temp = hd.GasTemp( U0, Nelec )

    vel0r = np.sqrt( vel0[:,0]*vel0[:,0] + vel0[:,1]*vel0[:,1] ) # all transverse velocity; radial and angular
    vel0z = np.where( xyz0[:,2]>0., vel0[:,2], -1.*vel0[:,2] )

    # Total Specific Energy
    v2 = vel0[:,0]*vel0[:,0] + vel0[:,1]*vel0[:,1] + vel0[:,2]*vel0[:,2]
    TotEnergy = (v2 + U0) # (km/s)^2

    #ERatio = v2/U0

    #if mode=='FAST':
        # mass per area; this is just morphology
        #mass_hist_xy = hd.ptcl_density_bin( xyz0[:,0], xyz0[:,1], mass, min1=-RMAX, max1=RMAX, min2=-RMAX, max2=RMAX, Nbins1=NBINS, Nbins2=NBINS ) # face on
        # mass*temp per area
        #temp_xy = hd.ptcl_density_bin( xyz0[:,0], xyz0[:,1], mass*temp, min1=-RMAX, max1=RMAX, min2=-RMAX, max2=RMAX, Nbins1=NBINS, Nbins2=NBINS ) # face on
        # mass * transverse velocity per area
        #vproj_xy = hd.ptcl_density_bin( xyz0[:,0], xyz0[:,1], mass*vel0r, min1=-RMAX, max1=RMAX, min2=-RMAX, max2=RMAX, Nbins1=NBINS, Nbins2=NBINS ) # face on
        # mass * perpendicular velocity per area
        #vz_xy = hd.ptcl_density_bin( xyz0[:,0], xyz0[:,1], mass*vel0z, min1=-RMAX, max1=RMAX, min2=-RMAX, max2=RMAX, Nbins1=NBINS, Nbins2=NBINS ) # face on
        #vz2_xy = hd.ptcl_density_bin( xyz0[:,0], xyz0[:,1], mass*vel0z*vel0z, min1=-RMAX, max1=RMAX, min2=-RMAX, max2=RMAX, Nbins1=NBINS, Nbins2=NBINS ) # face on

    #if mode=='FANCY':

    mass_hist_xy = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass, LOGMASS_MIN, LOGMASS_MAX )
    temp_xy      = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*temp,  LOGMASS_MIN+LOGTEMP_MIN, LOGMASS_MAX+LOGTEMP_MAX )
    vproj_xy     = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*vel0r, LOGMASS_MIN+LOGVEL_MIN,  LOGMASS_MAX+LOGVEL_MAX )
    vz_xy        = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*vel0z, LOGMASS_MIN+LOGVEL_MIN,  LOGMASS_MAX+LOGVEL_MAX )
    vz2_xy       = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*vel0z*vel0z, LOGMASS_MIN + 2.*LOGVEL_MIN, LOGMASS_MAX + 2.*LOGVEL_MAX )

    metals_xy = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*metals, LOGMASS_MIN + LOGMETALS_MIN, LOGMASS_MAX + LOGMETALS_MAX )
    energy_xy = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*TotEnergy, LOGMASS_MIN + LOGSPECEGY_MIN, LOGMASS_MAX + LOGSPECEGY_MAX )
    KE_xy     = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*v2, LOGMASS_MIN + LOGSPECEGY_MIN, LOGMASS_MAX + LOGSPECEGY_MAX )
    TE_xy     = 10.**fancy_pic( xyz0[:,0],xyz0[:,1],xyz0[:,2], mass*U0, LOGMASS_MIN + LOGSPECEGY_MIN, LOGMASS_MAX + LOGSPECEGY_MAX )

    # normalize
    vproj_xy /= mass_hist_xy
    temp_xy  /= mass_hist_xy
    vz_xy    /= mass_hist_xy
    vz2_xy   /= mass_hist_xy

    metals_xy /= mass_hist_xy
    energy_xy /= mass_hist_xy
    KE_xy     /= mass_hist_xy
    TE_xy     /= mass_hist_xy

    # z velocity dispersion
    sigproj_xy = np.sqrt( vz2_xy - vz_xy*vz_xy )

    # critical ratio theory
    eratio_xy = KE_xy / TE_xy

    if E_RATIO_PLOT:
        gas_map = [ np.log10(mass_hist_xy), np.log10(vproj_xy), np.log10(temp_xy), np.log10(metals_xy), np.log10(sigproj_xy), np.log10(eratio_xy) ]
    else:
        gas_map = [ np.log10(mass_hist_xy), np.log10(vproj_xy), np.log10(temp_xy), np.log10(metals_xy), np.log10(sigproj_xy), np.log10(energy_xy)+ENERGY_SHIFT ]

    #if mode=='FANCY':
    gas_map = [ np.transpose(gas_map_entry[:-1,:-1]) for gas_map_entry in gas_map ]

    ### RETURN ANSWER

    print( xyz4[kkAge] )
    return Snap.SimTime, gas_map, xyz4[kkAge]

j = 5
i = 723

iClus = 47
CoRotDat = CorotationDataAll[47]

Radius = CoRotDat[0]
RotRate = CoRotDat[1]
RotIcpt = CoRotDat[2]
#nStart = int(CoRotDat[3])
#nEnd   = int(CoRotDat[4])

# Calculate
SimTime, gas_map, young_stars = do_calculations(i,j,RotRate,RotIcpt,Radius)

# Plot
figR, axsR = plt.subplots(2,3,figsize=(12,6.5))

#m_alpha = (ALPHA_MAX-ALPHA_MIN) / (LOGMASS_MAX-LOGMASS_MIN)

#alpha_density = np.where( gas_map[0]>LOGMASS_MIN, m_alpha*(gas_map[0]-LOGMASS_MIN)+ALPHA_MIN , np.zeros_like(gas_map[0]) ) # min density is 0.3, max density is 1.0, below min density is 0.
#alphas = [ np.ones_like(alpha_density), alpha_density, alpha_density, alpha_density ]

nActual = np.shape(gas_map[0])[0]
x,y = np.meshgrid( np.linspace(-RMAX,RMAX,nActual), np.linspace(-RMAX,RMAX,nActual) )

for kcol in range(6):

    print('Map {}'.format(kcol))
    print(gas_map[kcol])

    #im = axsR[kcol//2,kcol%2].imshow( gas_map[kcol], extent=(-RMAX,RMAX,-RMAX,RMAX), origin='lower', cmap=cmaps[kcol], norm=norms[kcol], alpha=alphas[kcol] )
    im = axsR[kcol//3,kcol%3].imshow( gas_map[kcol], extent=(-RMAX,RMAX,-RMAX,RMAX), origin='lower', cmap=cmaps[kcol], norm=norms[kcol] )

    axsR[kcol//3,kcol%3].contour( x,y,gas_map[0], [-1.5,-0.5], linestyles=['solid','solid'], cmap='bone_r' )

    #axsR[kcol//3,kcol%3].scatter( young_stars[:,0],young_stars[:,1], marker='*', color='blue' )

    axsR[kcol//3,kcol%3].set_xticks([])
    axsR[kcol//3,kcol%3].set_yticks([])
    cbarR = figR.colorbar( im, ax=axsR[kcol//3,kcol%3], shrink=0.9 )
    cbarR.ax.tick_params(labelsize=11)

    axsR[kcol//3,kcol%3].set_xlim((-RMAX,RMAX))
    axsR[kcol//3,kcol%3].set_ylim((-RMAX,RMAX))

axsR[0,0].set_title(r'Column Density [$ {\rm M}_{\odot} / {\rm kpc}^2 $]')
axsR[0,1].set_title(r'Transverse Vel. $v_{\rm R}$ [km/s]')
axsR[0,2].set_title(r'Temperature [K]')

axsR[1,0].set_title(r'Metallicity $Z/Z_{\odot}$')
axsR[1,1].set_title(r'Vel. Dispersion ${\sigma}_{\rm z}$ [km/s]')
#axsR[1,2].set_title(r'Specific Energy $e$ [erg/${\rm M}_{\odot}$]')
axsR[1,2].set_title(r'Energy Ratio KE/TE')

feet = fm.FontProperties(size=15)
scalebar0 = AnchoredSizeBar( axsR[0,0].transData, 0.25, '250 pc', 'lower left', pad=0.15, color='white', frameon=False, size_vertical=0.01, fontproperties=feet )
axsR[0,0].add_artist(scalebar0)

figR.text( 0.925,0.45, 'log10 X', rotation='vertical', fontsize=15 )

figR.savefig('OfficialPlots/P4-BubblePicture.png',dpi=360)
plt.show()
