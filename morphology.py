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

import imageio.v2 as imageio

import contextlib

# test test test am i doing something here or like
# YOO NO WAY

arg_options = hd.handle_arguments()
ActiveRunIDs = arg_options[0]
print(ActiveRunIDs)
date = arg_options[1]

#MODE = hd.DEFAULT_MODE

#RENDER_SNAPS = None        # Choose the list of snapshots you want to see shown. If you want to just create the gif without plt.show(), set to None.
RENDER_SNAPS = [500]

SUBHALO_LIST = [0]

RMAX = 1.5
NBINS = 100

GRAPHICS_MODE = 'FANCY'
ALIGN_DISK = False

LOG_LO = -1.5 # msun/pc
LOG_HI = 1.5

SMOOTH_NGB = 16

FSNAP = 250 # do not go below this snap

###

cmap, norm = hd.create_cmap( 'magma', lo=LOG_LO, hi=LOG_HI, Nbins=20 )

###

def projection( xyz, m, graphics='FAST' ):

    '''
    Graphics = 'FAST' or 'FANCY'
    '''

    def fancy_pic( x,y,z,m, smooth_ngb=SMOOTH_NGB, func_scale=1.e-5 ):

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

    ###

    if graphics=='FAST':

        proj_all = np.zeros((3,NBINS,NBINS))

        proj_all[0] = hd.ptcl_density_bin( xyz[:,0], xyz[:,1], m, min1=-RMAX,max1=RMAX, min2=-RMAX,max2=RMAX, Nbins1=NBINS,Nbins2=NBINS ) #xy
        proj_all[1] = hd.ptcl_density_bin( xyz[:,0], xyz[:,2], m, min1=-RMAX,max1=RMAX, min2=-RMAX,max2=RMAX, Nbins1=NBINS,Nbins2=NBINS ) #xz
        proj_all[2] = hd.ptcl_density_bin( xyz[:,1], xyz[:,2], m, min1=-RMAX,max1=RMAX, min2=-RMAX,max2=RMAX, Nbins1=NBINS,Nbins2=NBINS ) #yz

        proj_all *= 1.e-6 # units are msun/kpc^2 by default. need them in msun/pc^2

        buffer_value = 10.**LOG_LO / 1.e3
        proj_all = np.log10(proj_all + buffer_value)

    if graphics=='FANCY':

        proj_all = np.zeros((3,NBINS,NBINS))

        #m /= 1.e10 # set m back to 1.e10 msun for some reason

        proj_all[0] = fancy_pic( xyz[:,0], xyz[:,1], xyz[:,2], m )
        proj_all[1] = fancy_pic( xyz[:,2], xyz[:,0], xyz[:,1], m )
        proj_all[2] = fancy_pic( xyz[:,2], xyz[:,1], xyz[:,0], m )

    return proj_all

def do_calculations(i,j,subhalo_id):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    s = Snap.s

    if Snap.MODE == 'COSMO':
        SIDarr = np.loadtxt('data/SubhaloIDData_run{}'.format(j),dtype='int')
        SID = SIDarr[i,subhalo_id]
    else:
        SID = 0

    print('NOW CALCULATING {} (RUN NO. {}) SNAPSHOT NO. {} SID={}'.format(RunName,j,i,SID))

    origin, vcom, Ltot = hd.adjustdata( Snap, subhalo_id=SID, rad=3.0 )
    print('Obtained Origin')

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### LOAD THINGS

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', verbose=True, **snap_kwargs )
    m0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )

    xyz4 = hd.extract( f,u'Coordinates',4, units='gal', verbose=True, **snap_kwargs )
    m4 = hd.extract( f,u'Masses',4, units='gal', **snap_kwargs )

    xyz0 -= origin
    xyz4 -= origin

    if ALIGN_DISK:
        xyz0 = hd.align_disk(xyz0,Ltot)
        xyz4 = hd.align_disk(xyz4,Ltot)

    kkRad0 = hd.calc_distance(xyz0) < RMAX*np.sqrt(3)
    kkRad4 = hd.calc_distance(xyz4) < RMAX*np.sqrt(3)

    xyz0 = xyz0[ kkRad0 ]
    m0   = m0[ kkRad0 ]
    xyz4 = xyz4[ kkRad4 ]
    m4   = m4[ kkRad4 ]

    if Snap.MODE=='ISO':
        rList = hd.calc_distance( xyz4 )
        stellar_rHalf = hd.half_mass_radius( rList, m4 )
    if Snap.MODE=='COSMO':
        stellar_rHalf = s[u'Subhalo'][u'SubhaloHalfmassRadType'][0,4] * 1000. * Snap.scalefac / Snap.h_small

    proj0_all = projection( xyz0, m0, graphics=GRAPHICS_MODE )
    proj4_all = projection( xyz4, m4, graphics=GRAPHICS_MODE )

    ### RETURN ANSWER

    return proj0_all, proj4_all, stellar_rHalf

Nruns = len(hd.runlabels)
for j,(RunName) in enumerate(hd.runlabels):
    if j in ActiveRunIDs:

        for subhalo_id in SUBHALO_LIST:

            print('run {}: {}, Halo {}'.format(j,RunName,subhalo_id))

            ImgList = []

            if RENDER_SNAPS == None:
                plot_snap_list = range(FSNAP,hd.last_snap_num[j])
            else:
                plot_snap_list = RENDER_SNAPS

            for i in plot_snap_list:

                try:

                    proj0_all, proj4_all, halfMassRad4 = do_calculations(i,j,subhalo_id)

                    fig,axs = plt.subplots(2,3,figsize=(13,8))
                    plt.subplots_adjust(wspace=0.13,hspace=0.13)

                    # Stars

                    for k in range(3):
                        im = axs[0,k].imshow( proj4_all[k][:-1,:-1], extent=(-RMAX,RMAX,-RMAX,RMAX), origin='lower', cmap=cmap, norm=norm )

                    axs[0,0].set_ylabel('Stars', fontsize=14)

                    # Gas

                    for k in range(3):
                        im = axs[1,k].imshow( proj0_all[k][:-1,:-1], extent=(-RMAX,RMAX,-RMAX,RMAX), origin='lower', cmap=cmap, norm=norm )

                    axs[1,0].set_ylabel('Gas', fontsize=14)
                    axs[1,1].set_xlabel('[kpc]', fontsize=13)

                    # Stellar Half Mass

                    theta = np.linspace(0,2*np.pi,1000)
                    xcirc = halfMassRad4 * np.cos(theta)
                    ycirc = halfMassRad4 * np.sin(theta)
                    for k in range(3):
                        axs[0,k].plot(xcirc,ycirc,linestyle='dashed',color='black')
                        axs[0,k].plot([0],[0], marker='+',color='black')

                    # Color Bar

                    fig.subplots_adjust(right=0.84)
                    cbar_ax = fig.add_axes([0.87,0.15,0.04,0.65])
                    fig.colorbar(im,cax=cbar_ax, shrink=0.9, label='log10 Column Density [$\mathrm{M}_{\odot}$ $\mathrm{pc^{-2}}$]')

                    fig.suptitle('{}, Mass Distribution at Snap No. {}'.format( RunName, i ), fontsize=16)
                    fig.savefig('zImgTmp/imageMorph.png')
                    ImgList.append( imageio.imread('zImgTmp/imageMorph.png') )

                    if RENDER_SNAPS == None:
                        plt.close()
                    else:
                        with contextlib.suppress(ValueError):
                            plt.show()

                except FileNotFoundError:

                    print('File Not Found')
                    pass

            if RENDER_SNAPS == None:
                imageio.mimsave( 'plots/projection_run{}_halo{}.gif'.format(j,subhalo_id), ImgList )
            else:
                pass
