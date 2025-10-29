import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import zHead as hd
from astropy.cosmology import FlatLambdaCDM

import imageio.v2 as imageio

arg_options = hd.handle_arguments()
#ActiveRunIDs = arg_options[0]
#ActiveRunIDs = [4,5]
ActiveRunIDs = [5]
print(ActiveRunIDs)
date = arg_options[1]

WEIGHT_MODE = 'vol'

ALIGN_DISK = False

RMAX = 1.5 # 1.25
ZMAX = 0.25

MINRHO = -5.5
MAXRHO = 5.5
MINTEMP = 0.5
MAXTEMP = 7.5

NBINS_RHO = 220
NBINS_TEMP = 140

SCATTER_RATIO_XY = ( NBINS_RHO + 1 ) / ( NBINS_TEMP + 1 )
HIST_RATIO_Y = 2.0 # ratio of the height of the histogram to the y-height of the scatter/image plot
REL_FIGSIZE = 1.5 # magnification, as defined, i guess

hist_hgt = 1.0

scatter_y = hist_hgt * HIST_RATIO_Y
scatter_x = scatter_y * SCATTER_RATIO_XY
figxsize = REL_FIGSIZE * ( scatter_x + hist_hgt )
figysize = REL_FIGSIZE * ( scatter_y + hist_hgt )

space_rho = np.linspace(MINRHO,MAXRHO, NBINS_RHO//2 )
space_temp  = np.linspace(MINTEMP,MAXTEMP, NBINS_TEMP//2 )

cmap,norm = hd.create_cmap('viridis',lo=-5.,hi=3.,Nbins=24)

###

npi=1

idens_Main_LYRA  = np.load('auxdata/LocalData_run0.npy')[0,npi]
idens_Main_RIGEL = np.load('auxdata/LocalData_run4.npy')[0,npi]

itemp_Main_LYRA  = np.load('auxdata/LocalData_run0.npy')[7,npi]
itemp_Main_RIGEL = np.load('auxdata/LocalData_run4.npy')[7,npi]

###

Nruns = len(hd.runlabels)
N_ACTIVE_RUNS = len(ActiveRunIDs)

def do_calculations(i,j,subhalo_id):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    s = Snap.s

    if Snap.MODE == 'COSMO':
        SIDarr = np.loadtxt('data/SubhaloIDData_run{}'.format(j),dtype='int')
        SID = SIDarr[i,subhalo_id]
    else:
        SID = 0

    print('NOW CALCULATING {} (RUN NO. {}) SNAPSHOT NO. {} '.format(RunName,j,i))

    origin, vcom, Ltot = hd.adjustdata( Snap, subhalo_id=SID )
    print('Obtained Origin')

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### LOAD THINGS

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', **snap_kwargs )
    mass0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )
    rho0 = hd.extract( f,u'Density',0, units='cgs', **snap_kwargs ) / hd.PROTONMASS_g
    U0 = hd.extract( f,u'InternalEnergy',0, **snap_kwargs )
    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )

    rho_galunit = hd.extract( f,u'Density',0, units='gal', **snap_kwargs )
    vol0 = mass0 / rho_galunit

    xyz0 -= origin

    if ALIGN_DISK:
        xyz0 = hd.align_disk(xyz0,Ltot)
        r0 = np.sqrt(sum([ xyz0[:,b]*xyz0[:,b] for b in range(2) ]))
        z0 = np.abs(xyz0[:,2])
        kk0 = (r0 < RMAX) * (z0 < ZMAX)
    else:
        r0 = np.sqrt(sum([ xyz0[:,b]*xyz0[:,b] for b in range(3) ]))
        kk0 = (r0 < RMAX)

    ### CALCULATE THINGS

    mass0 = mass0[kk0]
    vol0  = vol0[kk0]
    logrho = np.log10( rho0[kk0] )
    logtemp = np.log10( hd.GasTemp( U0[kk0], Nelec[kk0] ) )

    if WEIGHT_MODE == 'mass':
        pdb = hd.ptcl_density_bin( logrho,logtemp,mass0, min1=MINRHO,max1=MAXRHO, min2=MINTEMP,max2=MAXTEMP, Nbins1=NBINS_RHO+1,Nbins2=NBINS_TEMP+1)
        histwgt = mass0
    elif WEIGHT_MODE == 'vol':
        pdb = hd.ptcl_density_bin( logrho,logtemp,vol0,  min1=MINRHO,max1=MAXRHO, min2=MINTEMP,max2=MAXTEMP, Nbins1=NBINS_RHO+1,Nbins2=NBINS_TEMP+1)
        histwgt = vol0

    ### RETURN ANSWER

    return pdb, logrho, logtemp, histwgt

###

Nruns = len(hd.runlabels)
for j,(RunName) in enumerate(hd.runlabels):
    if j in ActiveRunIDs:
        print('run {}: {}'.format(j,RunName))

        #i = 480
        subhalo_id = 0

        #for i in range(250,750,10):
        for i in range(250,750,2):

            pdb, logrho, logtemp, histwgt = do_calculations(i,j,subhalo_id)

            fig, axs = plt.subplot_mosaic([['histx', '.'],
                                        ['scatter', 'histy']],
                                        figsize=(figxsize, figysize),
                                        width_ratios=(scatter_x, hist_hgt), height_ratios=(hist_hgt, scatter_y))

            plt.subplots_adjust(hspace=0.0,wspace=0.0)

            axs['histx'].tick_params(axis="x", labelbottom=False)
            axs['histy'].tick_params(axis="y", labelleft=False)

            axs['scatter'].imshow(np.log10(pdb+1.e-10),origin='lower',cmap=cmap,norm=norm,extent=(MINRHO,MAXRHO,MINTEMP,MAXTEMP))

            axs['histx'].hist(logrho, weights=histwgt, bins=space_rho, density=True, histtype='step',label='Gas')
            axs['histy'].hist(logtemp, weights=histwgt, bins=space_temp, orientation='horizontal', density=True, histtype='step')

            if j==4:
                axs['histx'].hist(idens_Main_RIGEL, bins=space_rho, density=True, histtype='step', label='SNe Env.', color='C2')
                axs['histy'].hist(itemp_Main_RIGEL - np.log10(1.989e43), bins=space_temp, density=True, histtype='step', orientation='horizontal', color='C2') # needed to correct for bug in GasTemp function input units
                axs['scatter'].text( 4.50, 7.25, 'RIGEL', color='white', fontsize=12, ha='center', va='top' )
            if j==5:
                axs['histx'].hist(idens_Main_LYRA, bins=space_rho, density=True, histtype='step', label='SNe Env.', color='C3')
                axs['histy'].hist(itemp_Main_LYRA, bins=space_temp, density=True, histtype='step', orientation='horizontal', color='C3')
                axs['scatter'].text( 4.50, 7.25, 'LYRA', color='white', fontsize=12, ha='center', va='top' )

            axs['histx'].set_yscale('log')
            axs['histy'].set_xscale('log')

            axs['histx'].set_ylim((5.e-2,2.e0))
            axs['histy'].set_xlim((5.e-4,2.e0))

            axs['histx'].legend(loc='upper right')

            axs['scatter'].set_ylabel('log10 Temperature [K]', fontsize=13)
            axs['scatter'].set_xlabel(r'log10 Density [${\rm cm}^{-3}$]', fontsize=13)

            axs['histx'].tick_params(labelsize=11)
            axs['histy'].tick_params(labelsize=11)
            axs['scatter'].tick_params(labelsize=11)

            fig.savefig('PhaseDiagrams/RadPhase_run{}_snap{}.png'.format(j,i))
            plt.show()
