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
ActiveRunIDs = arg_options[0]
print(ActiveRunIDs)
date = arg_options[1]

#MODE = hd.DEFAULT_MODE
#RENDER_SNAPS = None        # Choose the list of snapshots you want to see shown. If you want to just create the gif without plt.show(), set to None.
RENDER_SNAPS = [0]

RMIN = 0.2
RMAX = 10.0

VMAX = 90.0

N_PTS = 250

rad_log = 10.**np.linspace( np.log10(RMIN), np.log10(RMAX), N_PTS )
rad_lin = np.linspace(RMIN,RMAX,N_PTS)

FSNAP = 320
LSNAP = 640

#ACTIVE_P_TYPES = [0,1,4]

def fconc(x):
    return np.log(1+x) - x/(1+x)

def vcirc_NFW( r, Mvir=2.e10, Rvir=44., c=10. ):
    rs = Rvir/c
    M_r = Mvir * fconc(r/rs) / fconc(c)
    v_r = np.sqrt( hd.G * M_r / r ) # kpc, km/s, Msun
    return v_r

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
    m0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )

    xyz1 = hd.extract( f,u'Coordinates',1, units='gal', **snap_kwargs )
    m1 = hd.extract( f,u'Masses',1, units='gal', **snap_kwargs )

    xyz4 = hd.extract( f,u'Coordinates',4, units='gal', **snap_kwargs )
    m4 = hd.extract( f,u'Masses',4, units='gal', **snap_kwargs )

    xyz0 -= origin
    xyz1 -= origin
    xyz4 -= origin

    ### CALCULATE THINGS

    dist0 = hd.calc_distance(xyz0)
    dist1 = hd.calc_distance(xyz1)
    dist4 = hd.calc_distance(xyz4)

    #print( min(dist0) )

    MDlog0 = hd.mass_distribution_spherical( dist0, rad_log, m0 )
    MDlin0 = hd.mass_distribution_spherical( dist0, rad_lin, m0 )

    MDlog1 = hd.mass_distribution_spherical( dist1, rad_log, m1 )
    MDlin1 = hd.mass_distribution_spherical( dist1, rad_lin, m1 )

    MDlog4 = hd.mass_distribution_spherical( dist4, rad_log, m4 )
    MDlin4 = hd.mass_distribution_spherical( dist4, rad_lin, m4 )

    cummass_tot = MDlog0[1] + MDlog1[1] + MDlog4[1]
    density_tot = MDlog0[2] + MDlog1[2] + MDlog4[2]
    velocity_tot = np.sqrt( MDlin0[3]**2. + MDlin1[3]**2. + MDlin4[3]**2. )

    # Critical Density, Mvir, Rvir

    crit_dens = ( 3./8./np.pi/hd.G ) * (Snap.HubbleConst/1000.)**2.
    virial_density = 200. * crit_dens

    dens_below_rhovir = density_tot < virial_density
    index_of_max_dens = np.argmax(density_tot)
    radius_of_max_dens = rad_log[ index_of_max_dens ]
    print('Max Density at {} kpc'.format(radius_of_max_dens))
    further_than_densest_radius = ( rad_log > radius_of_max_dens ) * ( rad_log > 10.0 )
    #further_than_densest_radius = rad_log > 2.
    virial_index = np.argmax(dens_below_rhovir * further_than_densest_radius)

    Mvir = cummass_tot[ virial_index ]
    Rvir = rad_log[ virial_index ]
    Vvir = np.sqrt( hd.G * Mvir / Rvir )

    print( 'Virial Mass: {} Msun, Virial Radius: {} kpc, Virial Velocity: {} km/s'.format(Mvir,Rvir,Vvir) )

    ### RETURN ANSWER

    return MDlog0, MDlin0, MDlog1, MDlin1, MDlog4, MDlin4, Mvir, Rvir, Vvir, Snap.SimTime, Snap.redshift

Nruns = len(hd.runlabels)
for j,(RunName) in enumerate(hd.runlabels):
    if j in ActiveRunIDs:
        print('run {}: {}'.format(j,RunName))

        ImgList = []

        if RENDER_SNAPS == None:
            plot_snap_list = range( FSNAP, min(LSNAP,hd.last_snap_num[j]) )
        else:
            plot_snap_list = RENDER_SNAPS

        for i in plot_snap_list:

            try:

                # Calculate

                MDlog0, MDlin0, MDlog1, MDlin1, MDlog4, MDlin4, Mvir, Rvir, Vvir, time, z = do_calculations(i,j,0)

                # Plot

                fig,axs = plt.subplots(1,3,figsize=(17,6))

                y_line = 10.**np.linspace(2.0,15.0,100)
                y_line_2 = np.linspace(-5.0,VMAX+5.0,100)

                # Cumulative Mass Profile
                axs[0].plot( rad_log, MDlog0[1], color='blue', label='Gas', linewidth=1 )
                axs[0].plot( rad_log, MDlog1[1], color='green', label='DM', linewidth=1 )
                axs[0].plot( rad_log, MDlog4[1], color='red', label='Stars', linewidth=1 )
                axs[0].plot( rad_log, MDlog0[1]+MDlog1[1]+MDlog4[1], color='black', label='Total', linewidth=2 )
                axs[0].set_xlabel('Radius [kpc]',fontsize=14)
                axs[0].set_ylabel('Cumulative Mass [$\mathrm{M}_{\odot}$]',fontsize=14)
                axs[0].plot( rad_log, Mvir * np.ones_like(rad_log), linestyle='dashed', color='gray', alpha=0.69, linewidth=1 )
                axs[0].plot( Rvir * np.ones(100), y_line, linestyle='dashed', color='gray', alpha=0.69, linewidth=1 )
                axs[0].set_xscale('log')
                axs[0].set_yscale('log')
                axs[0].set_ylim((1.e3,1.e13))

                axs[0].legend(loc='lower right',fontsize=13)

                # Density Profile
                axs[1].plot( rad_log, MDlog0[2], color='blue', label='Gas', linewidth=1 )
                axs[1].plot( rad_log, MDlog1[2], color='green', label='DM', linewidth=1 )
                axs[1].plot( rad_log, MDlog4[2], color='red', label='Stars', linewidth=1 )
                axs[1].plot( rad_log, MDlog0[2]+MDlog1[2]+MDlog4[2], color='black', label='Total', linewidth=2 )
                axs[1].set_xlabel('Radius [kpc]',fontsize=14)
                axs[1].set_ylabel('Density [$\mathrm{M}_{\odot}$ $\mathrm{kpc^{-3}}$]',fontsize=14)
                axs[1].plot( Rvir * np.ones(100), y_line, linestyle='dashed', color='gray', alpha=0.69, linewidth=1 )
                axs[1].set_xscale('log')
                axs[1].set_yscale('log')
                axs[1].set_ylim((1.e4,1.e12))

                # Velocity Profile
                axs[2].plot( rad_lin, MDlin0[3], color='blue', label='Gas', linewidth=1 )
                axs[2].plot( rad_lin, MDlin1[3], color='green', label='DM', linewidth=1 )
                axs[2].plot( rad_lin, MDlin4[3], color='red', label='Stars', linewidth=1 )
                axs[2].plot( rad_lin, np.sqrt( MDlin0[3]*MDlin0[3] + MDlin1[3]*MDlin1[3] + MDlin4[3]*MDlin4[3] ), color='black', label='Total', linewidth=2 )
                axs[2].plot( rad_lin, vcirc_NFW(rad_lin, Mvir=2.e10, Rvir=44., c=10. ), color='gray', label='NFW', linewidth=1, alpha=0.5 )
                axs[2].set_xlabel('Radius [kpc]',fontsize=14)
                axs[2].set_ylabel('Circular Velocity [km/s]',fontsize=14)
                axs[2].plot( rad_lin, Vvir * np.ones_like(rad_lin), linestyle='dashed', color='gray', alpha=0.69, linewidth=1 )
                axs[2].plot( Rvir * np.ones(100), y_line_2, linestyle='dashed', color='gray', alpha=0.69, linewidth=1 )
                axs[2].set_ylim((0,VMAX))

                for k in range(3):
                    axs[k].tick_params(labelsize=14)
                    axs[k].set_xlim(RMIN,RMAX)
                axs[2].set_xlim(0,RMAX)

                fig.suptitle('{}, Mass Distribution at Snap No. {} (t={} Myr, z={})'.format( RunName, i, round(time,2), round(z,2) ), fontsize=16)
                fig.savefig('zImgTmp/image.png')
                ImgList.append( imageio.imread('zImgTmp/image.png') )

                if RENDER_SNAPS == None:
                    plt.close()
                else:
                    #plt.show()
                    plt.close()

            except FileNotFoundError:

                print('File Not Found')
                pass

        if RENDER_SNAPS == None:
            imageio.mimsave( 'plots/density{}.gif'.format(j), ImgList )
        else:
            imageio.mimsave( 'plots/density{}.gif'.format(j), ImgList )
            #pass
