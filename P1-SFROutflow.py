import numpy as np
import h5py
import matplotlib.pyplot as plt

import zHead as hd

arg_options = hd.handle_arguments()
ActiveRunIDs = arg_options[0]

ActiveRunIDs = [3,4,5]
#ActiveRunIDs = [3]

print(ActiveRunIDs)
date = arg_options[1]

print(hd.last_snap_num)

CALCULATE = False

NTIMES = 240
NSTARTS = [0,750,150,150,40,450]
NENDS = [700,1500,900,900,790,1200]

NAVG = 8

tStart = 0
tEnd   = 750.

FSNAPLIST = [ 0, 0, 150, 150, 40, 250 ]
LSNAPLIST = [ 1, 1, 900, 900, 790, 1000 ]
INTVLLIST = [ 1, 1, 1, 1, 10, 1 ]

HEIGHTS = [1.0,2.0,4.0]
REF_RAD = 2.0      # kpc, set
THICKNESS = 0.05   # kpc, set
TAU_AVG_SFR = 25.   # Myr; fixed

Smin = 5.e-7
Smax = 2.e-2
Mmin = 5.e-5
Mmax = 2.e-2
Emin = 3.e-5
Emax = 3.e0

nbn_sidehist = 20

def logfloor( x, floor=-10. ):
    return np.where( x > 10.**floor, np.log10(x), floor*np.ones_like(x) )

def histcut( x, Min, Max ):

    w = np.where( np.isnan(x), Min*np.ones_like(x), x )

    after_min = np.where( w>Min, w, Min*np.ones_like(w) )
    y = after_min
    after_max = np.where( y<Max, y, Max*np.ones_like(y) )

    return after_max

def loghistcut( x, Min, Max ):

    return histcut( np.log10(x), np.log10(Min), np.log10(Max) )

###

Nruns = len(hd.runlabels)
N_ACTIVE_RUNS = len(ActiveRunIDs)

def do_calculations_sfr(i,j):

    Snap = hd.Snapshot(i,j)
    f = Snap.f

    print('NOW CALCULATING {} (RUN NO. {}) SNAPSHOT NO. {} '.format(RunName,j,i))

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    age4_sim = hd.extract( f,u'GFM_StellarFormationTime',4, **snap_kwargs ) * Snap.TimeFac

    if j==4:
        dist_unit_fac = 0.001
        mass_unit_fac = 1.0
    else:
        dist_unit_fac = 1.0
        mass_unit_fac = 1.e10

    m4 = hd.extract( f,u'GFM_InitialMass',4, **snap_kwargs ) * mass_unit_fac

    hist = np.histogram( age4_sim, bins=np.linspace(FSNAPLIST[j],LSNAPLIST[j],NTIMES), weights=m4 )
    time_interval = (LSNAPLIST[j]-FSNAPLIST[j]) / NTIMES

    times = hist[1][:-1]
    sfr = hist[0] / time_interval / 1.e6

    times_avg, sfr_avg = hd.moving_average( times, sfr, n=NAVG )
    overall_sfr_avg = np.average(sfr)

    print('Averaging Interval: {} Myr'.format( time_interval * NAVG ))

    return times_avg, sfr_avg, overall_sfr_avg

def do_calculations_outflow(i,j,ref_height):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    s = Snap.s

    print('NOW CALCULATING RUN NO. {} SNAPSHOT NO. {} '.format(j,i))

    if j==4:
        origin = 300.
    elif j==5:
        origin = 5000.
    else:
        origin = 100.
    vcom = 0.0

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### Load Main Things, Adjust Coordinates

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', **snap_kwargs )
    vel0 = hd.extract( f,u'Velocities',0, units='gal', **snap_kwargs )
    mass0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )

    U0 = hd.extract( f,u'InternalEnergy',0, units='gal', **snap_kwargs )
    Dens0_gal = hd.extract( f,u'Density',0, units='gal', **snap_kwargs )

    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )

    #if Snap.MODE=='ISO':
    xyz4 = hd.extract( f,u'BirthPos',4, units='gal', **snap_kwargs ) # use birth pos for isolated
    mass4 = hd.extract( f,u'GFM_InitialMass',4, units='gal', **snap_kwargs )
    age4 = Snap.find_star_age( hd.extract( f,'GFM_StellarFormationTime',4, verbose=True, **snap_kwargs ) )

    xyz0 -= origin
    vel0 -= vcom
    xyz4 -= origin

    ### Find Outflow Values, Select Particles in Outflow Shell

    rCyl0 = np.sqrt( xyz0[:,0]*xyz0[:,0] + xyz0[:,1]*xyz0[:,1] )
    rCyl4 = np.sqrt( xyz4[:,0]*xyz4[:,0] + xyz4[:,1]*xyz4[:,1] )

    Z0 = np.abs( xyz0[:,2] )
    Z4 = np.abs( xyz4[:,2] )

    vr0 = ( xyz0[:,0]*vel0[:,0] + xyz0[:,1]*vel0[:,1] ) / (rCyl0 + 1.e-5)

    vz0plus = xyz0[:,2]*vel0[:,2] / (xyz0[:,2] + 1.e-5) # note; this is the component of the velocity in the POSITIVE Z direction.
    vz0minus = -xyz0[:,2]*vel0[:,2] / (xyz0[:,2] + 1.e-5) # note; this is the component of the velocity in the NEGATIVE Z direction.
    vz0 = np.where( xyz0[:,2] > 0., vz0plus, vz0minus )

    v2 = vel0[:,0]*vel0[:,0] + vel0[:,1]*vel0[:,1] + vel0[:,2]*vel0[:,2]

    gamma = 5.0 / 3.0
    Enthalpy = 0.5*mass0*vz0*vz0 + gamma*mass0*U0

    if j!=4:
        Pressure0 = hd.extract( f,u'Pressure',0, units='gal', **snap_kwargs )
        Vol0 = mass0 / Dens0_gal

    pr0 = mass0 * vr0
    pz0 = mass0 * vz0

    Eflux_z0 = Enthalpy * vz0

    kkCyl0_in = rCyl0 > REF_RAD - (THICKNESS/2.)
    kkCyl0_out = rCyl0 < REF_RAD + (THICKNESS/2.)
    kkCyl0 = kkCyl0_in * kkCyl0_out * ( Z0 < ref_height )

    kkZ0_in = Z0 > ref_height - (THICKNESS/2.)
    kkZ0_out = Z0 < ref_height + (THICKNESS/2.)
    kkZ0 = kkZ0_in * kkZ0_out * ( rCyl0 < REF_RAD )

    # Sum Up Outflow Values and Return

    outflowR = np.sum( pr0[kkCyl0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 ) # Msun / yr
    outflowZ = np.sum( pz0[kkZ0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 )   # Msun / yr 

    EnergyFluxZ = np.sum( Eflux_z0[kkZ0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 ) # in 2.e43 erg/yr replace Msun with the units of enthalpy; Msun (km/s)^2 = 2.e43 erg
    EnergyFluxZ *= ( 1.989e33 * 1.e10 * 1.e6 / 1.e51 ) # Converted to 1.e51 erg / Myr

    sfr_in_rad = np.sum( mass4 ) / ( TAU_AVG_SFR * 1.e6 )

    return Snap.SimTime, outflowR, outflowZ, EnergyFluxZ, sfr_in_rad

###

fig,axs = plt.subplots( 3,2, figsize=(8,7), width_ratios=(4,1) )
plt.subplots_adjust( hspace=0.0, wspace=0.0, left=0.2 )
for k in range(3):
    axs[k,0].tick_params(labelsize=13)
    axs[k,1].tick_params(labelsize=13)

jCol = [9,9,2,0,2,3]

LyraFixedSNe = np.genfromtxt('data/LyraFixedSNe.csv',delimiter=',')

Nruns = len(hd.runlabels)
for j,(RunName) in enumerate(hd.runlabels):
    if j in ActiveRunIDs:

        timeSFR, sfr, overall_avg = do_calculations_sfr( hd.last_snap_num[j], j )
        times_arr = np.linspace(tStart,tEnd,100)

        np.savetxt( 'data/sfr{}.txt'.format(j), np.transpose(np.array([ timeSFR-timeSFR[0], sfr ])) )

        if CALCULATE:

            outflowdata = []
            for ref_height in HEIGHTS:
                outflowdata_hgt = []

                for i in range(FSNAPLIST[j],LSNAPLIST[j],INTVLLIST[j]):

                    try:
                        timeOF, rflux, zflux, zEflux, sfrOF = do_calculations_outflow(i,j,ref_height)
                        outflowdata_hgt.append([ timeOF, rflux, zflux, zEflux, sfrOF ])
                    except FileNotFoundError:
                        pass
                outflowdata.append(outflowdata_hgt)

            outflowdata = np.array(outflowdata)
            np.save('data/OutflowData_Multilevel{}.npy'.format(j),outflowdata)

        else:

            outflowdata = np.load('data/OutflowData_Multilevel{}.npy'.format(j))

        hstkwgs = { 'orientation':'horizontal', 'histtype':'step', 'color':'C{}'.format(jCol[j]), 'density':True, 'log':True } #'density':True, 

        ### ### ###

        axs[0,0].plot( LyraFixedSNe[:,0]-250., LyraFixedSNe[:,1], color='red', linestyle='dashed', alpha=0.13 ) # Fixed SNe Curve
        axs[0,0].plot( timeSFR-timeSFR[0], sfr, label=RunName, color='C{}'.format(jCol[j]) )

        binsS = np.linspace( np.log10(Smin), np.log10(Smax), nbn_sidehist+1, endpoint=True )
        axs[0,1].hist( loghistcut(sfr,Smin,Smax), bins=binsS, **hstkwgs )

        ### ### ### ### ###

        nAvging = 30 // INTVLLIST[j]
        print(nAvging)
        iHgt = 0

        ### ### ###

        outflowtimeM, Moutflowrate = hd.moving_average( outflowdata[iHgt,:,0], outflowdata[iHgt,:,2], n=nAvging )
        axs[1,0].plot( outflowtimeM-outflowtimeM[0], Moutflowrate, color='C{}'.format(jCol[j]), label=RunName)

        binsM = np.linspace( np.log10(Mmin), np.log10(Mmax), nbn_sidehist+1, endpoint=True )
        axs[1,1].hist( loghistcut(Moutflowrate,Mmin,Mmax), bins=binsM, **hstkwgs )

        ### ### ###

        outflowtimeE, Eoutflowrate = hd.moving_average( outflowdata[iHgt,:,0], outflowdata[iHgt,:,3], n=nAvging )
        axs[2,0].plot( outflowtimeE-outflowtimeE[0], Eoutflowrate, color='C{}'.format(jCol[j]), label=RunName)

        binsE = np.linspace( np.log10(Emin), np.log10(Emax), nbn_sidehist+1, endpoint=True )
        axs[2,1].hist( loghistcut(Eoutflowrate,Emin,Emax), bins=binsE, **hstkwgs )

        #print( binsS, binsM, binsE )
        #print(sfr,Moutflowrate,Eoutflowrate)
        #print(RunName,len(sfr),len(Moutflowrate),len(Eoutflowrate))

axs[2,0].set_xlabel('Time [Myr]',fontsize=15)
axs[2,1].set_xlabel('Likelihood',fontsize=12)

axs[0,0].set_ylabel(r'SFR [${\rm M}_{\odot}$/yr]',fontsize=13)
axs[1,0].set_ylabel(r'Mass Flux [${\rm M}_{\odot}$/yr]',fontsize=13)
axs[2,0].set_ylabel('Energy Flux\n'+r'[${10}^{51}$ erg/Myr]',fontsize=13)

for k in range(3):

    axs[k,0].set_xlim((0.,720.))
    axs[k,0].set_yscale('log')

    axs[k,1].set_xlim(5.e-2,5.e0)

    hd.axisstyle(axs[k,0])
    hd.axisstyle(axs[k,1])

    axs[k,0].set_xticks([0,200,400,600])

    axs[k,1].set_xticks([1.e-1,1.e0])
    axs[k,1].tick_params(which='minor',top=False,bottom=False)

    axs[k,1].set_yticklabels([])

axs[0,0].set_ylim((Smin,Smax))
axs[1,0].set_ylim((Mmin,Mmax))
axs[2,0].set_ylim((Emin,Emax))

axs[0,1].set_ylim((np.log10(Smin),np.log10(Smax)))
axs[1,1].set_ylim((np.log10(Mmin),np.log10(Mmax)))
axs[2,1].set_ylim((np.log10(Emin),np.log10(Emax)))

axs[0,0].set_xticklabels([])
axs[1,0].set_xticklabels([])
axs[2,0].set_xticklabels([0,200,400,600])

axs[0,1].set_yticks([-5,-3])
axs[1,1].set_yticks([-4,-3,-2])
axs[2,1].set_yticks([-3,-1])

axs[0,1].set_xticklabels([])
axs[1,1].set_xticklabels([])
axs[2,1].set_xticklabels([r'$10^{-1}$',r'$10^{0}$'])

axs[1,0].tick_params(which='minor',left=False,right=False)

plt.savefig('OfficialPlots/P1-SFROutflow.png',dpi=300)
plt.show()

#axs[1].plot( outflowtimeM-outflowtimeM[0], logfloor(Moutflowrate,floor=ENERGY_MIN-5), color='C{}'.format(jCol[j]), linewidth=1, label=RunName)
#axs[2].plot( outflowtimeE-outflowtimeE[0], logfloor(Eoutflowrate,floor=ENERGY_MIN-1), color='C{}'.format(jCol[j]), linewidth=1, label=RunName)
#axs[k].xaxis.set_ticks_position('both')
#axs[k].yaxis.set_ticks_position('both')
#axs[k].tick_params(direction='in')

#####

#def axisstyle(ax):
#    ax.xaxis.set_ticks_position('both')
#    ax.yaxis.set_ticks_position('both')
#    ax.tick_params(which='major', direction='in')
#    ax.tick_params(which='minor', direction='in')

#####

#ENERGY_MAX = 2.0
#ENERGY_MIN = -7.0

#axs[2].legend(loc='lower right',fontsize=16)

        #print( timeSFR )
        #axs[0].plot( times_arr, overall_avg*np.ones_like(times_arr), color='C{}'.format(jCol[j]), linestyle='dashed', alpha=0.5 )

        #ySig1S = np.percentile(sfr,16)        
        #ySig2S = np.percentile(sfr,84)
        #axs[0].fill_between( timeSFR-timeSFR[0], ySig1S, ySig2S, color='C{}'.format(jCol[j]), alpha=0.37 )

        #print( np.diff(timeSFR) )

        #ySig1M = np.percentile(Moutflowrate,16)
        #ySig2M = np.percentile(Moutflowrate,84)
        #axs[1].fill_between( outflowtimeM-outflowtimeM[0], ySig1M, ySig2M, color='C{}'.format(jCol[j]), alpha=0.37 )

        #print( np.diff(outflowtimeM) )

        #ySig1E = np.percentile(Eoutflowrate,16)        
        #ySig2E = np.percentile(Eoutflowrate,84)
        #axs[2].fill_between( outflowtimeE-outflowtimeE[0], ySig1E, ySig2E, color='C{}'.format(jCol[j]), alpha=0.37 )

        #print( np.diff(outflowtimeE) )