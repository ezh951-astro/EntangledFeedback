import numpy as np
import h5py
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

import zHead as hd

arg_options = hd.handle_arguments()
ActiveRunIDs = arg_options[0]

ActiveRunIDs = [5]

print(ActiveRunIDs)
date = arg_options[1]

print(hd.last_snap_num)

#####

LOCDAT_L = np.load('../SupernovaExtract/LyraSNeExtract/data/LyraSNeLocationData_4096_5.npy')
ENVDAT_L = np.load('../SupernovaExtract/LyraSNeExtract/data/LyraSNeEnvironmentData_4096_5.npy')
SNID_to_time_ref = np.loadtxt('../SupernovaExtract/LyraSNeExtract/data/SNID_to_time_Lyra.txt')

kk_ZeroEnergy = LOCDAT_L[:,0,9]/1.e51 > 0.3
LOCDAT_L = LOCDAT_L[kk_ZeroEnergy]
ENVDAT_L = ENVDAT_L[kk_ZeroEnergy]

#####

#CALCULATE = False

FSNAP = 250
LSNAP = 1000

HIST_NBINS = 500

HEIGHTS = [1.0,2.0,4.0]

REF_RAD = 2.0      # kpc, set
THICKNESS = 0.05   # kpc, set

TAU_AVG_SFR = 25.   # Myr; fixed

tStart = FSNAP + 200.
tEnd   = LSNAP + 200.

###

time_of_SN = np.array([ 1000.*SNID_to_time_ref[:,1][ SNID_to_time_ref[:,0]==snid ] for snid in LOCDAT_L[:,0,0] ])[:,0]
E_of_SN = LOCDAT_L[:,0,9] / 1.e51

LogAvDens = []
for nSN,(locmass,x,y,z) in enumerate(zip( ENVDAT_L[:,0,6,:], ENVDAT_L[:,0,0,:], ENVDAT_L[:,0,1,:], ENVDAT_L[:,0,2,:] )): # 0 for first snapshot, 6 for mass. first colon is now the nSN, second colon is the particle

    NPart = np.argmax( np.cumsum(locmass) > 20. ) # Mscale = 20.

    gas_norm = np.sqrt( x*x + y*y + z*z )
    TrueDensity = 4.44e-8 * np.sum(locmass[ :NPart+1 ]) / ( 4.*np.pi/3. ) / gas_norm[NPart+1]**3.

    LogAvDens.append( np.log10(TrueDensity) )

LogAvDens= np.array(LogAvDens)

kk_hDens = ( LogAvDens >= 1.0 )
kk_lDens = ( LogAvDens < 1.0  )

histbins = np.linspace( tStart, tEnd, HIST_NBINS )
BinWidth = (tEnd-tStart)/HIST_NBINS
nAvging  = int(TAU_AVG_SFR/BinWidth)

hSN_ERate, hSN_times = np.histogram( time_of_SN[kk_hDens], bins=histbins, weights=E_of_SN[kk_hDens] )
lSN_ERate, lSN_times = np.histogram( time_of_SN[kk_lDens], bins=histbins, weights=E_of_SN[kk_lDens] )

hSN_ERate /= BinWidth # total energy in bin of x megayears, divided by x (in megayears); 1.e51 erg / Myr
lSN_ERate /= BinWidth

hSN_times = hSN_times[1:]
lSN_times = lSN_times[1:]

lSN_times, lSN_ERate = hd.moving_average(lSN_times, lSN_ERate, n=nAvging)
hSN_times, hSN_ERate = hd.moving_average(hSN_times, hSN_ERate, n=nAvging)

###

j = 5
outflowdata = np.load('data/OutflowData_Multilevel{}.npy'.format(j))

fig,axs = plt.subplots(1,1,figsize=(8,5))
#figCC,axsCC = plt.subplots(1,1,figsize=(7,6))

axs.plot( lSN_times-450., lSN_ERate, color='red', linewidth=1.5, alpha=0.7, label='Low-Dens. SNe' )
axs.plot( hSN_times-450., hSN_ERate, color='blue', linewidth=1.5, alpha=0.7, label='High-Dens. SNe' )

#for iHgt,(ref_height,linestyle) in enumerate(zip(HEIGHTS,HEIGHT_LINESTYLES)):

# 1kpc
outflowtime, outflowrate = hd.moving_average( outflowdata[0,:,0], outflowdata[0,:,3], n=nAvging )
axs.plot( outflowtime-450., outflowrate, color='orange', linewidth=1.5, linestyle='solid', label='Flux at 1 kpc', alpha=0.7 )

# 2kpc
#outflowtime, outflowrate = hd.moving_average( outflowdata[1,:,0], outflowdata[1,:,3], n=nAvging )
#axs.plot( outflowtime-450., outflowrate, color='#777777', linewidth=1, linestyle='dashed', alpha=0.5)

# 4kpc
#outflowtime, outflowrate = hd.moving_average( outflowdata[2,:,0], outflowdata[2,:,3], n=nAvging )
#axs.plot( outflowtime-450., outflowrate, color='#aaaaaa', linewidth=1, linestyle='dotted', alpha=0.4)

outflowrate_intp = np.interp( lSN_times, outflowdata[0,:,0], outflowdata[0,:,3] )
#axs.plot( lSN_times-450., outflowrate_intp, color='C1', linewidth=1, linestyle=linestyle, label='Energy Flux [|z|={} kpc]'.format(ref_height) )

#axs.text( 10., 6.8, r'$c_{\rm Lo} = 1.86 \cdot 10^{-3}$',  color='red',    fontsize=11, alpha=0.8, ha='left', va='top' )
#axs.text( 10., 6.5, r'$c_{\rm Hi} = 1.66 \cdot 10^{-3}$',  color='blue',   fontsize=11, alpha=0.8, ha='left', va='top' )
#axs.text( 10., 6.2, r'$c_{\rm Tot} = 1.82 \cdot 10^{-3}$', color='purple', fontsize=11, alpha=0.8, ha='left', va='top' )

#axs.text( 333., 1.43, '2 kpc', color='#777777', fontsize=11, alpha=0.5, ha='left', va='center' )
#axs.text( 292., 0.18, '4 kpc', color='#aaaaaa', fontsize=11, alpha=0.4, ha='left', va='center' )

tlagL, ccL = hd.cross_correlate( lSN_times-450., lSN_ERate, outflowrate_intp )
tlagH, ccH = hd.cross_correlate( lSN_times-450., hSN_ERate, outflowrate_intp )
tlagT, ccT = hd.cross_correlate( lSN_times-450., lSN_ERate + hSN_ERate, outflowrate_intp )

indL = np.argmax(ccL)
indH = np.argmax(ccH)
indT = np.argmax(ccT)
print( 'Low-Density Coefficient: {} at t={} Myr'.format( ccL[indL], tlagL[indL] ) )
print( 'High-Density Coefficient: {} at t={} Myr'.format( ccH[indH], tlagH[indH] ) )
print( 'All Coefficient: {} at t={} Myr'.format( ccL[indH], tlagL[indH] ) )

tlagHauto, ccHauto = hd.cross_correlate( lSN_times-450., hSN_ERate, hSN_ERate )
tlagLauto, ccLauto = hd.cross_correlate( lSN_times-450., lSN_ERate, lSN_ERate )
indHauto = np.argmax(ccHauto)
indLauto = np.argmax(ccLauto)
print('H autocorrelation {} at {}'.format(ccHauto[indHauto],tlagHauto[indHauto]))
print('L autocorrelation {} at {}'.format(ccLauto[indLauto],tlagLauto[indLauto]))

#axs.xaxis.set_ticks_position('both')
#axs.yaxis.set_ticks_position('left')
hd.axisstyle(axs)
axs.tick_params(labelsize=14,direction='in')

axs.set_xlabel('Time [Myr]',fontsize=16)
axs.set_ylabel(r'Energy Rate [${10}^{51}$ erg/Myr]',fontsize=16)

axs.set_xlim((30.,750.))
axs.set_ylim((0.,6.9))

axs.set_xticks([30,270,510,750])
axs.set_yticks([0,3,6])
axs.set_yticklabels([0.0,3.0,6.0])

axs.legend(loc='upper right',fontsize=14)

fig.savefig('GalFresca/P3-TemporalCorr.png',dpi=360)
plt.show()

###

'''

HEIGHT_LINESTYLES = ['solid','dashed','dotted']

Nruns = len(hd.runlabels)
N_ACTIVE_RUNS = len(ActiveRunIDs)

def do_calculations(i,j,ref_height):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    s = Snap.s

    if Snap.MODE == 'COSMO':
        SIDarr = np.loadtxt('data/SubhaloIDData_run{}'.format(j),dtype='int')
        SID = SIDarr[i,subhalo_id]
    else:
        SID = 0

    print('NOW CALCULATING {} (RUN NO. {}) SNAPSHOT NO. {} '.format(RunName,j,i))

    origin = 5000.
    vcom = 0.0
    Ltot = np.array([1.,0.,0.])

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### Load Main Things, Adjust Coordinates

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', **snap_kwargs )
    vel0 = hd.extract( f,u'Velocities',0, units='gal', **snap_kwargs )
    mass0 = hd.extract( f,u'Masses',0, units='gal', **snap_kwargs )

    U0 = hd.extract( f,u'InternalEnergy',0, units='gal', **snap_kwargs )
    Pressure0 = hd.extract( f,u'Pressure',0, units='gal', **snap_kwargs )
    Dens0_gal = hd.extract( f,u'Density',0, units='gal', **snap_kwargs )
    Vol0 = mass0 / Dens0_gal

    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )
    logtemp = np.log10( hd.GasTemp( U0, Nelec ) )

    if Snap.MODE=='ISO':
        xyz4 = hd.extract( f,u'BirthPos',4, units='gal', **snap_kwargs ) # use birth pos for isolated
    if Snap.MODE=='COSMO':
        xyz4 = hd.extract( f,u'Coordinates',4, units='gal', **snap_kwargs ) # if cosmo, don't account for birth position; its gonna be impossible
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

    Enthalpy = 0.5*mass0*vz0*vz0 + mass0*U0 + Pressure0*Vol0
    print( np.median(0.5*mass0 * (v2-vr0*vr0-vz0*vz0) ), np.median(mass0*U0), np.median(Pressure0*Vol0) )

    pr0 = mass0 * vr0
    pz0 = mass0 * vz0

    Eflux_z0 = Enthalpy * vz0

    kkCyl0_in = rCyl0 > REF_RAD - (THICKNESS/2.)
    kkCyl0_out = rCyl0 < REF_RAD + (THICKNESS/2.)
    kkCyl0 = kkCyl0_in * kkCyl0_out * ( Z0 < ref_height )

    kkZ0_in = Z0 > ref_height - (THICKNESS/2.)
    kkZ0_out = Z0 < ref_height + (THICKNESS/2.)
    kkZ0 = kkZ0_in * kkZ0_out * ( rCyl0 < REF_RAD )

    kk4 = ( rCyl4 < REF_RAD ) * ( Z4 < ref_height ) * (age4 < TAU_AVG_SFR)

    # Sum Up Outflow Values and Return

    outflowR = np.sum( pr0[kkCyl0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 ) # Msun / yr
    outflowZ = np.sum( pz0[kkZ0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 )   # Msun / yr 

    EnergyFluxZ = np.sum( Eflux_z0[kkZ0] ) * ( 525600. * 60. ) / ( THICKNESS * 3.e16 ) # in 2.e43 erg/yr replace Msun with the units of enthalpy; Msun (km/s)^2 = 2.e43 erg
    EnergyFluxZ *= ( 1.989e33 * 1.e10 * 1.e6 / 1.e51 ) # Converted to 1.e51 erg / Myr

    sfr_in_rad = np.sum( mass4[kk4] ) / ( TAU_AVG_SFR * 1.e6 )

    return Snap.SimTime, outflowR, outflowZ, EnergyFluxZ, sfr_in_rad

###

#SNIDs = LOCDAT_L[:,:,0]

Nruns = len(hd.runlabels)
for j,(RunName) in enumerate(hd.runlabels):
    if j in ActiveRunIDs:

        print('run {}: {}'.format(j,RunName))

        if CALCULATE:

            outflowdata = []
            for ref_height in HEIGHTS:
                outflowdata_hgt = []
                for i in range(FSNAP,LSNAP):

                    try:
                        time, rflux, zflux, zEflux, sfr = do_calculations(i,j,ref_height)
                        outflowdata_hgt.append([ time, rflux, zflux, zEflux, sfr ])
                    except FileNotFoundError:
                        pass
                outflowdata.append(outflowdata_hgt)

            outflowdata = np.array(outflowdata)
            np.save('data/OutflowData_Multilevel{}.npy'.format(j),outflowdata)

        else:

            outflowdata = np.load('data/OutflowData_Multilevel{}.npy'.format(j))

        print(np.shape(outflowdata))

        fig,axs = plt.subplots(1,1,figsize=(9.5,5))
        #figCC,axsCC = plt.subplots(1,1,figsize=(7,6))

        axs.plot( lSN_times-450., lSN_ERate, color='red', linewidth=1, alpha=0.5, label='Low-Dens. SNe' )
        axs.plot( hSN_times-450., hSN_ERate, color='blue', linewidth=1, alpha=0.5, label='High-Dens. SNe' )
        for iHgt,(ref_height,linestyle) in enumerate(zip(HEIGHTS,HEIGHT_LINESTYLES)):
            outflowtime, outflowrate = hd.moving_average( outflowdata[iHgt,:,0], outflowdata[iHgt,:,3], n=nAvging )
            axs.plot( outflowtime-450., outflowrate, color='orange', linewidth=1, linestyle=linestyle, label='Energy Flux [|z|={} kpc]'.format(ref_height) )

            outflowrate_intp = np.interp( lSN_times, outflowdata[iHgt,:,0], outflowdata[iHgt,:,3] )
            axs.plot( lSN_times-450., outflowrate_intp, color='C1', linewidth=1, linestyle=linestyle, label='Energy Flux [|z|={} kpc]'.format(ref_height) )

            tlagL, ccL = hd.cross_correlate( lSN_times-450., lSN_ERate, outflowrate_intp )
            tlagH, ccH = hd.cross_correlate( lSN_times-450., hSN_ERate, outflowrate_intp )
            tlagT, ccT = hd.cross_correlate( lSN_times-450., lSN_ERate + hSN_ERate, outflowrate_intp )

            indL = np.argmax(ccL)
            indH = np.argmax(ccH)
            indT = np.argmax(ccT)
            print( 'Low-Density Coefficient: {} at t={} Myr'.format( ccL[indL], tlagL[indL] ) )
            print( 'High-Density Coefficient: {} at t={} Myr'.format( ccH[indH], tlagH[indH] ) )
            print( 'All Coefficient: {} at t={} Myr'.format( ccL[indH], tlagL[indH] ) )

            #axsCC.plot( tlagL, ccL, color='C{}'.format(iHgt), linestyle='solid' )
            #axsCC.plot( tlagH, ccH, color='C{}'.format(iHgt), linestyle='dashed' )
            #axsCC.plot( tlagT, ccT, color='C{}'.format(iHgt), linestyle='dotted' )

        axs.xaxis.set_ticks_position('both')
        axs.yaxis.set_ticks_position('left')
        axs.tick_params(labelsize=14,direction='in')

        axs.set_xlabel('Time [Myr]',fontsize=16)
        axs.set_ylabel('Energy Rate [1.e51 erg/Myr]',fontsize=16)

        axs.set_xlim((0.,750.))
        axs.set_ylim((0.,8.))

        axs.legend(loc='upper right',fontsize=14)

        fig.savefig('OfficialPlots/P4-TemporalCorr.png')
        #figCC.savefig('plots/crosscorr.png')
        plt.show()
'''
