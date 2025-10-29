import numpy as np
import h5py
import matplotlib.pyplot as plt

import zHead as hd

#arg_options = hd.handle_arguments()
#ActiveRunIDs = arg_options[0]
#ActiveRunIDs = [2,3]
#print(ActiveRunIDs)
#date = arg_options[1]

#RunName = 'arb.'

CalcLYRA    = True
CalcRIGEL   = True
CalcSMUGGLE = True

PlotOnly = False

NFields = 8
MASS_SCALES = [10,20,50,100,200,500,1000,2000,4000]

MAX_PTCL_LIST = [ 1, 1, 64, 64, 4096, 2048 ]

CALCED_FSNAP_LIST = [ 0, 0, 150, 150, 40, 250 ]
CALCED_LSNAP_LIST = [ 0, 0, 900, 900, 790, 1000 ]
SNAP_INTVL_LIST   = [ 1, 1, 10, 10, 10, 10 ]

###
ActiveRunIDs = []
if CalcSMUGGLE:
    ActiveRunIDs.append(2)
    ActiveRunIDs.append(3)
if CalcRIGEL:
    ActiveRunIDs.append(4)    
if CalcLYRA:
    ActiveRunIDs.append(5)
CALCED_SNAPS_LIST = [ np.arange( CALCED_FSNAP_LIST[b], CALCED_LSNAP_LIST[b], SNAP_INTVL_LIST[b] ) for b in range(6) ]
print( CALCED_SNAPS_LIST )
###

#Nruns = len(hd.runlabels)
#N_ACTIVE_RUNS = len(ActiveRunIDs)

def do_calculations( i,j, max_ptcl ):

    Snap = hd.Snapshot(i,j)
    f = Snap.f
    #s = Snap.s
    #subhalo_id = 0

    #if Snap.MODE == 'COSMO':
        #SIDarr = np.loadtxt('data/SubhaloIDData_run{}'.format(j),dtype='int')
        #SID = SIDarr[i,subhalo_id]
    #else:
        #SID = 0

    #print('NOW CALCULATING {} (RUN NO. {}) SNAPSHOT NO. {} '.format(RunName,j,i))

    #origin, vcom, Ltot = hd.adjustdata( Snap, subhalo_id=SID )
    #print('Obtained Origin')

    snap_kwargs = { 'parts_per_snap':Snap.PartsPerSnap, 'cosmo_scaling':Snap.CosmoScaling }

    ### LOAD THINGS

    xyz0 = hd.extract( f,u'Coordinates',0, units='gal', **snap_kwargs )
    mass0 = hd.extract( f,u'Masses',0 , units='gal', **snap_kwargs)

    vel0 = hd.extract( f,u'Velocities',0, units='gal', **snap_kwargs )
    KE0 = 0.5 * mass0 * sum([ vel0[:,b]*vel0[:,b] for b in range(3) ])
    TE0 = mass0 * hd.extract( f,u'InternalEnergy',0, units='gal', **snap_kwargs )
    # Energies in Msun * (km/s)^2 = 1.989e43 erg

    dens0 = hd.extract( f,u'Density',0, units='gal', **snap_kwargs )
    u0 = hd.extract( f,u'InternalEnergy',0, **snap_kwargs )
    Nelec = hd.extract( f,u'ElectronAbundance',0, **snap_kwargs )
    temp0 = hd.GasTemp( u0, Nelec )

    age4 = Snap.find_star_age( hd.extract( f,u'GFM_StellarFormationTime',4, **snap_kwargs ) )
    xyz4 = hd.extract( f,u'Coordinates',4, units='gal', **snap_kwargs ) # if cosmo, don't account for birth position; its gonna be impossible
    m4 = hd.extract( f,u'Masses',4, units='gal', **snap_kwargs )

    ### Adjust & Exclude

    #xyz0 -= origin
    #xyz4 -= origin

    if j != 4: # non-RIGEL runs
        csfr0 = hd.extract( f,u'StarFormationRate',0 , units='gal', **snap_kwargs )
        kkSF = csfr0 > 0.0
    else:

        vDiv = 1.e3 * hd.extract( f,u'VelocityDivergence',0, **snap_kwargs ) # km/s / pc -> km/s / kpc
        vCurl = 1.e3 * hd.extract( f,u'VelocityCurl',0, **snap_kwargs ) # km/s / pc; based on sim numbers, im assuming parsecs. so convert to kpc
        VirParam = vDiv*vDiv + vCurl*vCurl

        kkCold = temp0 < 100 # Kelvin
        kkDens = dens0 * 4.44e-8 < 1.e4 # mH/cc
        kkDiv  = vDiv < 0
        kkVirP = VirParam < 2.*hd.G*dens0
        kkSF = kkCold * kkDens * kkDiv * kkVirP

        print( np.sum(kkCold), np.sum(kkDens), np.sum(kkDiv), np.sum(kkVirP), np.sum(kkSF) )

        csfr0 = np.where( kkSF, mass0 / np.sqrt( 3.*np.pi/32./hd.G/dens0 ), np.zeros_like(mass0) )

    ### CALCULATE THINGS

    xyz0_SF = xyz0[kkSF]
    mass0_SF = mass0[kkSF]

    All_NBHD_Data = []
    for iCell,(mCell,sfcell_pos) in enumerate(zip(mass0_SF,xyz0_SF)):

        xyz0rel = xyz0-sfcell_pos
        r0rel = np.sqrt(sum([ xyz0rel[:,b]*xyz0rel[:,b] for b in range(3) ]))

        xyz4rel = xyz4-sfcell_pos
        r4rel = np.sqrt(sum([ xyz4rel[:,b]*xyz4rel[:,b] for b in range(3) ]))

        ind_rads = np.argsort(r0rel)
        ordered_masses = mass0[ind_rads]
        ordered_masses_cum = np.cumsum(ordered_masses)

        NBHD_Data = []
        for nbhd_size in MASS_SCALES:

            MassReached = ( ordered_masses_cum > nbhd_size )
            AllFalse = not np.max( MassReached )
            AllTrue  = np.min( MassReached )
            if AllTrue:
                ptcl_est = 1
            elif AllFalse:
                ptcl_est = max_ptcl
            else:
                ptcl_est = np.argmax(MassReached) + 1

            #print(MassReached)

            kk_nbhd = ind_rads[:ptcl_est] # are you in the neighborhood of the SF-ing gas particle in question?

            rad_nbhd = r0rel[kk_nbhd][-1]

            mass_nbhd = np.sum(mass0[kk_nbhd])
            ke_nbhd = np.sum(KE0[kk_nbhd])
            te_nbhd = np.sum(TE0[kk_nbhd])
            csfr_nbhd = np.sum(csfr0[kk_nbhd])

            kk_SFing_in_nbhd = ( r0rel < rad_nbhd ) * ( csfr0 > 0 )
            sfing_mass_nbhd = np.sum( mass0[kk_SFing_in_nbhd] )

            kk4_YoungStarsInNbhd = ( r4rel < rad_nbhd ) * ( age4 < 50. ) # Only care about young stars; if its an old star, it might've just ended up in that neighborhood and we don't care

            mstar_young_nbhd = np.sum( m4[kk4_YoungStarsInNbhd] )

            appenddat = [ mCell, rad_nbhd, mass_nbhd, ke_nbhd, te_nbhd, csfr_nbhd, sfing_mass_nbhd, mstar_young_nbhd ]
            #NFields = len(appenddat)
            #print(NFields)
            NBHD_Data.append(appenddat)

        #NBHD_Data = np.array(NBHD_Data)
        All_NBHD_Data.append(NBHD_Data)

    All_NBHD_Data = np.array(All_NBHD_Data)
    #print(np.shape(All_NBHD_Data))

    return All_NBHD_Data

###

if not PlotOnly:
    for j in ActiveRunIDs:

        bfrw = 17
        ngbdat = np.zeros((bfrw,len(MASS_SCALES),NFields))

        CalcedSnaps = CALCED_SNAPS_LIST[j]
        MaxPtcls = MAX_PTCL_LIST[j]

        for i in CalcedSnaps:
            snap_nbhddat = do_calculations(i,j,MaxPtcls)
            if np.size(snap_nbhddat) != 0:
                ngbdat = np.concatenate( (ngbdat,snap_nbhddat), axis=0  )
            else:
                pass
            print(np.shape(ngbdat))

        ngbdat = ngbdat[bfrw:]
        ngbdat = np.transpose(ngbdat,axes=(2,1,0))
        np.save('data/TF4data/run{}.npy'.format(j),ngbdat)
        print(np.shape(ngbdat))

else:
    for j in ActiveRunIDs:
        ngbdat = np.load('data/TF4data/run{}.npy'.format(j))
        print(np.shape(ngbdat))

### ### ###

LocalDensity = ( ngbdat[1] / ( 4.*np.pi/3. ) / ngbdat[0]**3. ) * 4.44e-8
LocalEnergyRatio = ngbdat[2] / ngbdat[3]
LocalSpecSFR = ( ngbdat[4] / ngbdat[1] ) * 1.e-9
LocalSFMassProportion = ngbdat[4] / ngbdat[1]

plt.scatter( LocalDensity[0], LocalSpecSFR[0]+1.e-13, s=1 )
plt.scatter( LocalDensity[2], LocalSpecSFR[2]+1.e-13, s=1 )
plt.xscale('log')
plt.yscale('log')
plt.xlim((1.e-2,1.e5))
plt.show()
