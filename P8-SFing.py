import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import corner
import zHead as hd
import zGlobalParams as GP

#localdata_LYRA = np.load('auxdata/LocalData_run0.npy')
#localdata_RIGEL = np.load('auxdata/LocalData_run1.npy')
#localdata_SMUGGLE = np.load('auxdata/LocalData_run2.npy')
#localdata_LYRArt = np.load('auxdata/LocalData_run3.npy')
#localdata_RIGELrt = np.load('auxdata/LocalData_run4.npy')
#localdata_SMUGGLErt = np.load('auxdata/LocalData_run5.npy')
LocalDatas = GP.LocalDatas
LocalDataNames = GP.LocalDataNames

MassScales = GP.MassScales

###

CalcLYRA    = False
CalcRIGEL   = False
CalcSMUGGLE = False

CALCULATE = (CalcLYRA or CalcRIGEL) or CalcSMUGGLE

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

def do_calculations( i,j, max_ptcl ):

    Snap = hd.Snapshot(i,j)
    f = Snap.f

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
            NBHD_Data.append(appenddat)

        All_NBHD_Data.append(NBHD_Data)

    All_NBHD_Data = np.array(All_NBHD_Data)

    return All_NBHD_Data


###

if CALCULATE:

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
        np.save('data/SFRegDat{}.npy'.format(j),ngbdat)
        print(np.shape(ngbdat))

### ### ###

SFRDats = [ np.load('data/SFRegDat{}.npy'.format(b)) for b in range(2,6) ] # SMUGGLE-RT, SMUGGLE-noRT, RIGEL, LYRA
LocDensSFRs = [ ( 4.44e-8 * SFRDats[b][2] / ( 4.*np.pi/3. ) / SFRDats[b][1]**3. ) for b in range(4) ]
SpecSFRs = [ 1.e-9 * ( SFRDats[b][5] / SFRDats[b][2] ) for b in range(4) ]

figE, axsE = plt.subplots(3,2,figsize=(6,8))
plt.subplots_adjust(hspace=0.0,wspace=0.0,left=0.2)

for jSim,localdata in enumerate(LocalDatas):

    # Can be loaded with the correct jSim for every simulation.
    CSFR_aft_zero = localdata[2,1]==0. # npi=1, 2nd index
    CSFR_bef_zero = localdata[1,1]==0.
    CSFR_afz_not_bfz = np.logical_and(localdata[2,1]==0., localdata[1,1]>0. )

    if jSim==0: # LYRA (noRT)
        for iMass,npi in enumerate([1,4,7]):
            axsE[iMass,0].scatter( np.log10(LocDensSFRs[3][npi]), np.log10(SpecSFRs[3][npi]+1.e-13), c='red', s=1, alpha=0.5, label='LYRA' ) # Index of Simulation is MANUALLY LINKED!
            print('Sim LYRA: SFRBef Zero: {}, SFRAft Zero: {}, After but not Before: {}, NumSNeTot: {}'.format( np.sum(CSFR_bef_zero), np.sum(CSFR_aft_zero), np.sum(CSFR_afz_not_bfz), len(CSFR_aft_zero) ))
            axsE[iMass,0].axvline( np.median(np.log10(LocDensSFRs[3][npi])), alpha=0.37, linestyle='dashed', color='red' )

    if jSim==2: # SMUGGLE-noRT
        for iMass,npi in enumerate([7]):
            axsE[2,0].scatter( np.log10(LocDensSFRs[1][npi]), np.log10(SpecSFRs[1][npi]+1.e-13), c='blue', s=1, alpha=0.5, label='SMUGGLE' ) # Index of Simulation is MANUALLY LINKED!
            print('Sim SMUGGLE-noRT: SFRBef Zero: {}, SFRAft Zero: {}, After but not Before: {}, NumSNeTot: {}'.format( np.sum(CSFR_bef_zero), np.sum(CSFR_aft_zero), np.sum(CSFR_afz_not_bfz), len(CSFR_aft_zero) ))
            axsE[2,0].axvline( np.median(np.log10(LocDensSFRs[1][npi])), alpha=0.37, linestyle='dashed', color='blue' )

            # Secret Scatter
            axsE[0,0].scatter( np.log10(LocDensSFRs[1][npi])+50., np.log10(SpecSFRs[1][npi]+1.e-13), c='blue', s=1, alpha=0.5, label='SMUGGLE' ) # Index of Simulation is MANUALLY LINKED!

    if jSim==4: # RIGEL (RT)
        for iMass,npi in enumerate([1,4,7]):
            axsE[iMass,1].scatter( np.log10(LocDensSFRs[2][npi]), np.log10(SpecSFRs[2][npi]+1.e-13), c='green', s=1, alpha=0.5, label='RIGEL' ) # Index of Simulation is MANUALLY LINKED!
            print('Sim RIGEL: SFRBef Zero: {}, SFRAft Zero: {}, After but not Before: {}, NumSNeTot: {}'.format( np.sum(CSFR_bef_zero), np.sum(CSFR_aft_zero), np.sum(CSFR_afz_not_bfz), len(CSFR_aft_zero) ))
            axsE[iMass,1].axvline( np.median(np.log10(LocDensSFRs[2][npi])), alpha=0.37, linestyle='dashed', color='green' )

    if jSim==5: # SMUGGLE-RT
        for iMass,npi in enumerate([7]):
            axsE[2,1].scatter( np.log10(LocDensSFRs[0][npi]), np.log10(SpecSFRs[0][npi]+1.e-13), c='C1', s=1, alpha=0.5, label='SMUGGLE' ) # Index of Simulation is MANUALLY LINKED!
            print('Sim SMUGGLE-RT: SFRBef Zero: {}, SFRAft Zero: {}, After but not Before: {}, NumSNeTot: {}'.format( np.sum(CSFR_bef_zero), np.sum(CSFR_aft_zero), np.sum(CSFR_afz_not_bfz), len(CSFR_aft_zero) ))
            axsE[2,1].axvline( np.median(np.log10(LocDensSFRs[0][npi])), alpha=0.37, linestyle='dashed', color='C1' )

            # Secret Scatter
            axsE[0,1].scatter( np.log10(LocDensSFRs[0][npi])+50., np.log10(SpecSFRs[0][npi]+1.e-13), c='C1', s=1, alpha=0.5, label='SMUGGLE' ) # Index of Simulation is MANUALLY LINKED!


for kcol in range(2):
    for krow in range(3):
        hd.axisstyle(axsE[krow,kcol])
        axsE[krow,kcol].set_xlim(-2.5,5.0)
        axsE[krow,kcol].set_xticks([-2,0,2,4])
        axsE[krow,kcol].tick_params(labelsize=12)
        axsE[krow,kcol].set_ylim((-11.5,-6.5))
        axsE[krow,kcol].set_yticks(([-11,-9,-7]))

for krow in range(3):
    axsE[krow,1].set_yticklabels([])
for kcol in range(2):
    axsE[0,kcol].set_xticklabels([])
    axsE[1,kcol].set_xticklabels([])

axsE[0,0].set_ylabel(r'${\rm M}_{\rm env} = 20$ ' + hd.Msun, fontsize=12)
axsE[1,0].set_ylabel(r'${\rm M}_{\rm env} = 200$ ' + hd.Msun, fontsize=12)
axsE[2,0].set_ylabel(r'${\rm M}_{\rm env} = 2000$ ' + hd.Msun, fontsize=12)

axsE[0,0].set_title('No Stellar Radiation', fontsize=12)
axsE[0,1].set_title('Stellar Radiation', fontsize=12)

axsE[0,0].legend(loc='lower left', fontsize=11)
axsE[0,1].legend(loc='lower left', fontsize=11)

#figE.supxlabel(r'log10 Density [${\rm cm}^{-3}$]',fontsize=13.5)
figE.text( 0.55, 0.04,  r'log10 Density [${\rm cm}^{-3}$]', fontsize=13.5, ha='center' )
figE.supylabel(r'log10 Local SFR/$M_{\rm env}$ [${\rm yr}^{-1}$]',fontsize=13.5)

figE.savefig('OfficialPlots/P8-SFing.png', dpi=300)
plt.show()
