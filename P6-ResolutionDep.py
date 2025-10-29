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

cmap, norm = hd.create_cmap('RdYlGn',lo=-1.0,hi=1.0,Nbins=12)

MINDENS = -5.0
MAXDENS = 5.0

dens_space = np.linspace(MINDENS,MAXDENS,50)
def vtoE(v_kms):
    return np.log10( 1.989e43 * v_kms * v_kms )

figE, axsE = plt.subplots(3,3,figsize=(11,6.75),gridspec_kw={ 'height_ratios':[4,3,2], 'width_ratios':[2,2,2] })
plt.subplots_adjust(hspace=0.0,wspace=0.0)

#histcolors = GP.colors
runCmaps = GP.contourCmaps
runNorms = GP.contourNorms

#SFRDats = [ np.load('data/TF4data/run{}.npy'.format(b)) for b in range(2,6) ] # SMUGGLE-RT, SMUGGLE-noRT, RIGEL, LYRA

#print([ np.shape(SFRDats[b]) for b in range(4) ])

#LocDensSFRs = [ ( 4.44e-8 * SFRDats[b][2] / ( 4.*np.pi/3. ) / SFRDats[b][1]**3. ) for b in range(4) ]
#SpecSFRs = [ 1.e-9 * ( SFRDats[b][5] / SFRDats[b][2] ) for b in range(4) ]

# Plotting

for jSim,localdata in enumerate(LocalDatas):
    if jSim==0: # LYRA (noRT)
        print('LYRA')
        for iMass,npi in enumerate([1,4,7]): # bimodality can kind of be seen at Level 4 (64 particles), barely visible at Lv 5 (128 ptcl), completely gone by Lv 6 (256 ptcl)

            # Grab Data (Top Section)
            idens_Main = localdata[0,npi]
            energy_array = np.log10(localdata[3,npi])
            ratio = np.log10(localdata[6,npi])

            print(localdata[3,npi])
            print(localdata[6,npi])

            kk_posE = np.logical_not(np.isnan(energy_array))
            kk_posRatio = np.logical_not(np.isnan(ratio))

            # Find Contours (Top Section)
            percentile_levels = [0.68,0.95]
            contour_kwargs_energy = { 'xrange':[MINDENS,MAXDENS], 'yrange':[42,50], 'bins':[200,160], 'sigma':4 }
            contour_kwargs_ratio  = { 'xrange':[MINDENS,MAXDENS], 'yrange':[-3,3], 'bins':[200,80], 'sigma':4 }
            X1,Y1,Z1,levels1, ContourLabel1 = hd.scatter_to_contour( idens_Main[kk_posE],energy_array[kk_posE],np.ones_like(kk_posE,dtype='float'), percentile_levels, **contour_kwargs_energy )
            X2,Y2,Z2,levels2, ContourLabel2 = hd.scatter_to_contour( idens_Main[kk_posRatio],ratio[kk_posRatio],np.ones_like(kk_posRatio,dtype='float'), percentile_levels, **contour_kwargs_ratio )

            # Do the plotting
            CS1 = axsE[0,iMass].contour(X1,Y1,Z1,levels1, linewidths=1.5, cmap=runCmaps[jSim], norm=runNorms[jSim], linestyles=['dashed','solid'], alpha=1.0)
            CS2 = axsE[1,iMass].contour(X2,Y2,Z2,levels2, linewidths=1.5, cmap=runCmaps[jSim], norm=runNorms[jSim], linestyles=['dashed','solid'], alpha=1.0)
            axsE[2,iMass].hist( idens_Main, color='#f24332', bins=np.arange(MINDENS,MAXDENS,0.5), density=True, histtype='step', linewidth=2.0, alpha=1.0, label='LYRA (H.Res)')

            # Median Ratio
            xMed, yMed, ySigMed = hd.scatter_to_median( idens_Main[kk_posRatio], ratio[kk_posRatio] )
            axsE[1,iMass].plot( xMed, yMed, color='#f24332', alpha=1.0, linestyle='dotted' )

            # Secret Line
            axsE[0,iMass].plot( [0,1], [52,53], color='#f24332', linewidth=2.0, alpha=1.0, label='LYRA (H.Res)')

    if jSim==2: # SMUGGLE-noRT
        print('SMUGGLE no RT')
        for iMass,npi in enumerate([7]): # bimodality can kind of be seen at Level 4 (64 particles), barely visible at Lv 5 (128 ptcl), completely gone by Lv 6 (256 ptcl)

            # Grab Data (Top Section)
            idens_Main = localdata[0,npi]
            energy_array = np.log10(localdata[3,npi])
            ratio = np.log10(localdata[6,npi])

            print(localdata[3,npi])
            print(localdata[6,npi])

            kk_posE = np.logical_not(np.isnan(energy_array))
            kk_posRatio = np.logical_not(np.isnan(ratio))

            # Find Contours (Top Section)
            percentile_levels = [0.68,0.95]
            contour_kwargs_energy = { 'xrange':[MINDENS,MAXDENS], 'yrange':[42,50], 'bins':[200,160], 'sigma':4 }
            contour_kwargs_ratio  = { 'xrange':[MINDENS,MAXDENS], 'yrange':[-3,3], 'bins':[200,80], 'sigma':4 }
            X1,Y1,Z1,levels1, ContourLabel1 = hd.scatter_to_contour( idens_Main[kk_posE],energy_array[kk_posE],np.ones_like(kk_posE,dtype='float'), percentile_levels, **contour_kwargs_energy )
            X2,Y2,Z2,levels2, ContourLabel2 = hd.scatter_to_contour( idens_Main[kk_posRatio],ratio[kk_posRatio],np.ones_like(kk_posRatio,dtype='float'), percentile_levels, **contour_kwargs_ratio )

            xMed, yMed, ySigMed = hd.scatter_to_median( idens_Main[kk_posRatio], ratio[kk_posRatio] )
            axsE[1,2].plot( xMed, yMed, color='#8583bd', alpha=1.0, linestyle='dotted' )

            # Do the plotting
            CS1 = axsE[0,2].contour(X1,Y1,Z1,levels1, linewidths=1.5, cmap=runCmaps[jSim], norm=runNorms[jSim], linestyles=['dashed','solid'], alpha=1.0)
            CS2 = axsE[1,2].contour(X2,Y2,Z2,levels2, linewidths=1.5, cmap=runCmaps[jSim], norm=runNorms[jSim], linestyles=['dashed','solid'], alpha=1.0)
            axsE[2,2].hist( idens_Main, color='#8583bd', bins=np.arange(MINDENS,MAXDENS,0.5), density=True, histtype='step', linewidth=2.0, alpha=1.0, label='SMUGGLE (L.Res)')

            # Secret Line
            axsE[0,2].plot( [0,1], [52,53], color='#8583bd', linewidth=2.0, alpha=1.0, label='SMUGGLE (L.Res)')
            
for kcol in range(3):

    axsE[0,kcol].plot( dens_space, vtoE(30)*np.ones(50), color='gray', linestyle='dashed', alpha=0.13 )
    axsE[0,2].text( 4.9, 46.15, '30 km/s', color='gray', fontsize=11, va='top', ha='right', alpha=0.3 )
    axsE[1,kcol].plot( dens_space, np.log10(0.28/0.72)*np.ones(50), color='gray', linestyle='dotted', alpha=0.26 )
    axsE[1,2].text( 4.9, -0.5, '28% Kinetic', color='gray', fontsize=11, va='top', ha='right', alpha=0.3 )

    for krow in range(3):
        axsE[krow,kcol].set_xlim(MINDENS,MAXDENS)
        axsE[krow,kcol].tick_params(labelsize=12)
        hd.axisstyle(axsE[krow,kcol])

    axsE[0,kcol].set_ylim((42.5,50.5))
    axsE[0,kcol].set_yticks([44,46,48,50])
    axsE[0,kcol].set_yticklabels(['$10^{44}$','$10^{46}$','$10^{48}$','$10^{50}$'])

    axsE[1,kcol].set_ylim((-3.0,3.0))
    axsE[1,kcol].set_yticks([-2,0,2])
    axsE[1,kcol].set_yticklabels(['$10^{-2}$','$10^{0}$','$10^{2}$'])

    axsE[2,kcol].set_ylim((0,0.8))
    axsE[2,kcol].set_yticks([0,0.3,0.6])

for kcol in range(3):
    axsE[2,kcol].set_xticks([-4,-2,0,2,4])
    axsE[2,kcol].set_xticklabels(['$10^{-4}$','$10^{2}$','$10^{0}$','$10^{2}$','$10^{4}$'])

for kcol in range(3):
    axsE[0,kcol].set_xticklabels([])
    axsE[1,kcol].set_xticklabels([])

for krow in range(3):
    axsE[krow,1].set_yticklabels([])
    axsE[krow,2].set_yticklabels([])

# Axis Labels
#axsE[0,0].set_ylabel('Avg. '+r'$\Delta e$ [erg/${\rm M}_{\odot}$]',fontsize=12)
#axsE[1,0].set_ylabel(r'$\delta$KE / $\delta$TE',fontsize=12)
#axsE[2,0].set_ylabel(r'P [arb.]',fontsize=12)

figE.text( 0.062, 0.67, 'Avg. '+r'$\Delta e$ [erg/${\rm M}_{\odot}$]',fontsize=12, rotation=90, va='center')
figE.text( 0.062, 0.42, r'$\delta$KE / $\delta$TE',fontsize=12, rotation=90, va='center')
figE.text( 0.062, 0.19, r'Likelihood',fontsize=12, rotation=90, va='center')

axsE[0,0].set_title(r'20 {} Env.'.format(hd.Msun))
axsE[0,1].set_title(r'200 {} Env.'.format(hd.Msun))
axsE[0,2].set_title(r'2000 {} Env.'.format(hd.Msun))

axsE[0,2].legend(loc='upper right', fontsize=11)

#figE.supxlabel(r'Density [${\rm cm}^{-3}$]',fontsize=14)
#figE.text( 0.52, 0.045, r'Density [${\rm cm}^{-3}$]',fontsize=14, ha='center' )
axsE[2,1].set_xlabel( r'Density [${\rm cm}^{-3}$]',fontsize=14, ha='center' )

figE.savefig('OfficialPlots/P6-ResolutionDiff.png',dpi=300)
plt.show()
