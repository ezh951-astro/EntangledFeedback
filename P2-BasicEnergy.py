import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import corner
import zHead as hd
import zGlobalParams as GP

LocalDatas = GP.LocalDatas
LocalDataNames = GP.LocalDataNames

RigelSiteDens = GP.RigelSiteDens

MassScales = GP.MassScales

cmap, norm = hd.create_cmap('RdYlGn',lo=-1.0,hi=1.0,Nbins=12)

MINDENS = -5.0
MAXDENS = 5.0

EMIN = 42
EMAX = 50

NSNMAX = 600

dens_space = np.linspace(MINDENS,MAXDENS,50)
def vtoE(v_kms):
    return np.log10( 1.989e43 * v_kms * v_kms )
figE, axsE = plt.subplots(2,1,figsize=(7,7),gridspec_kw={'height_ratios':[4,1]})
plt.subplots_adjust(hspace=0.0)

histcolors = GP.colors

runCmaps = GP.contourCmaps
runNorms = GP.contourNorms

alphas = [0.7,0.4]

for jSim,localdata in enumerate(LocalDatas):
    if jSim in [0,1]:

        npi = 1

        idens_Main = localdata[0,npi]
        energy_array = localdata[3,npi]
        ratio = localdata[6,npi]

        kk_posE = energy_array > 0.0
        kk_negE = energy_array <= 0.0

        if jSim==0:
            axsE[0].scatter( idens_Main[kk_posE], np.log10(energy_array[kk_posE]), s=1, c=histcolors[jSim], alpha=0.5 )

        percentile_levels = [0.68,0.95]
        contour_kwargs_energy = { 'xrange':[MINDENS,MAXDENS], 'yrange':[42,50], 'bins':[200,160], 'sigma':4 }

        X1,Y1,Z1,levels1, ContourLabel1 = hd.scatter_to_contour( idens_Main[kk_posE],np.log10(energy_array[kk_posE]),np.ones_like(kk_posE,dtype='float'), percentile_levels, **contour_kwargs_energy )

        axsE[0].contour(X1,Y1,Z1,levels1,label=ContourLabel1, linewidths=1.5, cmap=runCmaps[jSim], norm=runNorms[jSim], linestyles=['dashed','solid'], alpha=alphas[jSim])

        if jSim==0:

            axsE[0].plot( dens_space, np.log10( 1.e51/1680./(10.**dens_space)**(-0.26) )*np.ones(50), color='purple', linestyle='dashed', alpha=0.7 )
            axsE[0].text( 2.5, 48.62, 'S-T Transition\n KO+15', color='purple', fontsize=12, ha='center', va='bottom', alpha=0.9 )

            ref_de = np.log10( 1.e51 / 1680. / (10.**idens_Main)**(-0.26) )

            of_All  = ( np.log10(energy_array) > ref_de )
            #of_PosE = ( np.log10(energy_array)[kk_posE] > ref_de[kk_posE] )
            of_PosE = ( np.log10(energy_array) > ref_de )[kk_posE]

            print( 'Above ST Line out of All SNe {}/{}'.format( np.sum(of_All), len(of_All) ))
            print( 'Above ST Line out of Positive SNe {}/{}'.format( np.sum(of_PosE), len(of_PosE) ))
            print( 'Number of positive SNe', sum(kk_posE) )

            axsE[0].plot( dens_space, vtoE(30)*np.ones(50), color='black', linestyle='dashed', alpha=0.55 )
            axsE[0].text( 1.5, 46.17, r'$v=30 {\rm km/s}$', color='black', fontsize=12, ha='center', va='top', alpha=0.6 )

            axsE[0].fill_between( [3,MAXDENS], [EMIN,EMIN], [EMAX,EMAX], color='pink', alpha=0.3 )
            axsE[1].fill_between( [3,MAXDENS], [0,0], [NSNMAX,NSNMAX], color='pink', alpha=0.3 )
            axsE[0].text( 2.95, 42.1, r'SF Thresh.', color='red', rotation=90, ha='right', va='bottom', alpha=0.4 )

            #axsE[0].vlines( [3], 42,50, linestyle='dotted', color='green', alpha=0.6 )
            #axsE[1].vlines( [3], 42,50, linestyle='dotted', color='green', alpha=0.6 )

            axsE[1].hist( idens_Main, color='red', bins=np.arange(MINDENS,MAXDENS,0.5), alpha=0.65, histtype='step', linewidth=3 )

            axsE[0].set_ylabel(r'Specific Energy $\Delta e$ [erg/${\rm M}_{\odot}$]',fontsize=14.5)
            axsE[1].set_ylabel('No. SNe',fontsize=14.5)
            axsE[1].set_xlabel(r'Local Density [${\rm cm}^{-3}$]',fontsize=14.5)

            for krow in [0,1]:
                hd.axisstyle(axsE[krow])
                axsE[krow].set_xlim(MINDENS,MAXDENS)
                axsE[krow].tick_params(labelsize=13)

            axsE[0].set_ylim((EMIN,EMAX))
            axsE[0].set_yticks([42,44,46,48,50])
            axsE[0].set_yticklabels([ r'${10}^{42}$', r'${10}^{44}$', r'${10}^{46}$', r'${10}^{48}$', r'${10}^{50}$' ])

            axsE[0].set_xticklabels([])

            axsE[1].set_xticks([-4,-2,0,2,4])
            axsE[1].set_xticklabels([ r'${10}^{-4}$', r'${10}^{-2}$', r'${10}^{0}$', r'${10}^{2}$', r'${10}^{4}$' ])

            axsE[1].set_ylim((0,NSNMAX))
            axsE[1].set_yticks([0,250,500])
 
figE.savefig('OfficialPlots/P2-BasicEnergy.png',dpi=300)
plt.show()

#localdata_LYRA = np.load('auxdata/LocalData_run0.npy')
#localdata_RIGEL = np.load('auxdata/LocalData_run1.npy')
#localdata_SMUGGLE = np.load('auxdata/LocalData_run2.npy')
#localdata_LYRArt = np.load('auxdata/LocalData_run3.npy')
#localdata_RIGELrt = np.load('auxdata/LocalData_run4.npy')
#localdata_SMUGGLErt = np.load('auxdata/LocalData_run5.npy')

#RigelStarMass = np.loadtxt('data/RIGELMassiveStarMass_noRT.txt')
#SNeOccurred = RigelStarMass[-1] == -2
#RigelSiteDens = np.loadtxt('data/RIGELSiteDens_noRT.txt')[-1,SNeOccurred]
