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

#RigelStarMass = np.loadtxt('data/RIGELMassiveStarMass_noRT.txt')
#SNeOccurred = RigelStarMass[-1] == -2
#RigelSiteDens = np.loadtxt('data/RIGELSiteDens_noRT.txt')[-1,SNeOccurred]
RigelSiteDens = GP.RigelSiteDens

UseRigelTrueDens = False

MassScales = GP.MassScales

#cmap, norm = hd.create_cmap('RdYlGn',lo=-1.0,hi=1.0,Nbins=12)

MINDENS = -3.0
MAXDENS = 2.0
dens_space = np.linspace(MINDENS,MAXDENS,50)

def CoolRad(nH):
    return 28.4 * nH**(-3./7.)

def DensityFormulaCurve(nH,M=2000.):
    rho = nH / 44.44444 # 4.44e-8 mH/cm3 = 1 Msun/kpc^3, 44 mH/cm3 = 1 Msun/pc^3
    return ( (3./4./np.pi)*(M/rho) )**(1./3.)

figE, axsE = plt.subplots(1,1,figsize=(7,6))
plt.subplots_adjust(hspace=0.0,wspace=0.15)

colors = GP.colors
#colors = ['Reds_r','Grays_r','Greens_r','Reds_r','Blues_r','Greens_r']
#runCmaps = [ hd.create_cmap(colorname,lo=-100.0,hi=200.0,Nbins=10)[0] for colorname in colors ]
#runNorms = [ hd.create_cmap(colorname,lo=-100.0,hi=200.0,Nbins=10)[1] for colorname in colors ]
runCmaps = GP.contourCmaps
runNorms = GP.contourNorms
alphas = [0.2,0.2,0,0,1.0,0]

#for jSim,localdata in enumerate([ localdata_LYRA, localdata_RIGEL, localdata_SMUGGLE, localdata_LYRArt, localdata_RIGELrt, localdata_SMUGGLErt ]):
#for jSim,(localdata,SimLabel) in enumerate(zip(LocalDatas,LocalDataNames)):
    #if jSim in [0,1,4]:

        #print(jSim)

npi_fine   = 1
npi_coarse = 7

percentile_kwargs = {'total_bins':15, 'Xmin':MINDENS-0.5, 'Xmax':MAXDENS+0.5}

# Run 1: RIGEL
jSim=1
dens20 = LocalDatas[jSim][0,npi_fine]
dens2000 = LocalDatas[jSim][0,npi_coarse]
if UseRigelTrueDens:
    dens20 = np.log10(RigelSiteDens*44.44444)

X1 = dens2000
Y1 = CoolRad(10.**LocalDatas[jSim][0,0])
#axsE.scatter( X1, Y1, s=1, alpha=0.13, color='gray' ) #label='RIGEL-noRT', color='gray'

xSig_1, ySigb_1 = hd.scatter_to_percentile( X1, Y1, 16, **percentile_kwargs )
xSig_1, ySigt_1 = hd.scatter_to_percentile( X1, Y1, 84, **percentile_kwargs )
axsE.fill_between( xSig_1, ySigb_1, ySigt_1, color='gray', alpha=0.07 )


# Run 0: LYRA
jSim=0
dens20 = LocalDatas[jSim][0,npi_fine]
dens2000 = LocalDatas[jSim][0,npi_coarse]

X0 = dens2000
Y0 = CoolRad(10.**LocalDatas[jSim][0,0])
#print(Y0)
axsE.scatter( X0, Y0, s=1.5, alpha=0.15, label='LYRA (noRT)', color='red' )

xSig_0, ySigb_0 = hd.scatter_to_percentile( X0, Y0, 16, **percentile_kwargs )
xSig_0, ySigt_0 = hd.scatter_to_percentile( X0, Y0, 84, **percentile_kwargs )
axsE.fill_between( xSig_0, ySigb_0, ySigt_0, color='red', alpha=0.1 )


# Run 4: RIGEL-RT
jSim=4
dens20 = LocalDatas[jSim][0,npi_fine]
dens2000 = LocalDatas[jSim][0,npi_coarse]

X4 = dens2000
Y4 = CoolRad(10.**LocalDatas[jSim][0,0])
#print(Y4)
axsE.scatter( X4, Y4, s=2, alpha=1.0, label='RIGEL (RT)', color='green' )

xSig_4, ySigb_4 = hd.scatter_to_percentile( X4, Y4, 16, **percentile_kwargs )
xSig_4, ySigt_4 = hd.scatter_to_percentile( X4, Y4, 84, **percentile_kwargs )
axsE.fill_between( xSig_4, ySigb_4, ySigt_4, color='green', alpha=0.1 )


### Other

axsE.plot( dens_space, CoolRad(10.**dens_space), color='black', label=r'Expected $r_{\rm cool}$' )

axsE.set_xlabel(r'Coarse-Estimate Density ${\rho}_{2000}$ [mH/cc]',fontsize=14)
axsE.set_ylabel(r'Fine-Estimate Cooling Radius $r_{\rm cool} ({\rho}_{20})$ [pc]',fontsize=14)

hd.axisstyle(axsE)

axsE.set_yscale('log')

axsE.set_xlim((MINDENS,MAXDENS))
axsE.set_ylim((3.e-1,3.e3))
axsE.tick_params(labelsize=13)

axsE.legend(loc='upper right',fontsize=13)

axsE.text( 0.37, 10.**(0.768), r'r=(22.6 pc)$ {\rho}_{2000}^{-0.42} $', fontsize=13, rotation=-23.7, alpha=0.69, ha='left', va='bottom' )

#axsE.set_title('Cooling Radius Comparison')

figE.savefig('OfficialPlots/P7-CoolingRadius.png',dpi=300)
plt.show()
