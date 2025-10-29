import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
import corner

import zMassScales as MS

#import zGlobalParams as GP

###

G = 6.6743015e-8 # cgs; cm, g, s
mH = 1.67262192e-24 # cgs

MSUN = r'${\rm M}_{\odot}$'

###

LOCDAT_L = np.load('../SupernovaExtract/LyraSNeExtract/data/LyraSNeLocationData_4096_5.npy')
ENVDAT_L = np.load('../SupernovaExtract/LyraSNeExtract/data/LyraSNeEnvironmentData_4096_5.npy')
kk_ZeroEnergy = LOCDAT_L[:,0,9]/1.e51 > 0.3
LOCDAT_L = LOCDAT_L[kk_ZeroEnergy]
ENVDAT_L = ENVDAT_L[kk_ZeroEnergy]

LOCDAT_R = np.transpose(np.load('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/Rigel_LocDat_2048_noRT.npy'),axes=(1,0,2))
ENVDAT_R = np.transpose(np.load('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/Rigel_EnvDat_2048_noRT.npy'),axes=(1,0,2,3))
RigelStarMass = np.loadtxt('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/RIGELMassiveStarMass_noRT.txt')
SNeOccurred = RigelStarMass[-1] == -2
RigelSiteDens = np.loadtxt('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/RIGELSiteDens_noRT.txt')[-1,SNeOccurred]

LOCDAT_S = np.transpose(np.load('../SupernovaExtract/SMUGGLESNeExtract/data/SMUGGLE_LocDat_N64_run1.npy'),axes=(1,0,2))
ENVDAT_S = np.transpose(np.load('../SupernovaExtract/SMUGGLESNeExtract/data/SMUGGLE_EnvDat_N64_run1.npy'),axes=(1,0,2,3))

LOCDAT_L_rt = LOCDAT_L # run does not exist
ENVDAT_L_rt = ENVDAT_L # run does not exist
kk_ZeroEnergy = LOCDAT_L_rt[:,0,9]/1.e51 > 0.3
LOCDAT_L_rt = LOCDAT_L_rt[kk_ZeroEnergy]
ENVDAT_L_rt = ENVDAT_L_rt[kk_ZeroEnergy]

LOCDAT_R_rt = np.transpose(np.load('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/Rigel_LocDat_2048_yesRT.npy'),axes=(1,0,2))
ENVDAT_R_rt = np.transpose(np.load('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/Rigel_EnvDat_2048_yesRT.npy'),axes=(1,0,2,3))

LOCDAT_S_rt = np.transpose(np.load('../SupernovaExtract/SMUGGLESNeExtract/data/SMUGGLE_LocDat_N64_run0.npy'),axes=(1,0,2))
ENVDAT_S_rt = np.transpose(np.load('../SupernovaExtract/SMUGGLESNeExtract/data/SMUGGLE_EnvDat_N64_run0.npy'),axes=(1,0,2,3))

LOCDAT_ALL = [LOCDAT_L,LOCDAT_R,LOCDAT_S,LOCDAT_L_rt,LOCDAT_R_rt,LOCDAT_S_rt]
ENVDAT_ALL = [ENVDAT_L,ENVDAT_R,ENVDAT_S,ENVDAT_L_rt,ENVDAT_R_rt,ENVDAT_S_rt]

###

#MassScales = [10,20,50,100,200,500,1000,2000,4000]
MassScales = MS.MassScales
print(MassScales)
NFields = 8

MINDENS = -5.5
MAXDENS = 4.5

for jSim,(LOCDAT,ENVDAT) in enumerate(zip(LOCDAT_ALL,ENVDAT_ALL)):
#for jSim,(LOCDAT,ENVDAT) in enumerate(zip([LOCDAT_L],[ENVDAT_L])):

    x_atSN = LOCDAT[:,0,1] - 100.
    y_atSN = LOCDAT[:,0,2] - 100.
    z_atSN = LOCDAT[:,0,3] - 100.
    R_at_SN = np.sqrt( x_atSN*x_atSN + y_atSN*y_atSN )

    NSN_tot = np.shape(LOCDAT)[0]
    NPart_TOT = np.shape(ENVDAT)[3]
    localdata = np.zeros((NFields,len(MassScales),NSN_tot))
    for nSN in range(NSN_tot):

        print('SN No. {}'.format(nSN))

        gas_pos = np.transpose( ENVDAT[nSN,:,0:3], axes=(1,0,2) ) # component, time, particle
        x = gas_pos[0]
        y = gas_pos[1]
        z = gas_pos[2]
        gas_norm = np.sqrt( sum([ gas_pos[b]*gas_pos[b] for b in range(3) ]) )
        rhat_gas = gas_pos / gas_norm # component, time, particle

        Mass = ENVDAT[nSN,:,6] # time, particle (like everything else)

        SpecificKineticEnergy = 1.989e43 * 0.5 * ( ENVDAT[nSN,:,3]*ENVDAT[nSN,:,3] + ENVDAT[nSN,:,4]*ENVDAT[nSN,:,4] + ENVDAT[nSN,:,5]*ENVDAT[nSN,:,5] )
        if jSim==0:
            SpecificThermalEnergy = ENVDAT[nSN,:,7] / ENVDAT[nSN,:,6]
        else:
            SpecificThermalEnergy = ENVDAT[nSN,:,7]
        SpecificTotalEnergy = SpecificKineticEnergy + SpecificThermalEnergy

        EnergyRatio = SpecificKineticEnergy / SpecificThermalEnergy

        #Temperature = ENVDAT[nSN,:,8]
        # Correct the RIGEL temperature
        if jSim in [1,4]:
            Temperature = ENVDAT[nSN,:,8] / 1.989e43
        else:
            Temperature = ENVDAT[nSN,:,8]

        if jSim in [2,5]:
            print(ENVDAT[nSN,0,7])
            print(ENVDAT[nSN,1,7])
            #print(Temperature)
            #print(ENVDAT[nSN,1,9])
        MassWeightedTemperature = Mass * Temperature

        if jSim in [0,3]: # LYRA
            CellSFR     = ENVDAT[nSN,:,11] * 60 * 525600 / 1.989e33 # Msun/yr
        if jSim in [2,5]: # SMUGGLE
            CellSFR     = ENVDAT[nSN,:,10] * 60 * 525600 / 1.989e33 # Msun/yr
        if jSim in [1,4]: # RIGEL; does not account for eligibility, so we will do that here

            DensSFCalc    = ENVDAT[nSN,:,9] * 1.e10 # cm3
            tdyn          = np.sqrt( 3.*np.pi/32./G/mH/DensSFCalc ) / ( 525600. * 60. ) # seconds to years

            kkRigelDensSF = ( DensSFCalc > 1.e4 )
            kkRigelTempSF = ( Temperature < 100. ) #  - np.log10(1.989e43) RIGEL temperatures (might be) wrong? Divide by 1.989e43 first
            kk = kkRigelDensSF * kkRigelTempSF

            CellSFR       = np.where( kk, Mass/tdyn, np.zeros_like(Mass) ) # Mass in Msun, time in yrs

        for i_NP,MScale in enumerate(MassScales):

            isnpb = 0
            if jSim==0: # LYRA
                isnpa = 1 # 4 for a 5 Myr dt; 1 for a 1 Myr dt.
            elif jSim==1: # RIGEL
                isnpa = 1
            elif jSim==2: # SMUGGLE
                isnpa = 1

            AllFalse_bef = not np.max( np.cumsum(Mass[isnpb])>MScale ) # maximum of everything is false so its all false; add a not to flip this to true (if everything is false)
            AllTrue_bef  = np.min( np.cumsum(Mass[isnpb])>MScale ) # minimum of everything is true so its all true
            if AllTrue_bef: # even the first particle exceeds the threshold; so just use 1 particle
                NPart_bef = 1
            elif AllFalse_bef: # no particle exceeds the threshold; so gotta use all particles
                NPart_bef = NPart_TOT
            else:
                NPart_bef = np.argmax( np.cumsum(Mass[isnpb])>MScale )

            AllFalse_aft = not np.max( np.cumsum(Mass[isnpa])>MScale )
            AllTrue_aft  = np.min( np.cumsum(Mass[isnpa])>MScale )
            if AllTrue_aft:
                NPart_aft = 1
            elif AllFalse_aft:
                NPart_aft = NPart_TOT
            else:
                NPart_aft = np.argmax( np.cumsum(Mass[isnpa])>MScale )

            print( 'NPart Before & After for Mass Scale {}: {} {}'.format(MScale,NPart_bef,NPart_aft) )

            TrueDensity = ( np.sum(Mass[isnpb,:NPart_bef]) / (4.*np.pi/3.) / gas_norm[isnpb,NPart_bef-1]**3. ) * 4.44e-8
            LocalTemperature = np.sum(MassWeightedTemperature[isnpb,:NPart_bef]) / np.sum(Mass[isnpb,:NPart_bef])

            TrueKEBef = np.sum( Mass[isnpb,:NPart_bef] * SpecificKineticEnergy[isnpb,:NPart_bef] ) / np.sum(Mass[isnpb,:NPart_bef])
            TrueKEAft = np.sum( Mass[isnpa,:NPart_aft] * SpecificKineticEnergy[isnpa,:NPart_aft] ) / np.sum(Mass[isnpa,:NPart_aft])

            TrueTEBef = np.sum( Mass[isnpb,:NPart_bef] * SpecificThermalEnergy[isnpb,:NPart_bef] ) / np.sum(Mass[isnpb,:NPart_bef])
            TrueTEAft = np.sum( Mass[isnpa,:NPart_aft] * SpecificThermalEnergy[isnpa,:NPart_aft] ) / np.sum(Mass[isnpa,:NPart_aft])

            TrueEnergyBef = np.sum( Mass[isnpb,:NPart_bef] * SpecificTotalEnergy[isnpb,:NPart_bef] ) / np.sum(Mass[isnpb,:NPart_bef])
            TrueEnergyAft = np.sum( Mass[isnpa,:NPart_aft] * SpecificTotalEnergy[isnpa,:NPart_aft] ) / np.sum(Mass[isnpa,:NPart_aft])

            localdata[0,i_NP,nSN] = np.log10(TrueDensity) # Local Density
            localdata[1,i_NP,nSN] = np.sum(CellSFR[isnpb,:NPart_bef]) / np.sum(Mass[isnpb,:NPart_bef]) # Specific (per gas) Cell SFR Before
            localdata[2,i_NP,nSN] = np.sum(CellSFR[isnpa,:NPart_aft]) / np.sum(Mass[isnpb,:NPart_bef]) # Speciifc (per gas) Cell SFR After
            localdata[3,i_NP,nSN] = TrueEnergyAft - TrueEnergyBef # True Energy Difference
            localdata[4,i_NP,nSN] = np.median(SpecificTotalEnergy[isnpa,:NPart_aft]) - np.median(SpecificTotalEnergy[isnpb,:NPart_bef]) # Median Energy Diff
            localdata[5,i_NP,nSN] = 10.**np.average( np.log10(SpecificTotalEnergy[isnpa,:NPart_aft]) ) - 10.**np.average( np.log10(SpecificTotalEnergy[isnpb,:NPart_bef]) ) # Log Energy Diff
            localdata[6,i_NP,nSN] = (TrueKEAft-TrueKEBef) / (TrueTEAft-TrueTEBef)
            localdata[7,i_NP,nSN] = np.log10(LocalTemperature)

    np.save('auxdata/LocalData_run{}.npy'.format(jSim),localdata)
    if jSim==0:
        localdata_LYRA = localdata
    elif jSim==1:
        localdata_RIGEL = localdata
    elif jSim==2:
        localdata_SMUGGLE = localdata
    elif jSim==3:
        localdata_LYRArt = localdata
    elif jSim==4:
        localdata_RIGELrt = localdata
    elif jSim==5:
        localdata_SMUGGLErt = localdata

'''
def ptcl_density_bin_smoothed( r1, r2, mass, MIN1=MINDENS, MAX1=MAXDENS, MIN2=42, MAX2=50, NBINS1=200, NBINS2=160, SIG1=2, SIG2=2, truncation=1.e-3 ): # Sigmas should be in pixels

    ptcl_density = np.zeros((NBINS2,NBINS1))

    xrange = np.linspace(MIN1,MAX1,NBINS1,endpoint=False)
    yrange = np.linspace(MIN2,MAX2,NBINS2,endpoint=False)

    xMesh, yMesh = np.meshgrid(xrange,yrange)

    for i_ptcl,(x,y,m) in enumerate(zip(r1,r2,mass)):

        sig1_scale = (MAX1-MIN1) / NBINS1 # actual x increment per pixel
        sig2_scale = (MAX2-MIN2) / NBINS2 # actual y increment per pixel

        z1 = ( xMesh - x ) / ( SIG1 * sig1_scale ) # sigma converted by pixels * actual x per pixel
        z2 = ( yMesh - y ) / ( SIG2 * sig2_scale ) # sigma converted by pixels * actual x per pixel
        Gaussian = np.exp( -0.5 * ( z1*z1 + z2*z2 ) )

        Gaussian_Cut = np.where( Gaussian > truncation, Gaussian, np.zeros_like(Gaussian) )
        Gaussian_Cut_Volume = np.sum(Gaussian_Cut)
        Gaussian_Cut_Normalized = Gaussian_Cut / Gaussian_Cut_Volume

        if (x>MIN1) and (x<MAX1) and (y>MIN2) and (y<MAX2):

            ptcl_density += Gaussian_Cut_Normalized

    return ptcl_density
'''