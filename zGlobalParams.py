import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
#from mpl_toolkits import mplot3d
#import corner
import zHead as hd

import zMassScales as MS
MassScales = MS.MassScales

localdata_LYRA_nort    = np.load('auxdata/LocalData_run0.npy')
localdata_RIGEL_nort   = np.load('auxdata/LocalData_run1.npy')
localdata_SMUGGLE_nort = np.load('auxdata/LocalData_run2.npy')
localdata_LYRArt    = np.load('auxdata/LocalData_run3.npy')
localdata_RIGELrt   = np.load('auxdata/LocalData_run4.npy')
localdata_SMUGGLErt = np.load('auxdata/LocalData_run5.npy')

LocalDatas = [ localdata_LYRA_nort, localdata_RIGEL_nort, localdata_SMUGGLE_nort, localdata_LYRArt, localdata_RIGELrt, localdata_SMUGGLErt ]
LocalDataNames = ['LYRA noRT','RIGEL noRT','SMUGGLE noRT','INVALID RUN','RIGEL RT','SMUGGLE RT']

colors = ['C3','gray','purple','brown','C1','green']

ContourCmapNames = ['Reds_r','Grays','Purples_r','viridis_r','Oranges_r','Greens_r']
contourCmaps = [ hd.create_cmap(colorname,lo=-100.0,hi=200.0,Nbins=10)[0] for colorname in ContourCmapNames ]
contourNorms = [ hd.create_cmap(colorname,lo=-100.0,hi=200.0,Nbins=10)[1] for colorname in ContourCmapNames ]

RigelStarMass = np.loadtxt('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/RIGELMassiveStarMass_noRT.txt')
SNeOccurred = RigelStarMass[-1] == -2
RigelSiteDens = np.loadtxt('../SupernovaExtract/RigelSNeExtract-SeeEmails/dataEmail/RIGELSiteDens_noRT.txt')[-1,SNeOccurred]
