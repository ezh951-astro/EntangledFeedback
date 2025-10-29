import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from astropy.cosmology import FlatLambdaCDM

zdirsname = 'zdirs'

### Global Constants

ageGyr = 13.77 # approx age of universe in Gyr

G = 4.302e-6 # kpc Msun^-1 (km/s)^2
BOLTZMANN_ergK = 1.3806e-16 # erg/K
PROTONMASS_g = 1.6726e-24 # grams

cm_in_kpc = 3.08568e21
g_in_Msun = 1.989e33
cms_in_kms = 1.e5

meters_in_parsec = 3.08568e16
yr_in_galunit = meters_in_parsec / 525600. / 60. # approximately 1 billion

Msun = r'${\rm M}_{\odot}$'

### Handle Input File Directories

zdirslist = open(zdirsname,'r').read().splitlines()

header_indices = []
lineskip_indices = []
for i,linestr in enumerate(zdirslist):
	if linestr == '':
		lineskip_indices.append(i)
	elif linestr[0] == '%':
		header_indices.append(i)

Nruns = lineskip_indices[0] - header_indices[0] - 1

runlabels            = zdirslist[ header_indices[0]+1 : header_indices[0]+Nruns+1 ]
simtypes             = zdirslist[ header_indices[1]+1 : header_indices[1]+Nruns+1 ]
snap_dirs            = zdirslist[ header_indices[2]+1 : header_indices[2]+Nruns+1 ]
last_snap_num        = zdirslist[ header_indices[3]+1 : header_indices[3]+Nruns+1 ]
parts_per_snaps_list = zdirslist[ header_indices[4]+1 : header_indices[4]+Nruns+1 ]

last_snap_num  = [ int(numstr) for numstr in last_snap_num ]
parts_per_snaps_list  = [ int(numstr) for numstr in parts_per_snaps_list ]

def handle_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--RunNums', help='which run (numbers) do we want to include')
    parser.add_argument('-v', '--version', help='string noting the version, use todays date')
    args = parser.parse_args()

    def akane_kurashiki( junpei ): # converts letters to numbers
        try:
            return int(junpei)
        except ValueError:
            return ord(junpei)-87

    if args.RunNums != None:
        ActiveRunIDs = [ akane_kurashiki(args.RunNums[i]) for i in range(len(args.RunNums)) ]
    else:
        ActiveRunIDs = range(Nruns)

    if args.version != None:
        date = args.version
    else:
        date = 'xxxx'

    return ActiveRunIDs, date

def get_istr3(i):
    # Assumes that i>=0
    if i<10:
        return '00'+str(i)
    if i>=10 and i<100:
        return '0'+str(i)
    if i>=100:
        return str(i)

#####
### Extraction of Snapshot Data
#####

class Snapshot:

    def __init__( self, i,j ):

        # i, j, subhalo_id
        self.i = i
        self.j = j

        # RunName, SnapDir, FofDir, PartsPerSnap, MODE
        self.RunName      = runlabels[j]
        self.SnapDir      = snap_dirs[j]
        self.FofDir       = snap_dirs[j]
        self.PartsPerSnap = parts_per_snaps_list[j]
        self.MODE         = simtypes[j]

        # SnapType, f, Header
        istr3 = get_istr3(i)
        if self.PartsPerSnap == 1:
            self.SnapType = 'SINGLE'
            self.f = h5py.File(self.SnapDir + 'snapshot_' + istr3 + '.hdf5')
            self.Header = self.f[u'Header'].attrs
        else:
            self.SnapType = 'MULTIPLE'
            self.f = [ h5py.File(self.SnapDir + 'snapdir_{}/snapshot_{}.{}.hdf5'.format(istr3,istr3,b)) for b in range(self.PartsPerSnap) ]
            self.Header = self.f[0][u'Header'].attrs

        # DistFac, MassFac, VelFac (in gal units)
        try:
            self.DistFac = self.Header[u'UnitLength_in_cm'] / cm_in_kpc
            self.MassFac = self.Header[u'UnitMass_in_g'] / g_in_Msun
            self.VelFac  = self.Header[u'UnitVelocity_in_cm_per_s'] / cms_in_kms
        except KeyError:
            if self.RunName=='RIGEL':
                self.DistFac = 0.001
                self.MassFac = 1.0
                self.VelFac  = 1.0
            else: # Default
                self.DistFac = 1.0
                self.MassFac = 1.e10
                self.VelFac  = 1.0

        # TimeFac, s, redshift, scalefac, h_small, SimTime (Myr), CosmoScaling (bool), HubbleConst
        if self.RunName=='RIGEL':
            self.TimeFac = 1.
        else:
            self.TimeFac = 1000.

        if self.MODE=='ISO':

            self.s = None

            self.redshift = 0.
            self.scalefac = 1.
            self.h_small = 1.

            self.cosmo = None
            self.SimTime = self.TimeFac * extract_header(self.f,u'Time',PartsPerSnap=self.PartsPerSnap) #Myr

            self.CosmoScaling = False

            self.HubbleConst = 70.

        if self.MODE=='COSMO':

            self.s = h5py.File(self.FofDir + 'fof_subhalo_tab_' + get_istr3(i) + '.hdf5')

            self.redshift = extract_header(self.f,u'Redshift',PartsPerSnap=self.PartsPerSnap)
            self.scalefac = 1. / (1.+self.redshift)
            self.h_small = extract_header(self.f,u'HubbleParam',PartsPerSnap=self.PartsPerSnap)

            Om0 = extract_header(self.f,u'Omega0',PartsPerSnap=self.PartsPerSnap)

            self.cosmo = FlatLambdaCDM( H0=100.*self.h_small, Om0=Om0 )
            self.SimTime = self.TimeFac * self.cosmo.age(self.redshift).value #Myr

            self.CosmoScaling = True

            self.HubbleConst = 100.*self.h_small * np.sqrt( Om0/self.scalefac/self.scalefac/self.scalefac + (1.0-Om0) )        

    def find_star_age( self, sim_birth_time ):
        #print( self )
        #print( self.SimTime )
        #print( self.TimeFac )
        #print( sim_birth_time )
        if self.MODE == 'ISO':
            return self.SimTime - ( self.TimeFac * sim_birth_time )
        elif self.MODE == 'COSMO':
            return self.SimTime - ( self.TimeFac * ( self.cosmo.age( (1.0/sim_birth_time)-1.0 ).value ) )

def extract_header( f, field, PartsPerSnap=1 ):
    if PartsPerSnap == 1:
        return f[u'Header'].attrs[field]
    else:
        return f[0][u'Header'].attrs[field]

def get_fieldobj_units( field_obj, units ):

    if units=='cgs':
        mass_fac = 1.0
        dist_fac = 1.0
        vel_fac  = 1.0
    elif units=='gal':
        mass_fac = g_in_Msun # g in a Msun
        dist_fac = cm_in_kpc # cm in a kpc
        vel_fac = cms_in_kms # cm/s in a km/s
    elif units=='hel':
        mass_fac = 1.989e33 # Msun
        dist_fac = 1.49597870691e13 # cm in a AU
        vel_fac = 474372. # cm/s in a AU/yr
    else:
        raise ValueError('Not a valid system of units')

    try:

        mass_scaling = field_obj.attrs[u'mass_scaling']
        dist_scaling = field_obj.attrs[u'length_scaling']
        vel_scaling  = field_obj.attrs[u'velocity_scaling']

        to_cgs = field_obj.attrs[u'to_cgs']
        conversion_factor = to_cgs / (mass_fac**mass_scaling) / (dist_fac**dist_scaling) / (vel_fac**vel_scaling)

        return conversion_factor

    except KeyError:
        print('UNITS INFORMATION NOT FOUND! Setting factor to 1.0')
        return 1.0

def get_cosmo_scaling( f, field, ptype, parts_per_snap ):

    pstr = u'PartType{}'.format(ptype)

    if parts_per_snap==1:
        field_obj = f[pstr][field]
    else:
        field_obj = f[0][pstr][field]

    try:

        a_scaling = field_obj.attrs[u'a_scaling']
        h_scaling = field_obj.attrs[u'h_scaling']

        redshift = extract_header( f,u'Redshift',PartsPerSnap = parts_per_snap )
        scalefac = 1.0 / ( 1.0 + redshift )
        h_small  = extract_header( f,u'HubbleParam',PartsPerSnap = parts_per_snap )

        comoving_scaling_factor = ( scalefac ** a_scaling ) * ( h_small ** h_scaling )

        return comoving_scaling_factor

    except KeyError:
        print('COSMO SCALING INFORMATION NOT FOUND! Setting factor to 1.0')
        return 1.0

def extract( f, field, ptype, parts_per_snap=1, units=None, cosmo_scaling=False, verbose=False ):

    if parts_per_snap == 1:
        snaptype='SINGLE'
    else:
        snaptype='MULTIPLE'

    if verbose:
        print('Now extracting {} for ptype {}'.format(field,ptype))
    pstr = u'PartType{}'.format(ptype)

    try:
    #if True:

        if snaptype=='SINGLE':
            field_obj = f[pstr][field]
            ans = np.array(field_obj) # if SINGLE we don't care about parts_per_snap, so default 8 is ok
        elif snaptype=='MULTIPLE':
            field_obj = f[0][pstr][field]
            dats = [ np.array( f[jp][pstr][field] ) for jp in range(parts_per_snap) ]
            ans = np.concatenate( dats, axis=0 )

        if verbose:
            print('Loaded array of size {}'.format(np.shape(ans)))

        if units == None:
            pass
        else:
            conversion_factor = get_fieldobj_units( field_obj, units )
            if verbose:
                print('conversion_factor {}'.format(conversion_factor))
            ans *= conversion_factor

        if cosmo_scaling:
            comoving_scaling_factor = get_cosmo_scaling( f, field, ptype, parts_per_snap )
            ans *= comoving_scaling_factor

        return ans

    except KeyError:
    #if False:

        if ( ptype in [1,2,3,5] ) and ( field=='Masses' ): # masses that use the mass table instead of arrays

            if verbose:
                print('Mass of PartType 1, 2, 3, or 5 requested but not found; referring to MassTable')

            if snaptype=='SINGLE':
                ptcl_list = np.ones_like(np.array(f[pstr][u'ParticleIDs'])) # if SINGLE we don't care about parts_per_snap, so default 8 is ok
            elif snaptype=='MULTIPLE':
                dats = [ np.ones_like(np.array(f[jp][pstr][u'ParticleIDs'])) for jp in range(parts_per_snap) ]
                ptcl_list = np.concatenate( dats, axis=0 )
            mass = extract_header( f,u'MassTable',PartsPerSnap = parts_per_snap )[ptype]
            ans = mass * ptcl_list

            if units == None:
                pass
            else:

                if units=='cgs':
                    mass_fac = 1.0
                elif units=='gal' or units=='hel':
                    mass_fac = 1.989e33 # g in a Msun

                if snaptype=='SINGLE':
                    conversion_factor = f[u'Parameters'].attrs[u'UnitMass_in_g'] / mass_fac
                if snaptype=='MULTIPLE':
                    conversion_factor = f[0][u'Parameters'].attrs[u'UnitMass_in_g'] / mass_fac

                ans *= conversion_factor

            if cosmo_scaling:
                h_small = extract_header( f,u'HubbleParam',PartsPerSnap = parts_per_snap )
                ans /= h_small

            return ans

        elif ( field=='GFM_StellarFormationTime' ):

            if verbose:
                print('GFM_StellarFormationTime not found, trying StellarFormationTime')

            try: # need this bc this doesnt rule out there simply being no star particles yet
                if snaptype=='SINGLE':
                    field_obj = f[pstr][u'StellarFormationTime']
                    ans = np.array(field_obj) # if SINGLE we don't care about parts_per_snap, so default 8 is ok
                elif snaptype=='MULTIPLE':
                    field_obj = f[0][pstr][u'StellarFormationTime']
                    dats = [ np.array( f[jp][pstr][u'StellarFormationTime'] ) for jp in range(parts_per_snap) ]
                    ans = np.concatenate( dats, axis=0 )

                return ans

            except KeyError: # if key error persists, that means the problem is with PartType4; just return zeros
                return 1.e-13 * np.ones(2)

        else:

            if verbose:
                print('Field empty, using zeros (1.e-13)')

            if field in [u'Coordinates',u'Velocities',u'BirthPos']:
                return 1.e-13*np.ones((2,3))
            else:
                return 1.e-13*np.ones(2)



#####
### Basic Kinematics / Mechanics Functions
#####

def order_masses( rList, mList ):
    ind = np.argsort(rList)
    rOrd = rList[ind]
    mOrd = mList[ind]
    cummass = np.cumsum(mOrd)
    return rOrd, cummass

def half_mass_radius( rList, mList ):

    ordered = order_masses(rList,mList)

    distOrd = ordered[0]
    cummass = ordered[1]
    #print(distOrd,cummass)

    TotMass = cummass[-1]
    HalfMass = TotMass / 2.
    indHalf = np.argmax( cummass>HalfMass )
    rHalf = distOrd[indHalf]

    return rHalf

def calc_distance( xyzarr, center=np.array([0.0,0.0,0.0]) ):

    xyzcen = xyzarr - center
    rad2 = sum([ xyzcen[:,b]*xyzcen[:,b] for b in range(3) ])
    return np.sqrt( rad2 )

def calc_com( xyz, mass ):

    totmass = np.sum(mass)

    com_x = np.sum(mass * xyz[:,0]) / totmass
    com_y = np.sum(mass * xyz[:,1]) / totmass
    com_z = np.sum(mass * xyz[:,2]) / totmass
    com = np.array([ com_x, com_y, com_z ])
    return com

def calc_total_angular_momentum( xyz, vel, mass ):

    Lx = mass * (xyz[:,1]*vel[:,2] - xyz[:,2]*vel[:,1])
    Ly = mass * (xyz[:,2]*vel[:,0] - xyz[:,0]*vel[:,2])
    Lz = mass * (xyz[:,0]*vel[:,1] - xyz[:,1]*vel[:,0])

    return np.array([ np.sum(Lx), np.sum(Ly), np.sum(Lz) ])

def find_alignment_matrix(L,verbose=True):

    Ltot = np.sqrt( np.sum(L*L) )

    phi   = np.arctan2( L[1], L[0] )
    theta = np.arccos( L[2] / Ltot )

    Rphi = np.array([ [ np.cos(phi), np.sin(phi), 0 ], [ -np.sin(phi), np.cos(phi), 0 ], [ 0, 0, 1 ] ])
    Rtheta = np.array([ [ np.cos(theta), 0, -np.sin(theta) ], [ 0, 1, 0 ], [ np.sin(theta), 0, np.cos(theta) ] ])
    RRot = np.matmul(Rtheta,Rphi)

    if verbose:
        print( 'Alignment: {} should be 0 0 1'.format( np.matmul(RRot,L) ) )

    return RRot

def apply_rotation_matrix( RRot, vecarr ): # can be applied to xyz or vel

    xNew = sum([ RRot[0,b] * vecarr[:,b] for b in range(3) ])
    yNew = sum([ RRot[1,b] * vecarr[:,b] for b in range(3) ])
    zNew = sum([ RRot[2,b] * vecarr[:,b] for b in range(3) ])

    return np.transpose(np.array([ xNew, yNew, zNew ]))

def align_disk( vecarr, L ): # rotates positions and velocities so that the total inner angular momentum points upwards
    return apply_rotation_matrix( find_alignment_matrix(L), vecarr )



#####
### Coordinate Adjustments
#####

def stellar_com_from_snapshot( f, parts_per_snap=1, units='gal', verbose=False ): # ISO

    xyz2 = extract( f,u'Coordinates',2, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    xyz3 = extract( f,u'Coordinates',3, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    xyz4 = extract( f,u'Coordinates',4, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    xyz = np.concatenate((xyz2,xyz3,xyz4),axis=0)

    vel2 = extract( f,u'Velocities',2, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    vel3 = extract( f,u'Velocities',3, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    vel4 = extract( f,u'Velocities',4, parts_per_snap=parts_per_snap, units=units, cosmo_scaling=False )
    vel = np.concatenate((vel2,vel3,vel4),axis=0)

    m2 = extract_header( f,u'MassTable', PartsPerSnap=parts_per_snap )[2] * np.ones_like(xyz2[:,0]) # it actually doesn't matter what units the mass are in, since it gets normalized
    m3 = extract_header( f,u'MassTable', PartsPerSnap=parts_per_snap )[3] * np.ones_like(xyz3[:,0])
    m4 = extract( f,u'Masses',4, parts_per_snap=parts_per_snap)
    m = np.concatenate((m2,m3,m4),axis=0)

    x_com = calc_com( xyz, m )
    v_com = calc_com( vel, m )

    if verbose:
        print('Stellar CoM is {}, drift velocity is {}'.format(x_com,v_com))

    return x_com, v_com

def SubhaloCenter( f, s, parts_per_snap=1, N=128, subhalo_id = 0 ): # in GAL units, assuming a single snapshot, cosmological sim

    #print(f,s)

    # CM velocity of the N=128 closest particles to the halo center. Returns it in PHYSICAL units

    redshift = f[u'Header'].attrs[u'Redshift']
    scalefac = 1. / (1.+redshift)
    h_small = f[u'Header'].attrs['HubbleParam']

    dist_unit_fac = f[u'Header'].attrs[u'UnitLength_in_cm'] / cm_in_kpc
    mass_unit_fac = f[u'Header'].attrs[u'UnitMass_in_g'] / g_in_Msun
    vel_unit_fac  = f[u'Header'].attrs[u'UnitVelocity_in_cm_per_s'] / cms_in_kms

    center = np.array(s[u'Subhalo'][u'SubhaloPos'][subhalo_id]) * dist_unit_fac * scalefac / h_small

    #xyz0 = np.array(f[u'PartType0'][u'Coordinates']) * dist_unit_fac * scalefac / h_small - center
    #xyz1 = np.array(f[u'PartType1'][u'Coordinates']) * dist_unit_fac * scalefac / h_small - center
    #xyz4 = np.array(f[u'PartType4'][u'Coordinates']) * dist_unit_fac * scalefac / h_small - center

    xyz0 = extract( f,u'Coordinates',0, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True ) - center
    xyz1 = extract( f,u'Coordinates',1, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True ) - center
    xyz4 = extract( f,u'Coordinates',4, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True ) - center

    dist0 = np.sqrt( xyz0[:,0]*xyz0[:,0] + xyz0[:,1]*xyz0[:,1] + xyz0[:,2]*xyz0[:,2] )
    dist1 = np.sqrt( xyz1[:,0]*xyz1[:,0] + xyz1[:,1]*xyz1[:,1] + xyz1[:,2]*xyz1[:,2] )
    dist4 = np.sqrt( xyz4[:,0]*xyz4[:,0] + xyz4[:,1]*xyz4[:,1] + xyz4[:,2]*xyz4[:,2] )

    #v0 = np.array(f[u'PartType0'][u'Velocities']) * vel_unit_fac * np.sqrt(scalefac) / h_small #units are now physical km/s
    #v1 = np.array(f[u'PartType1'][u'Velocities']) * vel_unit_fac * np.sqrt(scalefac) / h_small
    #v4 = np.array(f[u'PartType4'][u'Velocities']) * vel_unit_fac * np.sqrt(scalefac) / h_small

    v0 = extract( f,u'Velocities',0, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )
    v1 = extract( f,u'Velocities',1, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )
    v4 = extract( f,u'Velocities',4, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )

    mass0 = extract( f,u'Masses',0, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )
    mass1 = extract( f,u'Masses',1, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )
    mass4 = extract( f,u'Masses',4, parts_per_snap=parts_per_snap, units='gal', cosmo_scaling=True )

    #mass0 = np.array(f[u'PartType0'][u'Masses']) * mass_unit_fac / h_small
    #mass1 = np.array(f[u'Header'].attrs[u'MassTable'][1]) * mass_unit_fac / h_small * np.ones(np.shape(xyz1)[0])
    #mass4 = np.array(f[u'PartType4'][u'Masses']) * mass_unit_fac / h_small

    dist = np.concatenate((dist0,dist1,dist4))
    mass = np.concatenate((mass0,mass1,mass4))
    vel  = np.concatenate((v0,v1,v4))

    ind = np.argsort(dist)

    massNbhd = mass[ind][:N]
    velNbhd  = vel[ind][:N]

    Mtot = sum(massNbhd)
    velWeight = [ massNbhd[i]*velNbhd[i] for i in range(N) ]
    CMvel = sum(velWeight) / Mtot

    return center, CMvel

def get_center_snap_galunit( Snap, subhalo_id=0, Ncen=128 ):

    #print('Debug',Snap.MODE)

    if Snap.MODE == 'ISO':
        xcen, vcen = stellar_com_from_snapshot( Snap.f, parts_per_snap = Snap.PartsPerSnap )
    elif Snap.MODE == 'COSMO':
        xcen, vcen = SubhaloCenter( Snap.f, Snap.s, subhalo_id=subhalo_id, N=Ncen )

    return xcen, vcen

def adjustdata( Snap, subhalo_id=0, rad=10.0, ptypes=[0,4], Ncen=128 ):

    cen, vcen = get_center_snap_galunit( Snap, subhalo_id=subhalo_id, Ncen=Ncen )

    angmom_tot = np.array([0.0,0.0,0.0])
    for p in ptypes:

        xyz = extract( Snap.f, u'Coordinates', p, parts_per_snap=Snap.PartsPerSnap, units='gal', cosmo_scaling=Snap.CosmoScaling )
        vel = extract( Snap.f, u'Velocities',  p, parts_per_snap=Snap.PartsPerSnap, units='gal', cosmo_scaling=Snap.CosmoScaling )
        m   = extract( Snap.f, u'Masses',      p, parts_per_snap=Snap.PartsPerSnap, units='gal', cosmo_scaling=Snap.CosmoScaling )

        xyz -= cen
        vel -= vcen

        kk = calc_distance(xyz) < rad

        L = calc_total_angular_momentum( xyz[kk], vel[kk], m[kk] )
        angmom_tot += L

    return cen, vcen, angmom_tot

### ### ###



def moving_average(t,x,n=10):
    ret = np.cumsum(x,dtype='float')
    ret[n:] = ret[n:] - ret[:-n]
    return t[n-1:], ret[n-1:]/n

def cross_correlate(t,f,g):

    N = len(t)

    dt = np.average(np.diff(t)) # assuming constant width of t's.

    avgf = np.average(f)
    avgg = np.average(g)

    f -= avgf
    g -= avgg

    #normf = np.sum(f) * dt # f instead of f^2
    #normg = np.sum(g) * dt
    normf = np.sqrt( np.sum(f*f) * dt )
    normg = np.sqrt( np.sum(g*g) * dt )

    fn = f/normf
    gn = g/normg

    tlags = []
    ccs = []
    for i in range( -(N-1),N-1 ):
        h = gn * np.roll(fn,i)
        if i<0:
            cc = np.sum(h[0:N+i])
            tlag = t[0] - t[-i]
        else:
            cc = np.sum(h[i:N])
            tlag = t[i] - t[0]
        tlags.append(tlag)
        ccs.append(cc)

    return np.array(tlags), np.array(ccs) * dt

def ptcl_density_bin(r1,r2,mass,min1=-50.,max1=-50.,min2=-50.,max2=50.,Nbins1=100,Nbins2=100):

    ptcl_density = np.zeros((Nbins1,Nbins2))

    for i_ptcl, (x,y,m) in enumerate(zip(r1,r2,mass)):
        xbin = int( (x-min1) * Nbins1 / (max1-min1) )
        ybin = int( (y-min2) * Nbins2 / (max2-min2) )
        if xbin>=0 and ybin>=0:
            try:
                ptcl_density[xbin,ybin] += m
            except IndexError:
                pass
        else:
            pass

    ptcl_density = ptcl_density * Nbins1 * Nbins2 / (max1-min1) / (max2-min2) # divide by area of bin

    return np.transpose(ptcl_density)

def mass_distribution_spherical( r_Particles, r_Output, m_Particles ):

    m_all = np.concatenate(( m_Particles, 1.e-3*np.ones_like(r_Output), 1.e-3*np.ones_like(r_Output) ))
    r_all = np.concatenate(( r_Particles, r_Output+1.e-6, r_Output-1.e6 ))

    ordered = order_masses( r_all, m_all )
    distOrd = ordered[0]

    indrads = np.array([ np.argmax(distOrd>rad_j) for rad_j in r_Output ]) # index of 1st ptcl that exceeds measured radius
    distOrd = distOrd[indrads] # distOrd
    cummass = ordered[1][indrads] # cummass # sum masses of all existing particles first, then take the index

    vol = 4.*np.pi/3. * distOrd*distOrd*distOrd
    density = cummass / vol
    velocity = np.sqrt( G * cummass / distOrd )

    return distOrd, cummass, density, velocity

def GasTemp( SpecificInternalEnergy, ElectronAbundance ): # Specific Internal Energy in km/s, i.e. gal units

    UnitVelocity_in_cm_per_s = 1.e5  #  1 km/sec

    gamma= 5.0/3.

    Xh = 0.76 # mass fraction of hydrogen
    MeanWeightFac = 4.0 / ( 3.0*Xh + 1.0 + 4.0*Xh*ElectronAbundance )

    temp = MeanWeightFac * PROTONMASS_g / BOLTZMANN_ergK * (gamma-1) * SpecificInternalEnergy * (UnitVelocity_in_cm_per_s**2.)

    return temp

### ### ###

def create_cmap(cmapname,lo=4.0,hi=8.0,Nbins=20):
    levels = MaxNLocator(nbins=Nbins).tick_values(lo,hi)
    cmap = plt.get_cmap(cmapname)
    norm = BoundaryNorm(levels,ncolors=cmap.N,clip=True)
    return cmap, norm

def axisstyle(ax):
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(which='major', direction='in')
    ax.tick_params(which='minor', direction='in')

def add_colorbar( fig,im, ax=None, label='Colorbar', labelsize=18, ticksize=13, right_position=0.80,right_spacing=0.03, bar_width=0.04,bar_bottom=0.15,bar_height=0.65,shrink=0.9 ):

    if ax==None:
        fig.subplots_adjust(right=right_position)
        cbar_ax = fig.add_axes([ right_position+right_spacing, bar_bottom, bar_width, bar_height ])
        cbar = fig.colorbar(im,cax=cbar_ax, shrink=shrink)
    else:
        cbar = fig.colorbar(im,cax=ax, shrink=shrink)

    cbar.ax.tick_params(labelsize=ticksize)
    cbar.ax.set_ylabel(label, fontsize=labelsize)

### Contours Start

def ptcl_density_bin_smoothed( r1, r2, mass, MIN1=-5, MAX1=5, MIN2=42, MAX2=50, NBINS1=200, NBINS2=160, SIG1=2, SIG2=2, truncation=1.e-3 ): # Sigmas should be in pixels

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

            #print(Gaussian_Cut_Normalized)
            ptcl_density += m * Gaussian_Cut_Normalized

    return ptcl_density, xMesh, yMesh
    
def percentiles2D( ptcl_density, percentiles, num_densities = 100 ): # percentiles are the values at which i actually want to interpolate at

    #ptcl_density /= np.max(ptcl_density) # normalize so that the max density is 1.
    dens_vals = np.linspace(0,np.max(ptcl_density),num_densities)

    #print(dens_vals)

    percentiles_sampled = []
    for d in dens_vals:
        ptcl_density_above_d = np.where( ptcl_density > d, ptcl_density, np.zeros_like(ptcl_density) )
        percentiles_sampled.append( np.sum(ptcl_density_above_d) / np.sum(ptcl_density) )

    return percentiles, np.interp( percentiles, np.flip(percentiles_sampled), np.flip(dens_vals) )

def scatter_to_contour( x,y, weight, levels, xrange=[-5,5], yrange=[42,50], bins=[200,160], sigma=2, interp_res=100, truncation=1.e-3 ):

    ptcl_density, xMesh, yMesh = ptcl_density_bin_smoothed( x, y, weight, MIN1=xrange[0], MAX1=xrange[1], MIN2=yrange[0], MAX2=yrange[1], NBINS1=bins[0], NBINS2=bins[1], SIG1=sigma, SIG2=sigma, truncation=truncation )
    percentiles, interp_dens_vals = percentiles2D( ptcl_density, levels, num_densities=interp_res )

    return xMesh, yMesh, ptcl_density, np.flip(interp_dens_vals), np.flip(percentiles) # the levels of the contour are now the interpolated density values; percentiles are now the labels

def scatter_to_median(Xraw,Yraw, total_bins=20):

    X = Xraw[~np.isnan(Xraw)]
    Y = Yraw[~np.isnan(Yraw)]

    bins = np.linspace( np.min(X), np.max(X), total_bins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X,bins)

    running_median = [np.median(Y[idx==k]) for k in range(total_bins)]
    running_std    = [Y[idx==k].std() for k in range(total_bins)]

    xcurve = bins - (delta/2.)

    return xcurve, running_median, running_std

def scatter_to_percentile(Xraw,Yraw,level, total_bins=10, Xmin=None, Xmax=None):

    X = Xraw[~np.isnan(Xraw)]
    Y = Yraw[~np.isnan(Yraw)]

    if Xmin==None:
        Xmin = np.min(X)
    if Xmax==None:
        Xmax = np.max(X)

    bins = np.linspace( Xmin, Xmax, total_bins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X,bins)

    running_percentile_of_level = []
    for k in range(total_bins):

        #print(Y[idx==k])
        if len(Y[idx==k]) > 0:
            val = np.percentile(Y[idx==k],level)
        else:
            val = np.nan
        running_percentile_of_level.append(val)

    xcurve = bins - (delta/2.)

    return xcurve, running_percentile_of_level

### End Contours


### ### In Progress Do Not Use ### ###

def interpolate_motion_2pt_scalar(t_mid,t0,t1,x0,x1,v0,v1):

    dt = t1 - t0

    a0 = (-6./dt/dt)*x0 + (-4./dt)*v0 + (6./dt/dt)*x1 + (-2./dt)*v1 # checked wolfram alpha inverted matrix lmao
    j0 = (12./dt/dt/dt)*x0 + (6./dt/dt)*v0 + (-12./dt/dt/dt)*x1 + (6./dt/dt)*v1

    t = t_mid - t0

    x_mid = x0 + v0*t + a0*t*t/2. + j0*t*t*t/6.
    v_mid = v0 + a0*t + j0*t*t/2.

    return x_mid, v_mid

def interpolate_motion_2pt_vector( t_mid, t0, t1, x0arr, x1arr, v0arr, v1arr ):

    Npts = np.shape(x0arr)[0]

    phase_data_6D = np.zeros((Npts,6))
    for b in range( np.shape(x0arr)[1] ):

        if Npts > 1:

            x0 = x0arr[:,b]
            x1 = x1arr[:,b]
            v0 = v0arr[:,b]
            v1 = v1arr[:,b]

            x_mid, v_mid = interpolate_motion_2pt_scalar( t_mid, t0, t1, x0, x1, v0, v1 )

            phase_data_6D[:,b]   = x_mid
            phase_data_6D[:,3+b] = v_mid

        else:

            x0 = x0arr[b]
            x1 = x1arr[b]
            v0 = v0arr[b]
            v1 = v1arr[b]

            x_mid, v_mid = interpolate_motion_2pt_scalar( t_mid, t0, t1, x0, x1, v0, v1 )

            phase_data_6D[0,b]   = x_mid
            phase_data_6D[0,3+b] = v_mid

        return phase_data_6D

#def get_center( f, s=None, parts_per_snap=1, subhalo_id=0, Ncen=128 ): # units = gal, N_Nbhd=128, subhalo=0; cosmo assumed 1 part per snap

    #if parts_per_snap == 1:
        #snaptype = 'SINGLE'
    #else:
        #snaptype = 'MULTIPLE'

    #if s==None: # mode is ISO
        #xcen, vcen = stellar_com_from_snapshot( f, parts_per_snap=parts_per_snap )
    #else:
        #xcen, vcen = SubhaloCenter( f, s, subhalo_id=subhalo_id, N=Ncen )

    #return xcen, vcen
