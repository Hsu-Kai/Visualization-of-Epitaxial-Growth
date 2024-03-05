import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider



def get_energy_surface(yx_atoms, n_steps):
    uvecs = np.array([[0, 1], [3**0.5/2, 0.5]])
    n_atoms = yx_atoms.shape[1]
    ij = np.mgrid[0:1:(n_steps+1)*1j, 0:1:(n_steps+1)*1j] # 2 x n x n
    yx_offsets = (ij[:, None, ...] * uvecs[..., None, None]).sum(axis=0).copy()
    energies = []
    for yx in yx_atoms.T:
        E = Esub_yx(yx[:, None, None] + yx_offsets)
        energies.append(E)
    average_energy_surface = sum(energies) / n_atoms
    return average_energy_surface

def Esub_yx(yx_calc):
    r3o2 = 3**0.5 / 2.
    kay = 2 * np.pi * np.array([[-0.5, r3o2], [-0.5, -r3o2], [1, 0]]) / r3o2
    if yx_calc.ndim == 3:
        three = np.cos((kay[..., None, None] * yx_calc).sum(axis=1))
    elif yx_calc.ndim == 2:
        three = np.cos((kay[..., None] * yx_calc).sum(axis=1))
    else:
        print('uhoh!')
        three = None
    E = 1. - (1.5 + three.sum(axis=0)) / 4.5  # HEY!!!! negative sign for graphene
    return E

def dots(vecs, nmax=5):
    ij = np.mgrid[-nmax:nmax+1, -nmax:nmax+1]
    keep = np.abs(ij.sum(axis=0)) <= nmax
    ij = ij[:, keep]
    yx = (ij[:, None] * vecs[:, ::-1, None]).sum(axis=0)
    # yx = (ij[:, None] * vecs[..., None]).sum(axis=0)
    return yx

def hexvecs(a=1, R=0):
    uvecs = np.array([[1, 0], [0.5, 3**0.5/2]])
    s, c = [f(np.radians(R)) for f in (np.sin, np.cos)]
    rotm = np.array([[c, s], [-s, c]])
    urvecs = (rotm * uvecs[..., None]).sum(axis=1)
    vecs = a * urvecs
    return vecs


### Beware of the Negative sign (annoying!)
def Egrad_yx(yx_calc):
    r3o2 = 3**0.5 / 2.
    kay = 2 * np.pi * np.array([[-0.5, r3o2], [-0.5, -r3o2], [1, 0]]) / r3o2
    if yx_calc.ndim == 3:
        three = np.sin((kay[..., None, None] * yx_calc).sum(axis=1))
        E_partial =   (((kay.T[..., None, None]) * (three)).sum(axis=1)) / 4.5  
        return E_partial        
    elif yx_calc.ndim == 2:
        three = np.sin((kay[..., None] * yx_calc).sum(axis=1))
        E_partial =   (((kay.T[..., None]) * (three)).sum(axis=1)) / 4.5  
        return E_partial
    else:
        three = None
        print('something wrong!')


def atom_separation(index_1, index_2):
    d = np.sqrt((new_loc[0][index_1] - new_loc[0][index_2])**2+(new_loc[1][index_1] - new_loc[1][index_2])**2)
    return d


def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k/ncol, k%ncol
    

class Pair(): #there must be a better/elegant/systematic way to do this...
    def __init__(self, nmax=5):
        self.nmax = int(nmax)
        if self.nmax == 1:
            self.index = ((0,1),(0,2),(1,2))
        elif self.nmax == 2:
            self.index = ((0,1),(1,2),(3,4),(0,3),(3,5),(1,4),(1,3),(2,4),(4,5))
        elif self.nmax == 3:
            self.index = ((0,1),(1,2),(2,3),(4,5),(5,6),(7,8),(0,4),(4,7),(7,9),
                          (1,5),(5,8),(2,6),(1,4),(2,5),(5,7),(3,6),(6,8),(8,9))
        elif self.nmax == 4:
            self.index = ((0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(9,10),(10,11),(12,13),
                          (0,5),(5,9),(9,12),(12,14),(1,6),(6,10),(10,13),(2,7),(7,11),(3,8),
                          (1,5),(2,6),(6,9),(3,7),(7,10),(10,12),(4,8),(8,11),(11,13),(13,14))
        elif self.nmax == 5:
            self.index = ((0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(11,12),
                          (12,13),(13,14),(15,16),(16,17),(18,19),(0,6),(6,11),(11,15),(15,18),
                          (18,20),(1,7),(7,12),(12,16),(16,19),(2,8),(8,13),(13,17),(3,9),(9,14),
                          (4,10),(1,6),(2,7),(7,11),(3,8),(8,12),(12,15),(4,9),(9,13),(13,16),(16,18),
                          (5,10),(10,14),(14,17),(17,19),(19,20))
        elif self.nmax == 6:
            self.index = ((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(7,8),(8,9),(9,10),(10,11),(11,12),
                          (13,14),(14,15),(15,16),(16,17),(18,19),(19,20),(20,21),(22,23),(23,24),
                          (25,26),(0,7),(7,13),(13,18),(18,22),(22,25),(25,27),(1,8),(8,14),(14,19),
                          (23,26),(2,9),(9,15),(15,20),(20,24),(3,10),(10,16),(16,21),(4,11),(11,17),
                          (5,12),(1,7),(2,8),(8,13),(3,9),(9,14),(14,18),(4,10),(10,15),(15,19),(19,22),
                          (5,11),(11,16),(16,20),(20,23),(23,25),(6,12),(12,17),(17,21),(21,24),(24,26))
        elif self.nmax == 7:
            self.index = ((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(8,9),(9,10),(10,11),(11,12),
                          (12,13),(13,14),(15,16),(16,17),(17,18),(18,19),(19,20),(21,22),(22,23),
                          (23,24),(24,25),(26,27),(27,28),(28,29),(30,31),(31,32),(33,34),(0,8),(8,15),
                          (15,21),(21,26),(26,30),(30,33),(33,35),(1,9),(9,16),(16,22),(22,27),(27,31),(31,34),
                          (2,10),(10,17),(17,23),(23,28),(28,32),(3,11),(11,18),(18,24),(24,29),(4,12),(12,19),
                          (19,25),(5,13),(13,20),(6,14),(1,8),(2,9),(9,15),(3,10),(10,16),(16,21),(4,11),(11,17),
                          (17,22),(22,26),(5,12),(12,18),(18,23),(23,27),(27,30),(6,13),(13,19),(19,24),(24,28),
                          (28,31),(31,33),(7,14),(14,20),(20,25),(25,29),(29,32),(32,34))      
        else:
            print('Undefined!')
            
class Island():
    def __init__(self, a=1, R=0, kind='hex', nmax=3, n_longer=0):
        self.a = float(a)
        self.R = float(R)
        self.kind = str(kind).lower()[:3]
        self.nmax = int(nmax)
        self.n_longer = int(n_longer)
        self.vecs = hexvecs(a=self.a, R=self.R)
        if self.kind in ('hex', 'tri'):
            ij = np.mgrid[-self.nmax:self.nmax+1, -self.nmax:self.nmax+1]
            keep = np.abs(ij.sum(axis=0)) <= self.nmax
            if self.kind == 'tri':
                keep *= (ij[0] >= 0) * (ij[1] >= 0)
                keep += (ij == 0).all(axis=0)
        elif self.kind == 'bar':
            ij = np.mgrid[-self.nmax:self.nmax+self.n_longer+1,
                          -self.nmax:self.nmax+1]
            i, j = ij
            keep_j = np.abs(j) <= self.nmax
            keep_i_left = (i >= -self.nmax) * (i + j >= -self.nmax)
            keep_i_right = ((i <= self.nmax + self.n_longer) *
                            (i + j <= self.nmax + self.n_longer))
            keep = keep_j * keep_i_left * keep_i_right
        else:
            print('uhoh! unknown kind')            
        self.ij = ij[:, keep]
        self.yx = (self.ij[:, None] * self.vecs[:, ::-1, None]).sum(axis=0)


### BEGIN

a_2H_TaS2_nominal = 3.315
a_1T_TaS2_nominal = 3.36
a_graphene = 2.46
a_SiC  = 3.07       # as long as it's not 3C phase

CTE_bilayer_graphene = -10.9E-06
CTE_monolayer_graphene = -21.4E-06
CTE_1T_TaS2 = 60.E-06

dCTE = CTE_1T_TaS2 - CTE_bilayer_graphene  ####

a_TaS2_nominal = a_2H_TaS2_nominal    ####

print('a_TaS2_nominal: ', a_TaS2_nominal)
print('dCTE: ', dCTE)

pct_hi = 6.
pct_lo = -1.
pct_hi = 5.
pct_lo = -5.
r3o2 = np.sqrt(3) / 2.
rnom = a_TaS2_nominal/a_graphene
rmin = (1 + 0.01 * pct_lo) * rnom
rmax = (1 + 0.01 * pct_hi) * rnom



rots = np.linspace(0, 30, 31)
aas = np.linspace(rmin, rmax, 12)
nmaxes = np.arange(1, 6)
n_steps = 50

all_results = []
for nmax in nmaxes:
    print('nmax: ', nmax)
    results = []
    all_results.append(results)
    for a in aas:
        print('a: ', a)
        group = []
        results.append(group)
        for R in rots:
            # hexi = Island(a=a, R=R, kind='hex', nmax=nmax)
            tri = Island(a=a, R=R, kind='tri', nmax=nmax)
            # bar = Island(a=a, R=R, kind='bar', nmax=nmax, n_longer=4*nmax)
            # E = get_energy_surface(hexi.yx, n_steps=n_steps)
            E = get_energy_surface(tri.yx, n_steps=n_steps)
            # E = get_energy_surface(bar.yx, n_steps=n_steps)
            group.append([E.min(), E.max()])

all_results = np.array(all_results)
print('all_results.shape: ', all_results.shape)

mini, maxi = all_results.min(), all_results.max()


# Energy vs. angle
if False:
    d = 0.01
    fig, axes = plt.subplots(len(nmaxes), 1, figsize=[7, 7.5])
    for nmax, blob, ax in zip(nmaxes, all_results, axes.flatten()):
        for thing in blob:
            for trace in thing.T[:1]:
                ax.plot(rots, trace) # , c=color
        ax.set_ylim(mini-d, maxi+d)
    plt.show()

        
#3d Energy Map
X, Y = np.meshgrid(rots, aas)   
for thing in all_results:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Z = thing[..., 0]
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()


#3d Energy Map: view from top with some degrees off
if False:
    X, Y = np.meshgrid(rots, aas)   
    for thing in all_results:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        Z = thing[..., 0]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        ax.view_init(elev=70., azim=-90.)
        plt.show()

    
if False: 
    fig, ax = plt.subplots(1, 1)#  , figsize=[9, 7]
    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    E_sub = Esub_yx(yx)
    ax.imshow(E_sub, origin='lower', extent=extent)
    plt.show()


    #potential
    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    E_sub = Esub_yx(yx)
    plt.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')
    plt.show()
    
    #potential_gradient (numerical)
    E_dy, E_dx = (-1)*np.array(np.gradient(E_sub))
    E_delta = np.sqrt((E_dy)**2+(E_dx)**2)
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    plt.imshow(E_delta, origin='lower', extent=extent, cmap = 'Greens_r')
    plt.title('magnitude & numerical')
    plt.show()


    #potential with gradient magnitude as arrows (numerical)
    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    E_sub = Esub_yx(yx)
    plt.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')
    y, x = yx
    plt.quiver(x[::50,::50], y[::50,::50], E_dx[::50,::50], E_dy[::50,::50],
                edgecolor='k', facecolor='None', linewidth=1, scale=1)
    plt.show()

    #potential_gradient_x (re-derivation)
    plt.imshow(Egrad_yx(yx)[0,:,:], origin='lower', extent=extent, cmap = 'Greens_r')
    plt.title('gradient x & re-derivation')
    plt.show()
    
    #potential_gradient_y (re-derivation)
    plt.imshow(Egrad_yx(yx)[1,:,:], origin='lower', extent=extent, cmap = 'Greens_r')
    plt.title('gradient y & re-derivation')
    plt.show()

    #potential_gradient (re-derivation)
    E_grad = np.sqrt((Egrad_yx(yx)[0,:,:])**2+(Egrad_yx(yx)[1,:,:])**2)
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    plt.imshow(E_delta, origin='lower', extent=extent, cmap = 'Greens_r')
    plt.title('magnitude & re-derivation')
    plt.show()



# Relax adatoms: 1st stage relaxation
if True:
    fig, ax = plt.subplots(1, 1)#  , figsize=[9, 7]
    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    E_sub = Esub_yx(yx)
    extent = [yx[1].min(), yx[1].max(), yx[0].min(), yx[0].max()]
    plt.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')

    E_dy, E_dx = (-1)*np.array(np.gradient(E_sub))
    # a=
    # R=
    # nmax=
    tri = Island(a=a, R=R, kind='tri', nmax=nmax)
    y, x = tri.yx
    ax.plot(x, y, 'ob', ms=11)
    print('a:' ,a, ' R:' ,R, ' nmax:' ,nmax)

    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    y, x = yx
    plt.quiver(x[::50,::50], y[::50,::50], E_dx[::50,::50], E_dy[::50,::50],
                   edgecolor='k', facecolor='None', linewidth=1, scale=1)
    #E_dy, E_dx = (-1)*np.array(np.gradient(E_sub))

    ax.set_xlim(-2, 15)
    ax.set_ylim(-2, 15)
    plt.show()

if True:
    #new_locs = []
    delta = 0.2
    new_loc = tri.yx + (- delta) * Egrad_yx(tri.yx)
    #new_locs.append(new_loc)
    #new_locs = np.array(new_locs)

    fig, ax = plt.subplots(1, 1)#  , figsize=[9, 7]
    yx = np.mgrid[-2:22:1000j, -2:22:1000j]
    E_sub = Esub_yx(yx)
    plt.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')
    #y, x = yx
    Egrad_normx = 1.*Egrad_yx(tri.yx)[1] / np.sqrt((Egrad_yx(tri.yx)[1])**2 + (Egrad_yx(tri.yx)[0])**2)
    Egrad_normy = 1.*Egrad_yx(tri.yx)[0] / np.sqrt((Egrad_yx(tri.yx)[1])**2 + (Egrad_yx(tri.yx)[0])**2)

    plt.quiver(tri.yx[1], tri.yx[0], (-1.)*Egrad_normx, (-1.)*Egrad_normy,
                   edgecolor='k', facecolor='None',
                   linewidth=2, scale_units='width')
    tri = Island(a=a, R=R, kind='tri', nmax=nmax)
    y, x = tri.yx
    ax.plot(x, y, 'ob', ms=11)
    y, x = new_loc
    ax.plot(x, y, 'o', color='none', markeredgecolor='red', 
	markersize=11)
    ax.set_xlim(-2, 15)
    ax.set_ylim(-2, 15)
    plt.title('first stage relaxation')
    plt.show()


#calculate the new lattice constant: 1st stage relaxation
    #consider only neaby atom separation
if True:
    a_ratio = []
    pair = Pair(nmax = 5)
    for index_1, index_2 in pair.index:
        a_ratio.append(atom_separation(index_1, index_2))
        print(atom_separation(index_1, index_2))

    print('neaby atom separation:', a_ratio)
    a_ratio = np.array(a_ratio)
    print('raw mean value of neaby atom separation:', a)
    print('mean value of new neaby atom separation:', a_ratio.mean())
    print('root mean square value of new neaby atom separation:', np.sqrt((a_ratio**2).sum()/len(a_ratio**2)))




#Slider
if True:
    # The parametrized function to be plotted
    def tri_island(ratio, angle, nmax, n_y, n_x, delta):
        uvecs = np.array([[0, 1], [3**0.5/2, 0.5]])
        n_steps = 100
        ij = np.mgrid[0:1:(n_steps+1)*1j, 0:1:(n_steps+1)*1j]
        yx_offsets = (ij[:, None, ...] * uvecs[..., None, None]).sum(axis=0).copy()
        yx0 = Island(a=ratio, R=angle, kind='tri', nmax=nmax).yx + yx_offsets[:, int(n_y), int(n_x)][...,None]
        new_loc = yx0 + (- delta) * Egrad_yx(yx0)
        return yx0, new_loc

    
    # Define initial parameters
    init_ratio = 1.74
    init_angle = 0
    init_nmax = 1
    init_n_x = 0
    init_n_y = 0
    init_delta = 0
    
    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()#  , figsize=[16, 10]
    yx0 = np.mgrid[-2:22:1000j, -2:22:1000j]
    extent = [yx0[1].min(), yx0[1].max(), yx0[0].min(), yx0[0].max()]
    E_sub = Esub_yx(yx0)
    ax.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')
    
    ax.plot(tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y, init_delta)[0][1],
                 tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y, init_delta)[0][0],
                 'ob', ms=11)
    ax.plot(tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y, init_delta)[1][1],
                 tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y, init_delta)[1][0],
                 'o', color='none', markeredgecolor='red', markersize=11)         
    ax.set_xlim(-2, 15)
    ax.set_ylim(-2, 15)            
    ax.set_title(' E=' + str(round(Esub_yx(tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y,
                                            init_delta)[0]).sum()/len(tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y,
                                            init_delta)[0][1]), 4)) + ', E(translation)=' + str(round(get_energy_surface(tri_island(init_ratio, init_angle, init_nmax, init_n_x, init_n_y,
                                            init_delta)[0], n_steps=n_steps).min(), 4)) + ', E_relax=' + str(round(Esub_yx(tri_island(init_ratio, init_angle,
                                            init_nmax, init_n_x, init_n_y, init_delta)[1]).sum()/len(tri_island(init_ratio, init_angle, init_nmax, init_n_x,
                                            init_n_y, init_delta)[0][1]), 4)))
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    
    # Make a horizontal slider to control the ratio.
    axratio = plt.axes([0.25, 0.15, 0.65, 0.03])
    ratio_slider = Slider(
        ax=axratio,
        label='ratio',
        valmin=0.7,
        valmax=2,
        valinit=init_ratio,
    )

    # Make a horizontal slider to control the angle.
    axangle = plt.axes([0.25, 0.125, 0.65, 0.03])
    angle_slider = Slider(
        ax=axangle,
        label='angle',
        valmin=0,
        valmax=30,
        valinit=init_angle,
    )

    # Make a horizontal slider to control the nmax.
    axnmax = plt.axes([0.25, 0.1, 0.65, 0.03])
    nmax_slider = Slider(
        ax=axnmax,
        label='nmax',
        valmin=1,
        valmax=5,
        valinit=init_nmax,        
    )

    # Make a horizontal slider to control the n_x.
    axn_x = plt.axes([0.25, 0.075, 0.65, 0.03])
    n_x_slider = Slider(
        ax=axn_x,
        label='n_x',
        valmin=0,
        valmax=100,
        valinit=init_n_x,
    )
    
    # Make a horizontal slider to control the n_y.
    axn_y = plt.axes([0.25, 0.05, 0.65, 0.03])
    n_y_slider = Slider(
        ax=axn_y,
        label='n_y',
        valmin=0,
        valmax=100,
        valinit=init_n_y,
    )

    # Make a horizontal slider to control the delta.
    axdelta = plt.axes([0.25, 0.025, 0.65, 0.03])
    delta_slider = Slider(
        ax=axdelta,
        label='delta',
        valmin=0,
        valmax=1,
        valinit=init_delta,
    )
    
    
    # The function to be called anytime a slider's value changes
    def update(val):
        ax.clear()
        yx0 = np.mgrid[-2:22:1000j, -2:22:1000j]
        extent = [yx0[1].min(), yx0[1].max(), yx0[0].min(), yx0[0].max()]
        ax.imshow(E_sub, origin='lower', extent=extent, cmap = 'Greens_r')
        ax.plot(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val, delta_slider.val)[0][1],
                 tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val, delta_slider.val)[0][0],
                 'ob', ms=11)
        ax.plot(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val, delta_slider.val)[1][1],
                 tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val, delta_slider.val)[1][0],
                 'o', color='none', markeredgecolor='red', markersize=11) 
        ax.set_xlim(-2, 15)
        ax.set_ylim(-2, 15)
        ax.set_title(' E=' + str(round(Esub_yx(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val,
                    delta_slider.val)[0]).sum()/len(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val,
                    delta_slider.val)[0][1]), 4)) + ', E(translation)=' + str(round(get_energy_surface(tri_island(ratio_slider.val,
                    angle_slider.val, nmax_slider.val, n_x_slider.val, n_y_slider.val, delta_slider.val)[0],
                    n_steps=n_steps).min(), 4)) + ', E_relax=' + str(round(Esub_yx(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val,
                    n_x_slider.val, n_y_slider.val, delta_slider.val)[1]).sum()/len(tri_island(ratio_slider.val, angle_slider.val, nmax_slider.val,
                    n_x_slider.val, n_y_slider.val, delta_slider.val)[0][1]), 4)))
        
        plt.draw()



    # register the update function with each slider
    ratio_slider.on_changed(update)
    angle_slider.on_changed(update)
    nmax_slider.on_changed(update)
    n_x_slider.on_changed(update)
    n_y_slider.on_changed(update)
    delta_slider.on_changed(update)
    plt.show()
