import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad as jgrad
from jax import value_and_grad
from jax import grad, vmap
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Helvetica Neue')

from jax import jit
import jax
import jax
from functools import partial
from scipy import optimize
from scipy.spatial import SphericalVoronoi, geometric_slerp
from mpl_toolkits.mplot3d import proj3d
from jax.config import config
from numba import i8, f8, boolean
import self_propelled_power.tri_functions as trf
import self_propelled_power.periodic_functions as per

import itertools

import numpy as np
from matplotlib import pyplot as plot
from matplotlib.collections import LineCollection
# from numba import jit
from scipy.spatial import ConvexHull

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import time
import numba as nb


@jit
def per(dx,L):
    return jnp.mod(dx+L/2,L)-L/2

@jit
def get_vertices(tx, tR,L):
    tx_transf = jnp.mod(tx - jnp.expand_dims(tx[:,0],axis=1) + L/2,L) ##for practical reasons, shift the triad of centroids such that the first centroid lies in the middle of the domain
    txnorm_transf = jnp.sum(tx_transf ** 2, axis=2) - tR ** 2
    tx_lifted = jnp.concatenate((tx_transf, jnp.expand_dims(txnorm_transf,axis=2)), axis=2)
    triangle_normal = jnp.cross(tx_lifted[:,0], tx_lifted[:,1],axis=1) + jnp.cross(tx_lifted[:,1], tx_lifted[:,2],axis=1) + jnp.cross(tx_lifted[:,2], tx_lifted[:,0],axis=1)
    N = triangle_normal/jnp.expand_dims(jnp.linalg.norm(triangle_normal,axis=1),axis=1)
    vs = jnp.mod(jnp.expand_dims(-.5 / N[:,2],axis=1) * N[:,:2] + tx[:,0]- L/2,L)
    return vs

@jit
def tri_format(mesh_props=None,tri_props=None):
    assert mesh_props is not None
    if tri_props is None:
        tri_props = {}
    tri_props["tx"] = mesh_props["x"][mesh_props["tri"]]
    tri_props["tR"] = mesh_props["R"][mesh_props["tri"]]
    tri_props["vs"] = get_vertices(tri_props["tx"],tri_props["tR"],mesh_props["L"])
    tri_props["vn"]  = tri_props["vs"][mesh_props["neigh"]]
    tri_props["vp1"]  = jnp.roll(tri_props["vn"] ,axis=1,shift=-1)
    tri_props["vm1"]  = jnp.roll(tri_props["vn"] ,axis=1,shift=1)
    return tri_props

@jit
def tri_measure(tri_props=None,mesh_props=None):
    assert tri_props is not None, "must specify tri_props"
    assert mesh_props is not None, "must specify mesh_props"

    tri_props["v_vm1"] = per(jnp.expand_dims(tri_props["vs"],axis=1) - tri_props["vm1"],mesh_props["L"])
    tri_props["v_x"] = per(jnp.expand_dims(tri_props["vs"],axis=1) - tri_props["tx"],mesh_props["L"])
    tri_props["vm1_x"] = per(tri_props["vm1"]-tri_props["tx"],mesh_props["L"])
    tri_props["vp1_x"] = per(tri_props["vp1"]-tri_props["tx"],mesh_props["L"])

    tri_props["lm1"] = jnp.linalg.norm(tri_props["v_vm1"],axis=2)
    tri_props["A_component"] = 0.5 * jnp.cross(tri_props["vm1_x"], tri_props["v_x"])
    return tri_props

@partial(jit, static_argnums=(0,))
def cell_measure(nc,tri_props,cell_props,mesh_props):
    assert len(tri_props) is not 0, "must specify tri_props"
    assert len(mesh_props) is not 0, "must specify mesh_props"
    cell_props["A"] = get_A(tri_props,mesh_props,nc)
    cell_props["P"] = get_P(tri_props,mesh_props,nc)
    return cell_props

@partial(jit, static_argnums=(1,))
def _get_geometry(mesh_props,nc,tri_props=None,cell_props=None):
    if cell_props is None:
        cell_props = {}
    tri_props = tri_format(mesh_props,tri_props)
    tri_props = tri_measure(tri_props,mesh_props)
    cell_props = cell_measure(nc,tri_props,cell_props,mesh_props)
    return cell_props,tri_props

@partial(jit, static_argnums=(2,))
def assemble_scalar(tval,tri,nc):
    val = jnp.zeros((nc))
    for i in range(3):
        val = val.at[tri[...,i]].add(tval[..., i])
    return val

@partial(jit, static_argnums=(2,))
def get_P(tri_props,mesh_props,nc):
    return assemble_scalar(tri_props["lm1"],mesh_props["tri"],nc)

@partial(jit, static_argnums=(2,))
def get_A(tri_props,mesh_props,nc):
    return assemble_scalar(tri_props["A_component"],mesh_props["tri"],nc)


@jit
def get_flip_inequality(mesh_props,tri_props):
    d2_x = np.sum(tri_props["v_x"] ** 2, axis=-1) - tri_props["tR"] ** 2
    x_neigh = tri_props["tx"][mesh_props["neigh"], mesh_props["k2s"]]
    d2_xneigh = jnp.sum(per(x_neigh - jnp.expand_dims(tri_props["vs"], 1), mesh_props["L"]) ** 2, axis=-1) - tri_props["tR"][mesh_props["neigh"], mesh_props["k2s"]] ** 2
    flip_inequality = d2_x - d2_xneigh
    return flip_inequality



@jit
def get_flip_mask(mesh_props,tri_props):
    flip_mask = get_flip_inequality(mesh_props,tri_props)>0
    retriangulate = (flip_mask).any()
    tri_props["flip_mask"] = flip_mask
    tri_props["retriangulate"] = retriangulate
    return tri_props


@partial(jit, static_argnums=(6,7))
def get_geometry(x_hat,R_hat,L,tri,neigh,k2s,nv,nc, tri_props=None, cell_props=None):
    mesh_props = {}
    # mesh_props["x_true"] = x
    mesh_props["x"] = x_hat*L
    # mesh_props["R_true"] = R
    mesh_props["R"] = R_hat*L
    mesh_props["tri"] = tri
    mesh_props["neigh"] = neigh
    mesh_props["k2s"] = k2s
    mesh_props["nc"] = nc
    # mesh_props["L_true"] = L
    mesh_props["L"] = L
    mesh_props["nv"] = nv
    cell_props, tri_props = _get_geometry(mesh_props, nc, tri_props, cell_props)
    tri_props = get_flip_mask(mesh_props,tri_props)
    return mesh_props,cell_props,tri_props



@partial(jit, static_argnums=(6,7))
def energy(x,R,L,tri,neigh,k2s,nv,nc, tissue_params,tri_props=None, cell_props=None):
    mesh_props,cell_props,tri_props = get_geometry(x,R,L,tri,neigh,k2s,nv,nc, tri_props, cell_props)
    energy = tissue_params["kappa_A"]*(cell_props["A"]-tissue_params["A0"])**2 + tissue_params["kappa_P"]*(cell_props["P"]-tissue_params["P0"])**2
    return energy.sum() + tissue_params["pressure"]*(L**3 - tissue_params["V0"])**2,tri_props["retriangulate"]


@partial(jit, static_argnums=(6,7))
def nabla_energy(x,R,L,tri,neigh,k2s,nv,nc,tissue_params,tri_props=None, cell_props=None):
    return jgrad(energy, argnums=(0, 1, 2),has_aux=True)(x,R,L,tri,neigh,k2s,nv,nc,tissue_params,tri_props, cell_props)


@partial(jit, static_argnums=(6,7))
def nabla_energy_L(x,R,L,tri,neigh,k2s,nv,nc,tissue_params,tri_props=None, cell_props=None):
    return jgrad(energy, argnums=(2))(x,R,L,tri,neigh,k2s,nv,nc,tissue_params,tri_props, cell_props)

p_notch_range = np.linspace(0.1,1,8)
n_repeat = 2
L_out = np.zeros((len(p_notch_range),n_repeat))
for pi, p_notch in enumerate(p_notch_range):
    L = 10
    init_noise = 0.05
    A0 = 1
    x = trf.hexagonal_lattice(L, int(np.ceil(L)), noise=init_noise, A=A0)
    x += -x.min() + 1e-3
    x = x[np.argsort(x.max(axis=1))[:int(L ** 2 / A0)]]
    x_hat = x/L


    R_hat = np.ones(len(x))
    L = 1.
    mesh = Mesh(x_hat,R_hat,1.0,run_options={"equiangulate":True,"equi_nkill":10})
    mesh.triangulate()
    tri, neigh,k2s = mesh.tri, mesh.neigh,mesh.k2s
    nv = len(tri)
    nc = tri.max()+1

    nc = mesh.tri.max() + 1
    is_notch = np.zeros(nc, dtype=bool)
    is_notch[:int(np.round(p_notch * nc))] = True
    np.random.shuffle(is_notch)
    P0 = np.ones(nc)*3.5/10
    P0[is_notch] = 4.0/10
    A0 = np.ones(nc)*1/100
    A0[is_notch] = 1./100
    kappa_P = np.ones(nc)*1
    kappa_P[is_notch] = 0.2

    tissue_params = {"kappa_A": 0.1,
                     "A0": A0,
                     "P0": P0,
                     "kappa_P": kappa_P,
                     "mu_R":0,
                     "mu_L":0.1,
                     "pressure":0.,
                     "V0":8.**3}


    x0_hat = x_hat.copy()
    R0_hat = R_hat.copy()
    L0 = L


    # plt.scatter(*x_hat.T)
    # plt.show()
    #

    dt = 0.005
    tfin = 10
    t_span = np.arange(0,tfin,dt)
    nt = len(t_span)

    x_hat_save,R_hat_save,L_save = np.zeros((nt,nc,2)),np.zeros((nt,nc)),np.zeros((nt))
    E_save = np.zeros((nt))

    for i, t in enumerate(t_span):
        mesh_props,cell_props,tri_props = get_geometry(x_hat, R_hat, L, tri, neigh, k2s, nv, nc)

        (nabla_x, nabla_R, nabla_L),retriangulate = nabla_energy(x_hat, R_hat, L, tri, neigh, k2s, nv, nc, tissue_params)
        energy_val,retriangulate = energy(x_hat, R_hat, L, tri, neigh, k2s, nv, nc, tissue_params)
        E_save[i] = energy_val
        dt_x_hat = - nabla_x
        dt_R_hat = - tissue_params["mu_R"] * nabla_R
        dt_L = - tissue_params["mu_L"] * nabla_L

        x_hat = np.mod(x_hat + dt_x_hat*dt,1)
        R_hat += dt_R_hat*dt
        L += dt_L*dt
        x_hat_save[i] = x_hat.copy()
        R_hat_save[i] = R_hat.copy()
        L_save[i] = L
        if retriangulate:
            mesh = Mesh(np.array(x_hat), np.array(R_hat), 1.0, run_options={"equiangulate": True, "equi_nkill": 10})
            mesh.triangulate()
            tri, neigh, k2s = mesh.tri, mesh.neigh, mesh.k2s
            nv = len(tri)
            nc = tri.max() + 1


    tissue_params = {"kappa_A": 0.1,
                     "A0": A0,
                     "P0": P0,
                     "kappa_P": kappa_P,
                     "mu_R":0,
                     "mu_L":0.1,
                     "pressure":1e-3,
                     "V0": 8.0**3}

    dt = 0.001
    tfin = 10
    t_span = np.arange(0,tfin,dt)
    nt = len(t_span)

    x_hat_save,R_hat_save,L_save = np.zeros((nt,nc,2)),np.zeros((nt,nc)),np.zeros((nt))
    E_save = np.zeros((nt))

    for i, t in enumerate(t_span):
        mesh_props,cell_props,tri_props = get_geometry(x_hat, R_hat, L, tri, neigh, k2s, nv, nc)

        (nabla_x, nabla_R, nabla_L),retriangulate = nabla_energy(x_hat, R_hat, L, tri, neigh, k2s, nv, nc, tissue_params)
        energy_val,retriangulate = energy(x_hat, R_hat, L, tri, neigh, k2s, nv, nc, tissue_params)
        E_save[i] = energy_val
        dt_x_hat = - nabla_x
        dt_R_hat = - tissue_params["mu_R"] * nabla_R
        dt_L = - tissue_params["mu_L"] * nabla_L

        x_hat = np.mod(x_hat + dt_x_hat*dt,1)
        R_hat += dt_R_hat*dt
        L += dt_L*dt
        x_hat_save[i] = x_hat.copy()
        R_hat_save[i] = R_hat.copy()
        L_save[i] = L
        if retriangulate:
            mesh = Mesh(np.array(x_hat), np.array(R_hat), 1.0, run_options={"equiangulate": True, "equi_nkill": 10})
            mesh.triangulate()
            tri, neigh, k2s = mesh.tri, mesh.neigh, mesh.k2s
            nv = len(tri)
            nc = tri.max() + 1

    plt.plot(L_save)
    plt.show()

    L_out[pi] = L_save[-1]
    print(pi)


from scipy.sparse import coo_matrix,csr_matrix
from scipy.sparse.csgraph import connected_components

n_cc = np.zeros(p_notch_range.size,dtype=int)
giant_cluster = np.zeros(p_notch_range.size,dtype=int)
for pi, p_notch in enumerate(p_notch_range):
    L = 10
    init_noise = 0.05
    A0 = 1
    x = trf.hexagonal_lattice(L, int(np.ceil(L)), noise=init_noise, A=A0)
    x += -x.min() + 1e-3
    x = x[np.argsort(x.max(axis=1))[:int(L ** 2 / A0)]]
    x_hat = x/L


    R_hat = np.ones(len(x))
    L = 1.
    mesh = Mesh(x_hat,R_hat,1.0,run_options={"equiangulate":True,"equi_nkill":10})
    mesh.triangulate()
    tri, neigh,k2s = mesh.tri, mesh.neigh,mesh.k2s


    nc = mesh.tri.max() + 1
    is_notch = np.zeros(nc, dtype=bool)
    is_notch[:int(np.round(p_notch * nc))] = True
    np.random.shuffle(is_notch)
    P0 = np.ones(nc)*3.5/10
    P0[is_notch] = 4.0/10
    A0 = np.ones(nc)*1/100
    A0[is_notch] = 1./100
    kappa_P = np.ones(nc)*1
    kappa_P[is_notch] = 0.2
    edges = np.column_stack([[tri[:,i],tri[:,(i+1)%3]] for i in range(3)]).T
    is_notch_edge = is_notch[edges].all(axis=1)
    notch_edges = edges[np.where(is_notch_edge)[0]]
    self_edges = np.column_stack([np.nonzero(is_notch)[0],np.nonzero(is_notch)[0]])
    graph_edges = np.row_stack((notch_edges,self_edges))
    adj = csr_matrix(coo_matrix((np.ones(len(graph_edges),dtype=bool),(graph_edges[:,0],graph_edges[:,1])),shape=(nc,nc)))
    n_cc[pi] = connected_components(adj[is_notch].T[is_notch].T)[0]
    giant_cluster[pi] = np.bincount(connected_components(adj[is_notch].T[is_notch].T)[1]).max()

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(p_notch_range*100,2*np.pi**2*(L_out[:,0]/2)**3,color="teal")
ax.spines[['right', 'top']].set_visible(False)
fig.subplots_adjust(bottom=0.3,left=0.3)
ax.set(xlabel="Percentage Notch+",ylabel="Heart Volume")
fig.savefig("Heart volume versus notch positive.pdf")

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(p_notch_range*100,L_out[:,0],color="teal")
ax.spines[['right', 'top']].set_visible(False)
fig.subplots_adjust(bottom=0.3,left=0.3)
ax.set(xlabel="Percentage Notch+",ylabel="Heart Diameter")
fig.savefig("Heart diameter versus notch positive.pdf")


fig, ax = plt.subplots(figsize=(4,3))
scatter = ax.scatter(giant_cluster/(p_notch_range*nc),L_out[:,0],c=p_notch_range*100)
cbar = plt.colorbar(scatter)
cbar.set_label('Percentage Notch+')

ax.spines[['right', 'top']].set_visible(False)
fig.subplots_adjust(bottom=0.3,left=0.3)
ax.set(xlabel="Percentage of Notch+ cells\nin Giant Cluster",ylabel="Heart Diameter")
fig.savefig("Giant cluster vs heart diameter.pdf")
fig.savefig("Heart volume versus notch positive.pdf")


ax[1].plot(p_notch_range*100,(p_notch_range*nc)/n_cc)
ax[2].plot(p_notch_range*100,giant_cluster)


ax[2].set(xlabel="Percentage Notch+")
ax[0].set(ylabel="'Volume' of heart")
ax[1].set(ylabel=r"$N_{CC}/N_{Notch+}$")
plt.show()

"""
Seemingly roughly functioning code

Apart from the radius movement, which appears to be unstable. 

Seems like L is linear in p_notch with this set up. This could be parameter dependent or fundamental. 

Without vertex movements or intercalations, resistance of tissue = sum resistance of cells. 
For percolation, we expect that somehow softer cells surrounded by stiffer cells are not able to deform as much. 
If this is indeed the case, then we need to show why and how this would occur. 




"""


fig, ax = plt.subplots()
ax.scatter(*x_hat.T*L)
ax.scatter(*tri_props["vs"].T)
fig.show()

from scipy.spatial import Voronoi, voronoi_plot_2d
xprime, Rprime, dictionary = generate_triangulation_mask(np.array(x_hat*L), np.array(R_hat*L), float(L),float(L))
vor = Voronoi(xprime)
fig, ax = plt.subplots()
voronoi_plot_2d(vor, show_vertices=False, line_colors='gray', line_width=2, line_alpha=0.6, point_size=5,ax=ax,show_points=False)

# Overlay a custom scatter plot
custom_points = np.random.rand(5, 2)  # Replace with your custom scatter points
# ax.scatter(*tri_props["vs"].T)
ax.scatter(*x_hat[is_notch].T*L,color="red")
ax.scatter(*x_hat[~is_notch].T*L,color="grey")

ax.set(xlim=(0,L),ylim=(0,L),aspect=1)
fig.show()



fig, ax = plt.subplots()
ax.plot(E_save)
fig.show()


plt.scatter(*x_hat[:].T)

plt.show()

fig, ax = plt.subplots()
ax.plot(L_save)
ax.set(ylim=(0,1))
fig.show()



# @partial(jit, static_argnums=(5,6))
def dt_X(t,X, tri, neigh, k2s, nv, nc, tissue_params):
    x_hat_flat, R_hat, L = X[:2*nc],X[2*nc:3*nc],X[-1]
    x_hat = x_hat_flat.reshape(-1,2)
    nabla_x,nabla_R,nabla_L = nabla_energy(x_hat, R_hat, L, tri, neigh, k2s, nv, nc, tissue_params)
    dt_x_hat = - nabla_x
    dt_R_hat = - tissue_params["mu_R"]*nabla_R
    dt_L_hat = - tissue_params["mu_L"]*nabla_L
    dt_X = jnp.concatenate((dt_x_hat.ravel(),dt_R_hat.ravel(),jnp.array([dt_L_hat])))
    return dt_X

# @partial(jit, static_argnums=(5,6))
def retriangulate_event(t,X, tri, neigh, k2s, nv, nc, tissue_params):
    x_hat_flat, R_hat, L = X[:2*nc],X[2*nc:3*nc],X[-1]
    x_hat = x_hat_flat.reshape(-1,2)
    mesh_props,cell_props,tri_props = get_geometry(x_hat, R_hat, L, tri, neigh, k2s, nv, nc)
    return get_flip_inequality(mesh_props,tri_props).ravel()

retriangulate_event.terminal=True
retriangulate_event.direction = -1.0

X0 = np.zeros(3*nc+1)
x_hat_flat = x_hat.ravel()
X0[:2 * nc], X0[2 * nc:3 * nc], X0[-1] = x_hat_flat, R_hat, L

from scipy.integrate import solve_ivp

solve_ivp(dt_X,[0,0.1],X0,args=(tri, neigh, k2s, nv, nc, tissue_params))




fig, ax = plt.subplots()
ax.scatter(*mesh.x.T)
for i, tri_i in enumerate(mesh.tri):
    for j in range(3):
        y = np.row_stack([mesh.x[mesh.tri][i, j], mesh.x[mesh.tri][i, (j + 1) % 3]])
        ax.plot(*y.T)
fig.show()

p_notch_range = np.linspace(0,1,10)
n_rep = 5

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

L_opt = np.zeros((p_notch_range.size,n_rep))
for i, p_notch in enumerate(p_notch_range):
    for j in range(n_rep):

        nc = mesh.tri.max()+1
        # p_notch = 0.2
        is_notch = np.zeros(nc,dtype=bool)
        is_notch[:int(np.round(p_notch*nc))] = True
        np.random.shuffle(is_notch)
        A0 = np.ones(nc)
        A0[is_notch] = 1.5
        # print(P0.mean())

        tissue_params = {"kappa_A":0.2,
                         "A0":A0,
                         "P0":3.5,
                         "kappa_P":10}

        x,R,L,tri,neigh = mesh.x,mesh.R,mesh.L,mesh.tri,mesh.neigh
        k2s = mesh.k2s
        nv = len(tri)
        nc = np.max(tri)+1
        # cell_props,tri_props = get_geometry(x,R,L,tri,neigh,nv,nc,{},{})
        #
        # vs_vec = np.zeros((10,nv,2))
        # for i, l in enumerate(np.linspace(0.2,4.5,10)):
        #     mesh_props, cell_props, tri_props = get_geometry(x, R, l, tri, neigh, k2s, nv, nc)
        #     vs_vec[i] = tri_props["vs"]
        #
        # Ps_vec = np.zeros((19, nc))
        # for i, l in enumerate(np.arange(0.2, 4.0,0.2)):
        #     mesh_props, cell_props, tri_props = get_geometry(x, R, l, tri, neigh, k2s, nv, nc)
        #     Ps_vec[i] = cell_props["A"]
        #
        # plt.plot(Ps_vec)
        # plt.show()
        #
        # i = 2
        #
        #
        # # ax.scatter(*mesh_props["x"][i].T)
        # ax.scatter(*tri_props["vs"].T,color="purple",alpha=0.4)
        #
        # # vs_in_i = np.where((tri==i).any(axis=1))[0]
        # # ax.scatter(*tri_props["vs"][vs_in_i].T,color="black",alpha=0.2+0.8*np.linspace(0,1,len(vs_in_i)))
        # fig.show()
        #
        #
        #
        # lm1_vec = np.zeros((10, nv,3))
        # for i, l in enumerate(np.linspace(0.2, 4.5, 10)):
        #     mesh_props, cell_props, tri_props = get_geometry(x, R, l, tri, neigh, k2s, nv, nc)
        #     lm1_vec[i] = tri_props["lm1"]
        #
        #
        # for i in range(10):
        #     for j in range(nv):
        #         plt.scatter(i,lm1_vec[i,j, 0]/np.linspace(0.2, 4.5, 10)[i], c=plt.cm.plasma(np.linalg.norm(vs_vec[i,j], axis=-1)))
        # plt.show()
        L0 = 1.0
        def dtL(t,l):
            return - nabla_energy_L(x, R, l[0], tri, neigh, k2s, nv, nc, tissue_params)

        def energy_L(l):
            return energy(x, R, l, tri, neigh, k2s, nv, nc, tissue_params)

        L_range = np.linspace(8,12,200)
        energies = np.zeros_like(L_range)
        for k in range(len(L_range)):
            energies[k] = energy_L(L_range[k])

        jacobian = jgrad(energy_L)

        sol = minimize(energy_L,10.0,method="Newton-CG",jac=jacobian)

        L_opt[i,j] = sol.x[0]
    print("done")

"""
To do now: 

Build infrastructure to simulate, i.e. forward solving, power diagram checking, retriangulation if necessary etc. 


d = np.sqrt(|xi - x|^2 - R^2)

for expansion to be vertex position preserving, then d/L is constant 

so |xi-x|^2/L^2 - R^2/L^2 

so x_hat = x/L and R_hat = R/L



"""