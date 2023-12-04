import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad as jgrad
from jax import grad, vmap

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


class Mesh:
    def __init__(self, x=None, R=None, L=None, tri=None, run_options=None):
        assert run_options is not None
        self.run_options = run_options
        self.x = x
        self.R = R
        self.L = L
        self.tri = tri
        self.neigh = None
        self.A = None
        self.P = None
        self.k2s = []

    def triangulate(self):
        self._triangulate()
        self.k2s = get_k2(self.tri, self.neigh)
        ###
        # ###BETA: the below. For now, recalculate each time step. I get seg fault.
        # ###
        # if type(self.k2s) is list or not self.run_options["equiangulate"]:
        #     self._triangulate()
        #     self.k2s = get_k2(self.tri, self.neigh)
        # else:
        #     tri, neigh, k2s, failed = re_triangulate(self.x, self.R, self.tri, self.neigh, self.k2s, self.x[self.tri], self.R[self.tri],
        #                                              self.L, self.n_v,
        #                                              self.vs, max_runs=self.run_options["equi_nkill"])
        #     if failed:
        #         self._triangulate()
        #         self.k2s = get_k2(self.tri, self.neigh)
        #     else:
        #         self.tri, self.neigh, self.k2s = tri, neigh, k2s

    def _triangulate(self):

        # 1. Tile cell positions 9-fold to perform the periodic triangulation
        #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
        #   and the rest are translations

        if type(self.A) is np.ndarray:
            maxA = np.max(self.A)
            max_d = np.sqrt(maxA / np.pi) * 5  ##2.5 cell diameters on average
            if not max_d > self.L / 50:
                max_d = self.L
        else:
            max_d = self.L

        xprime, Rprime, dictionary = generate_triangulation_mask(self.x, self.R, self.L, max_d)

        # 2. Perform the power triangulation on xprime, Rprime
        tri, n_v = get_power_triangulation(xprime, Rprime)

        # Del = Delaunay(y)
        # tri = Del.simplices
        n_c = self.x.shape[0]

        # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
        #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
        #   Generate a mask -- one_in -- that considers such triangles
        #   Save the new triangulation by applying the mask -- new_tri
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < n_c).any(axis=1)
        new_tri = tri[one_in]

        # 4. Remove repeats in new_tri
        #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details

        n_tri = dictionary[new_tri]
        n_tri = trf.remove_repeats(n_tri, n_c)

        # 6. Store outputs
        self.n_v = n_tri.shape[0]
        self.tri = n_tri
        self.neigh = trf.get_neighbours(n_tri)


@nb.jit(nopython=True)
def generate_triangulation_mask(x, R, L, max_d):
    xprime = np.zeros((0, 2), dtype=np.float64)
    Rprime = np.zeros((0), dtype=np.float64)
    dictionary = np.zeros((0), dtype=np.int64)
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            xprime_i = (x + np.array((i, j)) * L).astype(np.float64)
            if j == 0:
                if i == 0:
                    mask = np.ones_like(x[:, 0], dtype=np.bool_)
                else:
                    val = L * (1 - i) / 2
                    mask = np.abs(x[:, 0] - val) < max_d
            elif i == 0:
                val = L * (1 - j) / 2
                mask = np.abs(x[:, 1] - val) < max_d
            else:
                val_x = L * (1 - i) / 2
                val_y = L * (1 - j) / 2
                mask = np.sqrt((x[:, 0] - val_x) ** 2 + (x[:, 1] - val_y) ** 2) < max_d
            xprime = np.row_stack((xprime, xprime_i[mask]))
            Rprime = np.concatenate((Rprime, R[mask]))

            dictionary = np.concatenate((dictionary, np.nonzero(mask)[0].astype(np.int64)))
    return xprime, Rprime, dictionary


@nb.jit(boolean[:, :](f8[:, :], f8[:], i8[:, :], f8[:, :], f8[:, :], i8[:, :], i8[:, :], i8, f8[:, :], f8), cache=True)
def get_retriangulation_mask(x, R, tri, tR, l2v_x, neigh, k2s, ntri, vs, L):
    d_cell = tri.take(neigh * 3 + k2s).reshape(ntri, 3)
    xd = trf.tri_call3(x, d_cell)
    Rd = trf.tri_call(R, d_cell)
    rad2_d = per.per3(xd - np.expand_dims(vs, 1), L, L)
    rad2_d = rad2_d[..., 0] ** 2 + rad2_d[..., 1] ** 2
    mask = rad2_d - Rd ** 2 < l2v_x - tR ** 2
    return mask


@nb.jit(i8(boolean[:]), cache=True)
def get_first_nonzero(flat_mask):
    i = 0
    while ~flat_mask[i]:
        i += 1
    return i


@nb.jit(i8(boolean[:]), cache=True)
def get_any_nonzero(flat_mask):
    i = int(np.random.random() * flat_mask.size)
    while ~flat_mask[i]:
        i = int(np.random.random() * flat_mask.size)
    return i


@nb.jit((i8[:, :], i8[:, :], i8[:, :], i8, i8), cache=True)
def get_quartet(tri, neigh, k2s, tri_0i, tri_0j):
    a, b, d = np.roll(tri[tri_0i], -tri_0j)
    tri_1i, tri_1j = neigh[tri_0i, tri_0j], k2s[tri_0i, tri_0j]
    c = tri[tri_1i, tri_1j]

    # quartet = np.array((a,b,c,d))

    tri0_da = (tri_0j + 1) % 3
    da_i = neigh[tri_0i, tri0_da]
    da_j = k2s[tri_0i, tri0_da]
    da = tri[da_i, da_j]

    tri0_ab = (tri_0j - 1) % 3
    ab_i = neigh[tri_0i, tri0_ab]
    ab_j = k2s[tri_0i, tri0_ab]
    ab = tri[ab_i, ab_j]

    tri1_cd = (tri_1j - 1) % 3
    cd_i = neigh[tri_1i, tri1_cd]
    cd_j = k2s[tri_1i, tri1_cd]
    cd = tri[cd_i, cd_j]

    tri1_bc = (tri_1j + 1) % 3
    bc_i = neigh[tri_1i, tri1_bc]
    bc_j = k2s[tri_1i, tri1_bc]
    bc = tri[bc_i, bc_j]

    return tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j


@nb.jit(nopython=True, cache=True)
def tri_update(val, quartet_info):
    val_new = val.copy()
    tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j = quartet_info
    val_new[tri_0i, (tri_0j - 1) % 3] = val[tri_1i, tri_1j]
    val_new[tri_1i, (tri_1j - 1) % 3] = val[tri_0i, tri_0j]
    return val_new


@nb.jit(nopython=True, cache=True)
def update_mesh(quartet_info, tri, neigh, k2s):
    """
    Update tri, neigh and k2. Inspect the equiangulation code for some inspo.
    :return:
    """

    tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j = quartet_info

    neigh_new = neigh.copy()
    k2s_new = k2s.copy()

    tri_new = tri_update(tri, quartet_info)

    neigh_new[tri_0i, tri_0j] = neigh[tri_1i, (tri_1j + 1) % 3]
    neigh_new[tri_0i, (tri_0j + 1) % 3] = neigh[bc_i, bc_j]
    neigh_new[tri_0i, (tri_0j + 2) % 3] = neigh[tri_0i, (tri_0j + 2) % 3]
    neigh_new[tri_1i, tri_1j] = neigh[tri_0i, (tri_0j + 1) % 3]
    neigh_new[tri_1i, (tri_1j + 1) % 3] = neigh[da_i, da_j]
    neigh_new[tri_1i, (tri_1j + 2) % 3] = neigh[tri_1i, (tri_1j + 2) % 3]

    k2s_new[tri_0i, tri_0j] = k2s[tri_1i, (tri_1j + 1) % 3]
    k2s_new[tri_0i, (tri_0j + 1) % 3] = k2s[bc_i, bc_j]
    k2s_new[tri_0i, (tri_0j + 2) % 3] = k2s[tri_0i, (tri_0j + 2) % 3]
    k2s_new[tri_1i, tri_1j] = k2s[tri_0i, (tri_0j + 1) % 3]
    k2s_new[tri_1i, (tri_1j + 1) % 3] = k2s[da_i, da_j]
    k2s_new[tri_1i, (tri_1j + 2) % 3] = k2s[tri_1i, (tri_1j + 2) % 3]

    neigh_new[bc_i, bc_j] = tri_0i
    k2s_new[bc_i, bc_j] = tri_0j
    neigh_new[da_i, da_j] = tri_1i
    k2s_new[da_i, da_j] = tri_1j

    return tri_new, neigh_new, k2s_new


@nb.jit((f8[:, :], f8[:], i8[:, :], i8[:, :], i8[:, :], f8[:, :, :], f8[:, :], f8, i8, f8[:, :], i8), cache=True)
def re_triangulate(x, R, _tri, _neigh, _k2s, tx0, tR, L, ntri, vs0, max_runs=10):
    tri, neigh, k2s = _tri.copy(), _neigh.copy(), _k2s.copy()
    # lv_x = trf.tnorm(disp23(vs0, tx0, L))
    v_x = per.per3(np.expand_dims(vs0, 1) - tx0, L, L)
    l2v_x = v_x[..., 0] ** 2 + v_x[..., 1] ** 2

    mask = get_retriangulation_mask(x, R, tri, tR, l2v_x, neigh, k2s, ntri, vs0, L)
    continue_loop = mask.any()
    failed = False
    n_runs = 0
    if continue_loop:
        tx = tx0.copy()
        vs = vs0.copy()
        while (continue_loop):
            mask_flat = mask.ravel()
            q = get_first_nonzero(mask_flat)
            tri_0i, tri_0j = q // 3, q % 3
            quartet_info = get_quartet(tri, neigh, k2s, tri_0i, tri_0j)
            tri, neigh, k2s = update_mesh(quartet_info, tri, neigh, k2s)
            tx = tri_update(tx, quartet_info)
            tR = tri_update(tR, quartet_info)

            tri_0i, tri_1i = quartet_info[0], quartet_info[2]
            tx_changed = np.stack((tx[tri_0i], tx[tri_1i]))
            vs_changed = trf.circumcenter(tx_changed, L)
            vs[tri_0i], vs[tri_1i] = vs_changed
            v_x_changed = per.per(vs_changed - tx_changed[:, 0], L, L)
            l2v_x_changed = v_x_changed[..., 0] ** 2 + v_x_changed[..., 1] ** 2
            l2v_x[tri_0i], l2v_x[tri_1i] = l2v_x_changed
            mask = get_retriangulation_mask(x, R, tri, tR, l2v_x, neigh, k2s, ntri, vs0, L)
            if n_runs > max_runs:
                failed = True
                continue_loop = False
            if not mask.any():
                continue_loop = False
            n_runs += 1
    return tri, neigh, k2s, failed
#

@nb.jit(i8[:, :](i8[:, :], i8[:, :]))
def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int64)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s


@jit
def normalized(X):
    return X / jnp.sqrt(jnp.sum(X ** 2))


@jit
def lift(x, R):
    x_norm = x[0] ** 2 + x[1] ** 2 - R ** 2
    return jnp.column_stack([x, x_norm])


@jit
def get_triangle_normal(A, B, C):
    return normalized(jnp.cross(A, B) + jnp.cross(B, C) + jnp.cross(C, A))


@jit
def get_power_circumcenter(A, B, C):
    N = get_triangle_normal(A, B, C)
    return (-.5 / N[2]) * N[:2]


@jit
def get_circumcentre(x, R):
    return get_power_circumcenter(*get_x_lifted(x, R))


@nb.jit(nopython=True, cache=True)
def nb_get_power_circumcenter(A, B, C):
    N = get_triangle_normal(A, B, C)
    return (-.5 / N[2]) * N[:2]


@nb.jit(nopython=True, cache=True)
def nb_get_circumcentre(x, R):
    return get_power_circumcenter(*get_x_lifted(x, R))


@jit
def get_x_lifted(x, R):
    x_norm = jnp.sum(x ** 2, axis=1) - R ** 2
    x_lifted = jnp.concatenate((x, x_norm.reshape(-1, 1)), axis=1)
    return x_lifted


@nb.jit(nopython=True, cache=True)
def get_vertices(x_lifted, tri, n_v):
    V = np.zeros((n_v, 2))
    for i, tri in enumerate(tri):
        A, B, C = x_lifted[tri]
        V[i] = nb_get_power_circumcenter(A, B, C)
    return V


@nb.jit(nopython=True, cache=True)
def is_ccw_triangle(A, B, C):
    m = np.stack((A, B, C))
    M = np.column_stack((m, np.ones(3)))
    return np.linalg.det(M) > 0


@nb.jit(nopython=True, cache=True)
def build_tri_and_norm(x, simplices, equations):
    saved_tris = equations[:, 2] <= 0
    n_v = saved_tris.sum()
    norms = equations[saved_tris]
    tri_list = np.zeros((n_v, 3), dtype=np.int64)
    i = 0
    for (a, b, c), eq in zip(simplices[saved_tris], norms):
        if is_ccw_triangle(x[a], x[b], x[c]):
            tri_list[i] = a, b, c
        else:
            tri_list[i] = a, c, b
        i += 1
    return tri_list, norms, n_v


def get_power_triangulation(x, R):
    # Compute the lifted weighted points
    x_lifted = get_x_lifted(x, R)
    # Compute the convex hull of the lifted weighted points
    hull = ConvexHull(x_lifted)
    #
    # # Extract the Delaunay triangulation from the lower hull
    tri, norms, n_v = build_tri_and_norm(x, hull.simplices, hull.equations)

    return tri, n_v


#
# l2v_x = jnp.sum(tri_props["v_x"]**2,axis=-1)
#
# get_retriangulation_mask(mesh.x, R, tri, tR, l2v_x, neigh, k2s, ntri, vs, L)
# get_retriangulation_mask(mesh.x,mesh.R,mesh.tri,mesh.R[mesh.tri],l2v_x,mesh.neigh,mesh.k2s,len(mesh.tri),tri_props["vs"],L)
#
#
# #
# x = np.random.random((20,2))
# R = np.ones(20)
# L = 1.
# mesh = Mesh(x,R,L,run_options={"equiangulate":True,"equi_nkill":10})
# mesh.triangulate()
