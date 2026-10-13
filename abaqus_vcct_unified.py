# ==============================================================================
#   3D VCCT Post-Processing Script for Monoclinic Bi-material Interface Crack
#   Unified (modes 1+2 and 1+3): Shira Atias, June 2026
#
#   PURPOSE:
#     Calculates stress intensity factors (K1, K2, K3) and energy release
#     rates (G_I, G_II, G_III) using the 3D VCCT method for a delamination
#     crack at a bi-material interface between two dissimilar composite
#     laminates (monoclinic symmetry via Stroh formalism).
#
#     The mode mixity (K1+iK2  vs  K1+iK3) is determined automatically from
#     the material constants via the d* vector and -- when both W12 and W23
#     are non-zero ("5 ks" monoclinic case) -- by comparing the oscillatory
#     and non-oscillatory displacement-jump ratios Du1/Du3.
#
#   HOW TO RUN:
#     ABAQUS/CAE  ->  File > Run Script...
#     An ODB file must be open in the viewport.
#
#
#   COORDINATE SYSTEM:
#     x1 = crack propagation direction
#     x2 = normal to interface (crack opening = Mode I)
#     x3 = thickness direction (along crack front)
#     k  = z-position index,  i = crack extension index
#
#   OUTPUT FILES:
#     - vcct_results_<model_name>.txt : Full VCCT results table
#     - displacements_from_odb.txt    : Crack face displacements (debug)
#
#   WARNING:  OUTPUT FORMAT:
#     All numeric values written as %.21e (tab-separated).
#     Any change to column order WILL break downstream Excel imports!
# ==============================================================================

# -- Standard & ABAQUS imports -------------------------------------------------
from abaqusConstants import *
from driverUtils import executeOnCaeStartup
import sys
import math
import cmath
from math import log, cos, sin, sqrt

import numpy as np
import numpy.linalg as npl
import sympy as sp
from scipy.linalg import svd
from scipy.special import gamma


# ==============================================================================
# SECTION 1: USER PARAMETERS
# ==============================================================================

max_elements  = 25          # Maximum number of crack-front elements to use
instanceLabel = 'PART-1-1'  # ABAQUS instance name
tolerance     = 1e-7        # Spatial tolerance for node-on-path detection


# ==============================================================================
# SECTION 2: MATERIAL PROPERTIES
# ==============================================================================
# UD composite, same ply material for both layers, rotated by theta1 / theta2.
#
# WARNING:  ACTIVE CONFIGURATION: +30 deg / -60 deg
#     To switch to +/-45 deg uncomment the appropriate block below.

# -- Layer 1 (+30 deg) ---------------------------------------------------------
E11_1  = 138e3
E22_1  = 9.7e3
E33_1  = 9.7e3
v23_1  = 0.46
v13_1  = 0.32
v12_1  = 0.32
G23_1  = E22_1 / (2 * (1 + v23_1))
G13_1  = 4.6e3
G12_1  = 4.6e3
theta1 = np.pi / 6    # +30 degrees

v21_1 = v12_1 * E22_1 / E11_1
v31_1 = v13_1 * E33_1 / E11_1
v32_1 = v23_1 * E33_1 / E22_1

# -- Layer 2 (-60 deg) ---------------------------------------------------------
E11_2  = 138e3
E22_2  = 9.7e3
E33_2  = 9.7e3
v23_2  = 0.46
v13_2  = 0.32
v12_2  = 0.32
G23_2  = E22_2 / (2 * (1 + v23_2))
G13_2  = 4.6e3
G12_2  = 4.6e3
theta2 = -np.pi / 3   # -60 degrees

v21_2 = v12_2 * E22_2 / E11_2
v31_2 = v13_2 * E33_2 / E11_2
v32_2 = v23_2 * E33_2 / E22_2


print '\n\n\n###################### new run #############################'


# ==============================================================================
# SECTION 3: ODB ACCESS -- OPEN DATABASE AND IDENTIFY CRACK GEOMETRY
# ==============================================================================

odb      = session.odbs.values()[0]
step     = odb.steps.values()[-1]
frame    = step.frames[-1]
assembly = odb.rootAssembly.instances[instanceLabel]

odb_name   = odb.name[:-4]
model_name = ''
for x in reversed(odb_name):
    if x != '/':
        model_name = model_name + x
    else:
        break
model_name = model_name[::-1]

J_nodes = [s for s in odb.rootAssembly.nodeSets.keys() if 'Contour_1' in s]

crack_tip_node  = J_nodes[0]
front_tip_node  = J_nodes[-1]

crack_tip_label       = []
crack_tip_coordinates = []
for j in odb.rootAssembly.nodeSets[crack_tip_node].nodes[0]:
    crack_tip_label.append(j.label)
    crack_tip_coordinates = j.coordinates.tolist()

front_tip_coordinates = []
for j in odb.rootAssembly.nodeSets[front_tip_node].nodes[0]:
    front_tip_coordinates = j.coordinates.tolist()

x_tip = crack_tip_coordinates[0]
y_tip = crack_tip_coordinates[1]
z_tip = crack_tip_coordinates[2]
z_end = front_tip_coordinates[2]


# ==============================================================================
# SECTION 4: NODE COLLECTION -- CRACK FRONT AND CRACK FACE NODES
# ==============================================================================

max_nodes = 2 * max_elements

start_nodef = [x_tip,            y_tip, z_tip]
end_nodef   = [x_tip + max_nodes, y_tip, z_end]

start_nodeu = [x_tip - max_nodes, y_tip, z_tip]
end_nodeu   = [x_tip - tolerance,  y_tip, z_end]

nodef_tulips_all = []
nodeu_tulips     = []

all_nodes  = assembly.nodes
node_count = len(all_nodes)

node_labels = np.empty(node_count, dtype=np.int32)
node_coords = np.empty((node_count, 3), dtype=np.float64)

for idx, node in enumerate(all_nodes):
    node_labels[idx]    = node.label
    node_coords[idx, 0] = node.coordinates[0]
    node_coords[idx, 1] = node.coordinates[1]
    node_coords[idx, 2] = node.coordinates[2]

mask_f = (
    (node_coords[:, 0] >= start_nodef[0]) &
    (node_coords[:, 0] <= end_nodef[0])   &
    (np.abs(node_coords[:, 1] - y_tip) < tolerance) &
    (node_coords[:, 2] >= start_nodef[2]) &
    (node_coords[:, 2] <= end_nodef[2])
)
for i in np.where(mask_f)[0]:
    nodef_tulips_all.append((node_coords[i, 2], node_coords[i, 0], int(node_labels[i])))

mask_u = (
    (node_coords[:, 0] >= start_nodeu[0]) &
    (node_coords[:, 0] <= end_nodeu[0])   &
    (np.abs(node_coords[:, 1] - y_tip) < tolerance) &
    (node_coords[:, 2] >= start_nodeu[2]) &
    (node_coords[:, 2] <= end_nodeu[2])
)
for i in np.where(mask_u)[0]:
    nodeu_tulips.append((node_coords[i, 2], node_coords[i, 0], int(node_labels[i])))

nodef_tulips_all.sort()
nodeu_tulips.sort()

first_nodes = nodef_tulips_all[2:13]
l = (first_nodes[10][1] - first_nodes[0][1]) / 5.0

nodef_tulips = []
seen_nodes   = {}

for tup in nodef_tulips_all:
    z_coord = tup[0]
    x_coord = tup[1]
    if x_coord < x_tip + l * max_elements - 0.25 * l:
        if z_coord not in seen_nodes:
            seen_nodes[z_coord] = 1
            nodef_tulips.append(tup)
        elif seen_nodes[z_coord] <= max_nodes:
            seen_nodes[z_coord] += 1
            nodef_tulips.append(tup)

nodeu_updown  = []
u_labels_dict = {}

for tup in nodeu_tulips:
    z_coord, x_coord, label = tup
    if x_coord >= x_tip - l * max_elements:
        u_labels_dict.setdefault(z_coord, []).append(tup)

for z_coord_labels in u_labels_dict.values():
    nodeu_updown.extend(z_coord_labels[-2 * max_nodes:])


# ==============================================================================
# SECTION 5: ELEMENT COLLECTION -- UPPER AND LOWER INTERFACE ELEMENTS
# ==============================================================================

start_el_up   = [x_tip - 1.5 * l * max_nodes, y_tip + l / 2.0, z_tip]
end_el_up     = (x_tip + 1.5 * l * max_nodes,  y_tip + l / 2.0, z_end)

start_el_down = [x_tip - 1.5 * max_nodes,      y_tip - l / 2.0, z_tip]
end_el_down   = (x_tip + 1.5 * max_nodes,       y_tip - l / 2.0, z_end)

elements_up   = []
elements_down = []

node_coord_dict = {int(node_labels[i]): node_coords[i] for i in range(node_count)}

for element in assembly.elements:
    connectivity = element.connectivity
    elem_coords  = [node_coord_dict[lbl] for lbl in connectivity if lbl in node_coord_dict]

    if any(
        start_el_up[0] <= c[0] <= end_el_up[0] and
        abs(c[1] - start_el_up[1]) < tolerance and
        start_el_up[2] <= c[2] <= end_el_up[2]
        for c in elem_coords
    ):
        elements_up.append(element.label)

    elif any(
        start_el_down[0] <= c[0] <= end_el_down[0] and
        abs(c[1] - start_el_down[1]) < tolerance and
        start_el_down[2] <= c[2] <= end_el_down[2]
        for c in elem_coords
    ):
        elements_down.append(element.label)


# ==============================================================================
# SECTION 6: SPLIT nodeu_updown INTO UPPER AND LOWER CRACK FACE NODES
# ==============================================================================

nodeup_tulips   = []
nodedown_tulips = []
seen_labels     = set()

nodeu_dict = {tup[2]: tup for tup in nodeu_updown}

for elem_label in elements_up:
    connectivity = assembly.elements[elem_label - 1].connectivity
    for node_label in connectivity:
        if node_label in nodeu_dict and node_label not in seen_labels:
            nodeup_tulips.append(nodeu_dict[node_label])
            seen_labels.add(node_label)

for elem_label in elements_down:
    connectivity = assembly.elements[elem_label - 1].connectivity
    for node_label in connectivity:
        if node_label in nodeu_dict and node_label not in seen_labels:
            nodedown_tulips.append(nodeu_dict[node_label])
            seen_labels.add(node_label)

nodeup_tulips.sort()
nodedown_tulips.sort()


# ==============================================================================
# SECTION 7: BUILD 2D NODE ARRAYS  (node_z_num x node_x_num)
# ==============================================================================

l_x = (nodef_tulips[-2][1] - nodef_tulips[0][1]) / (max_elements - 1)
l_z = (nodef_tulips[-1][0] - nodef_tulips[0][0]) / (len(J_nodes) - 1) * 2

node_x_num = max_nodes
node_z_num = len(J_nodes)

nodef    = [[-1] * node_x_num for _ in range(node_z_num)]
nodeup   = [[-1] * node_x_num for _ in range(node_z_num)]
nodedown = [[-1] * node_x_num for _ in range(node_z_num)]

nodef_tulips_iter    = iter(nodef_tulips)
nodeup_tulips_iter   = iter(nodeup_tulips)
nodedown_tulips_iter = iter(nodedown_tulips)

for j in range(node_z_num):
    for i in range(node_x_num):
        if (i % 2 == 1) and (j % 2 == 1):
            pass
        else:
            nodef[j][i]    = next(nodef_tulips_iter)[2]
            nodeup[j][i]   = next(nodeup_tulips_iter)[2]
            nodedown[j][i] = next(nodedown_tulips_iter)[2]


# ==============================================================================
# SECTION 8: NODAL FORCE EXTRACTION  (sumNforc1/2/3)
# ==============================================================================

nodef_label_to_ij = {}
for i_row in range(node_z_num):
    for j_col in range(node_x_num):
        lbl = nodef[i_row][j_col]
        if lbl != -1:
            nodef_label_to_ij[lbl] = (i_row, j_col)

sumNforc1 = [[0.0] * node_x_num for _ in range(node_z_num)]
sumNforc2 = [[0.0] * node_x_num for _ in range(node_z_num)]
sumNforc3 = [[0.0] * node_x_num for _ in range(node_z_num)]

for elem_label in elements_up:
    elem_region = assembly.elements[elem_label - 1]

    nforc1_vals = frame.fieldOutputs['NFORC1'].getSubset(region=elem_region).values
    nforc2_vals = frame.fieldOutputs['NFORC2'].getSubset(region=elem_region).values
    nforc3_vals = frame.fieldOutputs['NFORC3'].getSubset(region=elem_region).values

    for v in nforc1_vals:
        ij = nodef_label_to_ij.get(v.nodeLabel)
        if ij:
            sumNforc1[ij[0]][ij[1]] += v.dataDouble

    for v in nforc2_vals:
        ij = nodef_label_to_ij.get(v.nodeLabel)
        if ij:
            sumNforc2[ij[0]][ij[1]] += v.dataDouble

    for v in nforc3_vals:
        ij = nodef_label_to_ij.get(v.nodeLabel)
        if ij:
            sumNforc3[ij[0]][ij[1]] += v.dataDouble


# ==============================================================================
# SECTION 9: DISPLACEMENT JUMP EXTRACTION  (dis_x, dis_y, dis_z)
# ==============================================================================

dis_val = frame.fieldOutputs['U'].values
disp_dict = {}
for v in dis_val:
    disp_dict[v.nodeLabel] = v.dataDouble

disup_x = [[0.0] * node_x_num for _ in range(node_z_num)]
disup_y = [[0.0] * node_x_num for _ in range(node_z_num)]
disup_z = [[0.0] * node_x_num for _ in range(node_z_num)]
dis_x   = [[0.0] * node_x_num for _ in range(node_z_num)]
dis_y   = [[0.0] * node_x_num for _ in range(node_z_num)]
dis_z   = [[0.0] * node_x_num for _ in range(node_z_num)]

with open('displacements_from_odb.txt', 'w') as disp_file:
    disp_file.write('{:<10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}\n'.format(
        "Node(i,j)", "dis_x", "disup_x", "val_x",
        "dis_y", "disup_y", "val_y",
        "dis_z", "disup_z", "val_z"))

    for j in range(node_z_num):
        for i in range(node_x_num):
            if (i % 2 == 1) and (j % 2 == 1):
                continue

            up_lbl   = nodeup[j][i]
            down_lbl = nodedown[j][i]
            col      = node_x_num - 1 - i

            d_up   = disp_dict[up_lbl]
            d_down = disp_dict[down_lbl]

            disup_x[j][col] = d_up[0]
            disup_y[j][col] = d_up[1]
            disup_z[j][col] = d_up[2]

            dis_x[j][col]   = d_up[0] - d_down[0]
            dis_y[j][col]   = d_up[1] - d_down[1]
            dis_z[j][col]   = d_up[2] - d_down[2]

            disp_file.write('({},{}) {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e} {:12.4e}\n'.format(
                i, j,
                dis_x[j][col], disup_x[j][col], d_down[0],
                dis_y[j][col], disup_y[j][col], d_down[1],
                dis_z[j][col], disup_z[j][col], d_down[2]))


# ==============================================================================
# SECTION 10: MATERIAL MATRICES -- STROH / LEKHNITSKII FORMALISM
# ==============================================================================

def S_mat(E11, E22, E33, G23, G13, G12, v23, v13, v12, v32, v31, v21):
    """6x6 compliance matrix S (Voigt notation: 11,22,33,23,13,12)."""
    return np.array([
        [ 1/E11,   -v21/E22, -v31/E33,    0,        0,        0      ],
        [-v12/E11,  1/E22,   -v32/E33,    0,        0,        0      ],
        [-v13/E11, -v23/E22,  1/E33,      0,        0,        0      ],
        [ 0,        0,        0,       1/G23,       0,        0      ],
        [ 0,        0,        0,          0,     1/G13,       0      ],
        [ 0,        0,        0,          0,        0,     1/G12     ]
    ])

S1 = S_mat(E11_1, E22_1, E33_1, G23_1, G13_1, G12_1, v23_1, v13_1, v12_1, v32_1, v31_1, v21_1)
S2 = S_mat(E11_2, E22_2, E33_2, G23_2, G13_2, G12_2, v23_2, v13_2, v12_2, v32_2, v31_2, v21_2)


def rotate_mat_x2(theta, S):
    """Rotate 6x6 compliance matrix S by angle theta about the x2-axis."""
    m, n = np.cos(theta), np.sin(theta)
    K = np.array([
        [ m**2,  0,  n**2,  0,    2*m*n,     0],
        [ 0,     1,  0,     0,    0,         0],
        [ n**2,  0,  m**2,  0,   -2*m*n,     0],
        [ 0,     0,  0,     m,    0,         -n],
        [-m*n,   0,  m*n,   0,   m**2-n**2,  0],
        [ 0,     0,  0,     n,    0,          m]
    ])
    K_inv = npl.inv(K)
    return np.dot(K_inv.T, np.dot(S, K_inv))

S1 = rotate_mat_x2(theta1, S1)
S2 = rotate_mat_x2(theta2, S2)


def reduce_matrix(S):
    """Plane-strain reduced compliance: St[i,j] = S[i,j] - S[i,2]*S[2,j]/S[2,2]"""
    St = S.copy()
    for i in range(6):
        for j in range(6):
            St[i, j] = S[i, j] - S[i, 2] * S[2, j] / S[2, 2]
    return St

St1 = reduce_matrix(S1)
St2 = reduce_matrix(S2)


def eig_calc_ex(St):
    """Closed-form roots of decoupled in-plane and anti-plane eigenvalues."""
    disc_ip = (2*St[0,1] + St[5,5])**2 - 4*St[0,0]*St[1,1] + 0j
    p1 = np.sqrt((-(2*St[0,1] + St[5,5]) - np.sqrt(disc_ip)) / (2*St[0,0]) + 0j)
    p2 = np.sqrt((-(2*St[0,1] + St[5,5]) + np.sqrt(disc_ip)) / (2*St[0,0]) + 0j)
    disc_ap = (2*St[3,4])**2 - 4*St[4,4]*St[3,3] + 0j
    p3 = (-(2*St[3,4]) + np.sqrt(disc_ap)) / (2*St[4,4])
    return [p1, p2, p3]


def eig_calc1(St):
    """Numerical roots of full degree-6 characteristic polynomial."""
    p  = sp.symbols('p')
    l2 = St[4,4]*p**2 - 2*St[3,4]*p + St[3,3]
    l3 = St[0,4]*p**3 - (St[0,3]+St[4,5])*p**2 + (St[1,4]+St[3,5])*p - St[1,3]
    l4 = St[0,0]*p**4 - 2*St[0,5]*p**3 + (2*St[0,1]+St[5,5])*p**2 - 2*St[1,5]*p + St[1,1]
    eq = l2*l4 - l3*l3
    roots = sp.nroots(eq, n=6)
    return [complex(r.as_real_imag()[0], r.as_real_imag()[1] + 1e-15)
            for r in roots if r.as_real_imag()[1] > 0]


def rearrange_list(p, p_ex):
    """Reorder sympy roots p to match the ordering of closed-form roots p_ex."""
    rounded_p    = [complex(round(x.real, 2), round(x.imag, 2)) for x in p]
    rounded_p_ex = [complex(round(x.real, 2), round(x.imag, 2)) for x in p_ex]
    if set(rounded_p_ex).issubset(set(rounded_p)):
        p = sorted(p, key=lambda x: rounded_p_ex.index(
            complex(round(x.real, 2), round(x.imag, 2))))
    return p

p1    = eig_calc1(St1);   p2    = eig_calc1(St2)
p1ex  = eig_calc_ex(St1); p2ex  = eig_calc_ex(St2)
p1    = rearrange_list(p1, p1ex)
p2    = rearrange_list(p2, p2ex)


def lambda_calc(St, p):
    """Stroh coupling parameter Lambda."""
    Lam = [0, 0, 0]
    for a in range(2):
        num = (St[0,4]*p[a]**3 - (St[0,3]+St[4,5])*p[a]**2
               + (St[1,4]+St[3,5])*p[a] - St[1,3])
        den = St[4,4]*p[a]**2 - 2*St[3,4]*p[a] + St[3,3]
        Lam[a] = -num / den
    a = 2
    num = (St[0,4]*p[a]**3 - (St[0,3]+St[4,5])*p[a]**2
           + (St[1,4]+St[3,5])*p[a] - St[1,3])
    den = (St[0,0]*p[a]**4 - 2*St[0,5]*p[a]**3
           + (2*St[0,1]+St[5,5])*p[a]**2 - 2*St[1,5]*p[a] + St[1,1])
    Lam[2] = -num / den
    return Lam

Lam1 = lambda_calc(St1, p1)
Lam2 = lambda_calc(St2, p2)


def xi_eta(St, p, Lam):
    """Stress function vectors xi and eta."""
    xi = [[0]*3 for _ in range(6)]
    for i in range(6):
        for a in range(3):
            xi[i][a] = (p[a]**2 * St[i,0] - p[a]*St[i,5] + St[i,1]
                        + Lam[a] * (p[a]*St[i,4] - St[i,3]))
    eta = [Lam[2]*(p[2]**2*St[i,0] - p[2]*St[i,5] + St[i,1])
           + (p[2]*St[i,4] - St[i,3])
           for i in range(6)]
    return xi, eta

xi1, eta1 = xi_eta(St1, p1, Lam1)
xi2, eta2 = xi_eta(St2, p2, Lam2)


def mat_A(xi, eta, p):
    """Stroh displacement amplitude matrix A."""
    return [
        [xi[0][0],        xi[0][1],        eta[0]       ],
        [xi[1][0]/p[0],   xi[1][1]/p[1],   eta[1]/p[2]  ],
        [xi[3][0]/p[0],   xi[3][1]/p[1],   eta[3]/p[2]  ]
    ]

def mat_B(Lam, p):
    """Stroh traction amplitude matrix B."""
    return [
        [-p[0],    -p[1],    -p[2]*Lam[2]],
        [ 1,        1,        Lam[2]      ],
        [-Lam[0],  -Lam[1],  -1           ]
    ]

A1, B1 = mat_A(xi1, eta1, p1), mat_B(Lam1, p1)
A2, B2 = mat_A(xi2, eta2, p2), mat_B(Lam2, p2)


def mats_SL(A, B):
    """Returns (S_inv, L_inv) -- the Barnett-Lothe tensors."""
    B_inv  = npl.inv(np.array(B, dtype=complex))
    AB_inv = np.dot(np.array(A, dtype=complex), B_inv)
    S_inv  = -AB_inv.real
    L_inv  = -AB_inv.imag
    return S_inv, L_inv

SL1_inv, L1_inv = mats_SL(A1, B1)
SL2_inv, L2_inv = mats_SL(A2, B2)

DD   = L1_inv + L2_inv
Dinv = npl.inv(DD)
W    = SL1_inv - SL2_inv
Sw   = np.dot(Dinv, W)

beta = np.sqrt(-0.5 * np.trace(np.dot(Sw, Sw)))
eps  = (1.0 / (2.0 * np.pi)) * np.log((1 + beta) / (1 - beta))

D11, D13, D33 = DD[0,0], DD[0,2], DD[2,2]
W12, W23      = W[0,1],  W[1,2]
denom = beta * (D11 * D33 - D13**2)
omega =  (D33 * W12 + D13 * W23) / denom
lamb  = -(D13 * W12 + D11 * W23) / denom


def calc_H1(D, eps):
    if theta2 == np.pi/2:
        return 4.0 * np.cosh(np.pi * eps)**2 / D[0, 0]
    """H1 = 4 cosh^2(pi*eps) / D22   (Modes I and the oscillatory partner)"""
    else:
        return 4.0 * np.cosh(np.pi * eps)**2 / D[1, 1]

def calc_H2(D, W):
    """H2 for the non-oscillatory mode (Mode II or III depending on mixity)."""
    if W[0, 1] == 0:
        return 4.0 / D[0, 0]
    r = W[1, 2] / W[0, 1]
    return 4.0 / (D[0, 0]*r**2 + 2*D[0, 2]*r + D[2, 2])

H1 = calc_H1(DD, eps)
H2 = calc_H2(DD, W)


def calculate_P(eps_val):
    num = gamma(0.5 + eps_val * 1j)**2
    den = gamma(2.0 * (1.0 + eps_val * 1j))
    return num / den

P  = calculate_P(eps)
Pr = P.real
Pi = P.imag
C  = (cmath.cosh(math.pi * eps) * math.sqrt(Pr**2 + Pi**2) / math.pi).real


# ==============================================================================
# SECTION 10b: MODE MIXITY DETECTION via d* vector
# ==============================================================================
# Compute the d* vector (null-space of Sw - cot(delta*pi)*I for delta = -1/2)
# and use it together with the displacement-jump ratios to decide whether
# K_osc = K1 + i*K2  (in-plane)  or  K_osc = K1 + i*K3  (out-of-plane).
# ==============================================================================

def _nullspace(A, atol=1e-13, rtol=0):
    """Return the null-space of A as column vectors."""
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def d_vector(Sw_mat, delta, param=1):
    """
    Normalised null-space vector d of  (Sw - cot(delta*pi)*I),
    scaled so that the second component equals 1.
    """
    cot_d_pi = 1.0 / np.tan(delta * np.pi)
    A = Sw_mat - cot_d_pi * np.eye(Sw_mat.shape[0])
    ns = _nullspace(A)[:, 0]
    scale = 1.0 / ns[1] if ns[1] != 0 else 1.0
    return ns * scale * param


def dstar_vector(Sw_mat, delta, param=1):
    """
    Normalised null-space vector d* of  (Sw - cot(delta*pi)*I),
    scaled so that the third component equals 1 (or first if third is zero).
    """
    cot_d_pi = 1.0 / np.tan(delta * np.pi)
    A = Sw_mat - cot_d_pi * np.eye(Sw_mat.shape[0])
    ns = _nullspace(A)[:, 0]
    if ns[2] != 0:
        scale = 1.0 / ns[2]
    else:
        scale = 1.0 / ns[0]
    return ns * scale * param


# delta values: complex pair for oscillatory field, real -0.5 for d*
_delta_osc_p = -0.5 + eps * 1j
_delta_osc_m = -0.5 - eps * 1j
_delta_star  = -0.5

d_n      = d_vector(Sw, _delta_osc_p)
d_bar_n  = d_vector(Sw, _delta_osc_m)
d_star_n = dstar_vector(Sw, _delta_star)


def classify_mode_mixity(d_star):
    """
    Classify the basic mode mixity from the d* vector components.

    Returns one of:
      'k1+k2'  -- in-plane shear dominant  (K_osc = K1 + i*K2)
      'k1+k3'  -- out-of-plane dominant    (K_osc = K1 + i*K3)
      'k2+k3'  -- (degenerate case)
      '5ks'    -- monoclinic coupling: BOTH d*[0] and d*[2] non-zero;
                  requires displacement-ratio test to resolve.
    """
    nonzero = [abs(d_star[c]) > 1e-10 for c in range(3)]
    if     nonzero[0] and not nonzero[1] and not nonzero[2]:
        return 'k1+k3'
    elif not nonzero[0] and     nonzero[1] and not nonzero[2]:
        return 'k2+k3'
    elif not nonzero[0] and not nonzero[1] and     nonzero[2]:
        return 'k1+k2'
    elif    nonzero[0] and not nonzero[1] and     nonzero[2]:
        return '5ks'
    else:
        return '5ks'   # fallback for any other coupled case


def resolve_5ks(DD, W, Omega, Lambda):
    """
    For the monoclinic '5ks' case, compare the oscillatory ratio Du1/Du3
    (eq. Du1/Du3 = (D11*Omega + D13*Lambda) / (D13*Omega + D33*Lambda))
    with the non-oscillatory ratio Du1/Du3 = -Lambda/Omega.

    Returns 'k1+k2' when the oscillatory in-plane component dominates
    (|ratio_osc| >= 10 * |ratio_non_osc|), otherwise 'k1+k3'.

    Parameters
    ----------
    DD     : 3x3 bimaterial D matrix
    W      : 3x3 bimaterial W matrix
    Omega  : omega coupling parameter
    Lambda : lambda coupling parameter
    """
    D11 = DD[0, 0]
    D13 = DD[0, 2]
    D33 = DD[2, 2]

    ratio_osc_num = D11 * Omega + D13 * Lambda
    ratio_osc_den = D13 * Omega + D33 * Lambda

    if abs(ratio_osc_den) < 1e-30:
        # Denominator vanishingly small -> in-plane dominates
        return 'k1+k2'

    ratio_osc = ratio_osc_num / ratio_osc_den

    if abs(Lambda) < 1e-30 or abs(Omega) < 1e-30:
        # Non-oscillatory ratio undefined -- fall back to oscillatory sign
        ratio_non = 0.0
    else:
        ratio_non = -Lambda / Omega

    print '*** 5ks monoclinic case detected ***'
    print '    ratio_osc  (Du1/Du3 oscillatory)     = %.6e' % abs(ratio_osc)
    print '    ratio_non  (Du1/Du3 non-oscillatory) = %.6e' % abs(ratio_non)

    if abs(ratio_non) < 1e-30 or abs(ratio_osc) >= 100.0 * abs(ratio_non):
        print '    Decision: in-plane dominates  --> K_osc = K1 + i*K2'
        return 'k1+k2'
    elif abs(ratio_non) > 1e30 or 100 * abs(ratio_osc) <= abs(ratio_non):
        print '    Decision: out-of-plane dominates --> K_osc = K1 + i*K3'
        return 'k1+k3'
    else:
        print '    Decision: !!indecisive!! --> defaulting to k1+i*k2'
        return 'k1+k2'

# -- Determine final mode mixity -----------------------------------------------
_basic_mixity = classify_mode_mixity(d_star_n)

if _basic_mixity == '5ks':
    mode_mixity = resolve_5ks(DD, W, omega, lamb)
else:
    mode_mixity = _basic_mixity

print '*** Mode mixity determined: %s ***' % mode_mixity

# mode_mixity is now either 'k1+k2' or 'k1+k3'
# (k2+k3 is left as-is; extend if needed)


# ==============================================================================
# SECTION 11: VCCT ENERGY RELEASE RATE AND INTERACTION INTEGRAL COMPUTATION
# ==============================================================================

half_x = node_x_num // 2

g1      = [[0.0]*half_x for _ in range(node_z_num)]
g2      = [[0.0]*half_x for _ in range(node_z_num)]
g3      = [[0.0]*half_x for _ in range(node_z_num)]
gtot123 = [[0.0]*half_x for _ in range(node_z_num)]
gtot    = [[0.0]*half_x for _ in range(node_z_num)]
i1      = [[0.0]*half_x for _ in range(node_z_num)]
i2      = [[0.0]*half_x for _ in range(node_z_num)]
i3      = [[0.0]*half_x for _ in range(node_z_num)]
ii      = [[0.0]*half_x for _ in range(node_z_num)]
i_tot123= [[0.0]*half_x for _ in range(node_z_num)]
da      = [[0.0]*half_x for _ in range(node_z_num)]
psi     = [[0.0]*half_x for _ in range(node_z_num)]
K1      = [[0.0]*half_x for _ in range(node_z_num)]
K2      = [[0.0]*half_x for _ in range(node_z_num)]
K3      = [[0.0]*half_x for _ in range(node_z_num)]
K_osc   = [[0.0]*half_x for _ in range(node_z_num)]   # the oscillatory SIF partner
K_non   = [[0.0]*half_x for _ in range(node_z_num)]   # the non-oscillatory SIF
E_el    = [[0.0]*half_x for _ in range(node_z_num)]


def get_weights(k, node_z_num):
    """
    Return (w_k_always, w_k_jeven, neighbours_always, neighbours_jeven).
    Encodes the 3D trapezoidal integration rule along z.
    """
    N = node_z_num - 1
    if k == 0:
        return 1.0, 0.0, [], [(k+1, 0.5)]
    elif k == N:
        return 1.0, 0.0, [], [(k-1, 0.5)]
    elif k == 1 and node_z_num == 3:
        return 0.0, 1.0, [(k+1, 1.0), (k-1, 1.0)], []
    elif k == 1:
        return 0.0, 1.0, [(k+1, 0.5), (k-1, 1.0)], []
    elif k == N - 1:
        return 0.0, 1.0, [(k+1, 1.0), (k-1, 0.5)], []
    elif k % 2 == 0:
        return 1.0, 0.0, [], [(k+1, 0.5), (k-1, 0.5)]
    else:
        return 0.0, 1.0, [(k+1, 0.5), (k-1, 0.5)], []


def vcct_contrib(k, j, idx_du, du_fields, F_fields, node_z_num):
    """Compute the weighted VCCT contribution for one (k, i, j) step."""
    wa, we, nb_a, nb_e = get_weights(k, node_z_num)
    j_even = (j % 2 == 0)
    result = [0.0, 0.0, 0.0]

    for m in range(3):
        du = du_fields[m]
        F  = F_fields[m]
        val_k = du[k][idx_du] * F[k][j]
        result[m] += wa * val_k
        if j_even:
            result[m] += we * val_k
        for (kr, wr) in nb_a:
            result[m] += wr * du[kr][idx_du] * F[kr][j]
        if j_even:
            for (kr, wr) in nb_e:
                result[m] += wr * du[kr][idx_du] * F[kr][j]
    return result


# ------------------------------------------------------------------------------
# Select interaction-integral force fields depending on mode mixity.
#
#   k1+k2 (file 12):  i1 = du_x*F2,  i2 = du_y*F1,  i3 = du_z*F2
#                     F fields for i-loop: [sumNforc2, sumNforc1, sumNforc2]
#
#   k1+k3 (file 13):  i1 = du_x*F2,  i2 = du_y*F3,  i3 = du_z*F2
#                     F fields for i-loop: [sumNforc2, sumNforc3, sumNforc2]
# ------------------------------------------------------------------------------

if mode_mixity == 'k1+k2':
    _iF_fields = [sumNforc2, sumNforc1, sumNforc2]
else:   # k1+k3 (or k2+k3 fallback)
    _iF_fields = [sumNforc2, sumNforc3, sumNforc2]

for k in range(node_z_num):
    for i in range(half_x):
        for j in range(2*i + 2):
            idx_du = (2*(i+1)) - j - 1

            # G contributions (identical for both mixity cases)
            dg = vcct_contrib(k, j, idx_du,
                              [dis_y, dis_x, dis_z],
                              [sumNforc2, sumNforc1, sumNforc3],
                              node_z_num)
            g1[k][i] += dg[0]
            g2[k][i] += dg[1]
            g3[k][i] += dg[2]

            # Interaction integral contributions (force field depends on mixity)
            di = vcct_contrib(k, j, idx_du,
                              [dis_x, dis_y, dis_z],
                              _iF_fields,
                              node_z_num)
            i1[k][i] += di[0]
            i2[k][i] += di[1]
            i3[k][i] += di[2]


# ==============================================================================
# SECTION 12: NORMALISATION AND SIF CALCULATION
# ==============================================================================

for k in range(node_z_num):
    for i in range(half_x):
        if k == 0 or k == node_z_num - 1:
            norm = l_x * l_z * (i + 1)
        else:
            norm = l_x * l_z * 2.0 * (i + 1)

        g1[k][i]  /= norm
        g2[k][i]  /= norm
        g3[k][i]  /= norm

        if mode_mixity == 'k1+k2':
            # i1 = du_x*F2 * omega,  i2 = du_y*F1 / omega,  i3 = du_z*F2 * lamb
            i1[k][i]   = i1[k][i] / norm * omega
            i2[k][i]   = i2[k][i] / norm / omega
            i3[k][i]   = i3[k][i] / norm * lamb
            # Dominant pair: G_I + G_II;  balance: -(i1+i2)
            gtot[k][i]    = g1[k][i] + g2[k][i]
            i_tot123[k][i]= -(i1[k][i] + i2[k][i])
        else:
            # k1+k3
            # i1 = du_x*F2 * omega,  i2 = du_y*F3 / lamb,   i3 = du_z*F2 * lamb
            i1[k][i]   = i1[k][i] / norm * omega
            i2[k][i]   = i2[k][i] / norm / lamb
            i3[k][i]   = i3[k][i] / norm * lamb
            # Dominant pair: G_I + G_III;  balance: -(i2+i3)
            gtot[k][i]    = g1[k][i] + g3[k][i]
            i_tot123[k][i]= -(i2[k][i] + i3[k][i])

        gtot123[k][i] = g1[k][i] + g2[k][i] + g3[k][i]
        da[k][i]      = l_x * (i + 1)


# -- Imbalance indicator -------------------------------------------------------
for k in range(node_z_num):
    for i in range(half_x):
        if i2[k][i] != 0:
            ii[k][i] = (i1[k][i] + i3[k][i] - i2[k][i]) / i2[k][i] * 100.0


# -- Non-oscillatory (decoupled) SIF -------------------------------------------
# For k1+k2: K3r is the decoupled mode -- from g3
# For k1+k3: K2r is the decoupled mode -- from g2
for k in range(node_z_num):
    for i in range(half_x):
        if mode_mixity == 'k1+k2':
            sign = -1.0 if g3[k][i] < 0.0 else 1.0
            K_non[k][i] = sqrt(abs(g3[k][i]) * H2) * sign
        else:
            sign = -1.0 if g2[k][i] < 0.0 else 1.0
            K_non[k][i] = sqrt(abs(g2[k][i]) * H2) * sign


# -- Mode mixity angle psi and oscillatory SIFs K1 + K_osc -------------------
for k in range(node_z_num):
    for i in range(half_x):
        if mode_mixity == 'k1+k2':
            # psi from (g1 - g2)
            psi_D = math.atan2(i_tot123[k][i], (g1[k][i] - g2[k][i]))
        else:
            # psi from (g1 - g3)
            psi_D = math.atan2(i_tot123[k][i], (g1[k][i] - g3[k][i]))

        psi[k][i] = 0.5 * psi_D - 0.5 * math.atan2(Pi, Pr) - eps * log(da[k][i])

        if k == 19:
            print "%.21e\t%.21e\t" % (psi_D, psi[k][i])

        sign = -1.0 if gtot[k][i] < 0.0 else 1.0
        K1[k][i]     = (abs(gtot[k][i] * H1))**0.5 * cos(psi[k][i]) * sign
        K_osc[k][i]  = (abs(gtot[k][i] * H1))**0.5 * sin(psi[k][i]) * sign

        if mode_mixity == 'k1+k2':
            K2[k][i] = K_osc[k][i]
            K3[k][i] = K_non[k][i]
        else:
            K3[k][i] = K_osc[k][i]
            K2[k][i] = K_non[k][i]


# -- Normalised element number -------------------------------------------------
for k in range(node_z_num):
    for i in range(half_x):
        E_el[k][i] = da[k][i] / da[k][0]


# ==============================================================================
# SECTION 13: OUTPUT -- WRITE vcct_results_<model_name>.txt
# ==============================================================================
# Columns: da | N/2 | K1 | K2 | K3 | G1 | G2 | G3 | Gi | i1 | i2 | i3 | ii | du1 | du2 | du3
# K2 and K3 semantics depend on mode_mixity (see header line written below).

file_name = 'vcct_results_' + model_name + '.txt'
with open(file_name, 'w') as out_file:
    if mode_mixity == 'k1+k2':
        out_file.write('MODE_MIXITY: K1+iK2  (K3 is decoupled non-oscillatory)\n')
        out_file.write('da - N/2 - K1 - K2(osc) - K3(non-osc) - G1 - G2 - G3 - Gi - i1 - i2 - i3 - ii - du1 - du2 - du3\n')
    else:
        out_file.write('MODE_MIXITY: K1+iK3  (K2 is decoupled non-oscillatory)\n')
        out_file.write('da - N/2 - K1 - K2(non-osc) - K3(osc) - G1 - G2 - G3 - Gi - i1 - i2 - i3 - ii - du1 - du2 - du3\n')

    for k in range(node_z_num):
        if k % 2 == 0:
            out_file.write('{}\n'.format(0.5 * k))
            for i in range(half_x):
                out_file.write(
                    '{:.21e}\t{:.21e}\t{:.21e}\t{:.21e}\t'
                    '{:.21e}\t{:.21e}\t{:.21e}\t{:.21e}\t'
                    '{:.21e}\t{:.21e}\t{:.21e}\t{:.21e}\t'
                    '{:.21e}\t{:.21e}\t{:.21e}\t{:.21e}\n'.format(
                        da[k][i],       E_el[k][i],
                        K1[k][i],       K2[k][i],       K3[k][i],
                        g1[k][i],       g2[k][i],       g3[k][i],
                        gtot123[k][i],  i1[k][i],
                        i2[k][i],       i3[k][i],       ii[k][i],
                        dis_x[k][i],    dis_y[k][i],    dis_z[k][i]
                    )
                )
            out_file.write('\n' * 5)


# ==============================================================================
# SECTION 14: PRINTED SUMMARY -- SELECTED Z-POSITIONS AND CHOSEN K
# ==============================================================================




print '\n\n\nmodel_name=\'', model_name, '\';'
print 'mode_mixity=\'', mode_mixity, '\';'
