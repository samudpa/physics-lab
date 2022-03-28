import numpy as np

def get_expected_omega0():
    '''Returns the expected value of the
    angular frequency of the free pendulum'''
    return omega0, omega0_err

# data

l_rod     = 49.8e-2
s_rod     = 0.795e-2
l_rod_err = 0.1e-2
s_rod_err = 0.005e-2

r_disk     = 6.975e-2/2
s_disk     = 1.120e-2
r_disk_err = 0.005e-2/2
s_disk_err = 0.005e-2

# densities

density_alu = 2700 # https://it.wikipedia.org/wiki/Alluminio
density_brass = 8550 # https://it.wikipedia.org/wiki/Ottone_(lega)
density_alu_err = 100
density_brass_err = 100

# volumes

V_disk     = np.pi * s_disk * r_disk**2
V_disk_err = V_disk * np.sqrt((r_disk_err/r_disk)**2 + 2*(s_disk_err/s_disk)**2)

V_rod     = l_rod * s_rod**2
V_rod_err = V_rod * np.sqrt((l_rod_err/l_rod)**2 + 2*(s_rod_err/s_rod)**2)

# masses

m_rod = V_rod * density_alu
m_disk = V_disk * density_brass
m_rod_err = m_rod * np.sqrt(
    (V_rod_err/V_rod)**2 + (density_alu_err/density_alu)**2
)
m_disk_err = m_disk * np.sqrt(
    (V_disk_err/V_disk)**2 + (density_brass_err/density_brass)**2
)

# rod's moment of inertia

I_rod = 1/3 * m_rod * l_rod**2
I_rod_err = I_rod * np.sqrt(
    (m_rod_err/m_rod)**2 + 2*(l_rod_err/l_rod)**2
)

# disk moment of inertia

I_disk_c = 1/2 * m_disk * r_disk**2
I_disk_c_err = I_disk_c * np.sqrt(
    (m_disk_err/m_disk)**2 + 2*(r_disk_err/r_disk)**2
)

I_disk_corr = m_disk * (l_rod + r_disk)**2
I_disk_corr_err = I_disk_corr * np.sqrt(
    (m_disk_err/m_disk)**2 + (l_rod_err**2 + r_disk_err**2)/((l_rod+r_disk)**2)
)

I_disk = I_disk_c + I_disk_corr
I_disk_err = np.sqrt(I_disk_c_err**2 + I_disk_corr_err**2)

I = I_disk + I_rod
I_err = np.sqrt(I_disk_err**2 + I_rod_err**2)

# find omega0

g0 = 9.81
g0_err = 0.01

m = m_rod + m_disk
m_err = np.sqrt(m_rod_err**2 + m_disk_err**2)

# calculate center of mass distance (d)

A = m_rod * l_rod/2
A_err = A * np.sqrt((m_rod_err/m_rod)**2 + (l_rod_err/l_rod)**2)
B = m_disk * (l_rod + r_disk)
B_err = B * np.sqrt((m_disk_err/m_disk)**2 + (l_rod_err**2 + r_disk_err**2)/(l_rod + r_disk)**2)

d_num = A + B
d_num_err = np.sqrt(A_err**2 + B_err**2)

d_den = m_rod + m_disk
d_den_err = np.sqrt(m_rod_err**2 + m_disk_err**2)

d = d_num / d_den # center of mass
d_err = d * np.sqrt((d_num_err/d_num)**2 + (d_den_err/d_den)**2)
print(f'\nd = {d} ± {d_err:.2g} m')

# calculate omega 0

omega0_sq = m * g0 * d / I
omega0_sq_err = omega0_sq * np.sqrt(
    (m_err/m)**2 + (g0_err/g0)**2 + (d_err/d)**2 + (I_err/I)**2
)

omega0 = np.sqrt(omega0_sq)
omega0_err = omega0_sq_err * 0.5 * np.sqrt(1/omega0_sq)