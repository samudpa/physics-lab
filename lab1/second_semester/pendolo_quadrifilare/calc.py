import numpy as np
import json

def get_cm_distance():
    '''Returns the value of l and its error'''
    return l, l_err

def get_moment_of_inertia():
    '''Returns the value of I and its error'''
    return I, I_err

def get_measures():
    '''Returns data from measures.json'''
    return data

def get_total_mass():
    '''Returns the value of M and its error'''
    return M, M_err

with open('data/measures.json', 'r') as f:
    data = json.loads(f.read())

# finding the volumes
V = {}
V_err = {}
for mat in ['wood', 'lead', 'alum']:
    V[mat] = data[mat][0]['x'] * data[mat][0]['y'] * data[mat][0]['z'] / 1e6
    V_err[mat] = V[mat] * np.sqrt(
        (data[mat][0]['x_err']/data[mat][0]['x'])**2 +
        (data[mat][0]['y_err']/data[mat][0]['y'])**2 +
        (data[mat][0]['z_err']/data[mat][0]['z'])**2
    )

# density values from wikipedia:
#   https://it.wikipedia.org/wiki/Legno
#   https://it.wikipedia.org/wiki/Piombo
#   https://it.wikipedia.org/wiki/Alluminio
density = {
    'wood': np.mean([310, 980]),
    'lead': 11340,
    'alum': 2700
}
density_err = {
    'wood': (980-310)/np.sqrt(12),
    'lead': density['lead']/100,
    'alum': density['alum']/10
}

# finding the masses
m = {}
m_err = {}
for mat in ['wood', 'lead', 'alum']:
    m[mat] = V[mat] * density[mat]
    m_err[mat] = m[mat] * np.sqrt(
        (V_err[mat]/V[mat])**2 +
        (density_err[mat]/density[mat])**2
    )

# total mass of the pendulum
M = m['wood'] + m['lead'] + m['alum']
M_err = np.sqrt(m_err['wood']**2 + m_err['lead']**2 + m_err['alum']**2)

# positions of the parts of the pendulum
z_lead = data['wood'][0]['z']/2 + data['lead'][0]['z']/2
z_alum = data['wood'][0]['z']/2 + data['alum'][0]['z']/2 - data['alum'][0]['offset']
y_alum = data['wood'][0]['y']/2 + data['alum'][0]['y']/2
z_lead_err = np.sqrt(
    (data['wood'][0]['z_err']/2)**2 +
    (data['lead'][0]['z_err']/2)**2
)
z_alum_err = np.sqrt(
    (data['wood'][0]['z_err']/2)**2 +
    (data['alum'][0]['z_err']/2)**2 +
    (data['alum'][0]['offset_err']/2)**2
)
y_alum_err = np.sqrt(
    (data['wood'][0]['y_err']/2)**2 +
    (data['alum'][0]['y_err']/2)**2
)

# center of mass position (z)
A = z_lead * m['lead']
A_err = A * np.sqrt(
    (z_lead_err/z_lead)**2 +
    (m_err['lead']/m['lead'])**2
)
B = z_alum * m['alum']
B_err = A * np.sqrt(
    (z_alum_err/z_alum)**2 +
    (m_err['alum']/m['alum'])**2
)
AB = A + B
AB_err = np.sqrt(A_err**2 + B_err**2)
z_cm = AB / M
z_cm_err = z_cm * np.sqrt(
    (AB_err/AB)**2 +
    (M_err/M)**2
)

# center of mass position (y)
y_cm = y_alum * m['alum'] / M
y_cm_err = y_cm * np.sqrt(
    (y_alum_err/y_alum)**2 +    
    (m_err['alum']/m['alum'])**2 +    
    (M_err/M)**2    
)

# axis of rotation distance from the center of mass of the pendulum
l = (data['d_wood'] + data['wood'][0]['z']/2 + z_cm)/100
l_err = np.sqrt(data['d_wood_err']**2 + (data['wood'][0]['z_err']/2)**2 + z_cm_err**2)/100

# moments of inertia of the parts of the pendulum (the alumium part is negligible)

# moment of inertia (wood)
Rsq = (data['wood'][0]['x']/100)**2 + (data['wood'][0]['z']/100)**2
Rsq_err = np.sqrt(
    2 * (data['wood'][0]['x']/100)**2 +
    2 * (data['wood'][0]['z']/100)**2
)
I_wood = 1/12 * m['wood'] * Rsq
I_wood_err = I_wood * np.sqrt((m_err['wood']/m['wood'])**2 + (Rsq_err/Rsq)**2)

# moment of inertia (lead)
Rsq = (data['lead'][0]['x']/100)**2 + (data['lead'][0]['z']/100)**2
Rsq_err = np.sqrt(
    2 * (data['lead'][0]['x']/100)**2 +
    2 * (data['lead'][0]['z']/100)**2
)
I_lead = 1/12 * m['lead'] * ((data['lead'][0]['x']/100)**2 + (data['lead'][0]['z']/100)**2)
I_lead_err = I_lead * np.sqrt((m_err['lead']/m['lead'])**2 + (Rsq_err/Rsq)**2)

# moments of inertia in respect to the center of mass
Icm_wood = I_wood + m['wood'] * (z_cm/100)**2 # Hyugens-Steiner
Icm_lead = I_lead + m['lead'] * ((z_cm - z_lead)/100)**2
Icm_wood_err = np.sqrt(
    I_wood_err**2 +
    (m['wood'] * (z_cm/100)**2)**2 * (
        (m_err['wood']/m['wood'])**2 + 
        2 * (z_cm_err/z_cm)**2
    )
)
Icm_lead_err = np.sqrt(
    I_lead_err**2 +
    (m['lead'] * ((z_cm - z_lead)/100)**2)**2 * (
        (m_err['lead']/m['lead'])**2 + 
        2 * (z_cm_err**2 + z_lead_err**2)/((z_cm - z_lead)**2)
    )
)

I = Icm_wood + Icm_lead + M * l**2
I_err = np.sqrt(Icm_wood_err**2 + Icm_lead_err**2 + l_err**2)