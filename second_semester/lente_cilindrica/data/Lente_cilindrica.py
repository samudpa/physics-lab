import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# load data
data = pd.read_csv("data_1.csv")

q = data['lens_dist']/100   # tutto in metri
q_err = 0.004
muro_luce = data['image_dist']/100
muro_luce_err = 0.004
p = muro_luce - q
p_err = np.sqrt(q_err**2 + muro_luce_err**2)







#calcolo inverso lunghezza focale atteso
r = 0.044 # m raggio bottiglia
n = 1.33 # indice rifrazione acqua
sigma_r = 0.001
sigma_n = 0.01
f_inv_0 = 2*(n-1)*(1 - (n - 1)/n)/r
sigma_f_inv_0 = np.sqrt(((2*(n-1)*(1 - (n - 1)/n)/r**2)**2)*sigma_r**2 + ((2/(r*n**2))**2)*sigma_n**2)
print( f_inv_0, sigma_f_inv_0)

#inversi per fittare la retta y = mx + q
x_data = 1/q
x_err = q_err/(q**2)
y_data = 1/p
y_err = p_err/(p**2)


#fit model
def line(x, m, k):

     return m * x + k

# calcolo fit

popt, pcov = curve_fit(line, x_data, y_data, p0 = [-1,11], sigma=y_err)

# se l'errore sula x non dovesse essere trascurabile
for i in range(0, 5):
     sigma_eff = np.sqrt(y_err**2+ (popt[0]*x_err)**2)
     popt, pcov = curve_fit(line, x_data, y_data, sigma = y_err)
     Chi2 = (((y_data-line(x_data, *popt ))/sigma_eff)**2).sum()
     print(popt, np.sqrt(pcov.diagonal()))
     print(f'Step {i}...')
     print(f'Chisquare = {Chi2:.2f}')




#plot
gs = plt.GridSpec(2,1, height_ratios=[3,1])
plt.subplots_adjust(hspace=0)
plt.subplot(gs[0])
plt.errorbar(x_data, y_data, xerr = x_err, yerr = y_err, fmt= '.')
plt.plot(x_data, line(x_data, *popt))
plt.grid()
plt.title( 'Grafico di dispersione di 1/q vs. 1/p')
plt.ylabel('1/p [m^-1]')


#plot residui
res= y_data-line(x_data, *popt)
plt.subplot(gs[1],  )
plt.errorbar(x_data, res, yerr = y_err, fmt = '.')
plt.axhline(0)
plt.grid()
plt.xlabel('1/q [m^-1]' )

plt.show()

#errore chiquadro
dof = len(x_data) - 2
Chi2_err = np.sqrt(2*dof)

print(Chi2/dof, Chi2_err/dof )




