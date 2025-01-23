import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

# Exp. Data: x1 vs. ∆H1mix (kJ/mol)
x1_exp = np.array([0.030, 0.101, 0.146, 0.196, 0.307, 0.405, 0.504, 0.604, 
                    0.709, 0.803, 0.902, 0.973])
deltaHmix_exp = np.array([-0.24, -0.62, -0.67, -0.61, -0.44, -0.3, -0.13, 
                          0.10, 0.25, 0.28, 0.22, 0.11])  

T = 25+273.15

# NRTL model function for ∆H_mix
def deltaHmix_NRTL(x1, T, parameters):
    
    x2 = 1 - x1
    R = 0.008314  # kJ/(mol·K)

    # NRTL parameters
    c1, c2, c3, c4, c5, c6 = parameters
    #c1 and c3 are in kJ/mol, c2 and c4 are in kJ/molK, c5 is unitless, c6 is in K

    alpha12 = c5+c6/(T-273.15)
    alpha21 = c5+c6/(T-273.15)

    
    tau12 = (c3+c4/(T-273.15))/(R * T)
    tau21 = (c1+c2/(T-273.15))/(R * T)
    G12 = math.exp(-alpha12 * tau12)
    G21 = math.exp(-alpha21 * tau21)

    T_prime = (T-273.15)**2

    # ∆Hmix calculation
    deltaHmix = (T*x1*x2*G21/(T_prime*(x1+x2*G21))) * ((1-x1*alpha12*tau21/(x1+x2*G21))*(c2+R*T_prime*tau21)-(x1*T*tau21*tau21*c6*R)/(x1+x2*G21))+(T*x1*x2*G12/(T_prime*(x2+x1*G12))) * ((1-x2*alpha12*tau12/(x2+x1*G12))*(c4+R*T_prime*tau12)-(x2*T*tau12*tau12*c6*R)/(x2+x1*G12))
    return deltaHmix  # Convert J/mol to kJ/mol


def objective(params, x1_exp, T, deltaHmix_exp, w=100, N=1):
    """
    Objective function to minimize the weighted normalized squared difference between experimental and calculated ∆Hmix.

    Parameters:
        params : [c1, c2, c3, c4, c5, c6] 
        T : 298 K
        w: Weighting factor.
        N: Number of isothermal systems. Equal to 1 since we only exp. data at T=25 C

    Output:
        Objective function value F.
    """
    n = len(x1_exp)  

    deltaHmix_calc = np.array([deltaHmix_NRTL(x1, T, params) for x1 in x1_exp])

    relative_squared_errors = (w * (deltaHmix_calc - deltaHmix_exp) / deltaHmix_exp) ** 2

    F = (1 / N) * (np.sum(relative_squared_errors) / n)

    print(f"Objective function value F: {F}")

    return F


def Optimization_Results():
	# Initial guesses for parameters [c1, c2, c3, c4, c5, c6]
    initial_guess = [0.4, 2, 1.1, -5, -0.5, -0.1]  
    return minimize(objective, initial_guess, args=(x1_exp, T, deltaHmix_exp), method='Nelder-Mead', options={'maxiter': 100000})

print(Optimization_Results())

result = Optimization_Results()
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

deltaHmix_calc = np.array([deltaHmix_NRTL(x1, T, optimized_params) for x1 in x1_exp])

plt.figure(figsize=(8, 6))
plt.plot(x1_exp, deltaHmix_exp, 'o', label='Experimental ∆Hmix', color='red')
plt.plot(x1_exp, deltaHmix_calc, '-', label='Calculated ∆Hmix', color='blue')
plt.xlabel('x1 (Mole Fraction)', fontsize=12)
plt.ylabel('∆Hmix (kJ/mol)', fontsize=12)
plt.title('Experimental vs. Calculated ∆Hmix', fontsize=14)
plt.legend()
plt.grid()
plt.show()
