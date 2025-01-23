from collections import defaultdict
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

crit_T = { 		#in K
	'water':647.3,
	'carbondioxide':304.2,
    'ethane':305.4,
    'n-butane':425.2,
    'acetone':508.0
}
crit_T = defaultdict(lambda: 0.0, crit_T)

crit_P = { 		#in Pa
	'water':22048000,
	'carbondioxide':7376000,
    'ethane':4884000,
    'n-butane':3800000,
    'acetone':4800000 
}
crit_P = defaultdict(lambda: 0.0, crit_P)

molecular_weight = { 		#g/mol
	'water':18.015,
	'carbondioxide':44.01,
    'ethane':30.070,
    'n-butane':58.124,
    'acetone':58.08
}
molecular_weight = defaultdict(lambda: 0.0, molecular_weight)

AntoineConstant_A = { 		
	'water':11.6834,
    'acetone':10.0311
}
AntoineConstant_A = defaultdict(lambda: 0.0, AntoineConstant_A)

AntoineConstant_B = { 	
	'water':3816.44,
    'acetone':2940.46
}
AntoineConstant_B = defaultdict(lambda: 0.0, AntoineConstant_B)

AntoineConstant_C = { 		
	'water':-46.13,
    'acetone':-35.93
}
AntoineConstant_C = defaultdict(lambda: 0.0, AntoineConstant_C)

VdW_EOS_Parameters = { 		
	'kij':0.000    #unitless
}

VdW_EOS_Parameters = defaultdict(lambda: 0.0, VdW_EOS_Parameters) 


def calc_AntoineEqn_P_vap(T, name): #T in K

    A = AntoineConstant_A[name]
    B = AntoineConstant_B[name] 
    C = AntoineConstant_C[name]

    #print(A,B,C)

    Pvap = math.exp(A-B/(T+C)) # Pvap in bars

    return Pvap


def calc_NRTL_liquid_activity_coeff(T, liquid_comp, names):
    
    x1 = liquid_comp[0]
    x2 = liquid_comp[1]

    R = 0.008314  # kJ/(mol·K)

    # Optimized NRTL parameters for the acetone-water mixture
    c1 = 0.8598
    c2 = 1.2595
    c3 = 1.7892
    c4 = -6.3648
    c5 = -1.9831
    c6 = 0.0502

    alpha12 = c5+c6/(T-273.15)
    alpha21 = c5+c6/(T-273.15)

    
    tau12 = (c3+c4/(T-273.15))/(R * T)
    tau21 = (c1+c2/(T-273.15))/(R * T)
    G12 = math.exp(-alpha12 * tau12)
    G21 = math.exp(-alpha21 * tau21)

    # Activity coefficient calculation
    Gamma1 = np.exp(x2*x2*(tau21*(G21/(x1+x2*G21))*(G21/(x1+x2*G21))+tau12*(G12/((x2+x1*G12)*(x2+x1*G12)))))
    Gamma2 = np.exp(x1*x1*(tau12*(G12/(x2+x1*G12))*(G12/(x2+x1*G12))+tau21*(G21/((x1+x2*G21)*(x1+x2*G21)))))

    #print(Gamma1, Gamma2)
    return Gamma1, Gamma2


def calc_vapor_mixture_fugacities(T, P, vapor_comp, names):

    R = 8.314 #in J/molK 

    Tc1 = crit_T[names[0]]
    Pc1 = crit_P[names[0]] 

    Tc2 = crit_T[names[1]]
    Pc2 = crit_P[names[1]] 

    #print(Tc1, Tc2, Pc1, Pc2)

    b1 = 0.125*R*Tc1/Pc1
    b2 = 0.125*R*Tc2/Pc2

    a1 = 27*(R*R*Tc1*Tc1/Pc1)/64
    a2 = 27*(R*R*Tc2*Tc2/Pc2)/64

    A1 = a1*P/(R*R*T*T)
    A2 = a2*P/(R*R*T*T)

    B1 = b1*P/(R*T)
    B2 = b2*P/(R*T)

    y1 = vapor_comp[0]
    y2 = vapor_comp[1]
    
    b_mix = y1*b1 + y2*b2

    k12 = VdW_EOS_Parameters['kij']
    a12 = (1-k12)*math.sqrt(a1*a2)

    #print(k12)

    a_mix = y1*y1*a1+y2*y2*a2+2*y1*y2*a12

    #print(a1, a2, a12, a_mix)

    A_mix = a_mix*P/(R*R*T*T)
    B_mix = b_mix*P/(R*T)

    p = -1-B_mix
    q = A_mix
    r = -A_mix*B_mix
    
    eqn=[1,p,q,r]
    roots=np.roots(eqn) #z=Root(z**3+p*z**2+q*z+r1==0)
    z=np.real(roots[np.isreal(roots)])	
    #return z
    if len(z) == 1:
        zV = z
    else:
        zV = max(z)

    sum_y1_a12 = y1*a1+y2*a12
    sum_y1_A12 = sum_y1_a12*P/(R*R*T*T)

    sum_y2_a12 = y1*a12+y2*a2
    sum_y2_A12 = sum_y2_a12*P/(R*R*T*T)


    lnphi1V = B1/(zV-B_mix)-math.log(abs(zV-B_mix))-2*sum_y1_A12/zV		
    f1V = y1*P*math.exp(lnphi1V)    
    
    lnphi2V = B2/(zV-B_mix)-math.log(abs(zV-B_mix))-2*sum_y2_A12/zV		
    f2V = y2*P*math.exp(lnphi2V)  
    #print(f1V, f2V)
    return f1V/100000, f2V/100000   #converting fugacity units from Pa to bar


def calc_binary_mixture_points(names):

    T = [298.15, 323.15, 373.15, 423.15, 523.15] #in K

    print('Temperature [K] Pressure [bar] x1 y1\n')

    x1 = np.arange(0.000, 1.001, 0.020)
    
    P = np.zeros((len(T),len(x1)))
    #y2 = np.zeros((len(T),len(x1)))
    y1 = np.zeros((len(T),len(x1)))
    i = 0
    while i < len(T):

        P_vap = [0, 0]
        j = 0

        while j < len(P_vap):

            sat_points = calc_AntoineEqn_P_vap(T[i], names[j])
            P_vap[j] = sat_points

            j = j + 1
        #print(P_vap)
        #return 5

        k = 0
        #liquid_comp = [0,0]
        #liquid_gamma = [0,0]
        while k < len(x1):
            if x1[k] == 1:
                P[i][k] = P_vap[0]
                y1[i][k] = 1 
                print(T[i], P[i][k], x1[k], y1[i][k])
                k = k + 1
                continue
            elif x1[k] == 0:
                P[i][k] = P_vap[1]
                y1[i][k] = 0 
                print(T[i], P[i][k], x1[k], y1[i][k])
                k = k + 1
                continue
            else:
                liquid_comp = [x1[k], 1-x1[k]] #the first component is acetone, and the second one is water
                liquid_gamma = calc_NRTL_liquid_activity_coeff(T[i], liquid_comp, names)
                
                f1L = liquid_comp[0]*liquid_gamma[0]*P_vap[0]
                f2L = liquid_comp[1]*liquid_gamma[1]*P_vap[1]
                #print(liquid_gamma)
                #k = k+1
                #continue
                if T[i] < 355:
                    convergence_fuga = 1e-03
                    fuga_change_ratio = 1.0
                    PL = 0.0*100000 #in Pa
                    PH = 10*100000 #in Pa
                elif T[i] < 425:
                    convergence_fuga = 1e-02
                    fuga_change_ratio = 1.0
                    PL = 0.0*100000 #in Pa
                    PH = 20*100000 #in Pa
                else:
                    convergence_fuga = 1e-02
                    fuga_change_ratio = 1.0
                    PL = 40*100000 #in Pa
                    PH = 540*100000 #in Pa
                while (fuga_change_ratio > convergence_fuga):

                    P_guess = (PL+PH)/2

                    convergence_fuga2 = 1e-03
                    fuga_change_ratio2 = 1.0
                    y2L = 0.0000
                    y2H = 1.0000
                    while (fuga_change_ratio2 > convergence_fuga2):
                    
                        y2 = (y2L+y2H)/2

                        vapor_comp = [1-y2, y2]
                        fV = calc_vapor_mixture_fugacities(T[i], P_guess, vapor_comp, names)

                        f1V = fV[0]
                        f2V = fV[1]
                        #print(P, f1V, f2V)
                        fuga_change_ratio2 = abs((f2V-f2L)/f2V)
                        #print(y2, f2L, f2V)
                        if f2L < f2V:
                            y2H = y2
                        else:
                            y2L = y2
                    
                    fuga_change_ratio = abs((f1V-f1L)/f1V)
                    #print(P, f1L, f1V, fuga_change_ratio)
                    if f1L < f1V:
                        PH = P_guess
                    else:
                        PL = P_guess
                
            y1[i][k] = 1-y2
            P[i][k] = P_guess/100000  #to convert from Pa into bars
            print(T[i], P[i][k], x1[k], y1[i][k])

            k = k + 1

        i = i + 1
        
    return T, P, x1, y1

def plot_P_x_y_diagram(T, P, x1, y1, P_exp, y_exp, kij_label):

    plt.figure(figsize=(10, 8))
    
    for i, temp in enumerate(T):
        plt.subplot(3, 2, i + 1)
        
        # Exclude P values at x1 = y1 = 1 for temperatures 100°C, 150°C, and 250°C
        if temp in [373.15, 423.15, 523.15]:  # These are 100°C, 150°C, and 250°C in Kelvin
            mask = (x1 != 1) | (y1[i] != 1)  # Exclude x1 = y1 = 1 points
            plt.plot(x1[mask], P[i, :][mask], label=f'Predicted P at {temp-273.15:.2f}°C', color='blue')
            plt.plot(y1[i, :][mask], P[i, :][mask], label=f'Predicted P at {temp-273.15:.2f}°C (y1)', color='green')
        else:
            plt.plot(x1, P[i, :], label=f'Predicted P at {temp-273.15:.2f}°C', color='blue')
            plt.plot(y1[i, :], P[i, :], label=f'Predicted P at {temp-273.15:.2f}°C (y1)', color='green')

        # Plot experimental data for 298.15 K
        if temp == 298.15:
            plt.plot(x1, P_exp, 'ro', label='Experimental at 298.15 K')
            plt.plot(y_exp, P_exp, 'o', color='purple', label='Experimental at 298.15 K (y1)')

        plt.title(f'T = {temp-273.15:.2f}°C, kij = {kij_label}', fontsize=10)

        plt.xlabel('x1 (acetone)')
        plt.ylabel('Pressure (P) [bars]')
        plt.grid(True)
        
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    plt.show()

print('//Part (a) //')
print('\n')
print('**BINARY MIXTURE POINTS START** \n')
print(' kij=0 \n')
binary_mixture_points_kij_zero = calc_binary_mixture_points(names=['acetone', 'water'])
#print(binary_mixture_points_kij_zero[1][0])

#Experimental data at 298.15 K
P_exp = [0.0317, 0.0878, 0.1252, 0.1508, 0.1687, 0.1815, 0.1910, 0.1982, 0.2038, 0.2084, 0.2122, 0.2156, 0.2186, 0.2213, 0.2240, 0.2265, 0.2289, 0.2314, 0.2338, 0.2362, 0.2386, 0.2410, 0.2435, 0.2459, 0.2484, 0.2508, 0.2533, 0.2557, 0.2581, 0.2606, 0.2630, 0.2653, 0.2677, 0.2701, 0.2724, 0.2746, 0.2769, 0.2791, 0.2813, 0.2835, 0.2856, 0.2877, 0.2899, 0.2920, 0.2941, 0.2962, 0.2982, 0.3003, 0.3024, 0.3043, 0.3059]  # in bars
y_exp = [0.000, 0.645, 0.755, 0.799, 0.822, 0.836, 0.845, 0.852, 0.857, 0.861, 0.864, 0.867, 0.869, 0.871, 0.873, 0.876, 0.878, 0.880, 0.882, 0.884, 0.886, 0.888, 0.890, 0.892, 0.895, 0.897, 0.899, 0.901, 0.904, 0.906, 0.908, 0.911, 0.913, 0.916, 0.919, 0.921, 0.924, 0.927, 0.930, 0.933, 0.936, 0.939, 0.943, 0.947, 0.951, 0.956, 0.962, 0.968, 0.977, 0.987, 1.000]  # Experimental y-values at 298.15 K

#Call the plotting function for kij = 0
plot_P_x_y_diagram(binary_mixture_points_kij_zero[0], binary_mixture_points_kij_zero[1], binary_mixture_points_kij_zero[2], binary_mixture_points_kij_zero[3], P_exp, y_exp, kij_label="0")


print('\n**BINARY MIXTURE CURVE POINTS END** \n')
print('\n')

print('//End of Part (a)//')
print('\n')
print('//Part (b) //')
print('\n')
print('**kij Optimization Results Start** \n')

def F_obj(k_ij):
    VdW_EOS_Parameters['kij'] = k_ij
    print(VdW_EOS_Parameters['kij'])

    # Exp. data given at 25 C   

    x1_exp = np.arange(0.000, 1.001, 0.020)
    
    names = ['acetone', 'water']
    T_exp = 298.15 # in K
    
    P_exp = [0.0317, 0.0878, 0.1252, 0.1508, 0.1687, 0.1815, 0.1910, 0.1982, 0.2038, 0.2084, 0.2122, 0.2156, 0.2186, 0.2213, 0.2240, 0.2265, 0.2289, 0.2314, 0.2338, 0.2362, 0.2386, 0.2410, 0.2435, 0.2459, 0.2484, 0.2508, 0.2533, 0.2557, 0.2581, 0.2606, 0.2630, 0.2653, 0.2677, 0.2701, 0.2724, 0.2746, 0.2769, 0.2791, 0.2813, 0.2835, 0.2856, 0.2877, 0.2899, 0.2920, 0.2941, 0.2962, 0.2982, 0.3003, 0.3024, 0.3043, 0.3059] #in bars
    #return len(P_exp)
    P_predicted = np.zeros(len(x1_exp))
    #return x_predicted
    F_sum = 0
    count = 0
    i = 0
    binary_mixture_points_kij_opt = calc_binary_mixture_points(names)[1][0] #it returns the eqm P values at 25 C
    for P in P_exp:
        P_predicted[i] = binary_mixture_points_kij_opt[i]
        #print('P_predicted = ', P_predicted[i], T_exp, 'K', P, 'Pa')
        F_sum += 100*abs(P - P_predicted[i])/P
        count += 1
        i += 1
    #print(count)
    F = F_sum/(count)
    print (F, VdW_EOS_Parameters['kij']) 
    return F

def Optimization_Results():
	kij = -0.43
	#return F_obj(VdW_EOS_Parameters['kij'])
	return minimize(F_obj, kij, method='Nelder-Mead')

parameter_opt_results = Optimization_Results()
print(parameter_opt_results)

print('**kij Optimization Results End** \n')

print('**BINARY MIXTURE POINTS START** \n')
VdW_EOS_Parameters['kij'] = parameter_opt_results.x
kij_value = VdW_EOS_Parameters['kij'] 
print(f' kij={kij_value} \n')
binary_mixture_points_kij_optimized = calc_binary_mixture_points(names=['acetone', 'water'])
print('\n**BINARY MIXTURE CURVE POINTS END** \n')
print('\n//End of Part (b)//')


plot_P_x_y_diagram(binary_mixture_points_kij_optimized[0], binary_mixture_points_kij_optimized[1], binary_mixture_points_kij_optimized[2], binary_mixture_points_kij_optimized[3], P_exp, y_exp, kij_label=f"{VdW_EOS_Parameters['kij'][0]:.2f}")




