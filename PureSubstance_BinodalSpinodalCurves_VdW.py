from collections import defaultdict
import math
import numpy as np

crit_T = { 		#in K
	'water':647.3,
	'carbondioxide':304.2,
    'ethane':305.4,
    'n-butane':425.2 
}
crit_T = defaultdict(lambda: 0.0, crit_T)

crit_P = { 		#in Pa
	'water':22048000,
	'carbondioxide':7376000,
    'ethane':4884000,
    'n-butane':3800000 
}
crit_P = defaultdict(lambda: 0.0, crit_P)

molecular_weight = { 		#g/mol
	'water':18.015,
	'carbondioxide':44.01,
    'ethane':30.070,
    'n-butane':58.124
}
molecular_weight = defaultdict(lambda: 0.0, molecular_weight)

def calc_sat_P_sat_Vs(T, name): #T in K

    P = 1000000 #in Pa 
    R = 8.314 #in J/molK 
    Tc = crit_T[name]
    Pc = crit_P[name]   
    #print(Tc, Pc)
    
    b = 0.125*R*Tc/Pc
    a=27*(R*R*Tc*Tc/Pc)/64
    #print(b, a)

    convergence_criteria = 1e-06
    fuga_change_ratio = 1
    PL = 0  #in Pa
    PH = 25000000 #in Pa
    while(fuga_change_ratio > convergence_criteria):
        P = (PL+PH)/2

        A=a*P/(R*R*T*T)
        B=b*P/(R*T)

        p = -1-B
        q = A
        r = -A*B
    
        eqn=[1,p,q,r]
        roots=np.roots(eqn) #z=Root(z**3+p*z**2+q*z+r1==0)
        z=np.real(roots[np.isreal(roots)])	
        #return z
        if len(z) == 1:
            zL = z
            zV = z
            rho = P/(z*R*T)*molecular_weight[name]*0.001
            if rho > 150: #in kg/m3    #a limit is necessary to be able to define a new pressure guess
                PH=P
                continue
            else:
                PL=P
                continue
        else:
            zL = min(z) 
            zV = max(z) 
    
        
        lnphiL=zL-1-math.log(abs(zL-B))-A/zL	
        fL=P*math.exp(lnphiL)

        lnphiV=zV-1-math.log(abs(zV-B))-A/zV		
        fV=P*math.exp(lnphiV)
        
        fuga_change_ratio = abs((fV-fL)/fL)
        #print(P, fL, fV)
        if fL < fV:
            PH = P
        else:
            PL = P
        #print(T, fL, fV)
    VL = zL*R*T/P
    VV = zV*R*T/P
    return P, VL, VV

def calc_binodal_points(name):

    print('Temperature [K] Saturation Pressure [bar] Saturated Liquid Molar Volume [cm3/mol] Saturated Vapor Molar Volume [cm3/mol]\n', file=f)

    #T is in K
    T = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 275, 280, 285, 290, 295, 300, 301, 302, 303, 304, 305, 305.1, 305.2, 305.3, 305.35, 305.38]

    P_sat = np.zeros(len(T))
    V_sat_L = np.zeros(len(T))
    V_sat_V = np.zeros(len(T))
    i=0
    while i < len(T):
        P_sat[i] = calc_sat_P_sat_Vs(T[i], name)[0]
        V_sat_L[i] = calc_sat_P_sat_Vs(T[i], name)[1]
        V_sat_V[i] = calc_sat_P_sat_Vs(T[i], name)[2]
        print(T[i], P_sat[i]/100000, V_sat_L[i]*1000000, V_sat_V[i]*1000000, file=f)
        i += 1
    return T, P_sat, V_sat_L, V_sat_V

def calc_spinodal_points(name):

    print('Temperature [K] Minimum Pressure [bar] Minimum Molar Volume [cm3/mol] Maximum Pressure [bar] Maximum Molar Volume [cm3/mol] \n', file=f)

    #T is in K
    T = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 275, 280, 285, 290, 295, 300, 301, 302, 303, 304, 305, 305.1, 305.2, 305.3, 305.35, 305.38, 310, 330, 350, 400, 450, 500, 600]
    R = 8.314 #in J/molK
    Tc = crit_T[name]
    Pc = crit_P[name]
    
    b = 0.125*R*Tc/Pc
    a = 27*(R*R*Tc*Tc/Pc)/64

    C1 = np.zeros(len(T))
    P_min = np.zeros(len(T))
    V_min = np.zeros(len(T))
    P_max = np.zeros(len(T))
    V_max = np.zeros(len(T))
    i = 0
    while i < len(T):

        # P = RT/(V-b)-a/(V^2)
        # dP/dV = -RT/((V-b)^2)+2*a/(V^3) = 0 => small V corresponds to the minimum value of the curve, and large V corresponds to the maximum value.
        # -RT*(V^3)+2*a*((V-b)^2) = 0
        # -RT*(V^3)+2a(V^2)-4abV+2a(b^2) = 0

        C1 = -R*T[i]
        C2 = 2*a
        C3 = -4*a*b
        C4 = 2*a*b*b

        eqn = [C1, C2, C3, C4]
        roots = np.roots(eqn)
        V_roots = np.real(roots)
        # = sorted(V_roots)

        k = 0
        while k < len(V_roots):
            
            if V_roots[k] <= b:
                V_roots = np.delete(V_roots, k)
            else:
                k +=1
                continue
            k +=1
        
        j = 0
        P = np.zeros(len(V_roots))
        while j < len(V_roots):
            
            #P = R*T/(V_roots-b)-a/(V_roots**2)
            P[j] = R*T[i]/(V_roots[j]-b)-a/(V_roots[j]**2)

            j += 1
        
        P_min[i] = min(P)
        min_index = np.argmin(P)

        P_max[i] = max(P)
        max_index = np.argmax(P)
    
        V_min[i] = V_roots[min_index]
        V_max[i] = V_roots[max_index]

        print(T[i], P_min[i]/100000, V_min[i]*1000000, P_max[i]/100000, V_max[i]*1000000, file=f)
        i += 1

    return T, P_min, V_min, P_max, V_max

def calc_fugacity(T, P, name):
    
    R = 8.314 #in J/molK 
    Tc = crit_T[name]
    Pc = crit_P[name]   
    #print(Tc, Pc)

    b = 0.125*R*Tc/Pc
    a=27*(R*R*Tc*Tc/Pc)/64

    A=a*P/(R*R*T*T)
    B=b*P/(R*T)

    p = -1-B
    q = A
    r = -A*B
    
    eqn=[1,p,q,r]
    roots=np.roots(eqn) #z=Root(z**3+p*z**2+q*z+r1==0)
    z=np.real(roots[np.isreal(roots)])	
    #return z
    if len(z) == 1:
        zL = z
        zV = z

        if T > 1.001*Tc:
            if P > Pc:
                lnphiV=zV-1-math.log(abs(zV-B))-A/zV		
                fV=P*math.exp(lnphiV)
                Phase = 'Supercritical'
                return T, P, fV/100000, Phase
            else: 
                lnphiV=zV-1-math.log(abs(zV-B))-A/zV		
                fV=P*math.exp(lnphiV)
                Phase = 'Gas'
                return T, P, fV/100000, Phase
        
        P_sat = calc_sat_P_sat_Vs(T, name)[0]
        if P > P_sat:
            lnphiL=zL-1-math.log(abs(zL-B))-A/zL	
            fL=P*math.exp(lnphiL)
            Phase = 'Liquid'
            return T, P, fL/100000, Phase
        else:
            lnphiV=zV-1-math.log(abs(zV-B))-A/zV		
            fV=P*math.exp(lnphiV)
            Phase = 'Vapor'
            return T, P, fV/100000, Phase
    else:
        zL = min(z) 
        zV = max(z)

        lnphiL=zL-1-math.log(abs(zL-B))-A/zL	
        fL=P*math.exp(lnphiL)

        lnphiV=zV-1-math.log(abs(zV-B))-A/zV		
        fV=P*math.exp(lnphiV)
            
        if abs((fV-fL)/fV) < 1e-6:
            Phase = 'Vapor-Liquid'
            return T, P, fV/100000, Phase
        else:
            P_sat = calc_sat_P_sat_Vs(T, name)[0]
            if P > P_sat:
                Phase = 'Liquid'
                return T, P, fL/100000, Phase
            else:
                Phase ='Vapor'
                return T, P, fV/100000, Phase

def calc_fugacity_points(name):
    print('Temperature [K] Pressure [bar] Fugacity [bar] Phase\n', file=f)
    #T in K
    T = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 290, 300, 305, 305.38, 310, 330, 350, 400, 450, 500, 600]
    # P in Pa
    P = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 120000, 140000, 160000, 180000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000, 2200000, 2400000, 2600000, 2800000, 3000000, 3200000, 3400000, 3600000, 3800000, 4000000, 4200000, 4400000, 4600000, 4800000, 4884000, 5000000, 5500000, 6000000, 6500000, 7000000, 8000000, 9000000, 10000000]
    #return len(T), len(P), len(T), len(P)
    
    fL = np.zeros(len(T)*len(P))
    fV = np.zeros(len(T)*len(P))
    i = 0
    while i < len(T):
        j = 0
        while j < len(P):
            fuga_outputs = calc_fugacity(T[i], P[j], name)
            fL = fuga_outputs[2]
            fV = fuga_outputs[3]
            print(T[i], P[j]/100000, fL, fV, file=f)
            j += 1
        i += 1
    return 5
    
name = 'ethane'
with open("Binodal_Spinodal_Fugacity.txt", "w") as f:
    print('//Part (a)//', file=f)
    print('\n', file=f)
    print('**BINODAL CURVE POINTS START** \n', file=f)
    binodal_points = calc_binodal_points(name)
    print('\n**BINODAL CURVE POINTS END** \n', file=f)
    print('\n', file=f)
    print('**SPINODAL CURVE POINTS START** \n', file=f)
    spinodal_points = calc_spinodal_points(name)
    print('\n**SPINODAL CURVE POINTS END** \n', file=f)
    #print('**SPINODAL CURVE POINTS START** \n', 'Temperatures [K]\n', spinodal_points[0],'\n', 'Minimum Pressures [bar]\n', spinodal_points[1]/100000, '\n', 'Minimum Molar Volumes [cm3/mol]\n', spinodal_points[2]*1000000, '\n', 'Maximum Pressures [bar]\n', spinodal_points[3]/100000, '\n', 'Maximum Molar Volumes [cm3/mol] \n', spinodal_points[4]*1000000, '\n **SPINODAL CURVE POINTS END**', file=f)
    print('//End of Part (a)//', file=f)
    print('\n', file=f)
    print('//Part (b)//', file=f)
    print('\n', file=f)
    fugacity_points = calc_fugacity_points(name)
    #print('**FUGACITY POINTS START** \n', 'Temperatures [K] \n', fugacity_points[0],'\n', 'Pressures [bar] \n', fugacity_points[1]/100000, '\n', 'Liquid Fugacities [bar] \n', fugacity_points[2], '\n',  'Vapor Fugacities [bar] \n', fugacity_points[3], '\n **FUGACITY POINTS END**')
    print('\n//End of Part (b)//', file=f)
