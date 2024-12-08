"""
@author: L.C.M. Cramer
@editor: J.J.A. Vaes

Created on 15-02-2024 12:01:35
Edited on 08-12-2024 16:43:00 
"""

import CoolProp
import CoolProp.CoolProp as CP
from CoolProp.State import AbstractState
import numpy as np
from scipy.optimize import root
  

def correction_factor(massflow, pressure, temperature, power, isp, ambientpressure):
    #Constants
    gas_constant = 8.314                # J/Kmol
    g0 = 9.80665                        # m/s^2
    
    # Geometrical properties Vaes [30-09-2024]
    width_throat  = 245.6  *10**-6      # m
    width_exit    = 1189.8 *10**-6      # m
    height_throat = 499.1  *10**-6      # m
    height_exit   = 408.7  *10**-6      # m
    area_throat   = 122.6  *10**-9      # m^2
    area_exit     = 487.8  *10**-9      # m^2
    
    # Geometrical properties Vaes [05-08-2024]
    # width_throat  = 232.8  *10**-6      # m
    # width_exit    = 1179.1 *10**-6      # m
    # height_throat = 499.1  *10**-6      # m
    # height_exit   = 471.9  *10**-6      # m
    # area_throat   = 116.2  *10**-9      # m^2
    # area_exit     = 555.2  *10**-9      # m^2

    # area_throat = width_throat *  height_throat
    # area_exit = width_exit * height_exit
    # hydraulic_diameter_throat = 4 * area_throat / (2 * width_throat + 2 * height_throat)

    # Geometrical properties Vaes [05-06-2024]
    # width_throat  = 213.9  *10**-6      # m
    # width_exit    = 1145.5 *10**-6      # m
    # height_throat = 499.1  *10**-6      # m
    # height_exit   = 256.2  *10**-6      # m

    # Geometrical properties Cramer [2023/2024]
    # width_throat  = 200 *10**-6         # m
    # width_exit    = 1138 *10**-6        # m
    # height_throat = 521 *10**-6         # m
    # height_exit   = 326 *10**-6         # m
         
    #Adjust inputs and units accordingly
    # pressure_chamber = pressure * 100               #mbar to Pa
    # pressure_ambient = ambientpressure * 100        #mbar to Pa
    # temperature_chamber = temperature + 273.15      #C to K
    # temperature_propellant = 20 + 273.15            #C to K    
    # Isp_real = isp                                  #s
    power_real = power                              #W
    # massflow_real = massflow * 10**-6               #kg/s
    
    #Override to check with Hutten's values --> Same number of magnitude but never equal values?
    # pressure_chamber = 989 *100
    # temperature_chamber = 19 +273.15
    massflow_real = 15.38 * 10**-6 
    Isp_real = 35.53
    # width_throat = 140 *10**-6 
    # width_exit = 1078 *10**-6
    # height_throat = 538 *10**-6
    # height_exit = 538 *10**-6
    # area_throat = width_throat *  height_throat
    # area_exit = width_exit * height_exit
    # hydraulic_diameter_throat = 4 * area_throat / (2 * width_throat + 2 * height_throat)

    temperature_propellant = 20 + 273.15            #C to K    
    temperature_chamber = 199.59 + 273.15
    pressure_chamber    = 100.116 * 1000
    pressure_ambient    = 728
    # area_throat         = 70.6 * 10**-9
    # area_exit           = 531.7 * 10**-9
    # height_throat       = 496 *10**-6
    # height_exit         = 496 *10**-6
    hydraulic_diameter_throat = 4 * area_throat / (2 * area_throat)
    
    #Gather Coolprop properties
    propellant_name: str = 'Nitrogen'

    # Update propellant with chamber properties
    propellant_chamber = AbstractState("HEOS", propellant_name)
    propellant_chamber.update(CoolProp.PT_INPUTS, pressure_chamber, temperature_chamber)
    chamber_viscosity = propellant_chamber.viscosity()
    molar_mass = propellant_chamber.molar_mass()
    gamma = propellant_chamber.cpmass() / propellant_chamber.cvmass()
    
    #Check sonic conditions throat
    pressure_throat = pressure_chamber * (2/(gamma+1))**(gamma/(gamma-1))
    temperature_throat_ideal = temperature_chamber * (2 / (gamma + 1))
    density_nitrogen = CP.PropsSI('D', 'T', temperature_chamber, 'P', pressure_throat, propellant_name)
    speed_of_sound = CP.PropsSI('A', 'T', temperature_throat_ideal, 'P', pressure_throat, propellant_name)

    
    # Calculate Ideal Rocket Theory parameters
    c_star_ideal = (1 / Γ(gamma)) * np.sqrt(temperature_chamber * gas_constant / molar_mass)
    mass_flow_ideal = pressure_chamber * area_throat / c_star_ideal
    velocity_max_flow_ideal = ((2 * gamma / (gamma - 1)) * (temperature_chamber * gas_constant / molar_mass)) ** 0.5
    pressure_exit_ideal = exit_pressure(gamma, area_exit / area_throat, pressure_chamber)
    temperature_exit_ideal = temperature_chamber * (pressure_exit_ideal / pressure_chamber) ** ((gamma - 1) / gamma)
    velocity_exit_ideal = velocity_max_flow_ideal * (1 - (pressure_exit_ideal / pressure_chamber) ** ((gamma - 1) / gamma)) ** 0.5
    thrust_ideal = mass_flow_ideal * velocity_exit_ideal + (pressure_exit_ideal -  pressure_ambient) * area_exit
    isp_ideal = (thrust_ideal / mass_flow_ideal) / g0
    thrust_coefficient_ideal = thrust_ideal / (pressure_chamber * area_throat)
    temperature_throat_ideal = temperature_chamber * (2 / (gamma + 1))
    pressure_throat_ideal = pressure_chamber * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    velocity_throat_ideal = (2 * propellant_chamber.cpmass() * (temperature_chamber - temperature_throat_ideal)) ** 0.5
    re_throat_ideal = (mass_flow_ideal * hydraulic_diameter_throat) / (chamber_viscosity * area_throat)
    
    #Compute power efficiency
    propellant_stored = AbstractState("HEOS", propellant_name) 
    propellant_stored.update(CoolProp.PT_INPUTS, pressure_chamber, temperature_propellant)
    power_input_ideal = (propellant_chamber.hmass() - propellant_stored.hmass()) * mass_flow_ideal  # Ideal power
    if power_real != 0:
        eta_heat = power_input_ideal / power_real
    else:
        eta_heat = float('nan')  # Set eta_heat to NaN (Not a Number) or handle the case differently based on your requirements
        
    #Reynolds number
    Cd_est = Cd_KH(re_throat_ideal, gamma, hydraulic_diameter_throat)
    Cd_real = massflow_real/mass_flow_ideal
    Re_throat_real = calc_reynolds_number(area_throat, width_throat, height_throat, Cd_real, chamber_viscosity, mass_flow_ideal)
    Re_throat_est  = calc_reynolds_number(area_throat, width_throat, height_throat, Cd_est, chamber_viscosity, mass_flow_ideal)

    micrometer = 1e6                    # Conversion factor for meters to micrometers
    micrometer_squared = 1e12           # Conversion factor for meters² to micrometers²
    kPa = 1e-3                          # Conversion factor for Pa to kPa
    mg_per_s = 1e6                      # Conversion factor for kg/s to mg/s
    mN = 1e3                            # Conversion factor for N to mN
    
    # (Previous code for calculations remains the same)
    
    # Print calculated factors
    print("\n--- Calculated Factors ---\n")

    print("\n--- Geometrical Properties ---")
    print(f"Throat width [μm]: {width_throat * micrometer:.2f}")
    print(f"Throat height [μm]: {height_throat * micrometer:.2f}")
    print(f"Exit width [μm]: {width_exit * micrometer:.2f}")
    print(f"Exit height [μm]: {height_exit * micrometer:.2f}")
    print(f"Throat area [μm²]: {area_throat * micrometer_squared:.2f}")
    print(f"Exit area [μm²]: {area_exit * micrometer_squared:.2f}")
    print(f"Hydraulic diameter of throat [μm]: {hydraulic_diameter_throat * micrometer:.2f}")

    print("\n--- Chamber Conditions ---")
    print(f"Chamber pressure [kPa]: {pressure_chamber * kPa:.2f}")
    print(f"Chamber temperature [K]: {temperature_chamber}")
    print(f"Ambient pressure [Pa]: {pressure_ambient:.2f}")
    
    print("\n--- Ideal Rocket Theory Parameters ---")
    print(f"Ideal characteristic velocity (c*): {c_star_ideal}")
    print(f"Ideal mass flow rate [mg/s]: {mass_flow_ideal * mg_per_s:.2f}")
    print(f"Ideal max flow velocity [m/s]: {velocity_max_flow_ideal}")
    print(f"Ideal exit pressure [kPa]: {pressure_exit_ideal * kPa:.2f}")
    print(f"Ideal exit temperature [K]: {temperature_exit_ideal}")
    print(f"Ideal exit velocity [m/s]: {velocity_exit_ideal}")
    print(f"Ideal thrust [mN]: {thrust_ideal * mN:.2f}")
    print(f"Ideal specific impulse (Isp) [s]: {isp_ideal}")
    print(f"Ideal thrust coefficient: {thrust_coefficient_ideal}")
    
    print("\n--- Real Performance Metrics ---")
    print(f"Discharge coefficient [%]: {Cd_est * 100:.2f}")
    print(f"Propellant consumption quality (ξ_Isp) [%]: {(Isp_real / isp_ideal) * 100:.2f}")
    
    print(f"Mass flow rate [mg/s]: {massflow_real * mg_per_s:.2f}")

    print(f"Reynolds number at throat [-]: {Re_throat_est:.0f}")
    print(f"Heater efficiency [%]: {eta_heat * 100:.2f}")

    #Discharge coefficient
    print("\n")
    print("Pressure ratio (Pt/Pc) [-] =", pressure_throat/pressure_chamber)
    print("Pressure ratio (Pc/Pe) [-] =", pressure_chamber/pressure_exit_ideal)
    print("Critical [-] =",  ((gamma+1)/2)**(gamma/(gamma-1)))
    print("Flow velocity throat [m/s] =", velocity_throat_ideal)
    print("Flow velocity [m/s] =", massflow_real/(density_nitrogen * area_throat))
    print("Speed of sound [m/s] =", speed_of_sound)
    print("Speed of sound [m/s] =", np.sqrt(gamma * temperature_throat_ideal * gas_constant / molar_mass))
    print('Discharge coefficient =', massflow_real/mass_flow_ideal)
    print("Heater efficiency =", eta_heat)
    print("Reynolds number =", Re_throat_real)
    print("Propellant consumption quality (ξ_Isp) =", Isp_real/isp_ideal)

    Cd   = massflow_real/mass_flow_ideal
    eta  = eta_heat
    Re   = Re_throat_real
    eisp = Isp_real/isp_ideal

    return Cd, eta, Re, eisp


# Determine Cd with KH
def calc_reynolds_number(area_throat, width_throat, height_throat, cd, viscosity, massflow):
    """Calculates the real Reynolds number"""
    hydraulic_diameter_eff = effective_hydraulic_diameter(area_throat, width_throat, height_throat, cd)
    return (massflow * hydraulic_diameter_eff) / (viscosity * area_throat)

# Effective hydraulic diameter
def effective_hydraulic_diameter(area_throat, width_throat, height_throat, cd):
    """Calculates the effective hydraulic diameter of a square nozzle from the discharge coefficient"""
    return 2 * area_throat * cd / \
            (4 * area_throat * cd + (
                    width_throat - height_throat) ** 2) ** 0.5


def Cd_KH(re, y, hydraulic_diameter_throat):
    """Calculate discharge coefficient using Kuluva and Hosack estimation."""
    rc = 260 * 10**-6
    rt = hydraulic_diameter_throat / 2

    term1 = ((rc + 0.05 * rt) / (rc + 0.75 * rt)) ** 0.019
    term2 = ((rc + 0.1 * rt) / rt) ** 0.21
    term3 = (0.97 + 0.86 * y) / np.sqrt(re)
    cd = term1 * (1 - term2 * term3)
    return cd

def Γ(y):
    """Vandenkerckhove function"""
    return y**0.5 * (2 / (y + 1)) ** ((y + 1) / (2 * y - 2))


def exit_pressure(y, AeAt, Pc):
    """Calculates the exit pressure using just floats"""
    term1 = ((2 * y) / (y - 1))
    vdkerckhove = Γ(y)

    def find_value():
        """Finds the value of Pe"""
        def f(Pe):
            PePc = Pe / Pc
            term2 = (PePc ** (2 / y))
            term3 = 1 - PePc ** ((y - 1) / y)
            return vdkerckhove / (np.sqrt(term1 * term2 * term3)) - AeAt

        # Relatively close guess that I found that works for y=1.01...1.5 and Ae/At=3..20
        guess = (Pc/2) / AeAt ** (y + 0.5)

        # Start root finding function
        ans = root(f, guess)
        return ans.x[0]

    value = find_value()
    return value

def get_density_nitrogen(temperature, pressure):
    substance = 'Nitrogen'
    density = CP.PropsSI('D', 'T', temperature, 'P', pressure, substance)
    return density