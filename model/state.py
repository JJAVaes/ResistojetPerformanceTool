from numpy import pi, sqrt, isclose, std
from typing import Dict

import CoolProp
from CoolProp.State import AbstractState

from common import constants
from common.units import *
from common.functions import exit_pressure, Γ


class State:
    """
    CoolProp sites:
        - http://coolprop.sourceforge.net/apidoc/CoolProp.AbstractState.html
        - http://www.coolprop.org/_static/doxygen/html/namespace_cool_prop.html#a58e7d98861406dedb48e07f551a61efb
    """
    def __init__(self,
                 nozzle_props: Dict,
                 chamber_temperature: Temperature,
                 chamber_pressure: Pressure,
                 ambient_pressure: Pressure,
                 propellant: str = 'water',
                 cd_model: str = 'KH',
                 nr_iterations: int = 100 * 1000):
        """
        Calculates the ideal rocket parameters and estimates quality factors using the following models:
            Discharge coefficient:
                'KH' Kuluva and Hosack 1971
                'TF' Tang and Fenn 1978
            Thrust coefficient loss:
                Spisz et al. 1965

        :param propellant: string representing the propellant used eg: 'water', 'nitrogen', 'CO2', 'methane'
        :param cd_model: 'KH' for Kuluva & Hosack or 'TF' for Tang & Fenn
            Estimation model of the discharge coefficient.
        """
        # Input variables
        self.temperature_chamber = Temperature(chamber_temperature).x
        self.pressure_chamber = Pressure(chamber_pressure).x
        self.ambient_pressure = Pressure(ambient_pressure).x
        self.nozzle = NozzleProperties(**nozzle_props)

        self.quality_factors_calculated = False
        self.cd_model = cd_model
        self.propellant_name = propellant

        # Attributes existing before calculation of the model are not output variables
        existing_attrs = dir(self)

        # Calculate the model with the initial values
        self.calc()

        # Get the attributes (variable names) of the output variables
        result_attrs = [attr for attr in dir(self) if not attr.startswith('_') and isinstance(self.__getattribute__(attr), float) and attr not in existing_attrs]

        vals = {}  # Stores the values of the model for the original input parameters
        self.data = {}  # Holds all the values of the model of the sampled input parameters

        # Initialise both dicts
        for attr in result_attrs:
            self.data[attr] = []
            vals[attr] = self.__getattribute__(attr)

        # Run model multiple times to get the error
        for _ in range(nr_iterations):
            # Define input variables from random samples
            self.temperature_chamber = chamber_temperature.get_sample()
            self.pressure_chamber = chamber_pressure.get_sample()
            self.ambient_pressure = ambient_pressure.get_sample()
            prop = {k: v.get_sample() for k, v in nozzle_props.items()}  # Samples the nozzle properties
            self.nozzle = NozzleProperties(**prop)

            # Calculate the model with the newly defined input variables
            self.calc()

            # Store the results of the model in a list
            for attr in result_attrs:
                self.data[attr].append(self.__getattribute__(attr))

        # Restore input params
        self.temperature_chamber = Temperature(chamber_temperature)
        self.pressure_chamber = Pressure(chamber_pressure)
        self.ambient_pressure = Pressure(ambient_pressure)
        self.nozzle = NozzleProperties(use_base_units=True, **nozzle_props)

        self.Cd_est = Dimensionless(vals['Cd_est'], error=3 * std(self.data['Cd_est']))
        self.c_star_ideal = Speed(vals['c_star_ideal'], error=3 * std(self.data['c_star_ideal']))
        self.chamber_viscosity = Viscosity(vals['chamber_viscosity'], error=3*std(self.data['chamber_viscosity']))
        self.gamma = Dimensionless(vals['gamma'], error=3*std(self.data['gamma']))
        self.isp_ideal = Time(vals['isp_ideal'], error=3 * std(self.data['isp_ideal']))
        self.isp_quality_est = Dimensionless(vals['isp_quality_est'], error=3*std(self.data['isp_quality_est']))
        self.isp_est = Time(vals['isp_est'], error=3 * std(self.data['isp_est']))
        self.mass_flow_ideal = MassFlow(vals['mass_flow_ideal'], error=3 * std(self.data['mass_flow_ideal']))
        self.mass_flow_est = MassFlow(vals['mass_flow_est'], error=3 * std(self.data['mass_flow_est']))
        self.molar_mass = MolarMass(vals['molar_mass'], error=3*std(self.data['molar_mass']))
        self.power_input_ideal = Watt(vals['power_input_ideal'], error=3 * std(self.data['power_input_ideal']))
        self.pressure_exit_ideal = Pressure(vals['pressure_exit_ideal'], error=3 * std(self.data['pressure_exit_ideal']))
        self.pressure_throat_ideal = Pressure(vals['pressure_throat_ideal'], error=3 * std(self.data['pressure_throat_ideal']))
        self.reynolds_throat_ideal = Dimensionless(vals['reynolds_throat_ideal'], error=3*std(self.data['reynolds_throat_ideal']))
        self.reynolds_throat_est = Dimensionless(vals['reynolds_throat_est'], error=3 * std(self.data['reynolds_throat_est']))
        self.temperature_exit_ideal = Temperature(vals['temperature_exit_ideal'], error=3 * std(self.data['temperature_exit_ideal']))
        self.temperature_throat_ideal = Temperature(vals['temperature_throat_ideal'], error=3 * std(self.data['temperature_throat_ideal']))
        self.thrust_ideal = Force(vals['thrust_ideal'], error=3 * std(self.data['thrust_ideal']))
        self.thrust_coefficient_ideal = Dimensionless(vals['thrust_coefficient_ideal'], error=3*std(self.data['thrust_coefficient_ideal']))
        self.thrust_quality_est = Dimensionless(vals['thrust_quality_est'], error=3*std(self.data['thrust_quality_est']))
        self.thrust_est = Force(vals['thrust_est'], error=3 * std(self.data['thrust_est']))
        self.velocity_exit_ideal = Speed(vals['velocity_exit_ideal'], error=3 * std(self.data['velocity_exit_ideal']))
        self.velocity_max_flow_ideal = Speed(vals['velocity_max_flow_ideal'], error=3 * std(self.data['velocity_max_flow_ideal']))
        self.velocity_throat_ideal = Speed(vals['velocity_throat_ideal'], error=3 * std(self.data['velocity_throat_ideal']))

    def calc(self):
        """Calculates the entire model given the """
        # Update propellant with chamber properties
        propellant_chamber = AbstractState("HEOS", self.propellant_name)
        propellant_chamber.update(CoolProp.PT_INPUTS, self.pressure_chamber, self.temperature_chamber)
        self.chamber_viscosity = propellant_chamber.viscosity()
        self.molar_mass = propellant_chamber.molar_mass()
        self.gamma = propellant_chamber.cpmass() / propellant_chamber.cvmass()

        # Calculate Ideal Rocket Theory parameters
        self.c_star_ideal = (1 / Γ(self.gamma)) * (self.temperature_chamber * constants.gas_constant.x / self.molar_mass) ** 0.5
        self.mass_flow_ideal = self.pressure_chamber * self.nozzle.area_throat / self.c_star_ideal
        self.velocity_max_flow_ideal = ((2 * self.gamma / (self.gamma - 1)) * (self.temperature_chamber * constants.gas_constant.x / self.molar_mass)) ** 0.5
        self.pressure_exit_ideal = exit_pressure(self.gamma, self.nozzle.area_exit / self.nozzle.area_throat, self.pressure_chamber)
        self.temperature_exit_ideal = self.temperature_chamber * (self.pressure_exit_ideal / self.pressure_chamber) ** ((self.gamma - 1) / self.gamma)
        self.velocity_exit_ideal = self.velocity_max_flow_ideal * (1 - (self.pressure_exit_ideal / self.pressure_chamber) ** ((self.gamma - 1) / self.gamma)) ** 0.5
        self.thrust_ideal = self.mass_flow_ideal * self.velocity_exit_ideal + (self.pressure_exit_ideal - self.ambient_pressure) * self.nozzle.area_exit
        self.isp_ideal = (self.thrust_ideal / self.mass_flow_ideal) / constants.g0.x
        self.thrust_coefficient_ideal = self.thrust_ideal / (self.pressure_chamber * self.nozzle.area_throat)
        self.temperature_throat_ideal = self.temperature_chamber * (2 / (self.gamma + 1))
        self.pressure_throat_ideal = self.pressure_chamber * (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1))
        self.velocity_throat_ideal = (2 * propellant_chamber.cpmass() * (self.temperature_chamber - self.temperature_throat_ideal)) ** 0.5

        propellant_throat = AbstractState("HEOS", self.propellant_name)
        propellant_throat.update(CoolProp.PT_INPUTS, self.pressure_throat_ideal, self.temperature_throat_ideal)
        throat_viscosity = propellant_throat.viscosity()
        self.reynolds_throat_ideal = (self.mass_flow_ideal * self.nozzle.hydraulic_diameter_throat) / (self.chamber_viscosity * self.nozzle.area_throat)

        propellant_stored = AbstractState("HEOS", self.propellant_name)
        propellant_stored.update(CoolProp.PT_INPUTS, self.pressure_chamber, Temperature.from_celcius(20).x)
        self.power_input_ideal = (propellant_chamber.hmass() - propellant_stored.hmass()) * self.mass_flow_ideal

        # Determine Cd with TF
        if self.cd_model == "TF":
            self.reynolds_throat_est = self.reynolds_throat_ideal
            re_prev = 2 * self.reynolds_throat_est
            while not isclose(re_prev, self.reynolds_throat_est):
                rc = Length.from_um(260).x
                rt = self.nozzle.hydraulic_diameter_throat / 2
                uc = self.chamber_viscosity
                ut = throat_viscosity
                Pr = propellant_throat.Prandtl()
                Re_D = self.reynolds_throat_est * sqrt(rt * uc ** 2 / (Pr * rc * ut ** 2))
                self.Cd_est = self.Cd_TF(Re_D, self.gamma)
                re_prev = self.reynolds_throat_est
                self.reynolds_throat_est = self.calc_reynolds_number(self.nozzle, self.Cd_est, self.chamber_viscosity, self.mass_flow_ideal)

        # Determine Cd with KH
        elif self.cd_model == "KH":
            self.Cd_est = self.Cd_KH(self.reynolds_throat_ideal, self.gamma, self.nozzle)
            self.reynolds_throat_est = self.calc_reynolds_number(self.nozzle, self.Cd_est, self.chamber_viscosity, self.mass_flow_ideal)

        else:
            raise ValueError(f"Unknown Cd model {self.cd_model}")

        # Thrust coefficient loss
        divergence_loss = math.sin(self.nozzle.divergent_half_angle) / self.nozzle.divergent_half_angle
        thrust_coefficient_loss = 17.6 * math.exp(0.0032 * self.nozzle.area_ratio) * ((self.Cd_est * self.chamber_viscosity / throat_viscosity) * self.reynolds_throat_ideal) ** (-0.5) * (2 / (self.gamma + 1)) ** (-5 / 6)

        # Estimate quality factors
        thrust_coefficient_est = self.thrust_coefficient_ideal * divergence_loss - thrust_coefficient_loss
        self.isp_quality_est = thrust_coefficient_est / self.thrust_coefficient_ideal
        self.thrust_quality_est = self.isp_quality_est * self.Cd_est

        # Estimated performance values
        self.mass_flow_est = self.mass_flow_ideal * self.Cd_est
        self.thrust_est = self.thrust_ideal * self.thrust_quality_est
        self.isp_est = self.isp_ideal * self.isp_quality_est

    def print_stats(self):
        print("Input:")
        print("Propellant =", self.propellant_name)
        print("Chamber temperature =", self.temperature_chamber)
        print("Chamber pressure =", self.pressure_chamber)
        print("Ambient pressure =", self.ambient_pressure)
        print("Area throat =", self.nozzle.area_throat)
        print("Area exit =", self.nozzle.area_exit)
        print("Throat height =", self.nozzle.throat_height)
        print("θ =", self.nozzle.divergent_half_angle * 180 / pi)

        print("\nOutput Ideal:")
        print("γ =", self.gamma)
        print("Thrust =", self.thrust_ideal)
        print("Massflow =", self.mass_flow_ideal)
        print("Isp =", self.isp_ideal)
        print("c* =", self.c_star_ideal)
        print("Cf =", self.thrust_coefficient_ideal)
        print("Max flow velocity =", self.velocity_max_flow_ideal)
        print("Power input =", self.power_input_ideal)
        print("Re throat =", self.reynolds_throat_ideal)
        print("Throat pressure =", self.pressure_throat_ideal)
        print("Throat temperature =", self.temperature_throat_ideal)
        print("Velocity throat =", self.velocity_throat_ideal)
        print("Exit pressure =", self.pressure_exit_ideal)
        print("Exit temperature =", self.temperature_exit_ideal)
        print("Velocity exit =", self.velocity_exit_ideal)

        print("\nOutput Expected:")
        print(f"Re throat ({self.cd_model}) =", self.reynolds_throat_est)
        print(f"Cd ({self.cd_model}) =", self.Cd_est)
        print(f"Isp quality ({self.cd_model}) =", self.isp_quality_est)
        print(f"Thrust quality ({self.cd_model}) =", self.thrust_quality_est)
        print(f"Massflow ({self.cd_model}) =", self.mass_flow_est)
        print(f"Thrust ({self.cd_model}) =", self.thrust_est)
        print(f"Isp ({self.cd_model}) =", self.isp_est)

        if self.quality_factors_calculated:
            print("\nOutput Real:")
            print(f"Re throat =", self.reynolds_throat_real)
            print(f"Cd =", self.Cd_real)
            print(f"Isp quality =", self.isp_quality_real)
            print(f"Thrust quality =", self.thrust_quality_real)
            print(f"Massflow =", self.mass_flow_real)
            print(f"Thrust =", self.thrust_real)
            print(f"Isp =", self.isp_real)
            print(f"Power input =", self.power_input_real)
            print(f"Heating efficiency =", self.heating_efficiency)

    def quality_factors(self, measured_thrust: Force, measured_mass_flow: MassFlow, measured_isp: Time, heater_power: Watt):
        """Calculates quality factors given the measured quantities"""
        self.quality_factors_calculated = True

        # Measured values
        self.thrust_real = Force(measured_thrust)
        self.mass_flow_real = MassFlow(measured_mass_flow)
        self.isp_real = Time(measured_isp)
        self.power_input_real = Watt(heater_power)

        self.thrust_quality_real = self.thrust_real / self.thrust_ideal
        self.Cd_real = self.mass_flow_real / self.mass_flow_ideal
        self.isp_quality_real = self.isp_real / self.isp_ideal
        self.reynolds_throat_real = self.calc_reynolds_number(self.nozzle, self.Cd_real, self.chamber_viscosity.x, self.mass_flow_ideal.x)

        if heater_power.x < 0.1:  # No power input
            self.heating_efficiency = Dimensionless(1)
        else:
            power_increase = self.power_input_ideal * self.Cd_real
            self.heating_efficiency = power_increase / heater_power

    @staticmethod
    def calc_reynolds_number(nozzle_props: 'NozzleProperties', cd: float, viscosity: float, mass_flow_ideal: float) -> float:
        """Calculates the real Reynolds number"""
        hydraulic_diameter_eff = State.effective_hydraulic_diameter(nozzle_props, cd)
        return (mass_flow_ideal * hydraulic_diameter_eff) / (viscosity * nozzle_props.area_throat)

    @staticmethod
    def effective_hydraulic_diameter(nozzle_props: 'NozzleProperties', cd: float) -> float:
        """Calculates the effective hydraulic diameter of a square nozzle from the discharge coefficient"""
        return 2 * nozzle_props.area_throat * cd / \
               (4 * nozzle_props.area_throat * cd + (
                       nozzle_props.throat_width - nozzle_props.throat_height) ** 2) ** 0.5

    @staticmethod
    def Cd_KH(re: float, y: float, nozzle_props: 'NozzleProperties') -> float:
        """Calculate discharge coefficient using Kuluva and Hosack estimation."""
        rc = Length.from_um(260).x
        rt = nozzle_props.hydraulic_diameter_throat / 2

        term1 = ((rc + 0.05 * rt) / (rc + 0.75 * rt)) ** 0.019
        term2 = ((rc + 0.1 * rt) / rt) ** 0.21
        term3 = (0.97 + 0.86 * y) / sqrt(re)
        cd = term1 * (1 - term2 * term3)
        return cd

    @staticmethod
    def Cd_TF(Re_D: float, y: float) -> float:
        """Calculate discharge coefficient using Tang and Fenn estimation."""
        term1 = ((y + 1) / 2) ** 0.75
        term2 = (4 * sqrt(6) / 3) + (8 / 3) * (9 - 4 * sqrt(6)) / (y + 1)
        term3 = (2 * sqrt(2) / 3) * (y - 1) * (y + 2) / sqrt(y + 1)
        cd = 1 - term1 * term2 / sqrt(Re_D) + term3 / Re_D
        return cd


class NozzleProperties:
    # After reassembly
    hutten_profile_new = {
        'free_exit_width': Length.from_um(1073, error=0),
        'free_throat_width': Length.from_um(135, error=0),
        'exit_height': Length.from_um(538, error=0),
        'exit_width': Length.from_um(1078, error=0),
    }
    # Before reassembly
    rick_profile_old = {
        'free_exit_width': Length.from_um(1126, error=5),
        'free_throat_width': Length.from_um(166, error=5),
        'exit_height': Length.from_um(543, error=8),
        'exit_width': Length.from_um(1081, error=8),
    }
    # Such that we get Huibs results
    huib_profile_cold = {
        'free_exit_width': Length.from_um(1084, error=0),
        'free_throat_width': Length.from_um(154, error=8),
        'exit_height': Length.from_um(496, error=4),
        'exit_width': Length.from_um(1084, error=15),
    }

    versteeg_profile_hot = {
        'free_exit_width': Length.from_um(1071.9, error=0),
        'free_throat_width': Length.from_um(142.3, error=7.8),
        'exit_height': Length.from_um(496, error=4),
        'exit_width': Length.from_um(1071.9, error=5.6),
    }

    def __init__(self,
                 use_base_units: bool = False,
                 free_exit_width: Length = None,
                 free_throat_width: Length = None,
                 exit_height: Length = None,
                 exit_width: Length = None):
        if use_base_units:
            self.free_exit_width = Length(free_exit_width)
            self.free_throat_width = Length(free_throat_width)
            self.exit_height = Length(exit_height)
            self.exit_width = Length(exit_width)
            self.throat_height = Length(exit_height)
            self.divergent_half_angle = Dimensionless(20 * pi / 180)  # Radians
            self.convergent_half_angle = Dimensionless(35 * pi / 180)  # Radians
        else:
            self.free_exit_width = Length(free_exit_width).x
            self.free_throat_width = Length(free_throat_width).x
            self.exit_height = Length(exit_height).x
            self.exit_width = Length(exit_width).x
            self.throat_height = Length(self.exit_height).x
            self.divergent_half_angle = Dimensionless(20 * pi / 180).x  # Radians
            self.convergent_half_angle = Dimensionless(35 * pi / 180).x  # Radians

        self.throat_width = self.exit_width + (self.free_throat_width - self.free_exit_width)
        self.area_throat: Area = self.throat_width * self.throat_height
        self.area_exit: Area = self.exit_width * self.exit_height
        self.area_ratio = self.area_exit / self.area_throat
        self.hydraulic_diameter_throat = 4 * self.area_throat / (2 * self.throat_width + 2 * self.throat_height)
        self.hydraulic_diameter_exit = 4 * self.area_exit / (2 * self.exit_width + 2 * self.exit_height)
