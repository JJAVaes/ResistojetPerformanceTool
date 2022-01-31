from common import constants
from analysis import Measurement
from analysis import example_analyse_nitrogen
from model.state import State, NozzleProperties

# Load data files
Measurement("155007.tdms", type=Measurement.THRUST_TEST_N2,
            description="3 Nitrogen test of 300 seconds long with 5 minutes in "
                        "between. 25C chamber temperature.")

Measurement("142659.tdms", type=Measurement.THRUST_TEST_N2,
            description="3 Nitrogen test of 300 seconds long with 5 minutes in "
                        "between. 25C chamber temperature.")

# Run analysis on the measurements
data = example_analyse_nitrogen.run()

for file_name in data:
    results = data[file_name]
    print(f'File: {file_name}')

    for result in results:
        thrust = result['thrust']
        chamber_pressure = result['chamber_pressure']
        chamber_temperature = result['chamber_temperature']
        mass_flow = result['mass_flow']
        ambient_pressure = result['ambient_pressure']
        heater_power = result['heater_power']

        # Init analytical state to estimate ideal and expected thrust using the given input parameters
        state = State(
            nozzle_props=NozzleProperties.hutten_profile_new,
            chamber_temperature=chamber_temperature,
            chamber_pressure=chamber_pressure,
            ambient_pressure=ambient_pressure,
            propellant='Nitrogen',
            cd_model='KH',
            nr_iterations=1000  # Reduce number of iterations (for error estimation) to keep the program quick
        )

        # Set the measured values obtained from the experiment to get the quality factors
        state.quality_factors(
            measured_thrust=thrust,
            measured_mass_flow=mass_flow,
            measured_isp=(thrust / (mass_flow * constants.g0)),
            heater_power=heater_power
        )

        # Print out some obtained values and quality factors
        print(f'\tThrust: {state.thrust_real}')
        print(f'\tIsp: {state.isp_real}')
        print(f'\tIsp quality: {state.isp_quality_real}')
        print(f'\tThrust quality: {state.thrust_quality_real}')
        print(f'\tCd: {state.Cd_real}\n')


