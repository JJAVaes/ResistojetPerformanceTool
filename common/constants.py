from .units import *


sccm_to_mgps = 1 / 60 * 1.250386130338901  # [standard cc per minute] to [mg/s] at 101325 Pa and 273.15 K

gas_constant = BaseUnit.new(8.31446261815324, SI(J=1, K=-1, mol=-1))
g0 = Acceleration(9.80665)


class Lever:
    CONFIGURATION_VERSTEEG = 1
    CONFIGURATION_HUTTEN = 2

    def __init__(self, configuration=CONFIGURATION_HUTTEN):
        if configuration == self.CONFIGURATION_HUTTEN:
            self.factor = NormalDistExt(mu=2.80591, sigma=0.0078)
        elif configuration == self.CONFIGURATION_VERSTEEG:
            self.factor = NormalDistExt(mu=1.8137, sigma=0)

    def get_factor(self) -> NormalDistExt:
        return self.factor


class ActuatorCalibration:
    def __init__(self, lever: Lever):
        coil_calibration = NormalDistExt(mu=0.826, sigma=0.006 / 3)
        self.calibration = lever.get_factor() * coil_calibration

    def get_calibration(self) -> NormalDistExt:
        return self.calibration

    def get_factor(self) -> float:
        return self.calibration.mean
