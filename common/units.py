from math import fabs, hypot, log, nan
from random import gauss
from statistics import NormalDist
from typing import Union, Optional
import math

Number = Union['BaseUnit', float, int]


class SI:
    """
    Usage:
        >>> SI({"m": 1, "s": -1})
        >>> SI(m=1, s=-1)

    Mass            kilogram    kg
    Length          meter       m
    Current         ampere      A
    Time            seconds     s
    Temperature     Kelvin      K
    Amount          mole        mol
    """
    _units = ("kg", "m", "A", "s", "K", "mol")

    def __init__(self, type_dict=None, **kwargs):
        self.unit_count = [0] * len(SI._units)

        # Unpack kwargs if units given as kwargs, not type_dict
        if type_dict is None:
            type_dict = kwargs
        elif kwargs:
            raise ValueError("Can't give kwargs when given a type dictionary.")

        for key in type_dict:
            if key in SI._units:
                self.unit_count[SI._units.index(key)] += type_dict[key]
            else:
                for cls in BaseUnit.__subclasses__():  # type: BaseUnit
                    if hasattr(cls, 'name') and key == cls.name:
                        self.unit_count = [i + j * type_dict[key] for i, j in
                                           zip(self.unit_count, cls.units.unit_count)]
                        break
                else:
                    raise ValueError(f"Unit must be one of {self._units}, not \"{key}\".")

    def get_prefix(self, oom: int):
        """Get the order of magnitude prefix."""
        supported_oom = [-6, -3, 0, 3, 6]
        if oom not in supported_oom:
            raise ValueError(f"Order of magnitude not supported: {oom}")
        prefixes = ["u", "m", "", "k", "M"]
        for unit, count in zip(SI._units, self.unit_count):
            if count == 0:
                continue
            if count == 1:
                pass  # TODO
        return dict(zip(supported_oom, prefixes))[oom]

    def __eq__(self, other: 'SI'):
        return all([i == j for i, j in zip(self.unit_count, other.unit_count)])

    def __ne__(self, other: 'SI'):
        return not self.__eq__(other)

    def __add__(self, other: 'SI'):
        types = [i + j for i, j in zip(self.unit_count, other.unit_count)]
        return SI(dict(zip(SI._units, types)))

    def __sub__(self, other: 'SI'):
        types = [i - j for i, j in zip(self.unit_count, other.unit_count)]
        return SI(dict(zip(SI._units, types)))

    def __mul__(self, power):
        if type(power) not in [int, float]:
            raise TypeError(f"Can't raise {self.__class__.__name__} to the power of {power.__class__.__name__}")
        types = [i * power for i in self.unit_count]
        return SI(dict(zip(SI._units, types)))

    def __str__(self):
        unit_type = BaseUnit.get_type(self)
        if hasattr(unit_type, 'name'):
            return unit_type.name
        s = []
        for unit, count in zip(SI._units, self.unit_count):
            if count == 0:
                continue
            if count == 1:
                s.append(f"{unit}")
            else:
                s.append(f"{unit}{count}")
        return " ".join(s)

    def __repr__(self):
        return str(dict(zip(SI._units, self.unit_count)))


class BaseUnit(object):
    def __init__(self, x: Union[float, 'BaseUnit'], units: SI = None, error: float = 0,
                 _factor: Optional[float] = None, _constant: Optional[float] = None):
        if isinstance(x, BaseUnit):
            assert self.units == x.units, f'Could not init unit {self.units} from {x.units}'
            units = x.units
            error = x.error
            x = x.x

        if isinstance(error, Percentage):
            error *= fabs(x + _constant) if _constant else fabs(x)
        self.normal_dist = NormalDistExt(x, error)
        self.units: SI = units if units is not None else self.units

        # Apply optional conversion factor and/or constants
        if _factor:
            self.normal_dist *= _factor
        if _constant:
            self.normal_dist += _constant

    @property
    def x(self):
        return self.normal_dist.mean

    @property
    def error(self):
        return self.normal_dist.stdev

    @property
    def error_percentage(self):
        if self.x == 0:
            return nan
        return 100 * self.error / fabs(self.x)

    @property
    def max(self):
        """Maximum expected value"""
        return self.x + self.error

    @property
    def min(self):
        """Minimum expected value"""
        return self.x - self.error

    @classmethod
    def get_type(cls, units: SI):
        """Returns a subclass of BaseUnit if the units match. Otherwise returns BaseUnit"""
        for unit_type in cls.__subclasses__():  # type: BaseUnit
            if units == unit_type.units:
                return unit_type
        return BaseUnit

    @classmethod
    def new(cls, x: float, units: SI, error: float = 0):
        """Creates a new BaseUnit object. First tries its subclasses before resorting to default BaseUnit"""
        if isinstance(error, Percentage):
            error = error * fabs(x)
        unit_type = cls.get_type(units)
        if unit_type == cls:
            return cls(x, units, error)
        return unit_type(x, error=error)

    def add_error(self, error: float):
        """Add an additional flat error or percentage"""
        if isinstance(error, Percentage):
            error = error * fabs(self.x)
        self.normal_dist = self.normal_dist + NormalDistExt(0, error)

    def set_error(self, error: float):
        """Sets the error in baseunits"""
        self.normal_dist._sigma = error

    def get_sample(self):
        """
        Get a single sample from the distribution.
        Error is defined at 3 sigma, so sigma = error / 3
        """
        return gauss(self.x, self.error / 3)

    def __oom(self) -> int:
        """Returns the order of magnitude of the value. Value is in [-6, -3, 0, 3, 6]"""
        oom = int(math.log10(abs(self.x)) // 3 * 3)
        return min(6, max(-6, oom))

    def __eq__(self, other: 'BaseUnit'):
        if not isinstance(other, BaseUnit):
            return self.x == other and sum(self.units.unit_count) == 0
        return self.normal_dist == other.normal_dist and self.units == other.units

    def __ne__(self, other: 'BaseUnit'):
        return not self.__eq__(other)

    def __neg__(self):
        return BaseUnit.new(-self.x, self.units, self.error)

    def __mul__(self, other: Number):
        if not isinstance(other, BaseUnit):
            other = Dimensionless(other)
        val = self.normal_dist * other.normal_dist
        return BaseUnit.new(val.mean, self.units + other.units, val.stdev)

    def __rmul__(self, other: Number):
        return self.__mul__(other)

    def __truediv__(self, other: Number):
        if type(other) in [float, int]:
            other = Dimensionless(other)
        val = self.normal_dist / other.normal_dist
        return BaseUnit.new(val.mean, self.units - other.units, val.stdev)

    def __rtruediv__(self, other: Number):
        if type(other) in [float, int]:
            other = Dimensionless(other)
        return other.__truediv__(self)

    def __add__(self, other: Number):
        if type(other) in [float, int] and type(self) == Dimensionless:
            other = Dimensionless(other)
        if not isinstance(other, BaseUnit) or self.units != other.units:
            raise TypeError(f"Can't add {self.__class__.__name__} and {other.__class__.__name__}")
        val = self.normal_dist + other.normal_dist
        return BaseUnit.new(val.mean, self.units, val.stdev)

    def __radd__(self, other: Number):
        if type(other) in [float, int]:
            other = Dimensionless(other)
        return other.__add__(self)

    def __sub__(self, other: Number):
        return self.__add__(-other)

    def __rsub__(self, other: Number):
        if type(other) in [float, int]:
            other = Dimensionless(other)
        return other.__sub__(self)

    def __pow__(self, power, modulo=None):
        """Does nothing with the modulo"""
        if type(power) not in [int, float, Dimensionless]:
            raise TypeError(f"Can't raise {self.__class__.__name__} to the power of {power.__class__.__name__}")
        if type(power) == Dimensionless:
            val = self.normal_dist ** power.normal_dist
            power = power.x  # For multiplication with self.units
        else:
            val = self.normal_dist ** power
        return BaseUnit.new(val.mean, self.units * power, val.stdev)

    def __rpow__(self, other):
        if type(other) not in [int, float]:
            raise TypeError(f"Base type has to be a number, not of type {type(other)}")
        # Convert power base to Dimensionless and use internal __pow__ method
        return Dimensionless(other) ** self

    def __str__(self):
        p = 0 if self.error == 0 else 1
        return_str = f"{self.x} "
        if not isinstance(self, Dimensionless):
            return_str += f"{self.units} "
        return return_str + f"(±{self.error_percentage:.{p}f}%, ±{self.error})"
        # oom = self.__oom()
        # return f"{self.x / 10 ** oom} {self.units.get_prefix(oom)}{self.units}"

    def __repr__(self):
        if BaseUnit.get_type(self.units) == BaseUnit:
            return f"{self.__class__.__name__}({self.x}, {repr(self.units)}, error={self.error})"
        return f"{self.__class__.__name__}({self.x}, error={self.error})"


class Dimensionless(BaseUnit):
    units = SI({})


class Time(BaseUnit):
    units = SI({"s": 1})

    @classmethod
    def from_minute(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=60)


class Length(BaseUnit):
    units = SI({"m": 1})

    def get_mm(self) -> Number:
        return self.x * 1E3

    def get_um(self) -> Number:
        return self.x * 1E6

    @classmethod
    def from_mm(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-3)

    @classmethod
    def from_um(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-6)

    def __str__(self):
        if self.x < 1E2:
            return super().__str__() + f" ({self.get_mm():.6f} mm)"
        else:
            return super().__str__()


class Mass(BaseUnit):
    units = SI({"kg": 1})


class Temperature(BaseUnit):
    units = SI({"K": 1})

    def get_celcius(self) -> Number:
        return self.x - 273.15

    @classmethod
    def from_celcius(cls, t: float, error: float = 0):
        return cls(t, error=error, _constant=273.15)

    def __str__(self):
        return super().__str__() + f" ({self.get_celcius():.2f} C)"


class Area(BaseUnit):
    units = SI({"m": 2})

    def get_mm2(self):
        return self.x * 1E6

    def get_um2(self):
        return self.x * 1E12

    @classmethod
    def from_cm2(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-4)

    @classmethod
    def from_mm2(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-6)

    def __str__(self):
        if self.x < 1E4:
            return super().__str__() + f" ({self.get_mm2():.6f} mm2)"
        else:
            return super().__str__()


class Volume(BaseUnit):
    units = SI({"m": 3})
    name = "m3"

    def get_cm3(self):
        return self.x * 1E6

    def get_liter(self):
        return self.x * 1E3

    @classmethod
    def from_cm3(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-6)

    @classmethod
    def from_liter(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-3)


class MassFlow(BaseUnit):
    units = SI({"kg": 1, "s": -1})
    name = "kg/s"

    def get_mgps(self) -> Number:
        return self.x * 1E6

    @classmethod
    def from_mgps(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-6)

    def __str__(self):
        return super().__str__() + f" ({self.get_mgps():.4f} mg/s)"


class VolumetricFlow(BaseUnit):
    units = SI({"m3": 1, "s": -1})
    name = "m3/s"


class Speed(BaseUnit):
    units = SI({"m": 1, "s": -1})
    name = "m/s"


class Acceleration(BaseUnit):
    units = SI({"m": 1, "s": -2})


class Force(BaseUnit):
    units = SI({"kg": 1, "m": 1, "s": -2})
    name = "N"

    def get_mN(self):
        return self.x * 1E3

    @classmethod
    def from_mN(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-3)

    def __str__(self):
        return super().__str__() + f" ({self.get_mN():.2f} mN)"


class Joule(BaseUnit):
    units = SI({"N": 1, "m": 1})
    name = "J"


class Pressure(BaseUnit):
    units = SI({"N": 1, "m": -2})
    name = "Pa"

    def get_bar(self) -> Number:
        return self.x / 1E5

    def get_mbar(self) -> Number:
        return self.x / 1E2

    @classmethod
    def from_bar(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E5)

    @classmethod
    def from_atm(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=101325)

    @classmethod
    def from_mbar(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E2)

    @classmethod
    def from_kpa(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E3)

    @classmethod
    def from_hpa(cls, x: float, error: float = 0):
        return cls.from_mbar(x, error=error)

    def __str__(self):
        return super().__str__() + f" ({self.get_bar():.3f} bar)"


class Density(BaseUnit):
    units = SI({"kg": 1, "m3": -1})
    name = "kg/m3"

    @classmethod
    def from_gpcm3(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E3)


class Watt(BaseUnit):
    units = SI({"J": 1, "s": -1})
    name = "W"


class Current(BaseUnit):
    units = SI({"A": 1})


class Voltage(BaseUnit):
    units = SI({"W": 1, "A": -1})
    name = "V"


class Resistance(BaseUnit):
    units = SI({"V": 1, "A": -1})
    name = "Ω"


class Frequency(BaseUnit):
    units = SI({"s": -1})
    name = "Hz"


class SpecificEnergy(BaseUnit):
    units = SI({"J": 1, "kg": -1})
    name = "J/kg"


class SpecificHeatCapacity(BaseUnit):
    units = SI({"J": 1, "kg": -1, "K": -1})
    name = "J/(kg*K)"


class MolarSpecificHeatCapacity(BaseUnit):
    units = SI({"J": 1, "mol": -1, "K": -1})
    name = "J/(mol*K)"


class Viscosity(BaseUnit):
    units = SI({"Pa": 1, "s": 1})


class MolarMass(BaseUnit):
    units = SI({"kg": 1, "mol": -1})

    @classmethod
    def from_gpmol(cls, x: float, error: float = 0):
        return cls(x, error=error, _factor=1E-3)


class HeatCapacity(BaseUnit):
    units = SI(J=1, K=-1)
    name = "J/K"


class Percentage(float):
    """Wrapper class to denote a percentage instead of an absolute value"""
    def __new__(cls, value):
        return float.__new__(cls, value / 100.)

    def __init__(self, value):
        float.__init__(value / 100.)

    def __str__(self):
        return f"{self*100}%"


class NormalDistExt(NormalDist):
    """Extended NormalDist which allows additional operations of NormalDist"""

    # https://www.geol.lsu.edu/jlorenzo/geophysics/uncertainties/Uncertaintiespart2.html
    # http://ipl.physics.harvard.edu/wp-uploads/2013/03/PS3_Error_Propagation_sp13.pdf
    # https://en.wikipedia.org/wiki/Propagation_of_uncertainty

    def __init__(self, mu=0.0, sigma=1.0):
        super().__init__(mu, sigma)

    @classmethod
    def from_normaldist(cls, item):
        """Convert super class to this class"""
        return cls(item._mu, item._sigma)

    def __add__(self, other):
        return self.from_normaldist(super().__add__(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self.from_normaldist(super().__sub__(other))

    def __pos__(self):
        return self.from_normaldist(super().__pos__())

    def __neg__(self):
        return self.from_normaldist(super().__neg__())

    def __mul__(x1, x2):
        """Multiply both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        if isinstance(x2, NormalDist):
            x3_mu = x1._mu * x2._mu
            x3_sigma = hypot(x1._mu * x2._sigma, x2._mu * x1._sigma)
            return NormalDistExt(x3_mu, x3_sigma)
        return NormalDistExt(x1._mu * x2, x1._sigma * fabs(x2))

    __rmul__ = __mul__

    def __truediv__(x1, x2):
        """Divide both mu and sigma by a constant.

        Used for rescaling, perhaps to change measurement units.
        Sigma is scaled with the absolute value of the constant.
        """
        if isinstance(x2, NormalDist):
            x3_mu = x1._mu / x2._mu
            x3_sigma = hypot(x1._sigma / x2._mu, x1._mu * x2._sigma / x2._mu ** 2)
            return NormalDistExt(x3_mu, x3_sigma)

        return NormalDistExt(x1._mu / x2, x1._sigma / fabs(x2))

    def __pow__(x1, x2):
        if not isinstance(x2, NormalDist):
            x3_mu = x1._mu ** x2
            if x2 == -1:  # Special case
                x3_sigma = x1._sigma
            else:
                x3_sigma = fabs(x3_mu * x2 * x1._sigma / x1._mu)
            return NormalDistExt(x3_mu, x3_sigma)

        x3_mu = x1._mu ** x2._mu
        term1 = x2._mu * x1._sigma / x1._mu
        term2 = log(x1._mu) * x2._sigma
        x3_sigma = fabs(x3_mu) * hypot(term1, term2)
        return NormalDistExt(x3_mu, x3_sigma)

    def __str__(self):
        return super(NormalDistExt, self).__str__() + f" ({100 * self._sigma / self._mu:.2f}%)"


if __name__ == '__main__':
    """Test cases"""
    assert Length(2) + Length(4) == Length(6)
    assert Area(8) / Length(2) == Length(4)
    assert 1 / Time(10) == Frequency(0.1)
    assert MassFlow(2) * Time(4) == Mass(8)
    assert 1 + Dimensionless(5) == 6
    assert 1 - Dimensionless(3) == -2
    assert -Mass(2) == Mass(2) * -1
    assert MassFlow(6) * 2 == MassFlow(12)
    assert MassFlow(6) / 2 == MassFlow(3)
    assert Dimensionless(4) - 2 == 2
    assert Dimensionless(3) + 2 == Dimensionless(5)
    assert BaseUnit(2, SI({"J": 1, "s": -1})) == Watt(2)
    assert BaseUnit(2, SI(J=1, s=-1)) == Watt(2)
    assert Watt(10) == Current(2) * Voltage(5)
    assert Resistance(10) == Voltage(20) / Current(2)
    assert SI(J=1, s=-1) == Watt(1).units
    # assert Length(2) ** 2 == Area(4)
    assert Time(120) == Time.from_minute(2)
    assert Time.from_minute(2).x == 120
    assert Temperature(293.15) == Temperature.from_celcius(20)
    assert Pressure.from_bar(1) == Pressure.from_mbar(1000)
    assert Speed(10, error=Percentage(15)) == Speed(10, error=1.5)
    assert Speed(10, error=Percentage(20)) != Speed(10, error=1)
    assert Pressure.from_bar(1, error=Percentage(1)) == Pressure(1E5, error=Percentage(1)) == \
           Pressure.from_bar(1, error=0.01) == Pressure(1E5, error=0.01E5)
    assert -NormalDistExt(2, 1) == NormalDistExt(-2, 1)
    temp = Temperature.from_celcius(100, error=Percentage(10))
    assert temp.error / temp.x == 0.1
    assert Temperature(100, error=10).error == Temperature.from_celcius(100, error=10).error
    time = Time(100, error=0)
    time.add_error(Percentage(1))
    time.add_error(1)
    assert time.error == 2 ** 0.5
    a = Length(0, error=0.1)
    b = Length(4, error=0.2)
    assert (a * b).error == 0.4
    assert (a / b).error == 0.025
    assert Area(Area(5)) == Area(5)
    assert Area(Area.from_cm2(5)) == Area.from_cm2(5)
    assert Pressure.from_kpa(10).get_mbar() == 100
