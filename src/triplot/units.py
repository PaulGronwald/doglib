"""Unit presets for tripartite plots."""
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSystem:
    name: str
    freq_label: str
    vel_label: str
    disp_label: str
    accel_label: str
    g_value: float  # accel unit in native velocity-time units (vel_unit / s)
    accel_in_g: bool  # True => accel axis shown in g's


IMPERIAL = UnitSystem(
    name="imperial",
    freq_label="Frequency [Hz]",
    vel_label="Pseudo-Velocity [in/s]",
    disp_label="Displacement [in]",
    accel_label="Acceleration [g]",
    g_value=386.089,
    accel_in_g=True,
)

SI = UnitSystem(
    name="SI",
    freq_label="Frequency [Hz]",
    vel_label="Pseudo-Velocity [m/s]",
    disp_label="Displacement [m]",
    accel_label="Acceleration [m/s\u00b2]",
    g_value=1.0,
    accel_in_g=False,
)

PRESETS = {"imperial": IMPERIAL, "SI": SI, "si": SI}


def resolve(units):
    if units is None:
        return IMPERIAL
    if isinstance(units, UnitSystem):
        return units
    if isinstance(units, str):
        try:
            return PRESETS[units]
        except KeyError:
            raise ValueError(f"unknown unit system {units!r}; choose from {list(PRESETS)}")
    raise TypeError(f"units must be str or UnitSystem, got {type(units).__name__}")
