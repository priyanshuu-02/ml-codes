"""
Robust IO-VNBD column resolution.

The dataset's headers carry mojibake in their unit suffixes: the degree sign arrives as
the two-character sequence "Â°" and the superscript two survives as "²". Matching on
exact header text therefore breaks on some columns and not others, purely depending on
which unit symbol a column happens to use.

Resolving on the meaningful part of the name and ignoring the unit suffix entirely
removes that whole class of failure.
"""
import re

_KEEP = re.compile(r"[^a-z0-9]+")


def normalise(text):
    """Lowercase and drop everything that is not a letter or digit."""
    return _KEEP.sub("", str(text).lower())


def resolve(dataframe, name):
    """
    Find the column whose normalised header begins with the normalised `name`.

    Pass the semantic part only, without units. "Heading" resolves
    "Heading (degrees)"; "ORIENTATION (Yaw)" resolves "ORIENTATION (Yaw) (Â°)" while
    leaving "GPS ORIENTATION (Â°)" alone.
    """
    wanted = normalise(name)
    if not wanted:
        raise KeyError("Empty column name requested")

    candidates = [column for column in dataframe.columns if normalise(column) == wanted]
    if len(candidates) == 1:
        return candidates[0]

    prefixed = [column for column in dataframe.columns if normalise(column).startswith(wanted)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        # Prefer the shortest, which is the least decorated match.
        return min(prefixed, key=lambda column: len(normalise(column)))

    raise KeyError(
        f"Column not found: {name!r}\nNormalised as: {wanted!r}\n"
        f"Available: {list(dataframe.columns)}"
    )


def resolve_all(dataframe, names):
    return [resolve(dataframe, name) for name in names]


# Semantic column names, free of unit suffixes.

SMARTPHONE_ACCELEROMETER = ["ACCELEROMETER X", "ACCELEROMETER Y", "ACCELEROMETER Z"]
SMARTPHONE_GRAVITY = ["GRAVITY X", "GRAVITY Y", "GRAVITY Z"]
SMARTPHONE_GYROSCOPE = ["GYROSCOPE Yaw", "GYROSCOPE Pitch", "GYROSCOPE Roll"]
SMARTPHONE_ORIENTATION = ["ORIENTATION (Yaw)", "ORIENTATION (Pitch)", "ORIENTATION (Roll)"]

VEHICLE_LATITUDE = "Latitude"
VEHICLE_LONGITUDE = "Longitude"
VEHICLE_VELOCITY = "Velocity"
VEHICLE_HEADING = "Heading"
VEHICLE_YAW_RATE = "Yaw Rate"
VEHICLE_LONGITUDINAL_G = "Indicated Longitudinal Acceleration"
VEHICLE_LATERAL_G = "Indicated Lateral Acceleration"
VEHICLE_TIME = "Time Since Start of Day"
