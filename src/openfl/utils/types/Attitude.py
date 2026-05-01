from enum import Enum

class Attitude(Enum):
    Honest = 0
    FreeRider = 1
    Malicious = 2
    Inactive = 3

    @classmethod
    def from_string(cls, string):
        if isinstance(string, cls):
            return string

        if "." in string:
            string = string.split(".")[-1]

        if string == "Honest":
            return Attitude.Honest
        elif string == "FreeRider":
            return Attitude.FreeRider
        elif string == "Malicious":
            return Attitude.Malicious
        elif string == "Inactive":
            return Attitude.Inactive
        else:
            raise ValueError(f"Invalid Attitude string: {string}")