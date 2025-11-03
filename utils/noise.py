"""Utility classes and functions for noise models.

Defines various noise models used in simulations and decoding.

The following noise types are defined:
- sp: state preparation Pauli error probability
- sp-e: state preparation erasure probability
- sqg: single-qubit gate Pauli error probability
- sqg-e: single-qubit gate erasure probability
- tqg: two-qubit gate Pauli error probability
- tqg-e: two-qubit gate erasure probability
- meas: measurement Pauli error probability
- meas-e: measurement erasure probability

Each noise model class encapsulates a specific type of noise configuration. To build a custom noise class, 
inherit from the Noise_Model base class and define the appropriate noise parameters in the noise_dict argument.
"""

class Noise_Model:
    def __init__(self, noise_dict: dict = None, model_name="generic", **kwargs):
        if noise_dict is not None and kwargs:
            raise TypeError("Provide either noise_dict or keyword arguments, not both.")
        
        self.name = model_name
        self.pauli_bitmask = 0b0
        self.erasure_bitmask = 0b0

        self.noise_dict = noise_dict if noise_dict is not None else kwargs

    def set_bitmasks(self, pauli_mask: int, erasure_mask: int):
        self.pauli_bitmask = pauli_mask
        self.erasure_bitmask = erasure_mask

    def __repr__(self):
        return f"Noise Model: {self.name}\n" + "\n".join([f"{k}: p={v}" for k, v in self.noise_dict.items()])
