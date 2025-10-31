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
    def __init__(self, noise_dict: dict, type=None):
        self.noises = noise_dict
        self.type = type if type else "generic"
        self.erasure = noise_dict.get('sqg-e', 0) > 0 or noise_dict.get('tqg-e', 0) > 0 or noise_dict.get('sp-e', 0) > 0
        self.pauli = noise_dict.get('sqg', 0) > 0 or noise_dict.get('tqg', 0) > 0 or noise_dict.get('sp', 0) > 0

    def __repr__(self):
        return f"Noise Model: {self.type}\n" + "\n".join([f"{k}: p={v}" for k, v in self.noises.items()])

class Pure_Erasure_Noise_Model(Noise_Model):
    def __init__(self, p: float):
        super().__init__({'sqg': p, 'tqg': p, 'sp-e': p}, type="pure-erasure")