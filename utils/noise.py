class Noise_Model:
    def __init__(self, noise_dict: dict, type=None):
        self.noises = noise_dict
        self.type = type if type else "generic"
        self.erasure = noise_dict.get('sqg-e', 0) > 0 or noise_dict.get('tqg-e', 0) > 0 or noise_dict.get('sp-e', 0) > 0
        self.pauli = noise_dict.get('sqg', 0) > 0 or noise_dict.get('tqg', 0) > 0 or noise_dict.get('sp', 0) > 0

    def __repr__(self):
        return f"Noise Model: {self.type}\n" + "\n".join([f"{k}: p={v}" for k, v in self.noises.items()])

class Code_Capacity_Noise_Model(Noise_Model):
    def __init__(self, p: float):
        super().__init__({'sqg': p, 'tqg': p}, type="code-capacity")

class Pure_Erasure_Noise_Model(Noise_Model):
    def __init__(self, p: float):
        super().__init__({'sqg': p, 'tqg': p, 'sp-e': p}, type="pure-erasure")