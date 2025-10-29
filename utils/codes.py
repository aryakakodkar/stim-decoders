import numpy as np
import matplotlib.pyplot as plt
import stim
import itertools
import time

# TODO: work out how to do this more cleanly
from circuit import *
from noise import Noise_Model, Code_Capacity_Noise_Model, Pure_Erasure_Noise_Model

POSSIBLE_CNOT_ERRORS = list(itertools.product('IJXYZ', repeat=2))

def print_binary_bitmask(bitmask):
    # write a function that prints all indices of a bitmask that are set to 1
    indices = []
    for i in range(bitmask.bit_length()):
        if (bitmask >> i) & 1:
            indices.append(i)
    print(f"Bitmask {bin(bitmask)} has bits set at indices: {indices}")

def indices_to_mask(indices):
    m = 0
    shift = (1).__lshift__  # localize to avoid global lookup
    or_ = int.__or__        # same trick
    for i in indices:
        m = or_(m, shift(i))
    return m

def split_support_list_fast(support_indices, pauli_mask, erasure_mask):
    """
    Optimized version with localized bit operations.
    """
    pauli_indices = []
    erasure_indices = []
    
    pm = pauli_mask
    em = erasure_mask
    
    for idx in support_indices:
        if (pm >> idx) & 1:
            pauli_indices.append(idx)
        if (em >> idx) & 1:
            erasure_indices.append(idx)
    
    return pauli_indices, erasure_indices

class _Qubit:
    """
    Utility class representing a qubit

    Args:
        type (str): Type of qubit ('data', 'x-ancilla', 'z-ancilla')
        coords (tuple): Coordinates of the qubit in the lattice
        id (int, optional): Unique identifier for the qubit. Defaults to None.
    """
    def __init__(self, type: str, coords: tuple, id: int = None):
        self._id = id
        self._coords = coords
        self._type = type

    def __repr__(self):
        return f"Qubit ID: {self.id}, Type: {self.type}, Coords: {self.coords}"

    @property
    def type(self):
        return self._type

    @property
    def coords(self):
        return self._coords
    
    @property
    def id(self):
        return self._id

## Generic stabilizer code class
class _Stabilizer_Code():
    def __init__(self, distance: int, check_density: int):
        self.distance = distance
        self.gates = [[] for _ in range(check_density)]

        self.observable = []
        self.plaquettes = {}

        self.data_qubits = {}
        self.x_ancillas = {}
        self.z_ancillas = {}

        self.all_qubit_ids = []
        self.data_qubit_ids = []
        self.ancilla_ids = []
        self.x_ancilla_ids = []
        self.z_ancilla_ids = []

        self.qubit_ids = {}

    def add_data_qubit(self, coord: tuple, id: int):
        self.data_qubits[coord] = _Qubit(type='data', coords=coord, id=id)
        self.qubit_ids[id] = self.data_qubits[coord]
        self.all_qubit_ids.append(id)
        self.data_qubit_ids.append(id)

    def add_x_ancilla(self, coord: tuple, id: int):
        self.x_ancillas[coord] = _Qubit(type='x-ancilla', coords=coord, id=id)
        self.qubit_ids[id] = self.x_ancillas[coord]
        self.plaquettes[id] = []
        self.all_qubit_ids.append(id)
        self.ancilla_ids.append(id)
        self.x_ancilla_ids.append(id)

    def add_z_ancilla(self, coord: tuple, id: int):
        self.z_ancillas[coord] = _Qubit(type='z-ancilla', coords=coord, id=id)
        self.qubit_ids[id] = self.z_ancillas[coord]
        self.plaquettes[id] = []
        self.all_qubit_ids.append(id)
        self.ancilla_ids.append(id)
        self.z_ancilla_ids.append(id)

    def qubit_with_id(self, id: int) -> _Qubit:
        return self.qubit_ids.get(id)
    
    def state_preparation_with_erasure(self, qubit_ids: list, p: float) -> str:
        """
        Generates a heralded erasure error (w/ detector) for state preparation on qubit_ids with probability p.

        Args:
            qubit_ids (list): List of qubit IDs to apply the heralded erasures to.
            p (float): Probability of erasure error.

        Returns:
            str: A string representing the heralded erasure error in the circuit.
        """
        prep_string = ""
        targets = ["X", "Y", "Z"]

        for qubit_id in qubit_ids:
            p_space = 1 - (p/4)
            err_string = f"CORRELATED_ERROR({p/4:.8f}) X{qubit_id + self.eq_diff}\n"
            for target in targets:
                err_string += f"ELSE_CORRELATED_ERROR({(p/4)/(p_space):.8f}) {target}{qubit_id} X{qubit_id + self.eq_diff}\n"
                p_space -= (p/4)

            prep_string += err_string

        prep_string += "MR " + " ".join([f"{qid+self.eq_diff}" for qid in qubit_ids]) + "\n"
        for i in range(len(qubit_ids), 0, -1):
            prep_string += f"DETECTOR rec[-{i}]\n"

        return prep_string

    @staticmethod
    def depolarize1(qubit_ids: list, p: float) -> str:
        """
        Generates a depolarizing error string for single-qubit depolarizing noise with support on qubit_ids and probability p.

        Args:
            qubit_ids (list): List of qubit IDs to apply the depolarizing errors to.
            p (float): Probability of depolarizing error.

        Returns:
            str: A string representing the depolarizing error in the circuit.
        """
        if p <= 0:
            return ""
        
        return f"DEPOLARIZE1({p}) " + " ".join([f"{qid}" for qid in qubit_ids]) + "\n"
    
    @staticmethod
    def depolarize2(qubit_ids: list, p: float) -> str:
        """
        Generates a depolarizing error string for two-qubit depolarizing noise with support on qubit_ids and probability p.

        Args:
            qubit_ids (list): List of tuples containing the two qubit IDs to apply the depolarizing errors to.
            p (float): Probability of depolarizing error.

        Returns:
            str: A string representing the depolarizing error in the circuit.
        """
        return f"DEPOLARIZE2({p}) " + " ".join([f"{q1} {q2}" for q1, q2 in qubit_ids]) + "\n"

    @staticmethod
    def erase1(qubit_ids: list, p: float) -> str:
        """
        Generates a heralded erasure error (w/o detector) for single-qubit depolarizing noise with support on qubit_ids and probability p.

        Args:
            qubit_ids (list): List of qubit IDs to apply the heralded erasures to.
            p (float): Probability of erasure error.

        Returns:
            str: A string representing the heralded erasure error in the circuit.
        """
        return f"HERALDED_ERASE({p}) " + " ".join([f"{qid}" for qid in qubit_ids]) + "\n"

    @staticmethod
    def noiseless_cnot(gates: list) -> str:
        """
        Generates a noiseless CNOT gate string for the given gates.

        Args:
            gates (list): List of tuples containing target and control qubit IDs for each CNOT gate.
        """
        return f"CX " + " ".join([f"{target} {control}" for target, control in gates]) + "\n"

    def noisy_cnot(self, gates: list, p: float) -> str:
        """
        Generates a noisy CNOT gate string with depolarizing noise for the given gates and probability p.

        Args:
            gates (list): List of tuples containing target and control qubit IDs for each CNOT gate.
            gate_index (float): Index to track the gate position (not used in this implementation).
            p (float): Probability of depolarizing error.
        """
        error_string = ""
        error_string += f"CX " + " ".join([f"{target} {control}" for target, control in gates]) + "\n"
        if p > 0:
            error_string += self.depolarize2(gates, p)
        
        return error_string

    def erased_cnot(self, gates: list, p: float) -> str:
        """
        Generates a noisy CNOT gate string with erasure noise for the given gates and probability p.

        Args:
            gates (list): List of tuples containing target and control qubit IDs for each CNOT gate.
            gate_index (float): Index to track the gate position (not used in this implementation).
            p (float): Probability of erasure error.
        """
        gate_string = ""
        gate_string += f"CX " + " ".join([f"{target} {control}" for target, control in gates]) + "\n"
        if p > 0:
            err_string = ""

            error_probs = {}

            for error in POSSIBLE_CNOT_ERRORS:
                error_probs[error] = p/15 # assumes perfectly depolarizing errors

            del error_probs[('I', 'I')]

            for gate in gates:
                else_space = 1
                correlated_err_string = ""

                for err, prob in error_probs.items():
                    normalized_prob = prob / else_space
                    else_space -= prob

                    target_string = ""
                    for i, target in enumerate(err):
                        if target not in ["I", "J"]:
                            target_string += f"{target}{gate[i]} X{gate[i]+self.eq_diff} "
                        elif target == "J":
                            target_string += f"X{gate[i]+self.eq_diff} "

                    else_marker = "ELSE_"
                    if correlated_err_string == "":
                        else_marker = ""
                    correlated_err_string += f"{else_marker}CORRELATED_ERROR({normalized_prob:.8f}) {target_string}\n"
                
                err_string += correlated_err_string

        return gate_string + err_string
    
    @staticmethod
    def noisy_hadamard(qubit_ids: list, p: float) -> str:
        """
        Generates a noisy Hadamard gate string with depolarizing noise for the given gates and probability p.

        Args:
            qubit_ids (list): List of qubit IDs to apply the noisy Hadamard gates to.
            p (float): Probability of depolarizing error.
        """
        error_string = f"H " + " ".join([f"{qubit_id}" for qubit_id in qubit_ids]) + "\n"
        if p > 0:
            error_string += _Stabilizer_Code.depolarize1(qubit_ids, p)

        return error_string

    @staticmethod
    def erased_hadamard(qubit_ids: list, p) -> str:
        """
        Generates a heralded erasure error (w/ detector) for single-qubit depolarizing noise with support on qubit_ids and probability p.

        Args:
            qubit_ids (list): List of qubit IDs to apply the heralded erasures to.
            p (float): Probability of erasure error.
        """
        error_string = f"H " + " ".join([f"{qubit_id}" for qubit_id in qubit_ids]) + "\n"
        if p > 0:
            error_string += _Stabilizer_Code.erase1(qubit_ids, p) + "DETECTOR rec[-1]\n"

        return error_string

class RSC(_Stabilizer_Code):
    def __init__(self, distance: int):
        super().__init__(distance, check_density=4)
        self._build_lattice()
        self._build_checks()

        self.eq_diff = max(self.qubit_ids.keys())

    def _build_lattice(self):
        for index in range((2*self.distance + 1)*(self.distance + 1)):
            x = index % (2*self.distance + 1)
            y = 2*(index // (2*self.distance + 1)) + (x % 2)

            if x % 2 == 1 and y % 2 == 1 and x < 2*self.distance and y < 2*self.distance:
                self.add_data_qubit((x, y), index)
                if y == 1:
                    self.observable.append(index)
            elif (x + y) % 4 != 0 and x > 1 and x < (2*self.distance - 1) and y < (2*self.distance + 1):
                self.add_x_ancilla((x, y), index)
            elif (x + y) % 4 == 0 and x < (2*self.distance + 1) and y > 1 and y < (2*self.distance - 1):
                self.add_z_ancilla((x, y), index)

        self.eq_diff = self.x_ancilla_ids[-1]

    def _build_checks(self):
        self.cnot_bitmasks = [0b0, 0b0, 0b0, 0b0]
        x_order = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        z_order = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        for check_num in range(4):
            for x_ancilla in self.x_ancillas.values():
                if qubit := self.data_qubits.get((x_ancilla.coords[0] + x_order[check_num][0], x_ancilla.coords[1] + x_order[check_num][1])):
                    self.gates[check_num].append((x_ancilla.id, qubit.id))
                    self.plaquettes[x_ancilla.id].append(qubit.id)
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << x_ancilla.id)) << self.eq_diff)

            for z_ancilla in self.z_ancillas.values():
                if qubit := self.data_qubits.get((z_ancilla.coords[0] + z_order[check_num][0], z_ancilla.coords[1] + z_order[check_num][1])):
                    self.gates[check_num].append((qubit.id, z_ancilla.id))
                    self.plaquettes[z_ancilla.id].append(qubit.id)
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << z_ancilla.id)) << self.eq_diff)

    def draw_lattice(self, ax=None, numbering=True):
        """
        Draws the lattice of the RSC code.

        Args:
            ax: Matplotlib axis to draw on. If None, a new figure and axis are created.
            numbering (bool): Whether to number the qubits with their IDs.

        Returns:
            Matplotlib axis: The axis with the drawn lattice.
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.invert_yaxis()

        for coord in self.data_qubits.keys():
            ax.scatter(coord[0], coord[1], c='blue', s=200)
            if numbering: ax.text(coord[0], coord[1], f"{self.data_qubits[coord].id}", fontsize=10, color='white', ha='center', va='center')

        for coord in self.x_ancillas.keys():
            ax.scatter(coord[0], coord[1], c='green', s=200)
            if numbering: ax.text(coord[0], coord[1], f"{self.x_ancillas[coord].id}", fontsize=10, color='white', ha='center', va='center')

        for coord in self.z_ancillas.keys():
            ax.scatter(coord[0], coord[1], c='red', s=200)
            if numbering: ax.text(coord[0], coord[1], f"{self.z_ancillas[coord].id}", fontsize=10, color='white', ha='center', va='center')

        if ax is None:
            plt.show()

        return ax

    def draw_checks(self):
        """
        Draws the CNOT gates for each check of the RSC code.

        Returns:
            ax: Matplotlib axis with the drawn checks.
        """
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for check_num, ax in enumerate(axs.flatten()):
            ax.set_title(f"Check {check_num + 1}")
            self.draw_lattice(ax=ax, numbering=False)
            for gate in self.gates[check_num]:
                control = self.qubit_with_id(gate[0])
                target = self.qubit_with_id(gate[1])

                color = 'green' if control.type == 'x-ancilla' else 'red'

                ax.annotate('', xy=target.coords, xycoords='data',
                            xytext=control.coords, textcoords='data',
                            arrowprops=dict(arrowstyle="->", color=color))

        plt.show()

        return axs
        
    # an attempt to create the stim circuit more efficiently
    def build_stim_circuit(self, noise_dict: dict = None):
        circuit = Circuit()

        post_hadamard_errors = 0

        ## Create bitmasks for erasure and pauli errors (e.g. erasure_qubits = 0b10101 means qubits 0, 2, 4 have erasure errors)
        pauli_qubits = noise_dict.get('pauli-qubits', 0)
        erasure_qubits = noise_dict.get('erasure-qubits', 0)

        ## State preparation
        circuit.add_reset(self.all_qubit_ids)
        sp_pauli, sp_erasure = split_support_list_fast(self.data_qubit_ids, pauli_qubits, erasure_qubits)

        if sp_pauli and (p_sp_pauli := noise_dict.get('sp', 0)) > 0:
            circuit.add_depolarize1(sp_pauli, p_sp_pauli)
        if sp_erasure and (p_sp_erasure := noise_dict.get('sp-e', 0)) > 0:
            circuit.add_erasure1(sp_erasure, p_sp_erasure)

        ## Hadamards on X-ancillas
        circuit.add_h_gate(self.x_ancilla_ids, index=True)
    
        hadamard_pauli, hadamard_erasure = split_support_list_fast(self.x_ancilla_ids, pauli_qubits, erasure_qubits)
        if hadamard_pauli and (p_had_pauli := noise_dict.get('sqg', 0)) > 0:
            circuit.add_depolarize1(hadamard_pauli, p_had_pauli)
            post_hadamard_errors += 1
        if hadamard_erasure and (p_had_erasure := noise_dict.get('sqg-e', 0)) > 0:
            circuit.add_erasure1(hadamard_erasure, p_had_erasure)
            post_hadamard_errors += 1

        ## CNOT rounds
        fic_ancillas_bitmask = erasure_qubits << self.eq_diff
        for i, check_gates in enumerate(self.gates):
            circuit.add_symmetric_pauli_erasure_cnot(check_gates, erasure_qubits, noise_dict.get('tqg-e', 0), self.eq_diff)
            fic_ancillas_intersect_bitmask = fic_ancillas_bitmask & self.cnot_bitmasks[i]
            fic_ancillas_intersect = list(mask_iter_indices(fic_ancillas_intersect_bitmask))
            if (p_meas_pauli := noise_dict.get('meas', 0)) > 0:
                circuit.add_depolarize1(fic_ancillas_intersect, p_meas_pauli)
            circuit.add_measurements(fic_ancillas_intersect)

        ## Hadamards on X-ancillas again
        circuit.append_to_circ_str(circuit.circ_str[circuit.hadamard_index:circuit.hadamard_index + post_hadamard_errors + 1])  # Reuse the hadamard gate string

        ## Measure all ancillas
        circuit.add_measurements(self.x_ancilla_ids)
        circuit.add_measurements(self.z_ancilla_ids)
        ancilla_meas_pauli, ancilla_meas_erasure = split_support_list_fast(self.ancilla_ids, pauli_qubits, erasure_qubits)
        if ancilla_meas_pauli and (p_anc_meas_pauli := noise_dict.get('meas', 0)) > 0:
            circuit.add_depolarize1(ancilla_meas_pauli, p_anc_meas_pauli)
        if ancilla_meas_erasure and (p_anc_meas_erasure := noise_dict.get('meas', 0)) > 0:
            circuit.add_erasure1(ancilla_meas_erasure, p_anc_meas_erasure)

        ## Measure all data qubits
        n_z_ancillas = (self.distance**2 - 1) // 2
        circuit.add_detectors(range(n_z_ancillas, 0, -1))

        ## Measure plaquettes
        
        return circuit.to_stim_circuit()

    def to_stim_circuit(self, noise_model: Noise_Model = None):
        start = time.time()
        circuit_string = ""

        gate_index = 0
        ancillas_fic = []

        # gates 
        h_gate = self.noisy_hadamard
        cnot_gate = self.noisy_cnot
        sp_gate = self.depolarize1

        if type(noise_model) == Pure_Erasure_Noise_Model:
            ancillas_fic = [qubit.id+self.eq_diff for qubit in self.qubit_ids.values()] # fictitious ancillas for CNOT erasure detection

            h_gate = self.erased_hadamard
            cnot_gate = self.erased_cnot
            sp_gate = self.state_preparation_with_erasure

        # noise model
        noise_dict = {"sqg": 0.001, "tqg": 0.001} if noise_model is None else noise_model.noises

        # some useful quantities
        qubit_ids = [qubit.id for qubit in (list(self.x_ancillas.values()) + list(self.z_ancillas.values()) + list(self.data_qubits.values()))]
        n = len(qubit_ids) # total number of qubits
        n_ancillas = n - len(self.data_qubits) # total number of ancillas
        gates_per_check = len(self.gates[0])

        # instantiate all qubits
        for qubit_id in qubit_ids:
            circuit_string += f"QUBIT_COORDS{self.qubit_ids[qubit_id].coords} {qubit_id}\n"

        # reset all qubits
        circuit_string += "R " + " ".join([f"{qubit_id}" for qubit_id in qubit_ids]) + " " + " ".join([f"{qubit_id}" for qubit_id in ancillas_fic]) + "\n"
        
        # add state-preparation noise
        if "sp" in noise_dict: circuit_string += sp_gate([qubit.id for qubit in self.data_qubits.values()], noise_dict["sp"])
        circuit_string += "TICK\n"

        # perform starting hadamard gates
        h_string = ""
        for x_ancilla in self.x_ancillas.values():
            h_string += h_gate([x_ancilla.id], noise_dict.get("sqg", 0))
            gate_index += 1

        circuit_string += h_string
        circuit_string += "TICK\n"

        # add gate rounds
        for check_gates in self.gates:
            # apply cnot gates for this check
            circuit_string += cnot_gate(check_gates, noise_dict.get("tqg", 0))
            checked_ancillas = [i + self.eq_diff for i in sum(check_gates, ())]
            circuit_string += "MR " + " ".join([f"{id}" for id in checked_ancillas]) + "\n"
            for i in range(len(checked_ancillas), 0, -1):
                circuit_string += f"DETECTOR rec[-{i}]\n" # TODO: FIX
            gate_index += gates_per_check
            circuit_string += "TICK\n"

        # hadamards again
        circuit_string += h_string
        circuit_string += "TICK\n"

        # measure all ancillas
        # TODO: add measurement noise

        circuit_string += "M " + " ".join([f"{qubit.id}" for qubit in (list(self.x_ancillas.values()) + list(self.z_ancillas.values()))]) + "\n"

        plaquette_string = ""
        for z_ancilla in self.z_ancillas.values():
            circuit_string += f"DETECTOR rec[-{n_ancillas - qubit_ids.index(z_ancilla.id)}]\n"
            plaquette_string += f"DETECTOR " + " ".join(f"rec[-{n - qubit_ids.index(data_qubit_id)}]" for data_qubit_id in self.plaquettes[z_ancilla.id]) + f" rec[-{n - qubit_ids.index(z_ancilla.id)}]" + "\n"

        circuit_string += "M " + " ".join([f"{qubit.id}" for qubit in self.data_qubits.values()]) + "\n" + plaquette_string

        circuit_string += "OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{n - qubit_ids.index(qubit_id)}]" for qubit_id in self.observable) + "\n"

        print("Generated stim circuit in", time.time() - start)
        return stim.Circuit(str(circuit_string))
    
    def to_stim_file(self, noise_model: Noise_Model = None, filename: str = "rsc_circuit.txt"):
        circuit = self.to_stim_circuit(noise_model=noise_model)
        with open(filename, "w") as f:
            f.write(str(circuit))
        return str(circuit)
    
    def to_stim_circuit_with_erasures(self, erasures: list):
        circuit_string = ""
        det_index = 0

        qubit_ids = [qubit.id for qubit in (list(self.x_ancillas.values()) + list(self.z_ancillas.values()) + list(self.data_qubits.values()))]
        indices = {qubit_id: idx for idx, qubit_id in enumerate(qubit_ids)}
        n = len(qubit_ids)
        n_ancillas = n - len(self.data_qubits)

        # instantiate all qubits
        for qubit_id in qubit_ids:
            circuit_string += f"QUBIT_COORDS{self.qubit_ids[qubit_id].coords} {qubit_id}\n"

        # reset all qubits
        circuit_string += "R " + " ".join([f"{qubit_id}" for qubit_id in qubit_ids]) + "\n"

        # TODO: add state-preparation noise
        for data_qubit in self.data_qubits.values():
            p = 0.75*erasures[det_index]
            circuit_string += self.depolarize1([data_qubit.id], p)
            det_index += 1

        # Ancilla noise
        for x_ancilla in self.x_ancillas.values():
            p = 0.75*erasures[det_index]
            circuit_string += self.noisy_hadamard([x_ancilla.id], p)
            det_index += 1

        circuit_string += "TICK\n"

        for round in range(4):
            # apply cnot gates for this check
            for gate in self.gates[round]:  
                circuit_string += self.noiseless_cnot([gate])
                circuit_string += self.depolarize1([gate[0]], 0.75*erasures[det_index])
                circuit_string += self.depolarize1([gate[1]], 0.75*erasures[det_index + 1])
                det_index += 2

            circuit_string += "TICK\n"

        # hadamards again
        for x_ancilla in self.x_ancillas.values():
            p = 0.75*erasures[det_index]
            circuit_string += self.noisy_hadamard([x_ancilla.id], p)
            det_index += 1

        circuit_string += "TICK\n"

        circuit_string += "M " + " ".join([f"{qubit.id}" for qubit in (list(self.x_ancillas.values()) + list(self.z_ancillas.values()))]) + "\n"
        
        plaquette_string = ""
        for z_ancilla in self.z_ancillas.values():
            circuit_string += f"DETECTOR rec[-{n_ancillas - indices[z_ancilla.id]}]\n"
            plaquette_string += f"DETECTOR " + " ".join(f"rec[-{n - indices[data_qubit_id]}]" for data_qubit_id in self.plaquettes[z_ancilla.id]) + f" rec[-{n - indices[z_ancilla.id]}]" + "\n"

        circuit_string += "M " + " ".join([f"{qubit.id}" for qubit in self.data_qubits.values()]) + "\n" + plaquette_string

        circuit_string += "OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{n - indices[qubit_id]}]" for qubit_id in self.observable) + "\n"

        return stim.Circuit(circuit_string)

    def to_stim_file_with_erasures(self, erasures: list, filename: str = "rsc_circuit_erased.txt"):
        circuit = self.to_stim_circuit_with_erasures(erasures, filename=filename)
        with open(filename, "w") as f:
            f.write(str(circuit))
        return str(circuit)