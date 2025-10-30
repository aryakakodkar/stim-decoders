import numpy as np
import matplotlib.pyplot as plt
import stim
import time

# TODO: work out how to do this more cleanly
from circuit import *
from bitops import *

# TODO: Docstring
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

# TODO: Docstring
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

class RSC(_Stabilizer_Code):
    def __init__(self, distance: int):
        super().__init__(distance, check_density=4)
        self.num_ancillas = distance**2 - 1
        self.num_qubits = 2*(distance**2) - 1
        self.measurement_indices = {}

        self.sp_support = 0b0
        self.hadamard_support = 0b0
        self.ancilla_measure_support = 0b0

        self.cached_strings = []

        self._build_lattice()
        self._build_checks()

    def _build_lattice(self):
        """Builds the rotated surface code lattice."""
        x_index = 0
        z_index = self.num_ancillas // 2
        data_index = self.num_ancillas

        for index in range((2*self.distance + 1)*(self.distance + 1)):
            x = index % (2*self.distance + 1)
            y = 2*(index // (2*self.distance + 1)) + (x % 2)

            if x % 2 == 1 and y % 2 == 1 and x < 2*self.distance and y < 2*self.distance:
                self.add_data_qubit((x, y), index)
                self.measurement_indices[index] = data_index
                self.sp_support |= (1 << index)
                data_index += 1
                if y == 1:
                    self.observable.append(index)
            elif (x + y) % 4 != 0 and x > 1 and x < (2*self.distance - 1) and y < (2*self.distance + 1):
                self.add_x_ancilla((x, y), index)
                self.measurement_indices[index] = x_index
                self.hadamard_support |= (1 << index)
                self.ancilla_measure_support |= (1 << index)
                x_index += 1
            elif (x + y) % 4 == 0 and x < (2*self.distance + 1) and y > 1 and y < (2*self.distance - 1):
                self.add_z_ancilla((x, y), index)
                self.measurement_indices[index] = z_index
                self.ancilla_measure_support |= (1 << index)
                z_index += 1

        self.eq_diff = self.x_ancilla_ids[-1]

    def _build_checks(self):
        """Builds the CNOT gates for each check of the RSC code."""
        self.cnot_bitmasks = [0b0, 0b0, 0b0, 0b0]
        x_order = [(-1, 1), (1, 1), (-1, -1), (1, -1)]
        z_order = [(-1, 1), (-1, -1), (1, 1), (1, -1)]
        for check_num in range(4):
            for x_ancilla in self.x_ancillas.values():
                if qubit := self.data_qubits.get((x_ancilla.coords[0] + x_order[check_num][0], x_ancilla.coords[1] + x_order[check_num][1])):
                    self.gates[check_num].append((x_ancilla.id, qubit.id))
                    # self.plaquettes[x_ancilla.id].append(qubit.id) # unnecessary for code initialization in Z eigenstate
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << x_ancilla.id)) << self.eq_diff)

            for z_ancilla in self.z_ancillas.values():
                if qubit := self.data_qubits.get((z_ancilla.coords[0] + z_order[check_num][0], z_ancilla.coords[1] + z_order[check_num][1])):
                    self.gates[check_num].append((qubit.id, z_ancilla.id))
                    self.plaquettes[z_ancilla.id].append(qubit.id)
                    self.cnot_bitmasks[check_num] |= (((1 << qubit.id) | (1 << z_ancilla.id)) << self.eq_diff)

    def _supports(self, pauli_bitmask):
        """Determines support lists for various operations based on error bitmasks.
        
        Args:
            pauli_bitmask: Integer bitmask for Pauli errors. (WARNING: assumes that all non-Pauli qubits are Erasures)
        
        Returns:
            Tuple[List[int], List[int], List[int], List[int], List[int], List[int]]: Support lists for the following error operations: sp_pauli
                sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure
        """
        sp_pauli = []
        sp_erasure = []
        hadamard_pauli = []
        hadamard_erasure = []
        ancilla_meas_pauli = []
        ancilla_meas_erasure = []

        for i in self.all_qubit_ids:
            if (pauli_bitmask >> i) & 1:
                if (self.sp_support >> i) & 1: 
                    sp_pauli.append(i)
                    continue

                ancilla_meas_pauli.append(i)
                if (self.hadamard_support >> i) & 1: hadamard_pauli.append(i)
            else:
                if (self.sp_support >> i) & 1: 
                    sp_erasure.append(i)
                    continue

                ancilla_meas_erasure.append(i)
                if (self.hadamard_support >> i) & 1: hadamard_erasure.append(i)

        return sp_pauli, sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure

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
        
    def build_circuit(self, noise_dict: dict = None):
        """Builds the Stim circuit for the RSC code with specified noise.

        Args:
            noise_dict (dict): Dictionary specifying noise parameters. For possible keys, see documentation API.

        Returns:
            The associated Circuit object
        """
        circuit = Circuit()

        post_hadamard_errors = 0 # measure number of different error processes after Hadamards for reusing gate strings

        ## Create bitmasks for erasure and pauli errors (e.g. erasure_qubits = 0b10101 means qubits 0, 2, 4 have erasure errors)
        pauli_qubits = noise_dict.get('pauli-qubits', 0)
        erasure_qubits = noise_dict.get('erasure-qubits', 0)

        sp_pauli, sp_erasure, hadamard_pauli, hadamard_erasure, ancilla_meas_pauli, ancilla_meas_erasure = self._supports(pauli_qubits)

        ## State preparation
        circuit.add_reset(self.all_qubit_ids)
        # sp_pauli, sp_erasure = split_support_list_fast(self.data_qubit_ids, pauli_qubits, erasure_qubits) # can be made quicker
        if sp_pauli and (p_sp_pauli := noise_dict.get('sp', 0)) > 0:
            circuit.add_depolarize1(sp_pauli, p_sp_pauli)
        if sp_erasure and (p_sp_erasure := noise_dict.get('sp-e', 0)) > 0:
            circuit.add_erasure1(sp_erasure, p_sp_erasure)

        ## Hadamards on X-ancillas
        circuit.add_h_gate(self.x_ancilla_ids, index=True)
    
        # hadamard_pauli, hadamard_erasure = split_support_list_fast(self.x_ancilla_ids, pauli_qubits, erasure_qubits)
        if hadamard_pauli and (p_had_pauli := noise_dict.get('sqg', 0)) > 0:
            circuit.add_depolarize1(hadamard_pauli, p_had_pauli)
            post_hadamard_errors += 1
        if hadamard_erasure and (p_had_erasure := noise_dict.get('sqg-e', 0)) > 0:
            circuit.add_erasure1(hadamard_erasure, p_had_erasure)
            post_hadamard_errors += 1

        ## CNOT rounds
        fic_ancillas_bitmask = erasure_qubits << self.eq_diff
        for i, check_gates in enumerate(self.gates):
            if erasure_qubits > 0 and (p_tqg_e := noise_dict.get('tqg-e', 0)) > 0:
                circuit.add_symmetric_pauli_erasure_cnot(check_gates, erasure_qubits, p_tqg_e, self.eq_diff)
                fic_ancillas_intersect_bitmask = fic_ancillas_bitmask & self.cnot_bitmasks[i]
                fic_ancillas_intersect = list(mask_iter_indices(fic_ancillas_intersect_bitmask))
                if (p_meas_pauli := noise_dict.get('meas', 0)) > 0:
                    circuit.add_depolarize1(fic_ancillas_intersect, p_meas_pauli)
                circuit.add_measurements(fic_ancillas_intersect, reset=True)
            else:
                circuit.add_cnot(check_gates)
                circuit.add_depolarize2(check_gates, noise_dict.get('tqg', 0)) # this process is inefficient (loops over check_gates twice)

        ## Hadamards on X-ancillas again
        circuit.append_to_circ_str(circuit.circ_str[circuit.hadamard_index:circuit.hadamard_index + post_hadamard_errors + 1])  # Reuse the hadamard gate string

        ## Add detectors for all measurements so far
        circuit.detect_all_measurements()

        ## Measure all ancillas
        circuit.add_measurements(self.x_ancilla_ids + self.z_ancilla_ids)
        # ancilla_meas_pauli, ancilla_meas_erasure = split_support_list_fast(self.ancilla_ids, pauli_qubits, erasure_qubits)
        if ancilla_meas_pauli and (p_anc_meas_pauli := noise_dict.get('meas', 0)) > 0:
            circuit.add_depolarize1(ancilla_meas_pauli, p_anc_meas_pauli)
        if ancilla_meas_erasure and (p_anc_meas_erasure := noise_dict.get('meas', 0)) > 0:
            circuit.add_erasure1(ancilla_meas_erasure, p_anc_meas_erasure)

        ## Z-ancilla detectors
        circuit.add_detectors(range(self.num_ancillas // 2, 0, -1))
        self.cached_strings.append(circuit.circ_str[-1])

        ## Measure all data qubits
        circuit.add_measurements(self.data_qubit_ids)
        self.cached_strings.append(circuit.circ_str[-1])
        # TODO: add measurement noise

        ## Measure plaquettes
        circuit.add_detectors([[self.num_qubits - self.measurement_indices[anc_id] for anc_id in self.plaquettes[z_anc_id] + [z_anc_id]] for z_anc_id in self.z_ancilla_ids], parity=True)
        self.cached_strings.extend(circuit.circ_str[-self.num_ancillas//2:])
        
        ## Observable
        circuit.add_observable([self.num_qubits - self.measurement_indices[qubit_id] for qubit_id in self.observable])
        self.cached_strings.append(circuit.circ_str[-1])

        return circuit

    def build_stim_circuit(self, noise_dict: dict = None):
        """Builds the Stim circuit for the RSC code with specified noise.

        Args:
            noise_dict (dict): Dictionary specifying noise parameters. For possible keys, see documentation API.

        Returns:
            The constructed stim circuit.
        """

        return self.build_circuit(noise_dict).to_stim_circuit()

    def erasure_syndrome_to_stabilizer_circuit(self, erasure_circuit: Circuit, syndrome: list, noise_dict: dict = None):
        """Converts an erasure syndrome into a stabilizer circuit.

        Args:
            circuit: The erasure circuit
            syndrome: The complete syndrome of the erasure circuit.
        
        Returns:
            The stabilizer circuit corresponding to the erasure syndrome.
        """
        circuit = Circuit()

        erasure_qubits = noise_dict.get('erasure-qubits', 0)

        curr_meas_set_index = 0
        curr_meas_index = 0
        meas_sets, meas_sets_norm = erasure_circuit.get_measurement_sets()

        circuit.add_reset(self.all_qubit_ids)
        
        if self.sp_support & erasure_qubits > 0 and noise_dict.get('sp-e', 0) > 0:
            if (sp_erasure := [meas_sets[curr_meas_set_index][i] for i, m in enumerate(syndrome[curr_meas_index:curr_meas_index + meas_sets_norm[curr_meas_set_index]]) if m]) and any(sp_erasure):
                circuit.add_depolarize1(sp_erasure, p=0.75)
            curr_meas_index += meas_sets_norm[curr_meas_set_index]
            curr_meas_set_index += 1

        circuit.add_h_gate(self.x_ancilla_ids)
        if self.hadamard_support & erasure_qubits > 0 and noise_dict.get('sqg-e', 0) > 0:
            if (hadamard_erasure := [meas_sets[curr_meas_set_index][i] for i, m in enumerate(syndrome[curr_meas_index:curr_meas_index + meas_sets_norm[curr_meas_set_index]]) if m]) and any(hadamard_erasure):
                circuit.add_depolarize1(hadamard_erasure, p=0.75)
            curr_meas_index += meas_sets_norm[curr_meas_set_index]
            curr_meas_set_index += 1

        for check_num, gate_check in enumerate(self.gates):
            circuit.add_cnot(gate_check)
            if (self.cnot_bitmasks[check_num] >> self.eq_diff) & erasure_qubits > 0 and noise_dict.get('tqg-e', 0) > 0:
                if (tqg_erasure := [meas_sets[curr_meas_set_index][i] - self.eq_diff for i, m in enumerate(syndrome[curr_meas_index:curr_meas_index + meas_sets_norm[curr_meas_set_index]]) if m]) and any(tqg_erasure):
                    circuit.add_depolarize1(tqg_erasure, p=0.75)
                curr_meas_index += meas_sets_norm[curr_meas_set_index]
                curr_meas_set_index += 1

        circuit.add_h_gate(self.x_ancilla_ids)
        if self.hadamard_support & erasure_qubits > 0 and noise_dict.get('sqg-e', 0) > 0:
            if (hadamard_erasure := [meas_sets[curr_meas_set_index][i] for i, m in enumerate(syndrome[curr_meas_index:curr_meas_index + meas_sets_norm[curr_meas_set_index]]) if m]) and any(hadamard_erasure):
                circuit.add_depolarize1(hadamard_erasure, p=0.75)
            curr_meas_index += meas_sets_norm[curr_meas_set_index]
            curr_meas_set_index += 1

        ## Measure all ancillas
        circuit.add_measurements(self.x_ancilla_ids + self.z_ancilla_ids)
        if self.ancilla_measure_support & erasure_qubits > 0 and noise_dict.get('meas-e', 0) > 0:
            if (ancilla_meas_erasure := [meas_sets[curr_meas_set_index][i] for i, m in enumerate(syndrome[curr_meas_index:curr_meas_index + meas_sets_norm[curr_meas_set_index]]) if m]) and any(ancilla_meas_erasure):
                circuit.add_depolarize1(ancilla_meas_erasure, p=0.75)
            curr_meas_index += meas_sets_norm[curr_meas_set_index]
            curr_meas_set_index += 1

        circuit.append_to_circ_str(self.cached_strings)

        return circuit
    
    def erasure_syndrome_to_stim_circuit(self, erasure_circuit: Circuit, syndrome: list, noise_dict: dict = None):
        """Converts an erasure syndrome into a stabilizer stim circuit.

        Args:
            circuit: The erasure circuit
            syndrome: The complete syndrome of the erasure circuit.
        
        Returns:
            The stabilizer stim circuit corresponding to the erasure syndrome.
        """
        circuit = self.erasure_syndrome_to_stabilizer_circuit(erasure_circuit, syndrome, noise_dict=noise_dict)
        return circuit.to_stim_circuit()

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