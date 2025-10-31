from typing import List
import stim
import time

from ..utils import bitops

CORRELATED_CNOT_CACHE = {}

def pauli_erasure_cnot_optimized(gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int = 0):
    """Builds optimized string list for symmetric Pauli + Erasure errors on CNOTs.

    Args:
        gates: List of (control, target) qubit index tuples for CNOTs.
        erasure_bitmask: Python int bitmask indicating which qubits have erasure errors.
        p: Probability of error occurring on each CNOT.
        eq_diff: Difference between data and fictitious ancilla qubit indices.

    Returns:
        List of strings representing the error operations to be appended to a Circuit._circ_str.
    """
    err_string = []
    PAULI_STR = ('I', 'X', 'Y', 'Z', 'J')
    
    for gate in gates:
        g0, g1 = gate[0], gate[1]
        erasure0 = (erasure_bitmask >> g0) & 1
        erasure1 = (erasure_bitmask >> g1) & 1
        gate_properties = (erasure0, erasure1, p)

        if gate_properties not in CORRELATED_CNOT_CACHE:
            # Build cache for this configuration
            else_space = 1.0
            num_errors = (4 + erasure0) * (4 + erasure1) - 1
            err_probs = p / num_errors
            
            cache_entries = []
            
            for q1 in range(4 + erasure0):
                for q2 in range(4 + erasure1):
                    if (q1, q2) == (0, 0): 
                        continue
                    
                    cache_entries.append((
                        q1, q2,
                        erasure0 and q1 != 0,
                        erasure1 and q2 != 0,
                        err_probs / else_space
                    ))
                    else_space -= err_probs
            
            CORRELATED_CNOT_CACHE[gate_properties] = cache_entries
        
        # Build error strings using precomputed cache
        cache_entries = CORRELATED_CNOT_CACHE[gate_properties]
        
        for i in range(len(cache_entries)):
            q1, q2, e0, e1, err_prob = cache_entries[i]
            
            parts = ""
            
            # Error prefix
            if i == 0:
                parts += f"CORRELATED_ERROR({err_prob}) "
            else:
                parts += f"ELSE_CORRELATED_ERROR({err_prob}) "

            # Pauli errors (skip identity and erasure marker)
            if 0 < q1 < 4:
                parts += f"{PAULI_STR[q1]}{g0} "
            if 0 < q2 < 4:
                parts += f"{PAULI_STR[q2]}{g1} "
            
            # Erasure markers
            if e0:
                parts += f"X{g0 + eq_diff} "
            if e1:
                parts += f"X{g1 + eq_diff} "

            # Single join and append
            err_string.append(parts)

    return err_string
    
class Circuit:
    def __init__(self):
        self._measurements = 0
        self._circ_str = []

        self._hadamard_index = None

        self._observable_index = 0

        self._measurement_sets = []
        self._measurement_sets_norm = []
        self._current_measurement_set = []
    
    @property
    def hadamard_index(self):
        return self._hadamard_index
    
    @hadamard_index.setter
    def hadamard_index(self, value):
        self._hadamard_index = value

    @property
    def circ_str(self):
        return self._circ_str
    
    def append_to_circ_str(self, lines: List[str]):
        """Appends multiple lines to the circuit string representation.
        
        Args:
            lines: List of strings to append.
        """
        self._circ_str.extend(lines)

    def add_reset(self, qubits: List[int]):
        """Adds reset operations for specified qubits.

        Args:
            qubits: List of qubit indices to reset.
        """
        self._circ_str.append("R " + " ".join(str(q) for q in qubits))

    def add_cnot(self, gates):
        """Adds CNOT operations for specified gate pairs.
        
        Args:
            gates: List of (control, target) qubit index tuples for CNOTs.
        """
        self._circ_str.append("CX " + " ".join(f"{control} {target}" for control, target in gates))

    def add_depolarize1(self, qubits: List[int], p: float):
        """Adds single-qubit depolarizing errors for specified qubits.

        Args:
            qubits: List of qubit indices to apply depolarizing errors.
            p: Probability of depolarizing error.
        """
        self._circ_str.append(f"DEPOLARIZE1({p}) " + " ".join(str(q) for q in qubits))

    def add_depolarize2(self, pairs: List[tuple], p: float):
        """Adds two-qubit depolarizing errors for specified qubit pairs.

        Args:
            pairs: List of (q1, q2) qubit index tuples for depolarizing errors.
            p: Probability of depolarizing error.
        """
        self._circ_str.append(f"DEPOLARIZE2({p}) " + " ".join(f"{pair[0]} {pair[1]}" for pair in pairs))

    def add_erasure1(self, qubits: List[int], p: float):
        """Adds single-qubit erasure errors for specified qubits.

        Args:
            qubits: List of qubit indices to apply erasure errors.
            p: Probability of erasure error.
        """
        measure_string = f"HERALDED_ERASE({p}) "
        num_qubits_measured = 0
        for q in qubits:
            measure_string += f"{q} "
            self._current_measurement_set.append(q)
            num_qubits_measured += 1

        self._circ_str.append(measure_string)
        self._measurements += num_qubits_measured
        self._measurement_sets.append(self._current_measurement_set)
        self._measurement_sets_norm.append(num_qubits_measured)
        self._current_measurement_set = []

    def add_symmetric_pauli_erasure_cnot(self, gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int=0):
        """Adds CNOT operations with symmetric Pauli + Erasure errors.

        Args:
            gates: List of (control, target) qubit index tuples for CNOTs.
            erasure_bitmask: Python int bitmask indicating which qubits have erasure errors. (WARNING: assumes all non-erasure qubits are Pauli qubits)
            p: Probability of error occurring on each CNOT.
            eq_diff: Difference between data and fictitious ancilla qubit indices.
        """
        self._circ_str.append("CX " + " ".join(f"{gate[0]} {gate[1]}" for gate in gates))
        if p > 0.0:
            self._circ_str.extend(pauli_erasure_cnot_optimized(gates, erasure_bitmask, p, eq_diff))

    def add_cnot(self, gates: List[tuple]):
        """Adds CNOT operations without errors.
        
        Args:
            gates: List of (control, target) qubit index tuples for CNOTs.
        """
        self._circ_str.append("CX " + " ".join(f"{gate[0]} {gate[1]}" for gate in gates))
            
    def add_measurements(self, qubits: List[int], reset: bool = False): # inefficient
        """Adds measurement operations for specified qubits.

        Args:
            qubits: List of qubit indices to measure.
        """
        measure_string = f"M{"R" if reset else ""} "
        num_qubits_measured = 0
        for q in qubits:
            measure_string += f"{q} "
            self._current_measurement_set.append(q)
            num_qubits_measured += 1

        self._circ_str.append(measure_string)
        self._measurements += num_qubits_measured
        self._measurement_sets.append(self._current_measurement_set)
        self._measurement_sets_norm.append(num_qubits_measured)
        self._current_measurement_set = []

    def add_h_gate(self, qubits: List[int], index: bool = False):
        """Adds Hadamard gates for specified qubits.

        Args:
            qubits: List of qubit indices to apply Hadamard gates.
            index: If True, records the index of this Hadamard operation in the circuit string (for later re-use).
        """
        self._circ_str.append(f"H " + " ".join(str(q) for q in qubits))
        if index: self.hadamard_index = len(self._circ_str) - 1

    def add_detectors(self, coords: List, parity: bool=False):
        """Adds detectors to the circuit.
        Args:
            coords: List of coordinates for detectors, or list of lists of coordinates for parity grouping.
            parity: If True, adds detectors with parity grouping.
        """
        if not parity:
            self._circ_str.append("\n".join([f"DETECTOR rec[-{coord}]" for coord in coords]))
        if parity:
            for coord in coords:
                self._circ_str.append(f"DETECTOR " + " ".join(f"rec[-{i}]" for i in coord))

    def detect_all_measurements(self):
        """Adds detectors for all measurements in the circuit, first-in first-out."""
        self.add_detectors(range(self._measurements, 0, -1))

    def add_observable(self, coords):
        """Adds an observable to the circuit.

        Args:
            coords: List of coordinates for the observable.
        """
        self._circ_str.append(f"OBSERVABLE_INCLUDE({self._observable_index}) " + " ".join(f"rec[-{coord}]" for coord in coords))
        self._observable_index += 1

    def get_measurement_sets(self):
        """Returns the measurement sets and their normalization factors.

        Returns:
            Tuple of (measurement_sets, measurement_sets_norm).
        """
        return self._measurement_sets, self._measurement_sets_norm

    def to_stim_circuit(self):
        """Converts the internal string representation to a stim.Circuit object."""
        return stim.Circuit("\n".join(self._circ_str))