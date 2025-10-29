from collections import deque
import itertools
from typing import List
import stim
import time

PAULI_GATES = ['I', 'X', 'Y', 'Z', 'J']

possible_erasure_configs = [(0, 0), (0, 1), (1, 0), (1, 1)]

ERROR_PROB_FOR_ERASURE_CONFIG = {}

CORRELATED_CNOT_CACHE = {}

def pauli_erasure_cnot_optimized(gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int = 0):
    """
    Optimized pure Python version for building CNOT error strings.
    
    Key optimizations:
    1. Cache error descriptors per gate configuration to avoid recomputation
    2. Use local references to avoid attribute lookups
    3. Minimize list allocations and operations
    4. Use tuple for PAULI_STR (slightly faster than list)
    5. Build strings efficiently with f-strings and single join
    
    Stim's circuit string parser (C++) is ~337x faster than the Python append() API,
    so we optimize string building and let Stim's parser handle the heavy lifting.
    """
    err_string = []
    err_string_append = err_string.append  # Local reference for speed
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
            
            # Build string parts efficiently
            parts = []
            parts_append = parts.append  # Local reference
            
            # Error prefix
            if i == 0:
                parts_append(f"CORRELATED_ERROR({err_prob})")
            else:
                parts_append(f"ELSE_CORRELATED_ERROR({err_prob})")
            
            # Pauli errors (skip identity and erasure marker)
            if 0 < q1 < 4:
                parts_append(f"{PAULI_STR[q1]}{g0}")
            if 0 < q2 < 4:
                parts_append(f"{PAULI_STR[q2]}{g1}")
            
            # Erasure markers
            if e0:
                parts_append(f"X{g0 + eq_diff}")
            if e1:
                parts_append(f"X{g1 + eq_diff}")
            
            # Single join and append
            err_string_append(' '.join(parts))
    
    return err_string

def mask_iter_indices(mask: int):
    """Yield set-bit indices (ascending) for Python int mask (lsb -> msb)."""
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        yield idx
        mask ^= lsb

def split_noise_masks(pauli_mask: int, erasure_mask: int):
    """
    Given integer bitmasks for Pauli and Erasure qubits,
    return two lists (pauli_indices, erasure_indices).

    - Each list contains the indices of qubits (ascending order).
    - Overlapping bits (if any) will appear in both lists.
    - Very fast: iterates only set bits.
    """
    pauli_indices = []
    erasure_indices = []

    # iterate only bits in union of pauli and erasure masks
    union = pauli_mask | erasure_mask
    if union == 0:
        return pauli_indices, erasure_indices

    pm = pauli_mask
    em = erasure_mask
    um = union
    _bit_length = int.bit_length

    while um:
        lsb = um & -um
        idx = _bit_length(lsb) - 1
        um ^= lsb  # clear that bit
        if (pm >> idx) & 1:
            pauli_indices.append(idx)
        if (em >> idx) & 1:
            erasure_indices.append(idx)

    return pauli_indices, erasure_indices

def split_support_masks(clean_mask: int, pauli_mask: int, erasure_mask: int):
    """
    Given three integer bitmasks (bits set indicate membership),
    return (support_list, pauli_list, erasure_list).
    Lists are in ascending index order (0..).
    """
    union = clean_mask | pauli_mask | erasure_mask
    if union == 0:
        return [], [], []

    support = []
    pauli = []
    erasure = []

    # localize for speed
    _bit_length = int.bit_length
    um = union
    pm = pauli_mask
    em = erasure_mask

    while um:
        lsb = um & -um
        idx = _bit_length(lsb) - 1  # index of the lsb
        um ^= lsb  # clear that bit

        support.append(idx)
        if (pm >> idx) & 1:
            pauli.append(idx)
        if (em >> idx) & 1:
            erasure.append(idx)

    return support, pauli, erasure

## State preparation gate
class NoiselessStatePrepGate:
    def __init__(self, qubits):
        self.qubits = qubits

    def to_string(self):
        return [f"R " + " ".join([str(q) for q in self.qubits])], 0

class GeneralStatePrepGate:
    def __init__(self, clean_bitmask=0b0, pauli_bitmask=0b0, erasure_bitmask=0b0, p_pauli=0.0, p_erasure=0.0):
        self.clean_bitmask = clean_bitmask
        self.pauli_bitmask = pauli_bitmask
        self.erasure_bitmask = erasure_bitmask
        self.p_pauli = p_pauli
        self.p_erasure = p_erasure

    def to_string(self):
        measurements = 0
        support_qubits, pauli_qubits, erasure_qubits = split_support_masks(self.clean_bitmask, self.pauli_bitmask, self.erasure_bitmask)

        base_str = [f"R " + " ".join([str(q) for q in support_qubits])]

        if self.p_pauli > 0.0 and self.pauli_bitmask != 0b0:
            base_str.append(f"DEPOLARIZE1({self.p_pauli}) " + " ".join([str(q) for q in pauli_qubits]))
        
        if self.p_erasure > 0.0 and self.erasure_bitmask != 0b0:
            base_str.append(f"ERASE1({self.p_erasure}) " + " ".join([str(q) for q in erasure_qubits]))
            measurements = len(erasure_qubits)

        return base_str, measurements

## Single-qubit gates
class GeneralSQGate:
    def __init__(self, gate_type: str, clean_bitmask=0b0, pauli_bitmask=0b0, erasure_bitmask=0b0, p_pauli: float = 0.0, p_erasure: float = 0.0):
        self.gate_type = gate_type
        self.clean_bitmask = clean_bitmask
        self.pauli_bitmask = pauli_bitmask
        self.erasure_bitmask = erasure_bitmask
        self.p_pauli = p_pauli
        self.p_erasure = p_erasure

    def to_string(self):
        measurements = 0
        support_qubits, pauli_qubits, erasure_qubits = split_support_masks(self.clean_bitmask, self.pauli_bitmask, self.erasure_bitmask)

        base_str = [f"{self.gate_type} " + " ".join([str(q) for q in support_qubits])]

        if self.p_pauli > 0.0 and self.pauli_bitmask != 0b0:
            base_str.append(f"DEPOLARIZE1({self.p_pauli}) " + " ".join([str(q) for q in pauli_qubits]))

        if self.p_erasure > 0.0 and self.erasure_bitmask != 0b0:
            base_str.append(f"ERASE1({self.p_erasure}) " + " ".join([str(q) for q in erasure_qubits]))
            measurements = len(erasure_qubits)

        return base_str, measurements

## Two-qubit gates
# TODO: Allow for clean qubits too
class SymmetricPauliErasureCNOT:
    def __init__(self, gates: list, erasure_bitmask=0b0, p: float=0.0,  eq_diff: int=0):
        self.gates = gates
        self.erasure_bitmask = erasure_bitmask
        self.p = p
        self.eq_diff = eq_diff

    def to_string(self):
        measurements = 0

        base_str = [f"CX " + " ".join([f"{gate[0]} {gate[1]}" for gate in self.gates])]

        if self.p > 0.0:
            base_str.extend(pauli_erasure_cnot_optimized(self.gates, self.erasure_bitmask, self.p, self.eq_diff))
            measurements += self.erasure_bitmask.bit_count()

        return base_str, measurements

## Measurement gates
class HeraldedAncillaMeasurementGate:
    def __init__(self, qubits: List[int], eq_diff: int=0):
        self.qubits = qubits
        self.eq_diff = eq_diff
    
    def to_string(self):
        return [f"MR " + " ".join([str(qid+self.eq_diff) for qid in self.qubits])], len(self.qubits)

class GeneralMeasurementGate:
    def __init__(self, clean_bitmask=0b0, pauli_bitmask=0b0, erasure_bitmask=0b0, p_pauli=0.0, p_erasure=0.0, reset: bool=False):
        self.clean_bitmask = clean_bitmask
        self.pauli_bitmask = pauli_bitmask
        self.erasure_bitmask = erasure_bitmask
        self.p_pauli = p_pauli
        self.p_erasure = p_erasure
        self.reset = reset

    def to_string(self):
        support_qubits, pauli_qubits, erasure_qubits = split_support_masks(self.clean_bitmask, self.pauli_bitmask, self.erasure_bitmask)

        base_str = [f"M{"R" if self.reset else ""} " + " ".join([str(q) for q in support_qubits])]

        if self.p_pauli > 0.0 and self.pauli_bitmask != 0b0:
            base_str.append(f"DEPOLARIZE1({self.p_pauli}) " + " ".join([str(q) for q in pauli_qubits]))
        
        if self.p_erasure > 0.0 and self.erasure_bitmask != 0b0:
            base_str.append(f"ERASE1({self.p_erasure}) " + " ".join([str(q) for q in erasure_qubits]))
            
        measurements = len(support_qubits)

        return base_str, measurements

# class Circuit:
#     def __init__(self):
#         self._gates = deque()
#         self._measurements = 0

#     @property
#     def gates(self):
#         return self._gates

#     def add_gate(self, gate):
#         self._gates.append(gate)

#     def __str__(self):
#         t0 = time.time()
#         gate_string = []
#         for gate in self._gates:
#             gate_str, measurements = gate.to_string()
#             gate_string.extend(gate_str)
#             self._measurements += measurements

#         print("Generated circuit string in", time.time() - t0)

#         return '\n'.join(gate_string)
    
class Circuit:
    def __init__(self):
        self._measurements = 0
        self.circ_str = []

    def add_reset(self, qubits: List[int]):
        self.circ_str.append("R " + " ".join(str(q) for q in qubits))

    def add_cnot(self, gates):
        self.circ_str.append("CX " + " ".join(f"{control} {target}" for control, target in gates))

    def add_depolarize1(self, qubits: List[int], p: float):
        self.circ_str.append(f"DEPOLARIZE1({p}) " + " ".join(str(q) for q in qubits))

    def add_erasure1(self, qubits: List[int], p: float):
        self.circ_str.append(f"HERALDED_ERASE({p}) " + " ".join(str(q) for q in qubits))

    def add_symmetric_pauli_erasure_cnot(self, gates: List[tuple], erasure_bitmask: int, p: float, eq_diff: int=0):
        self.circ_str.append("CX " + " ".join(f"{gate[0]} {gate[1]}" for gate in gates))
        if p > 0.0:
            self.circ_str.extend(pauli_erasure_cnot_optimized(gates, erasure_bitmask, p, eq_diff))

    def to_stim_circuit(self):
        return stim.Circuit("\n".join(self.circ_str))