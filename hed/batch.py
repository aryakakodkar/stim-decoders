"""Batch processing utilities for heralded erasure decoder circuits.

This module provides optimized batch building of Clifford circuits from erasure syndromes.
"""

from stimdecoders.utils import codes, circuits, bitops
import stim
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class BatchCircuitBuilder:
    """Optimized batch builder for Clifford circuits from erasure syndromes.
    
    This class pre-computes invariant circuit components and uses templates to
    quickly build circuits for multiple syndromes.
    """
    
    def __init__(self, rsc: codes.RSC, erasure_circuit: circuits.Circuit, noise_dict: dict):
        """Initialize the batch builder.
        
        Args:
            rsc: The RSC code object
            erasure_circuit: The erasure circuit
            noise_dict: Noise parameters dictionary
        """
        self.rsc = rsc
        self.erasure_circuit = erasure_circuit
        self.noise_dict = noise_dict
        
        # Get measurement set information
        self.meas_sets, self.meas_sets_norm = erasure_circuit.get_measurement_sets()
        self.erasure_qubits = noise_dict.get('erasure-qubits', 0)
        
        # Pre-compute circuit template components
        self._precompute_template()
        
    def _precompute_template(self):
        """Pre-compute the invariant parts of the circuit."""
        # Store indices for each measurement set segment
        self.meas_indices = []
        curr_idx = 0
        for norm in self.meas_sets_norm:
            self.meas_indices.append((curr_idx, curr_idx + norm))
            curr_idx += norm
        
        # Pre-compute which stages are active (have erasure errors possible)
        self.active_stages = {
            'sp': self.rsc.sp_support & self.erasure_qubits > 0 and self.noise_dict.get('sp-e', 0) > 0,
            'hadamard1': self.rsc.hadamard_support & self.erasure_qubits > 0 and self.noise_dict.get('sqg-e', 0) > 0,
            'cnots': [(self.rsc.cnot_bitmasks[i] >> self.rsc.eq_diff) & self.erasure_qubits > 0 and 
                     self.noise_dict.get('tqg-e', 0) > 0 for i in range(4)],
            'hadamard2': self.rsc.hadamard_support & self.erasure_qubits > 0 and self.noise_dict.get('sqg-e', 0) > 0,
            'ancilla_meas': self.rsc.ancilla_measure_support & self.erasure_qubits > 0 and 
                           self.noise_dict.get('meas-e', 0) > 0,
        }
        
        # Pre-build the base circuit string parts (parts that don't depend on syndrome)
        self.base_reset = f"R {' '.join(str(q) for q in self.rsc.all_qubit_ids)}"
        self.base_hadamard = f"H {' '.join(str(q) for q in self.rsc.x_ancilla_ids)}"
        self.base_cnots = [f"CX {' '.join(f'{g[0]} {g[1]}' for g in gate_check)}" 
                          for gate_check in self.rsc.gates]
        self.base_measurements = f"M {' '.join(str(q) for q in self.rsc.x_ancilla_ids + self.rsc.z_ancilla_ids)}"
        
    def _extract_erasures_vectorized(self, syndromes: np.ndarray) -> List[Dict[str, List[int]]]:
        """Extract erasure qubit indices from syndromes in a vectorized manner.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            
        Returns:
            List of dictionaries mapping stage names to erasure qubit indices
        """
        erasure_patterns = []
        
        for syndrome in syndromes:
            pattern = {}
            stage_idx = 0
            
            # State preparation erasures
            if self.active_stages['sp']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['sp'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['sp'] = []
            
            # First Hadamard erasures
            if self.active_stages['hadamard1']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['hadamard1'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['hadamard1'] = []
            
            # CNOT erasures for each check
            pattern['cnots'] = []
            for check_num in range(4):
                if self.active_stages['cnots'][check_num]:
                    start, end = self.meas_indices[stage_idx]
                    erased = [self.meas_sets[stage_idx][i] - self.rsc.eq_diff 
                             for i, m in enumerate(syndrome[start:end]) if m]
                    pattern['cnots'].append(erased if erased else [])
                    stage_idx += 1
                else:
                    pattern['cnots'].append([])
            
            # Second Hadamard erasures
            if self.active_stages['hadamard2']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['hadamard2'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['hadamard2'] = []
            
            # Ancilla measurement erasures
            if self.active_stages['ancilla_meas']:
                start, end = self.meas_indices[stage_idx]
                erased = [self.meas_sets[stage_idx][i] for i, m in enumerate(syndrome[start:end]) if m]
                pattern['ancilla_meas'] = erased if erased else []
                stage_idx += 1
            else:
                pattern['ancilla_meas'] = []
            
            erasure_patterns.append(pattern)
        
        return erasure_patterns
    
    def _build_circuit_from_pattern(self, pattern: Dict[str, List[int]]) -> circuits.Circuit:
        """Build a circuit from a pre-computed erasure pattern.
        
        Args:
            pattern: Dictionary mapping stage names to erasure qubit indices
            
        Returns:
            The constructed Circuit object
        """
        circuit = circuits.Circuit()
        
        # Reset
        circuit._circ_str.append(self.base_reset)
        
        # State preparation errors
        if pattern['sp']:
            circuit.add_depolarize1(pattern['sp'], p=0.75)
        
        # First Hadamard
        circuit._circ_str.append(self.base_hadamard)
        if pattern['hadamard1']:
            circuit.add_depolarize1(pattern['hadamard1'], p=0.75)
        
        # CNOTs
        for check_num in range(4):
            circuit._circ_str.append(self.base_cnots[check_num])
            if pattern['cnots'][check_num]:
                circuit.add_depolarize1(pattern['cnots'][check_num], p=0.75)
        
        # Second Hadamard
        circuit._circ_str.append(self.base_hadamard)
        if pattern['hadamard2']:
            circuit.add_depolarize1(pattern['hadamard2'], p=0.75)
        
        # Measurements
        circuit._circ_str.append(self.base_measurements)
        if pattern['ancilla_meas']:
            circuit.add_depolarize1(pattern['ancilla_meas'], p=0.75)
        
        # Append cached strings (detectors and observables)
        circuit.append_to_circ_str(self.rsc.cached_strings)
        
        return circuit
    
    def build_batch(self, syndromes: np.ndarray) -> List[stim.Circuit]:
        """Build Clifford circuits for a batch of syndromes.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            
        Returns:
            List of stim.Circuit objects, one for each syndrome
        """
        # Extract erasure patterns in a vectorized manner
        patterns = self._extract_erasures_vectorized(syndromes)
        
        # Build circuits from patterns
        circuits_list = []
        for pattern in patterns:
            circuit = self._build_circuit_from_pattern(pattern)
            circuits_list.append(circuit.to_stim_circuit())
        
        return circuits_list
    
    def build_batch_with_cache(self, syndromes: np.ndarray, 
                               circuit_cache: Dict[bytes, stim.Circuit] = None) -> Tuple[List[stim.Circuit], Dict[bytes, stim.Circuit]]:
        """Build Clifford circuits for a batch with caching.
        
        Args:
            syndromes: Array of syndromes (num_syndromes, syndrome_length)
            circuit_cache: Optional cache dictionary to use/update
            
        Returns:
            Tuple of (list of circuits, updated cache)
        """
        if circuit_cache is None:
            circuit_cache = {}
        
        circuits_list = []
        patterns = self._extract_erasures_vectorized(syndromes)
        
        for syndrome, pattern in zip(syndromes, patterns):
            syndrome_key = syndrome.tobytes()
            
            if syndrome_key in circuit_cache:
                circuits_list.append(circuit_cache[syndrome_key])
            else:
                circuit = self._build_circuit_from_pattern(pattern)
                stim_circuit = circuit.to_stim_circuit()
                circuit_cache[syndrome_key] = stim_circuit
                circuits_list.append(stim_circuit)
        
        return circuits_list, circuit_cache


def batch_build_clifford_circuits(rsc: codes.RSC, 
                                  erasure_circuit: circuits.Circuit,
                                  syndromes: np.ndarray,
                                  noise_dict: dict,
                                  use_cache: bool = True,
                                  circuit_cache: Dict[bytes, stim.Circuit] = None) -> Tuple[List[stim.Circuit], Dict[bytes, stim.Circuit]]:
    """Build Clifford circuits for multiple syndromes in batch mode.
    
    This is the main entry point for batch circuit building. It creates a
    BatchCircuitBuilder and uses it to efficiently build circuits.
    
    Args:
        rsc: The RSC code object
        erasure_circuit: The erasure circuit
        syndromes: Array of syndromes (num_syndromes, syndrome_length)
        noise_dict: Noise parameters dictionary
        use_cache: Whether to use caching
        circuit_cache: Optional pre-existing cache
        
    Returns:
        Tuple of (list of circuits, cache dictionary)
    """
    builder = BatchCircuitBuilder(rsc, erasure_circuit, noise_dict)
    
    if use_cache:
        return builder.build_batch_with_cache(syndromes, circuit_cache)
    else:
        circuits_list = builder.build_batch(syndromes)
        return circuits_list, {}
