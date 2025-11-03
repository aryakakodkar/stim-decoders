"""Heralded erasure decoder functions.

This module contains functions to build heralded erasure circuits,
sample syndromes, build corresponding Clifford circuits, and decode
them using PyMatching.
"""

from stimdecoders.utils import codes, circuits, bitops, noise
from stimdecoders.hed import batch as batch_builder
import stim
import pymatching
import numpy as np

# # Debug utility
# def print_syndrome(syndrome):
#     print("Syndrome indices:", end=' ')
#     for i, element in enumerate(syndrome):
#         if element: print(f"[{i}]", end=' ')
#     print()

def build_heralded_erasure_circuit(code: codes.Stabilizer_Code, noise_model: noise.Noise_Model, to_stim: bool = False):
    """Build a heralded erasure circuit for the given stabilizer code and noise parameters.
    
    Args:
        code: The stabilizer code object (currently only RSC supported)
        noise_dict: Noise parameters dictionary
        to_stim: If True, return stim.Circuit; if False, return circuits.Circuit
    
    Returns:
        The heralded erasure circuit (circuits.Circuit)
    """
    if not isinstance(code, codes.RSC):
        raise NotImplementedError("Currently only RSC code is supported for heralded erasure circuits.")
    
    circuit = circuits.build_rsc_erasure_circuit(code, noise_model=noise_model)

    if to_stim:
        circuit = circuit.to_stim_circuit()

    return circuit

def sample_erasure_circuit(circuit: stim.Circuit | circuits.Circuit, num_shots: int):
    """Sample the given erasure circuit for a number of shots.

    Args:
        circuit: The heralded erasure circuit (stim.Circuit or circuits.Circuit)
        num_shots: Number of shots to sample

    Returns:
        syndromes: Sampled syndromes (np.ndarray)
    """
    if isinstance(circuit, circuits.Circuit):
        circuit = circuit.to_stim_circuit()

    sampler = circuit.compile_detector_sampler()
    syndromes, observable_flips = sampler.sample(num_shots, separate_observables=True)
    return syndromes, observable_flips

def build_clifford_circuit(code: codes.Stabilizer_Code, erasure_circuit: circuits.Circuit, syndrome: np.ndarray, noise_dict: dict):
    """(DEPRECATED) Build the Clifford circuit corresponding to the given erasure circuit and syndrome.
    
    Args:
        code: The stabilizer code object (currently only RSC supported)
        erasure_circuit: The heralded erasure circuit
        syndrome: The measured syndrome (np.ndarray)
        noise_dict: Noise parameters dictionary
        
    Returns:
        clifford_circuit: The corresponding Clifford circuit (circuits.Circuit)
    """
    clifford_circuit = code.erasure_syndrome_to_stabilizer_circuit(erasure_circuit, syndrome, noise_dict)
    return clifford_circuit

# TODO: noise_dict should be part of erasure circuit. Should not have to pass it again here.
def build_clifford_stim_circuit(code: codes.Stabilizer_Code, erasure_circuit: circuits.Circuit, syndrome: np.ndarray, noise_dict: dict):
    """(DEPRECATED) Build the Clifford stim circuit corresponding to the given erasure circuit and syndrome.

    Args:
        code: The stabilizer code object (currently only RSC supported)
        erasure_circuit: The heralded erasure circuit
        syndrome: The measured syndrome (np.ndarray)
        noise_dict: Noise parameters dictionary

    Returns:
        clifford_stim_circuit: The corresponding Clifford stim circuit (stim.Circuit)
    """
    clifford_circuit = build_clifford_circuit(code, erasure_circuit, syndrome, noise_dict)
    clifford_stim_circuit = clifford_circuit.to_stim_circuit()
    return clifford_stim_circuit

def decode_erasure_circuit(rsc: codes.RSC, erasure_circuit: circuits.Circuit, num_trials: int, return_data: bool = False, noise_dict: dict = {}):
    """(DEPRECATED) Decode erasure circuit using basic method.
    
    Args:
        rsc: The RSC code object
        erasure_circuit: The heralded erasure circuit
        num_trials: Number of trials to decode
        return_data: If True, return predictions; if False, return error count
        noise_dict: Noise parameters dictionary

    Returns:
        Either predictions list or number of errors, depending on return_data
    """
    erasure_stim_circuit = erasure_circuit.to_stim_circuit()

    sampler = erasure_stim_circuit.compile_detector_sampler()
    syndromes, observable_flips = sampler.sample(num_trials, separate_observables=True)

    # Ensure noise_dict has erasure-qubits and pauli-qubits keys
    # Assume all qubits are erasure qubits for now
    erasure_mask = bitops.indices_to_mask(rsc.all_qubit_ids)
    pauli_mask = 0b0
    
    noise_dict_updated = noise_dict.copy()
    noise_dict_updated['erasure-qubits'] = erasure_mask
    noise_dict_updated['pauli-qubits'] = pauli_mask

    circuit_cache = {}

    predictions = []

    num_errors = 0
    for trial in range(num_trials):
        print("Trial", trial + 1, "of", num_trials, end='\r', flush=True)
        syndrome = syndromes[trial]
        observable_flip = observable_flips[trial]

        if syndrome.tobytes() in circuit_cache:
            clifford_stim_circuit = circuit_cache[syndrome.tobytes()]
        else:
            clifford_stim_circuit = rsc.erasure_syndrome_to_stim_circuit(erasure_circuit, syndrome=syndrome, noise_dict=noise_dict_updated)
            circuit_cache[syndrome.tobytes()] = clifford_stim_circuit

        clifford_decoder = pymatching.Matching.from_stim_circuit(clifford_stim_circuit)
        prediction = clifford_decoder.decode(syndrome[-rsc.num_ancillas:])

        if return_data:
            predictions.append(prediction)
            continue

        if not np.array_equal(observable_flip, prediction):
            num_errors += 1

    if return_data:
        return predictions
    else:
        return num_errors

def batch_decode_pure_erasure_circuit(rsc: codes.RSC, erasure_circuit: circuits.Circuit, 
                                 num_trials: int, batch_size: int = 1000,
                                 return_data: bool = False, noise_dict: dict = {}, verbose: bool = False):
    """Decode erasure circuit using optimized batch processing.
    
    This function builds Clifford circuits in batches for improved performance.
    It pre-computes circuit templates and processes multiple syndromes at once.
    
    Args:
        rsc: The RSC code object
        erasure_circuit: The heralded erasure circuit
        num_trials: Number of trials to decode
        batch_size: Number of syndromes to process in each batch
        return_data: If True, return predictions; if False, return error count
        noise_dict: Noise parameters dictionary
        
    Returns:
        Either predictions list or number of errors, depending on return_data
    """
    erasure_stim_circuit = erasure_circuit.to_stim_circuit()

    sampler = erasure_stim_circuit.compile_detector_sampler()
    syndromes, observable_flips = sampler.sample(num_trials, separate_observables=True)

    # Ensure noise_dict has erasure-qubits and pauli-qubits keys
    erasure_mask = bitops.indices_to_mask(rsc.all_qubit_ids)
    pauli_mask = 0b0
    
    noise_dict_updated = noise_dict.copy()
    noise_dict_updated['erasure-qubits'] = erasure_mask
    noise_dict_updated['pauli-qubits'] = pauli_mask

    # Create batch builder
    builder = batch_builder.BatchCircuitBuilder(rsc, erasure_circuit, noise_dict_updated)
    
    circuit_cache = {}
    predictions = []
    num_errors = 0
    
    # Process in batches
    num_batches = (num_trials + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_trials)
        
        batch_syndromes = syndromes[start_idx:end_idx]
        batch_observable_flips = observable_flips[start_idx:end_idx]
        
        if verbose:
            print(f"Processing batch {batch_idx + 1}/{num_batches} (trials {start_idx}-{end_idx})...", 
                end='\r', flush=True)
        
        # Build circuits for this batch (with caching)
        clifford_circuits, circuit_cache = builder.build_batch_with_cache(batch_syndromes, circuit_cache)
        
        # Decode each circuit in the batch
        for trial_in_batch, (clifford_circuit, syndrome, observable_flip) in enumerate(
            zip(clifford_circuits, batch_syndromes, batch_observable_flips)
        ):
            clifford_decoder = pymatching.Matching.from_stim_circuit(clifford_circuit)
            prediction = clifford_decoder.decode(syndrome[-rsc.num_ancillas:])
            
            if return_data:
                predictions.append(prediction)
            elif not np.array_equal(observable_flip, prediction):
                num_errors += 1
    
    print()  # Clear the progress line
    
    if return_data:
        return predictions
    else:
        return num_errors
