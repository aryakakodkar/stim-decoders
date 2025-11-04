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

def build_heralded_erasure_circuit(code: codes.Stabilizer_Code, noise_model: noise.Noise_Model, to_stim: bool = False):
    """Build a pure heralded erasure circuit for the given stabilizer code and noise parameters.
    
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

def batch_decode_pure_erasure_circuit(erasure_circuit: circuits.Circuit, 
                                 num_trials: int, batch_size: int = 1000,
                                 return_data: bool = False, verbose: bool = False):
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
        Either (observable flips, predictions) lists or (number of errors) depending on return_data
    """
    # Create batch builder
    builder = batch_builder.BatchCircuitBuilder(erasure_circuit)

    syndromes, observable_flips = sample_erasure_circuit(erasure_circuit.to_stim_circuit(), num_trials)

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
            prediction = clifford_decoder.decode(syndrome[-erasure_circuit.code.num_ancillas:])
            
            if return_data:
                predictions.append(prediction)
            elif not np.array_equal(observable_flip, prediction):
                num_errors += 1
    
    print() 
    
    if return_data:
        return observable_flips, predictions
    else:
        return num_errors
