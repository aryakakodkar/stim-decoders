"""Multi-core parallel decoder for heralded erasure circuits.

This module provides Sinter-like parallel execution across multiple noise parameters
and automatically distributes work across available CPU cores.
"""

from stimdecoders.utils import codes, circuits, bitops
from stimdecoders.hed import hed, batch as batch_builder
import stim
import pymatching
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import time


@dataclass
class DecoderTask:
    """Represents a single decoding task (one noise parameter set)."""
    distance: int
    noise_params: Dict[str, float]
    num_shots: int
    task_id: int = 0
    
    def __hash__(self):
        # Make hashable for deduplication
        params_tuple = tuple(sorted(self.noise_params.items()))
        return hash((self.distance, params_tuple, self.num_shots, self.task_id))

@dataclass
class DecoderResult:
    """Results from a single decoding task."""
    task: DecoderTask
    num_errors: int
    num_shots: int
    logical_error_rate: float
    elapsed_time: float
    num_unique_syndromes: int
    cache_hit_rate: float
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'distance': self.task.distance,
            'noise_params': self.task.noise_params,
            'num_shots': self.task.num_shots,
            'num_errors': self.num_errors,
            'logical_error_rate': self.logical_error_rate,
            'elapsed_time': self.elapsed_time,
            'num_unique_syndromes': self.num_unique_syndromes,
            'cache_hit_rate': self.cache_hit_rate
        }


def _worker_decode_task(task: DecoderTask, timed: bool = False) -> DecoderResult:
    """Worker function to decode a single task.
    
    This runs in a separate process. Each process creates its own code,
    circuit, and decoder objects.
    
    Args:
        task: The decoding task to execute
        timed: Whether to time the decoding process

    Returns:
        DecoderResult with statistics
    """
    t_start = time.time()
    
    # Build code and circuit for this task
    rsc = codes.RSC(distance=task.distance)
    # erasure_circuit = hed.build_heralded_erasure_circuit(rsc, task.noise_params)
    
    # Prepare noise dict with required keys
    erasure_mask = bitops.indices_to_mask(rsc.all_qubit_ids) # assume pure erasure noise. TODO: make configurable
    pauli_mask = 0b0
    
    noise_dict_updated = task.noise_params.copy()
    noise_dict_updated['erasure-qubits'] = erasure_mask
    noise_dict_updated['pauli-qubits'] = pauli_mask

    erasure_circuit = rsc.build_circuit(noise_dict=noise_dict_updated)
    
    # Sample syndromes
    erasure_stim_circuit = erasure_circuit.to_stim_circuit()
    sampler = erasure_stim_circuit.compile_detector_sampler()
    syndromes, observable_flips = sampler.sample(task.num_shots, separate_observables=True)
    
    # Build circuits using batch builder for efficiency
    builder = batch_builder.BatchCircuitBuilder(rsc, erasure_circuit, noise_dict_updated)
    clifford_circuits, circuit_cache = builder.build_batch_with_cache(syndromes)
    
    num_errors = 0
    
    # Decode all shots using pre-built circuits
    for trial in range(task.num_shots):
        syndrome = syndromes[trial]
        observable_flip = observable_flips[trial]
        clifford_stim_circuit = clifford_circuits[trial]
        
        clifford_decoder = pymatching.Matching.from_stim_circuit(clifford_stim_circuit)
        prediction = clifford_decoder.decode(syndrome[-rsc.num_ancillas:])
        
        if not np.array_equal(observable_flip, prediction):
            num_errors += 1
    
    # Compute statistics
    elapsed_time = time.time() - t_start
    logical_error_rate = num_errors / task.num_shots
    num_unique_syndromes = len(circuit_cache)
    cache_hit_rate = 1.0 - (num_unique_syndromes / task.num_shots)
    
    return DecoderResult(
        task=task,
        num_errors=num_errors,
        num_shots=task.num_shots,
        logical_error_rate=logical_error_rate,
        elapsed_time=elapsed_time,
        num_unique_syndromes=num_unique_syndromes,
        cache_hit_rate=cache_hit_rate
    )


class ParallelDecoder:
    """Multi-core parallel decoder for heralded erasure circuits.
    
    Similar to Sinter, this automatically distributes decoding tasks across
    available CPU cores. You can specify multiple noise parameters and it
    will run them in parallel.
    """
    
    def __init__(self, num_workers: Optional[int] = None, verbose: bool = True):
        """Initialize the parallel decoder.
        
        Args:
            num_workers: Number of worker processes. If None, uses cpu_count()-1
            verbose: Whether to print progress updates
        """
        if num_workers is None:
            # Leave one core free for system
            num_workers = max(1, cpu_count() - 1)
        
        self.num_workers = num_workers
        self.verbose = verbose
        
    def decode_tasks(self, tasks: List[DecoderTask]) -> List[DecoderResult]:
        """Decode multiple tasks in parallel.
        
        Args:
            tasks: List of decoding tasks to execute
            
        Returns:
            List of results, one per task
        """
        if self.verbose:
            print(f"Starting parallel decoder with {self.num_workers} workers")
            print(f"Total tasks: {len(tasks)}")
            print()
        
        # Run tasks in parallel
        with Pool(self.num_workers) as pool:
            results = []
            
            # Use imap_unordered for progress updates
            for i, result in enumerate(pool.imap_unordered(_worker_decode_task, tasks)):
                results.append(result)
                
                if self.verbose:
                    print(f"[{i+1}/{len(tasks)}] Completed d={result.task.distance}, "
                          f"p={result.task.noise_params.get('sp-e', 0):.4f}, "
                          f"errors={result.num_errors}/{result.task.num_shots} "
                          f"({result.logical_error_rate:.6f}), "
                          f"time={result.elapsed_time:.2f}s")
        
        if self.verbose:
            print()
            print(f"All {len(tasks)} tasks completed!")
            print()
        
        return results
    
    def collect_statistics(
        self, 
        distance: int,
        noise_probabilities: List[float],
        num_shots_per_prob: int,
        noise_type: str = 'erasure',
        noise_keys: Optional[List[str]] = None
    ) -> List[DecoderResult]:
        """Collect decoder statistics across multiple noise probabilities.
        
        This is the main entry point, similar to Sinter's API.
        
        Args:
            distance: Code distance
            noise_probabilities: List of noise probabilities to test
            num_shots_per_prob: Number of shots per probability
            noise_type: Type of noise ('erasure', 'depolarizing', 'custom')
            noise_keys: For 'custom', specify which noise keys to set
            
        Returns:
            List of results for all probabilities
        """
        # Build tasks
        tasks = []
        
        for i, p in enumerate(noise_probabilities):
            # Build noise dict based on type
            if noise_type == 'erasure':
                noise_params = {
                    'sp-e': p,
                    'sqg-e': p,
                    'tqg-e': p
                }
            elif noise_type == 'depolarizing':
                noise_params = {
                    'sp': p,
                    'sqg': p,
                    'tqg': p
                }
            elif noise_type == 'custom' and noise_keys:
                noise_params = {key: p for key in noise_keys}
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")
            
            task = DecoderTask(
                distance=distance,
                noise_params=noise_params,
                num_shots=num_shots_per_prob,
                task_id=i
            )
            tasks.append(task)
        
        # Decode in parallel
        return self.decode_tasks(tasks)
    
    def save_results(self, results: List[DecoderResult], filename: str):
        """Save results to JSON file.
        
        Args:
            results: List of decoder results
            filename: Output filename
        """
        data = {
            'results': [r.to_dict() for r in results],
            'metadata': {
                'num_workers': self.num_workers,
                'num_tasks': len(results),
                'total_shots': sum(r.num_shots for r in results)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {filename}")


def collect_decoder_statistics(
    distance: int,
    noise_probabilities: List[float],
    num_shots: int,
    noise_type: str = 'erasure',
    num_workers: Optional[int] = None,
    verbose: bool = True
) -> List[DecoderResult]:
    """Convenience function to collect decoder statistics.
    
    This is a simple, Sinter-like API for running parallel decoding.
    
    Args:
        distance: Code distance
        noise_probabilities: List of noise probabilities to test
        num_shots: Number of shots per probability
        noise_type: Type of noise ('erasure', 'depolarizing')
        num_workers: Number of worker processes (None = auto)
        verbose: Print progress updates
        
    Returns:
        List of results for all probabilities
        
    Example:
        >>> results = collect_decoder_statistics(
        ...     distance=3,
        ...     noise_probabilities=[0.001, 0.002, 0.005, 0.01],
        ...     num_shots=10000,
        ...     noise_type='erasure'
        ... )
    """
    decoder = ParallelDecoder(num_workers=num_workers, verbose=verbose)
    return decoder.collect_statistics(
        distance=distance,
        noise_probabilities=noise_probabilities,
        num_shots_per_prob=num_shots,
        noise_type=noise_type
    )
