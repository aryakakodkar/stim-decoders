def mask_iter_indices(mask: int):
    """Yield set-bit indices (ascending) for Python int mask
    
    Args:
        mask: Integer bitmask.
        
    Returns: 
        Generator of indices where bits are set.
    """
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        yield idx
        mask ^= lsb

def split_support_list_fast(support_indices, pauli_mask, erasure_mask):
    """(DEPRECATED) Splits support indices into Pauli and Erasure lists using bitmasks.

    Args:
        support_indices: List of qubit indices in the support.
        pauli_mask: Integer bitmask for Pauli errors.
        erasure_mask: Integer bitmask for Erasure errors.
    
    Returns:
        Tuple of (pauli_indices, erasure_indices) lists containing the respective qubit indices.
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

def indices_to_mask(indices):
    """(DEPRECATED) Converts a list of indices to a Python int bitmask.
    
    Args:
        indices: List of integer indices.
        
    Returns:
        Integer bitmask with bits set at the given indices.
    """
    m = 0
    shift = (1).__lshift__  # localize to avoid global lookup
    or_ = int.__or__        # same trick
    for i in indices:
        m = or_(m, shift(i))
    return m
