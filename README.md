# stimdecoders

This package is an attempt to build some alternative decoders for stim stabilizer circuits. This project is a work in progress.

Currently working on:
- Heralded Decoder: decoder which takes account of erasure qubits and converts erasure samples to stabilizer circuits, then decodes using PyMatching

Note that while I have done my best to optimize the decoders, they are still written in Python, and are nowhere near as fast as Stim and PyMatching on their own. Modifications and improvements are very welcome.
