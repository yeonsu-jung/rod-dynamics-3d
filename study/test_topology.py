#!/usr/bin/env python3
"""
Unit tests for topological invariants computation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import (
    compute_linking_number,
    compute_linking_matrix,
    compute_vorticity,
    compute_total_chirality,
    compute_per_rod_chirality,
    compute_per_pair_chirality
)


def test_linking_number_antisymmetry():
    """Test that linking number is antisymmetric: lk(A,B) = -lk(B,A)"""
    print("Testing linking number antisymmetry...")
    
    # Create two skew rods
    rod_a = np.array([0, 0, 0, 1, 0, 0])  # Along x-axis
    rod_b = np.array([0.5, -0.5, -0.5, 0.5, 0.5, 0.5])  # Vertical through (0.5, 0, 0)
    
    lk_ab = compute_linking_number(rod_a, rod_b)
    lk_ba = compute_linking_number(rod_b, rod_a)
    
    assert lk_ab == -lk_ba, f"Antisymmetry failed: lk(A,B)={lk_ab}, lk(B,A)={lk_ba}"
    print(f"  ✓ lk(A,B) = {lk_ab}, lk(B,A) = {lk_ba}")


def test_linking_matrix_antisymmetry():
    """Test that linking matrix is antisymmetric: X^T = -X"""
    print("Testing linking matrix antisymmetry...")
    
    # Create a simple configuration
    rods = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    ])
    
    X = compute_linking_matrix(rods)
    
    # Check antisymmetry
    assert np.allclose(X, -X.T), "Linking matrix is not antisymmetric"
    
    # Check diagonal is zero
    assert np.all(np.diag(X) == 0), "Diagonal is not zero"
    
    print(f"  ✓ X is antisymmetric and diagonal is zero")
    print(f"  Linking matrix:\n{X}")


def test_vorticity_orientation_invariance():
    """Test that vorticity is independent of orientation choices"""
    print("Testing vorticity orientation invariance...")
    
    # Create three rods
    rods = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    ])
    
    X1 = compute_linking_matrix(rods)
    v1 = compute_vorticity(X1, 0, 1, 2)
    
    # Flip orientation of first rod
    rods_flipped = rods.copy()
    rods_flipped[0] = rods[0, [3, 4, 5, 0, 1, 2]]  # Swap endpoints
    
    X2 = compute_linking_matrix(rods_flipped)
    v2 = compute_vorticity(X2, 0, 1, 2)
    
    # Vorticity should be the same
    assert v1 == v2, f"Vorticity changed after orientation flip: {v1} != {v2}"
    print(f"  ✓ Vorticity unchanged after orientation flip: v_012 = {v1}")


def test_simple_configuration():
    """Test computation on a simple known configuration"""
    print("Testing simple configuration...")
    
    # Create a simple configuration of 4 rods
    rods = np.array([
        [0, 0, 0, 1, 0, 0],      # Rod 0: along x-axis
        [0, 0, 0, 0, 1, 0],      # Rod 1: along y-axis
        [0, 0, 0, 0, 0, 1],      # Rod 2: along z-axis
        [0.5, 0.5, -1, 0.5, 0.5, 1]  # Rod 3: vertical through (0.5, 0.5, 0)
    ])
    
    X = compute_linking_matrix(rods)
    C = compute_total_chirality(X)
    c_i = compute_per_rod_chirality(X)
    c_ij = compute_per_pair_chirality(X)
    
    print(f"  Linking matrix:\n{X}")
    print(f"  Total chirality C: {C}")
    print(f"  Per-rod chirality c_i: {c_i}")
    print(f"  ✓ Computation completed successfully")


def test_eigenvalue_properties():
    """Test that eigenvalues of antisymmetric matrix are purely imaginary"""
    print("Testing eigenvalue properties...")
    
    # Create a configuration
    rods = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    ])
    
    X = compute_linking_matrix(rods)
    eigenvalues = np.linalg.eigvals(X)
    
    # For antisymmetric matrices, eigenvalues should be purely imaginary
    # (or zero for odd-dimensional matrices)
    real_parts = eigenvalues.real
    max_real = np.max(np.abs(real_parts))
    
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Max |Re(λ)|: {max_real:.2e}")
    
    # Allow small numerical error
    assert max_real < 1e-10, f"Eigenvalues have significant real parts: {max_real}"
    print(f"  ✓ Eigenvalues are purely imaginary (within numerical tolerance)")


def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("RUNNING UNIT TESTS FOR TOPOLOGICAL INVARIANTS")
    print("="*60)
    print()
    
    tests = [
        test_linking_number_antisymmetry,
        test_linking_matrix_antisymmetry,
        test_vorticity_orientation_invariance,
        test_simple_configuration,
        test_eigenvalue_properties
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            print()
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
