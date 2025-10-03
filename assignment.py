"""The function implements the FFT using the divide and conquer approach. """
import cmath
import numpy as np

def fft_recursive(x: list[float]) -> list[complex]:
    """
    Computes the Discrete Fourier Transform (DFT) of a list of numbers
    using a recursive Fast Fourier Transform (FFT) algorithm.
    """
    N = len(x)

    # Base case for the recursion
    if N <= 1:
        return x

    # Divide: Split the list into even and odd indexed elements
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])

    # Conquer: Combine the results of the sub-problems
    # Calculate the twiddle factors and combine
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    
    # Apply the butterfly equations
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]


"Iterative FFT Implementation"
def fft_iterative(x: list[float]) -> list[complex]:
    """
    Computes the Discrete Fourier Transform (DFT) of a list of numbers
    using an iterative Cooley-Tukey FFT algorithm.
    """
    N = len(x)
    
    # Input size must be a power of 2
    if (N & (N - 1)) != 0 and N != 0:
        raise ValueError("Input size must be a power of 2 for this FFT implementation.")

    # --- Bit-Reversal Permutation ---
    # Convert input to complex numbers for calculations
    A = [complex(val) for val in x]
    log2_N = N.bit_length() - 1
    for i in range(N):
        # Reverse the bits of the index i
        rev_i = int(format(i, f'0{log2_N}b')[::-1], 2)
        # Swap elements if the reversed index is greater
        if i < rev_i:
            A[i], A[rev_i] = A[rev_i], A[i]
            
    # --- Bottom-Up Computation ---
    m = 2  # Start with transforms of size 2
    while m <= N:
        # Twiddle factor for the current transform size
        wm = cmath.exp(-2j * cmath.pi / m)
        # Iterate through the blocks of size m
        for j in range(0, N, m):
            w = 1.0 + 0j  # Initialize twiddle factor for the block
            # Perform butterfly operations within the block
            for k in range(m // 2):
                t = w * A[j + k + m // 2]
                u = A[j + k]
                A[j + k] = u + t
                A[j + k + m // 2] = u - t
                w *= wm
        m *= 2  # Move to the next transform size
        
    return A

"Testing and Verification"

# --------- Quick Check ---------
if __name__ == "__main__":
    # Small real signal
    f = [1, 2, 3, 4, 2, 0, 1, -1]

    # --- Run Recursive FFT ---
    print("## Recursive FFT Results:")
    F_recursive = fft_recursive(f)
    for k, v in enumerate(F_recursive):
        # Format the complex number for cleaner output
        print(f"{k}: ({v.real:.4f} + {v.imag:.4f}j)")

    print("\n" + "="*30 + "\n")

    # --- Run Iterative FFT ---
    print("## Iterative FFT Results:")
    F_iterative = fft_iterative(f)
    for k, v in enumerate(F_iterative):
        print(f"{k}: ({v.real:.4f} + {v.imag:.4f}j)")

    print("\n" + "="*30 + "\n")

    # --- Verification ---
    # Use numpy's allclose to handle potential floating-point inaccuracies
    are_close = np.allclose(F_recursive, F_iterative)
    print(f"Are the results from both methods identical? {are_close}")

    