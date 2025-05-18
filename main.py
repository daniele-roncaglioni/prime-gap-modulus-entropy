import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os

def get_primes(n):
    """
    Returns all primes smaller than n using the Sieve of Eratosthenes.
    """
    print(f"Computing primes up to {n}...")
    start_time = time.time()
    
    # Initialize array with all numbers considered prime
    sieve = np.ones(n, dtype=bool)
    # 0 and 1 are not prime
    sieve[0:2] = False
    
    # Start with 2 and mark all its multiples as non-prime
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            # Mark all multiples of i as non-prime
            sieve[i*i:n:i] = False
    
    # Return indices where sieve is True (the primes)
    primes = np.where(sieve)[0]
    
    elapsed = time.time() - start_time
    print(f"Found {len(primes)} primes in {elapsed:.2f} seconds")
    
    return primes

def get_prime_gaps(primes):
    """
    Returns the gaps between consecutive primes.
    """
    if len(primes) <= 1:
        return []
    
    return [primes[i+1] - primes[i] for i in range(len(primes)-1)]

def get_gap_distribution(gaps, m):
    """
    Counts how often each gap occurs, modulo m, and normalizes to get a probability distribution.
    """
    # Apply modulo m to all gaps
    gaps_mod_m = [gap % m for gap in gaps]
    
    # Count occurrences of each remainder (0 to m-1)
    counts = np.zeros(m)
    for gap in gaps_mod_m:
        counts[gap] += 1
    
    # Normalize to get probability distribution
    total = np.sum(counts)
    if total > 0:  # Avoid division by zero
        distribution = counts / total
    else:
        distribution = np.zeros(m)
    
    return distribution

def calculate_entropy(distribution):
    """
    Calculates the Shannon entropy for the given probability distribution.
    """
    # Filter out zero probabilities to avoid log(0)
    non_zero_probs = distribution[distribution > 0]
    
    # Calculate entropy: -sum(p_i * log(p_i))
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    return entropy

def plot_entropy_curves(n_values, m_values, distributions, results_dir):
    """
    Plots the entropy vs n for different values of m.
    """
    # Create figure for entropy plots
    plt.figure(figsize=(10, 6))
    
    for m_idx, m in enumerate(m_values):
        print(f"Plotting entropy curve for m={m} ({m_idx+1}/{len(m_values)})...")
        entropies = []
        
        for n_idx, n in enumerate(n_values):
            distribution = distributions[m][n]  
            entropy = calculate_entropy(distribution)
            entropies.append(entropy)
        
        # Calculate maximum possible entropy for this m: log2(m)
        max_entropy = np.log2(m)
        
        # Plot the entropy values
        plt.plot(n_values, entropies, marker='o', label=f'm = {m}')
        
        # Add a horizontal line for the maximum entropy
        plt.axhline(y=max_entropy, linestyle='--', color=plt.gca().lines[-1].get_color(), 
                   alpha=0.5, label=f'Max entropy for m={m}: {max_entropy:.3f}')
    
    # Configure the plot
    plt.xscale('log')
    plt.xlabel('n (upper bound of primes, log scale)')
    plt.ylabel('Shannon Entropy (bits)')
    plt.title('Shannon Entropy of Prime Gap Distribution (mod m) vs n')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'entropy_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved entropy curves to {save_path}")

def plot_distributions(m_values, distributions, max_n, results_dir):
    """
    Plots the probability distributions of prime gaps modulo m for the largest n value.
    """
    # Calculate number of columns needed, aiming for a balanced layout
    num_m = len(m_values)
    num_cols = min(3, num_m)  # Max 3 columns for better visibility
    num_rows = (num_m + num_cols - 1) // num_cols  # Ceiling division
    
    # Create figure for distribution plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    # Make axes iterable even if there's only one row or column
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"Plotting distributions for largest n={max_n}...")
    
    # Plot each distribution
    for i, m in enumerate(m_values):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        
        # Get the distribution for the largest n
        distribution = distributions[m][max_n]
        
        # Plot as bar chart
        ax.bar(range(m), distribution, alpha=0.7)
        ax.set_title(f'Gap Distribution mod {m} (n={max_n})')
        ax.set_xlabel('Remainder mod ' + str(m))
        ax.set_ylabel('Probability')
        ax.set_xticks(range(m))
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add entropy value to the plot
        entropy = calculate_entropy(distribution)
        max_entropy = np.log2(m)
        ax.text(0.5, 0.9, f'Entropy: {entropy:.3f} bits\nMax: {max_entropy:.3f} bits', 
                transform=ax.transAxes, ha='center', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Hide any unused subplots
    for i in range(num_m, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    # Save the plot
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'gap_distributions.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved distribution plots to {save_path}")

def analyze_prime_gaps(n_values, m_values):
    """
    Analyze prime gaps and their distributions.
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    # Precompute primes and gaps for the largest n value
    max_n = max(n_values)
    all_primes = get_primes(max_n)
    all_gaps = get_prime_gaps(all_primes)
    
    # Store distributions for all n values and m values
    # Using a nested dictionary: distributions[m][n]
    distributions = {m: {} for m in m_values}
    
    # Process all data
    for m_idx, m in enumerate(m_values):
        print(f"Processing modulus m={m} ({m_idx+1}/{len(m_values)})...")
        
        for n_idx, n in enumerate(n_values):
            print(f"  - Computing for n={n} ({n_idx+1}/{len(n_values)})")
            
            # Filter primes less than the current n
            primes_for_n = all_primes[all_primes < n]
            
            # We need to recalculate gaps if this is not the max n
            if n < max_n:
                # Only take gaps between primes less than n
                # The last prime we consider is the last one < n
                num_primes = len(primes_for_n)
                if num_primes > 1:
                    gaps_for_n = all_gaps[:num_primes-1]
                else:
                    gaps_for_n = []
            else:
                gaps_for_n = all_gaps
            
            # Calculate distribution and store it
            distribution = get_gap_distribution(gaps_for_n, m)
            distributions[m][n] = distribution
    
    # Plot entropy curves
    plot_entropy_curves(n_values, m_values, distributions, results_dir)
    
    # Plot distributions for the largest n
    plot_distributions(m_values, distributions, max_n, results_dir)
    
    return distributions

# Example usage
if __name__ == "__main__":
    # Values of n to try (upper bounds for primes)
    n_values = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000, 10_000_000_000]
    
    # Different moduli to try
    m_values = [6]
    
    total_start_time = time.time()
    distributions = analyze_prime_gaps(n_values, m_values)
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds")
    
    # Print example for a specific case
    n = 1000
    m = 6
    print(f"\nGenerating example for n={n} and m={m}:")
    primes = get_primes(n)
    gaps = get_prime_gaps(primes)
    distribution = get_gap_distribution(gaps, m)
    entropy = calculate_entropy(distribution)
    max_entropy = np.log2(m)
    
    print(f"For n={n} and m={m}:")
    print(f"Number of primes: {len(primes)}")
    print(f"First few primes: {primes[:10]}...")
    print(f"First few gaps: {gaps[:10]}...")
    print(f"Gap distribution (mod {m}): {distribution}")
    print(f"Shannon entropy: {entropy} bits")
    print(f"Maximum possible entropy: {max_entropy} bits")
    print(f"Normalized entropy: {entropy/max_entropy:.4f} (entropy/max_entropy)")
