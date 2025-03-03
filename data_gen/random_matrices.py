import numpy as np

# Number of elements in the array
# num_elements = 25

# Dimensions of the matrices
rows, cols = 4, 3

# Number of matrices to generate
num_matrices = 50

# Generate matrices with rank 2
matrices = np.array(
    [
        (
            np.random.rand(rows, cols)  # Create a random matrix
            if min(rows, cols) >= 2  # Ensure at least rank 2 is possible
            else np.zeros((rows, cols))
        )  # Otherwise, make it a zero matrix
        for _ in range(num_matrices)
    ]
)

# Adjust each matrix to ensure it has rank 2
for matrix in matrices:
    if min(rows, cols) >= 2:
        # Make sure two rows (or columns) are linearly independent
        u = np.random.rand(rows)
        v = np.random.rand(cols)
        matrix[:1, :] = np.outer(u[:1], v)  # Ensure rank 1
        matrix[1:, :] += np.outer(u[1:], v[::-1])  # Add independent contribution


# print("Generated matrices with rank 2:")
# print(matrices)

# Splitting the matrices object into a new numpy array
split_array = np.array([(matrices[i], matrices[i + 25]) for i in range(25)])

print(f"Split array: {split_array}")
# np.save('data_gen/random_matrices', result)
