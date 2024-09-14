import numpy as np
import psutil
import time
import os
import shutil  # For disk space checking
import gc      # For garbage collection

# Global variables
DTYPE = np.float32

# Global functions
def calculate_memory_usage(matrix_shape, dtype=DTYPE):
    """
    Calculate the memory usage for a matrix given its shape and data type.

    Parameters:
    - matrix_shape (tuple): Shape of the matrix (rows, columns).
    - dtype (data-type): Data type of the matrix elements (default: np.float32).

    Returns:
    - memory_usage (float): Memory usage in MB.
    """
    element_size = np.dtype(dtype).itemsize  # Size of each element in bytes
    total_elements = matrix_shape[0] * matrix_shape[1]
    total_bytes = total_elements * element_size
    return total_bytes / (1024 ** 2)  # Convert bytes to MB

def check_available_memory(required_memory):
    """
    Check if the available memory is sufficient to perform an operation.

    Parameters:
    - required_memory (float): The memory required in MB.

    Returns:
    - bool: True if there is sufficient memory, False otherwise.
    """
    available_memory = psutil.virtual_memory().available / (1024 ** 2)  # Available memory in MB
    print(f"Available memory: {available_memory:.2f} MB")

    if available_memory > required_memory:
        print("Sufficient memory is available for the operation.")
        return True
    else:
        print("Insufficient memory available. The operation may not proceed with the specified method.")
        return False

def estimate_disk_space(matrix_shape, dtype=DTYPE):
    """
    Estimate the disk space required to store a matrix.

    Parameters:
    - matrix_shape (tuple): Shape of the matrix (rows, columns).
    - dtype (data-type): Data type of the matrix elements.

    Returns:
    - disk_space (float): Disk space in MB.
    """
    return calculate_memory_usage(matrix_shape, dtype)

def check_available_disk_space(required_space, directory='.'):
    """
    Check if there is enough available disk space in the given directory.

    Parameters:
    - required_space (float): The disk space required in MB.
    - directory (str): The directory path where the file will be saved.

    Returns:
    - bool: True if there is sufficient disk space, False otherwise.
    """
    total, used, free = shutil.disk_usage(directory)
    available_disk_space = free / (1024 ** 2)  # Convert bytes to MB
    print(f"Available disk space in '{os.path.abspath(directory)}': {available_disk_space:.2f} MB")

    if available_disk_space > required_space:
        print("Sufficient disk space is available for the operation.")
        return True
    else:
        print("Insufficient disk space available. The operation cannot proceed.")
        return False

def perform_chunked_multiplication_over_A(A_shape, B_shape, chunk_size, dtype=DTYPE):
    """
    Perform matrix multiplication by chunking over A (rows of A).

    Parameters:
    - A_shape (tuple): Shape of matrix A.
    - B_shape (tuple): Shape of matrix B.
    - chunk_size (int): Number of rows to process at a time.
    - dtype (data-type): Data type of the matrix elements.
    """
    # Adjust chunk_size if it's larger than A_shape[0]
    if chunk_size > A_shape[0]:
        print(f"Specified chunk_size ({chunk_size}) is larger than the number of rows in matrix A ({A_shape[0]}).")
        chunk_size = A_shape[0]
        print(f"Chunk size adjusted to {chunk_size}.\n")

    # Estimate RAM usage for chunked solution over A
    memory_A_chunk = calculate_memory_usage((chunk_size, A_shape[1]), dtype)
    memory_B_full = calculate_memory_usage(B_shape, dtype)
    memory_result_chunk = calculate_memory_usage((chunk_size, B_shape[1]), dtype)
    total_estimated_ram_chunking_A = memory_A_chunk + memory_B_full + memory_result_chunk

    print(f"Estimated maximum RAM required for solution with chunking over A: {total_estimated_ram_chunking_A:.2f} MB\n")

    # Check if the operation can be done with chunking over A
    if check_available_memory(total_estimated_ram_chunking_A):
        # Proceed with chunked multiplication over A
        print("Sufficient memory available for chunked multiplication over A.")
        print("Initializing matrix B for chunked multiplication over A...")

        # Initialize full matrix B
        B = np.random.rand(*B_shape).astype(dtype)

        # Create a memory-mapped file to save the results in chunks
        result_filename = 'matrix_multiplication_result.npy'
        result_shape = (A_shape[0], B_shape[1])

        # Create an empty file with the correct shape
        np.lib.format.open_memmap(
            result_filename,
            mode='w+',
            dtype=dtype,
            shape=result_shape
        )

        total_chunks = (A_shape[0] + chunk_size - 1) // chunk_size
        total_start_time = time.time()

        # Multiply in chunks over A and save to the memory-mapped file
        for chunk_index, i in enumerate(range(0, A_shape[0], chunk_size), start=1):
            chunk_start_time = time.time()
            # Select a chunk of rows from A
            end_i = min(i + chunk_size, A_shape[0])
            current_chunk_size = end_i - i
            print(f"Processing chunk {chunk_index}/{total_chunks} (rows {i} to {end_i - 1})...")

            # Load or generate the chunk of A
            A_chunk = np.random.rand(current_chunk_size, A_shape[1]).astype(dtype)  # Simulate loading or generating A_chunk

            # Multiply A_chunk with B
            result_chunk = np.dot(A_chunk, B)

            # Open the result memmap in 'r+' mode
            result_memmap = np.lib.format.open_memmap(
                result_filename,
                mode='r+',
                dtype=dtype,
                shape=result_shape
            )

            # Write the result chunk to the memory-mapped file
            result_memmap[i:end_i, :] = result_chunk
            result_memmap.flush()  # Flush changes to disk

            # Delete the memmap object to free memory
            del result_memmap

            # Delete temporary variables and collect garbage
            del A_chunk
            del result_chunk
            gc.collect()

            # Monitor current memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

            chunk_end_time = time.time()
            chunk_time = chunk_end_time - chunk_start_time
            elapsed_time = chunk_end_time - total_start_time
            estimated_total_time = (elapsed_time / chunk_index) * total_chunks
            remaining_time = estimated_total_time - elapsed_time

            print(f"Chunk {chunk_index} completed in {chunk_time:.2f} seconds.")
            print(f"Current memory usage: {memory_usage:.2f} MB")
            print(f"Estimated time remaining: {remaining_time / 60:.2f} minutes\n")

        # Delete B and collect garbage
        del B
        gc.collect()

        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        print(f"Matrix multiplication over A completed in {total_elapsed_time / 60:.2f} minutes.")
        print(f"Results saved to disk at '{os.path.abspath(result_filename)}'.")
        print(f"Final result file size: {os.path.getsize(result_filename) / (1024 ** 2):.2f} MB")
    else:
        print("Insufficient memory for chunked multiplication over A. Operation cannot proceed.")

def perform_multiplication(A_shape, B_shape, chunk_size, dtype=DTYPE):
    """
    Perform matrix multiplication, attempting full multiplication or chunking over A.

    Parameters:
    - A_shape (tuple): Shape of matrix A.
    - B_shape (tuple): Shape of matrix B.
    - chunk_size (int): Chunk size for rows (A).
    - dtype (data-type): Data type of the matrix elements.
    """
    # Calculate and report memory usage for full matrices
    memory_A_full = calculate_memory_usage(A_shape, dtype)
    memory_B_full = calculate_memory_usage(B_shape, dtype)
    memory_result_full = calculate_memory_usage((A_shape[0], B_shape[1]), dtype)
    total_estimated_ram_no_chunking = memory_A_full + memory_B_full + memory_result_full

    # Estimate disk space required for the result
    estimated_disk_space = estimate_disk_space((A_shape[0], B_shape[1]), dtype)

    print(f"Memory required for full matrix A: {memory_A_full:.2f} MB")
    print(f"Memory required for full matrix B: {memory_B_full:.2f} MB")
    print(f"Memory required for result matrix: {memory_result_full:.2f} MB")
    print(f"Total estimated RAM required for full solution: {total_estimated_ram_no_chunking:.2f} MB")
    print(f"Estimated disk space required for result: {estimated_disk_space:.2f} MB\n")

    # Check available disk space before proceeding
    if not check_available_disk_space(estimated_disk_space):
        print("Operation aborted due to insufficient disk space.")
        return

    # Attempt full multiplication
    print("Checking if full matrix multiplication can be performed...")
    if check_available_memory(total_estimated_ram_no_chunking):
        try:
            # Initialize matrices A and B with random values
            print("Initializing full matrices A and B...")
            A = np.random.rand(*A_shape).astype(dtype)
            B = np.random.rand(*B_shape).astype(dtype)

            # Try performing full multiplication
            print("Attempting full matrix multiplication...")
            start_time = time.time()
            result_full = np.dot(A, B)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Full matrix multiplication completed successfully in {total_time:.2f} seconds.")
            return  # Exit if the full multiplication was successful
        except MemoryError as e:
            print("\nMemoryError encountered during full matrix multiplication.")
            print(f"Error message: {str(e)}")
            print("Not enough memory to perform full matrix multiplication.")

    # Attempt chunking over A
    print("\nAttempting chunked multiplication over A...")
    perform_chunked_multiplication_over_A(A_shape, B_shape, chunk_size, dtype)

# Main calculation routine
if __name__ == "__main__":
    A_shape = (20000, 40)       # Shape of matrix A
    B_shape = (40, 1000000)     # Shape of matrix B
    chunk_size = 500          # Chunk size for rows of A

    perform_multiplication(A_shape, B_shape, chunk_size)
