def numim(n):
    """
    Create a 3-character string from an integer.
    
    Args:
    n: Integer to convert
    
    Returns:
    textnum: 3-character string representation of the input integer
    """
    # Convert the integer to a string
    n_str = str(n)
    
    # Calculate the number of leading zeros needed
    nz = 3 - len(n_str)
    
    # Create a string of leading zeros
    z = '0' * nz
    
    # Concatenate the zeros and the original number
    textnum = z + n_str
    
    return textnum