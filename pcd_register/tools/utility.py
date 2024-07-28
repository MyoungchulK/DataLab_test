"""All the convenience functions are saved in here

    * size_checker - check the size of the file 
"""
import os
import numpy as np

def size_checker(d_path: str, use_byte: bool = False) -> str:
    """ Check the size of the file

    Parameters
    ----------
    d_path : str
        The data path that want to measure the size
    use_byte : bool
        Whether user want to print the size in the byte ot not
        (default is False)

    Returns
    -------
    str
        The size of the file with a unit
    """
    
    # unit list
    dat_size = ['Bytes', 'KB', 'MB', 'GB', 'TB!?!?!', 'PB!?!?!']
    size_idx = 0 # Index for choosing right unit

    # file size
    file_size = os.path.getsize(d_path)

    # Round up the data size for more convenient unit
    if use_byte:
        pass # Just use Btyte
    else:
        n = len(str(file_size).split(".")[0]) # Counts the length of over decimal
        # while loop to update the size by each unit
        while n > 3:
            file_size /= 1024
            n = len(str(file_size).split(".")[0])
            size_idx += 1
        file_size = np.round(file_size, 2)

    # final message
    msg = f'file size is {file_size} {dat_size[size_idx]}'

    return msg 
