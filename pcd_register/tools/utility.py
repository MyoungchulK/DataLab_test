"""All the convenience functions are saved in here

    * h5_savor - save the results in hdf5 format 
    * size_checker - check the size of the file 
"""
import os
import numpy as np
import h5py

def h5_savor(output: str, 
             file_name: str,
             results: dict,
             verbose: bool = False):
    """Save the data into a hdf5 format. This saving part becomes own function,
    Since it is used on the main and wrappers. But I don't like to duplicate the 
    results, which can be huge dictionary, just for saving function. 

    Parameters
    ----------
    output : str
        The output path for the hdf5 file.
    file_name : str
        The name of the hdf5 file.
    results : dict
        The contents of the hdf5 file.
    verbose : bool
        Boolean statement to control the print (default is False)
    """

    # Make output path
    if len(output) == 0:
        # If output path is not specified, use default output path
        code_path = os.path.dirname(os.path.realpath(__file__)) # bit old method
        output = os.path.join(code_path, '../../output')
    if not os.path.exists(output):
        os.makedirs(output)

    # Creates the hdf5 file.
    output_name = os.path.join(output, file_name)
    hf = h5py.File(output_name, 'w')
    for r in results:
        if verbose:
            print(r, results[r].shape) # Checks what is saving in the file.
        hf.create_dataset(r, data=results[r], compression="gzip"
                          , compression_opts=9)
    hf.close()
    print(f'Output is in {output_name}. {size_checker(output_name)}')
    
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
