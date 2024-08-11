"""All the convenience functions are saved in here

    * h5_savor - save the results in hdf5 format 
    * size_checker - check the size of the file 
"""
import os
import numpy as np
import h5py

def get_tools_abspath() -> str:
    """Finds out the absolute path of the tools path.

    Returns
    -------
    tool_path : str
        The absolute path of the tools path.
    """
    
    tools_path = os.path.dirname(os.path.realpath(__file__)) 
    
    return tools_path

def h5_savor(dat_dict: dict, results: dict):
    """Save the data into a hdf5 format. This saving part becomes own function,
    Since it is used on the main and wrappers. But I don't like to duplicate the
    results, which can be huge dictionary, just for saving function. 

    Parameters
    ----------
    dat_dict : dict
        The variables for the pipeline process.
    results : dict
        The contents of the hdf5 file.
    """

    # Make output path
    output = dat_dict["output"]
    if len(output) == 0:
        # If output path is not specified, use default output path
        code_path = get_tools_abspath()
        file_path = f'../../output/{dat_dist["pipe"]}_example.h5'
        output = os.path.join(code_path, file_path)
    output_path = os.path.dirname(output)  
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Creates the hdf5 file.
    hf = h5py.File(output, 'w')
    for r in results:
        if dat_dict['verbose']:
            print(r, results[r].shape) # Checks what is saving in the file.
        hf.create_dataset(r, data=results[r], compression="gzip"
                          , compression_opts=9)
    hf.close()
    print(f'Output is in {output}. {size_checker(output)}')
    
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
