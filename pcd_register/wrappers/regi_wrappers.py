"""registration wrapper

This script is designed to call necessary classes to calculate the registration
process for multiple point cloud data. It will first perform the preprocessing,
such as down sampleing by voxeling and extracting feature by FPFH. Then, It 
will apply RANSAC and ICP for alignment. The results will be saved on pcd or
hdf5 format based on the informations.

    * main - Wrap pcd_loader and proc_3d_loader foe calculation 
"""

import os
import sys
import click
import numpy as np

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.pcd_loader import get_data_info
from tools.pcd_loader import pcd_loader
from tools.utility import h5_savor

# The arguments are controlled by the click package.
@click.command()
@click.option('-s', '--dat_src', default='', type=str)
@click.option('-t', '--dat_tar', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-e', '--use_dat_ex', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
def regi_main(dat_src: str,
              dat_tar: str,
              output: str, 
              verbose: bool, 
              use_dat_ex: bool,
              use_debug: bool) -> dict:
    """Designed to call necessary classes for calculation. If it is executed by
    itself, It will save the results in pcd ot hdf5 format. If not, It will 
    return the results in a dictionary format.

    Parameters
    ----------
    dat_src : str
        The input source file path (default is '')
    dat_tar : str
        The input target file path (default is '')
    output : str
        The path for storing output file. If user doesn't specify the path,
        It saves the output in the DataLab_test/output/ path. (default is '')
    verbose : bool
        Boolean statement to control the print (default is False)
    use_dat_ex : bool
        Boolean statement for using example ICP dataset (default is False)
        If use_dat_ex is True, the dat_src and dat_tar will be overwritten by
        ICP dataset.
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging (default is False)

    Returns
    -------
    dict
        The results will be linked to a dictionary format
    """

    # Check the sanity of the data path when it is main.
    if __name__ == "__main__":
        pipe = 'regi'
        dat_src, dat_tar, dat_key = get_data_info(pipe, dat_src, dat_tar, 
                                                  use_dat_ex=use_dat_ex, 
                                                  verbose=verbose)

    # Loads pcd file.
    pcd = pcd_loader([dat_src, dat_tar], verbose=verbose)
    pcd_src = pcd.pcd_list[0]
    pcd_tar = pcd.pcd_list[1]







    
    return {"1":np.array([1])}
    """
    if __name__ == "__main__":
        # Save the results.
        # Until I confirm the conventional file format for saving the results,
        # it will be saved in the hdf5 format.
        file_name = f'{dat_key}_{pipe}.h5'
        h5_savor(output, file_name, results, verbose=verbose)
    else: 
        # Return the results.
        return results
    """
if __name__ == "__main__":

    proc_3d_main()










