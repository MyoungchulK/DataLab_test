"""registration wrapper

This script is designed to call necessary classes to calculate the registration
process for multiple point cloud data. It will first perform the preprocessing,
such as down sampleing by voxeling and extracting feature by FPFH. Then, It 
will apply RANSAC and ICP for alignment. The results will be saved on pcd or
hdf5 format based on the informations.

    * main - Wrap pcd_loader and regi_loader for calculation 
"""

import os
import sys
import click
import numpy as np

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.pcd_loader import get_data_info, pcd_loader, save_pcd_info
from tools.regi_loader import regi_loader, draw_regi_result
from tools.utility import h5_savor

# The arguments are controlled by the click package.
@click.command()
@click.option('-v', '--dat_var', default='', type=str)
@click.option('-d', '--dat_dict', default={'':''}, type=dict)
def regi_main(dat_var: str, dat_dict: dict) -> dict:
    """Designed to call necessary classes for calculation. If it is executed by
    itself, It will save the results in pcd ot hdf5 format. If not, It will 
    return the results in a dictionary format.

    Parameters
    ----------
    dat_var : str
        The text file path that contains all the variables for the pipeline 
        process (default is ''). User can control the script by changing the 
        contents of the text file. By doing this way, we don't have to create
        infinite arguments at the terminal. The variables will be stored in the
        dictionary. If dat_var is empty, use icp examples in the examples path. 
    dat_dict : dict
        The variables for the pipeline process (Default is {'':''}).

    Returns
    -------
    dict
        The results will be linked to a dictionary format
    """

    # Check the sanity of the data path when it is main.
    if __name__ == "__main__":
        dat_dict = get_data_info(dat_var)
    verbose = dat_dict['verbose']
    output_dir = os.path.dirname(dat_dict['output'])
    
    # Loads pcd file.
    pcd = pcd_loader(dat_dict['dat_list'], verbose=verbose)

    # For this registration test, I only choose first two pcd data in the list
    # for the calculation. In the real case, the script need to be smarter to do
    # calculation for all the input files.
    pcd_list = pcd.pcd_list[:2] # source and target pcd files.

    # Constructs the class
    regi = regi_loader(pcd_list, verbose=verbose, 
                       use_debug=dat_dict['use_debug'])

    # Performs pre processing
    regi.get_pre_process(0, 1)
    pcd_down = regi.pcd_down
    pcd_fpfh = regi.pcd_fpfh
   
    # RANSAC registration
    ransac_regi = regi.get_ransac_regi(0, 1) 

    # ICP registration
    icp_regi = regi.get_icp_regi(0, 1)
    
    # Saving the results of down sample, RANSAC, ICP
    pcds = ['src', 'tar']
    res = ['preprocess', 'ransac', 'icp']
    results = [pcd_down, pcd_down, pcd_list]
    tans = [np.identity(4, dtype=float), 
            ransac_regi.transformation, 
            icp_regi.transformation]

    # for loop for the results type.
    for re_indi in range(len(res)):
        # Saves in the png format.
        plot_path = os.path.join(
            output_dir, 
            f'{res[re_indi]}_{dat_dict["pipe_name"]}.png')
        draw_regi_result(results[re_indi][0], results[re_indi][1], plot_path,
                         trans=tans[re_indi], verbose=verbose)

        # for loop for source or target pcd.
        for p_indi in range(len(pcds)):
            if p_indi == 0: # apply transformation 
                results[re_indi][p_indi].transform(tans[re_indi])

            # Saves in pcd format.
            file_path = os.path.join(
                output_dir,
                f'{res[re_indi]}_{dat_dict["pipe_name"]}_{pcds[p_indi]}.pcd') 
            save_pcd_info(results[re_indi][p_indi], file_path, verbose=verbose) 

    return {"1":np.array([1])}
    """
    if __name__ == "__main__":
        # Save the results.
        # Until I confirm the conventional file format for saving the results,
        # it will be saved in the hdf5 format.
        h5_savor(dat_dict, results)
    else: 
        # Return the results.
        return results
    """
if __name__ == "__main__":

    regi_main()










