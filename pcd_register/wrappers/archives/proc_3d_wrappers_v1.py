"""3d process wrapper

This script is designed to call necessary classes to calculate the covariance 
matrix, approximate curvature, and projection of point to the plane without 
using the Open3d package.

    * main - Wrap pcd_loader and proc_3d_loader for calculation 
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
from tools.proc_3d_loader import proc_3d_loader
from tools.utility import h5_savor

# The arguments are controlled by the click package.
@click.command()
@click.option('-l', '--dat_list', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-e', '--use_dat_ex', default=False, type=bool)
@click.option('-d', '--use_debug', default=False, type=bool)
def proc_3d_main(dat_list: str,
                 output: str, 
                 index: int, 
                 radius: float, 
                 verbose: bool, 
                 use_dat_ex: bool,
                 use_debug: bool) -> dict:
    """Designed to call necessary classes for calculation. If it is executed by
    itself, It will save the results in hdf5 format. If not, It will return the
    results in a dictionary format.

    Parameters
    ----------
    dat_list : str
        The list of input file paths (default is ''). User can input multiple
        files by saparating comma without space.
    output : str
        The path for storing output file. If user doesn't specify the path,
        It saves the output in the DataLab_test/output/ path. (default is '')
    index : int
        The index for selecting the point (default is 0)
    radius : float
        The boundary condition to select the neighboring points
        (default is 0.1)
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
        pipe = 'proc_3d'
        dat_list, dat_key = get_data_info(pipe, dat_list, 
                                          use_dat_ex=use_dat_ex, 
                                          verbose=verbose)

    # Loads pcd file.
    pcd = pcd_loader(dat_list, verbose=verbose)

    # Get the all points.
    # For this 3d process test, I only choose first pcd data in the list for the
    # calculation. In the real case, the script need to be smarter to do 
    # calculation for all the input files.
    pcd.get_pts(pcd_dat_idx=0, use_np=True) # Store the points in a NumPy array.
    pts = pcd.pts # 3d points in 2d array. Shape: (# of points, xyz)
 
    # Constructs the 3d process class. 
    proc_3d = proc_3d_loader(verbose=verbose, 
                             use_debug=use_debug, 
                             use_KDTree=False) 

    # Perform the calculations.
    proc_3d.get_3d_process(pts, index, radius)  
    covar_mtx = proc_3d.pts_nei_cov # The covariance matrix.
    approx_curv = np.array([proc_3d.pts_nei_curv]) # The approximate curvature.
    proj_pts = proc_3d.pts_nei_proj # The projection of point to the plane.

    # Save the middle steps, if use_debug is true.
    if use_debug:
        debug_lists = [proc_3d.pts_nei_cen, proc_3d.eig_val_nei, 
                       proc_3d.eig_vec_nei, proc_3d.nor_vec_nei,
                       proc_3d.dis_vec_nei, proc_3d.pts_i, proc_3d.pts_nei]
        debug_keys = ['centriod', 'eigen_val', 'eigen_vec', 'nomal_vec', 
                     'displace_vec', 'pts_i', 'pts_nei']
    
    # Makes a dictionary for the results.
    results = {'covar_mtx': covar_mtx, 
               'approx_curv': approx_curv, 
               'proj_pts': proj_pts}  
    if use_debug: # Add the middle steps.
        for idx, de in enumerate(debug_keys):
            results[de] = debug_lists[idx]
    
    # Print the results. 
    np.set_printoptions(threshold=0) # Turncates numpy print. Just for cosmetic.
    print(f'Covariance matrix @ index {index}:\n{covar_mtx}')
    print(f'Approximate Curvature @ index {index}: {approx_curv}')
    print(f'Projection of points to the plane @ index {index}:\n{proj_pts}')
    print(f'Size of array: {proj_pts.shape}')

    if __name__ == "__main__":
        # Save the results.
        # Until I confirm the conventional file format for saving the results,
        # it will be saved in the hdf5 format.
        file_name = f'{dat_key}_{pipe}_idx{index}_rad{radius}.h5'
        h5_savor(output, file_name, results, verbose=verbose)
    else: 
        # Return the results.
        return results
 
if __name__ == "__main__":

    proc_3d_main()










