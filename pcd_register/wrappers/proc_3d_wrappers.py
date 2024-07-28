"""3d process wrapper

This script is designed to call necessary classes to calculate the covariance 
matrix, approximate curvature, and projection of point to the plane without 
using the Open3d package.

    * main - Wrap pcd_loader and proc_3d_loader foe calculation 
"""

import os
import sys
import click
import numpy as np

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.pcd_loader import pcd_loader
from tools.proc_3d_loader import proc_3d_loader
from tools.utility import size_checker

# The arguments are controlled by the click package.
@click.command()
@click.option('-d', '--data', default='', type=str)
@click.option('-i', '--index', default=0, type=int)
@click.option('-r', '--radius', default=0.1, type=float)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-u', '--use_debug', default=False, type=bool)
def main(data: str, 
         index: int, 
         radius: float, 
         verbose: bool, 
         use_debug: bool) -> dict:
    """Designed to call necessary classes for calculation. If it is executed by
    itself, It will save the results in hdf5 format. If not, It will return the
    results in a dictionary format.

    Parameters
    ----------
    data : str
        The input file path (default is '')
    index : int
        The index for selecting the point (default is 0)
    radius : float
        The boundary condition to select the neighboring points
        (default is 0.1)
    verbose : bool
        Boolean statement to control the print (default is False)
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging (default is False)

    Returns
    -------
    dict
        The results will be linked to a dictionary format
    """

    # Loads pcd file.
    pcd = pcd_loader(data, verbose=verbose)
    pcd.get_pts_in_np() # Store the points in a NumPy array.

    # Get the all points.
    pcd.get_pts_in_np() # Store the points in a NumPy array.
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
        # It will be saves in the hdf5 format.
        import h5py       

        # Make output path 
        output_path = '../../output' # TODO: make package to control the path.
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Set the file name 
        if len(data) == 0:
            dat_name = 'EaglePointCloud'
        else:
            dat_base = os.path.basename(data)
            dat_name = os.path.splitext(dat_base)[0]
        dat_name_full = f'{dat_name}_proc_3d_idx{index}_rad{radius}.h5'
        output_name = os.path.join(output_path, dat_name_full)
       
        # Creates the hdf5 file.
        hf = h5py.File(output_name, 'w')
        for r in results:
            if verbose:
                print(r, results[r].shape) # Checking what is saving in the file.
            hf.create_dataset(r, data=results[r], compression="gzip"
                              , compression_opts=9)
        hf.close()
        print(f'Output is in {output_name}. {size_checker(output_name)}')
    else: 
        # Return the results.
        return results
 
if __name__ == "__main__":

    main()










