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
from tools.pcd_loader import get_data_info
from tools.pcd_loader import pcd_loader
from tools.utility import h5_savor

# The arguments are controlled by the click package.
@click.command()
@click.option('-l', '--dat_list', default='', type=str)
@click.option('-o', '--output', default='', type=str)
@click.option('-v', '--verbose', default=False, type=bool)
@click.option('-e', '--use_dat_ex', default=False, type=bool)
@click.option('-d', '--use_debug', default=False, type=bool)
def regi_main(dat_list: str,
              output: str, 
              verbose: bool, 
              use_dat_ex: bool,
              use_debug: bool) -> dict:
    """Designed to call necessary classes for calculation. If it is executed by
    itself, It will save the results in pcd ot hdf5 format. If not, It will 
    return the results in a dictionary format.

    Parameters
    ----------
    dat_list : str
        The list of input file paths (default is ''). User can input multiple
        files by saparating comma without space.
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
        dat_list, dat_key = get_data_info(pipe, dat_list, use_dat_ex=use_dat_ex, 
                                          verbose=verbose)

    # Loads pcd file.
    # For this registration test, I only choose first two pcd data in the list 
    # for the calculation. In the real case, the script need to be smarter to do 
    # calculation for all the input files.
    pcd = pcd_loader(dat_list, verbose=verbose)
    pcd_src = pcd.pcd_list[0] # source pcd file.
    pcd_tar = pcd.pcd_list[1] # target pcd file.

    pts_np = np.asarray(pcd_src.points)

    print(np.asarray(pcd_src.points).shape)
    dis = np.asarray(pcd_src.compute_nearest_neighbor_distance())
    print(dis.shape)
    print(np.nanmean(dis))
    print(np.nanmedian(dis))
    print(np.nanstd(dis))
    print(np.nanmin(dis))
    print(np.nanmax(dis))

    bbox = pcd_src.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    initial_voxel_size = np.mean(bbox_extent) / 32
    print(bbox)

    print(np.nanmin(pts_np, axis = 0))
    print(np.nanmax(pts_np, axis = 0))
    print(bbox_extent)
    print(initial_voxel_size)

    dis_len = len(dis)
    dis_idx = int(float(dis_len) * 0.95)
    dis_sort = np.sort(dis)
    dis_95 = dis_sort[dis_idx]
    medi = np.nanmedian(dis)
    print()
    print(dis_idx)
    print(dis_95)
    print(dis_sort[dis_idx - 10 : dis_idx + 10])
    print(medi)
    print(dis_95 - medi)


    
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










