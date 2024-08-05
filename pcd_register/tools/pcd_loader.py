"""Pcd data loader

This script is designed to load the pcd file and provide the points, colors, 
and normals in a NumPy array format.

    * get_data_info - Checks input file is pcd format or not
    * pcd_loader - Class that loading/reading a pcd file
"""

import os
import sys
import numpy as np
import open3d as o3d

def get_data_info(pipe: str,
                  dat_src: str,
                  dat_tar: str = '',
                  use_dat_ex: bool = False,
                  verbose: bool = False) -> str:
    """Checks input files are pcd format or not. If one of the input file is not 
    pcd format, exit the script. If use_dat_ex is True, the dat_src and dat_tar 
    will be overwritten by ICP dataset.

    Parameters
    ----------
    pipe: str
        The name of the pipeline. regi (registration) or proc_3d (3d process).
    dat_src : str
        The input source file path.
    dat_tar : str
        The input target file path (default is '').
    use_dat_ex : bool
        Boolean statement for using example ICP dataset (default is False).
    verbose : bool
        Boolean statement to control the print (default is False).

    Returns
    -------
    dat_src: str
        The modified input source path after sanity check.
    dat_tar: str
        The modified input target path after sanity check.
    dat_key : str
        The keyward to discribe the data.
    """

    # Check example option and replace data path if it is true.
    if use_dat_ex:
        icp_paths = o3d.data.DemoICPPointClouds().paths[:2] # only 2 paths
        dat_src = icp_paths[0]
        dat_tar = icp_paths[1]
        dat_key = 'icp'
    else:
        # First, check whether string has same end with '.pcd'.
        pos_formats = ('.pcd', '.ply')
        is_pcd_src = dat_src.endswith(pos_formats)
        # If pipeline is registration, then we actually check the sanity of 
        # input target file.
        if pipe == 'regi':
            is_pcd_tar = dat_tar.endswith(pos_formats)
        else:
            is_pcd_tar = True
        
        if not is_pcd_src or not is_pcd_tar:
            print('Something is wrong in the input file!')
            sys.exit(1)
        else:
            # Use file names for the data keyward.
            dat_src_key = os.path.splitext(os.path.basename(dat_src))[0]
            dat_key = f'{dat_src_key}'
            if pipe == 'regi': # Add taget file name into keyward.
                dat_tar_key = os.path.splitext(os.path.basename(dat_tar))[0]
                dat_key += f'_{dat_tar_key}'
    
    if verbose:
        print(f'Source data path: {dat_src}')
        if pipe == 'regi':
            print(f'Target data path: {dat_tar}')
        print(f'Data keyword: {dat_key}')

    return dat_src, dat_tar, dat_key
    

class pcd_loader:
    """Designed to load the pcd file and provide the points, colors, 
    and normals in a NumPy array format.

    ...

    Attributes
    ----------
    verbose : bool
        Boolean statement to control the print
    pcd_dat : PointCloud
        Open3d data class
    pts : ndarray
        Points from the pcd file in NumPy array
    col : ndarray
        Colors from the pcd file in NumPy array
    nor : ndarray 
        Colors from the pcd file in NumPy array

    Methods
    -------
    get_pts()
        Extract the informations from pcd file and wrap with NumPy array
    """

    def __init__(self, data: list, verbose: bool = False):
        """Loads the pcd files. 
        If there is noting in the input file, exit the script.

        Parameters
        ----------
        data : str
            The list of input file paths
        verbose : bool
            Boolean statement to control the print (default is False)
        """

        self.verbose = verbose

        # Load the all 3d data
        self.pcd_list = []
        for idx, indi in enumerate(data):
            self.pcd_list.append(o3d.io.read_point_cloud(indi))
            if self.verbose:
                print(f'Input file: {indi}')
            
            # If file is empty exit the script/
            if self.pcd_list[idx].is_empty():
                print('Something is wrong in the input file!')
                sys.exit(1)
    
    def get_pts(self, pcd_idx: int = 0, use_np: bool = False):
        """Extract the informations from pcd file and wrap with NumPy array

        Parameters
        ----------
        pcd_idx : int
            The index for pcd data list (default is 0).
        use_np : bool
            Boolean statement to wrap the point information with 
            NumPy array (default is False)
        """        
        
        self.pts = self.pcd_list[pcd_idx].points
        self.col = self.pcd_list[pcd_idx].colors
        self.nor = self.pcd_list[pcd_idx].normals
        if use_np:
            self.pts = np.asarray(self.pts, dtype = float)
            self.col = np.asarray(self.col, dtype = float)
            self.nor = np.asarray(self.nor, dtype = float)
