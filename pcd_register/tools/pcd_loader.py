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
                  dat_list: str,
                  use_dat_ex: bool = False,
                  verbose: bool = False) -> str:
    """Checks input files are pcd format or not. If one of the input file is not 
    pcd format, exit the script. If use_dat_ex is True, the dat_list will be  
    overwritten by ICP dataset.

    Parameters
    ----------
    pipe: str
        The name of the pipeline. regi (registration) or proc_3d (3d process).
    dat_list : str
        The list of input file paths.
    use_dat_ex : bool
        Boolean statement for using example ICP dataset (default is False).
    verbose : bool
        Boolean statement to control the print (default is False).

    Returns
    -------
    dat_list: str
        The modified list of input paths after sanity check.
    dat_key : str
        The keyward to discribe the data.
    """

    # Check example option and replace data path if it is true.
    if use_dat_ex:
        dat_list = o3d.data.DemoICPPointClouds().paths
        dat_key = 'icp'
    else:
        # Saparates by comma
        dat_list = dat_list.split(',') 
        dat_key_list = []        

        # First, check whether string has same end with '.pcd'.
        pos_formats = ('.pcd', '.ply')
        for dat in dat_list:
            if not dat.endswith(pos_formats):
                print('Something is wrong in the input file!')
                sys.exit(1)   

            # Use file names for the data keyward.
            file_name = os.path.splitext(os.path.basename(dat))[0]
            dat_key_list.append(file_name)

        dat_key = '_'.join(dat_key_list)
        
    if verbose:
        for idx, dat in enumerate(dat_list):
            print(f'Data path #{idx}: {dat}')
        print(f'Data keyword: {dat_key}')

    return dat_list, dat_key
    

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
            pcd_indi = o3d.io.read_point_cloud(indi)
            self.pcd_list.append(pcd_indi)
            if self.verbose:
                print(f'Input file #{idx}: {indi}')
            
            # If file is empty exit the script
            if pcd_indi.is_empty():
                print('Something is wrong in the input file!')
                sys.exit(1)
    
    def get_pts(self, pcd_dat_idx: int = 0, use_np: bool = False):
        """Extract the informations from pcd file and wrap with NumPy array

        Parameters
        ----------
        pcd_dat_idx : int
            The index for pcd data list (default is 0).
        use_np : bool
            Boolean statement to wrap the point information with 
            NumPy array (default is False)
        """        
        
        self.pts = self.pcd_list[pcd_dat_idx].points
        self.col = self.pcd_list[pcd_dat_idx].colors
        self.nor = self.pcd_list[pcd_dat_idx].normals
        if use_np:
            # Since each property is tuple, use np.asarray() to access data.
            self.pts = np.asarray(self.pts, dtype=float)
            self.col = np.asarray(self.col, dtype=float)
            self.nor = np.asarray(self.nor, dtype=float)
