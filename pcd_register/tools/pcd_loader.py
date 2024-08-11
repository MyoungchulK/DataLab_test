"""Pcd data loader

This script is designed to load the pcd file and provide the points, colors, 
and normals in a NumPy array format.

    * get_data_info - Checks input file is pcd format or not.
    * save_pcd_info - Saves the point cloud information into the pcd format.
    * pcd_loader - Class that loading/reading a pcd file.
"""

import os
import sys
import json
import numpy as np
import open3d as o3d

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.utility import get_tools_abspath, size_checker

def get_data_info(dat_var : str) -> dict:
    """Extracts the variables in the text file and store in the dictionary.
    Checks input files are pcd format or not. If one of the input file is not
    pcd format, exit the script. If use_dat_ex is True, the dat_list will be  
    overwritten by ICP dataset.

    Parameters
    ----------
    dat_var : str
        The text file path that contains all the variables for the pipeline 
        process (default is '').

    Returns
    -------
    dat_dict : dict
        The variables for the pipeline process.
    """
    
    # If dat_var is empty, use icp examples in the examples path.
    if len(dat_var) == 0:
        code_path = get_tools_abspath()
        file_path = f'../../examples/icp_var_ex.txt'
        dat_var = os.path.join(code_path, file_path) 

    # Opens the text file and stores in the dictionary.
    with open(dat_var, 'r') as f:
        data = f.read()
        dict_idx = data.find('"""')
    dat_dict = json.loads(data[:dict_idx])

    # Check example option and replace data path if it is true.
    if dat_dict['use_dat_ex']:
        dat_dict['dat_list'] = o3d.data.DemoICPPointClouds().paths
    else:
        # First, check whether string has same end with '.pcd'.
        pos_formats = ('.pcd', '.ply')
        for dat in dat_dict['dat_list']:
            if not dat.endswith(pos_formats):
                print('Something is wrong in the input file!')
                sys.exit(1)   

    # Check the variables in the dictionary.
    if dat_dict['verbose']:
        print(f'Data variable path: {dat_var}')
        print(json.dumps(dat_dict, indent=4))

    return dat_dict
    
def save_pcd_info(pcd: o3d.geometry.PointCloud, 
                  output: str, 
                  verbose: bool = False):
    """Saves the point cloud information into the pcd format.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        The point cloud information that need to be saved in the pcd format.
    output : str
        The output path for saving the drawing results.
    verbose : bool
        Boolean statement to control the print (Default is False).
    """

    # Saves the pcd infomation.
    o3d.io.write_point_cloud(output, pcd, compressed=True, 
                             print_progress=verbose)

    # Print the message
    if verbose:
        print(f'Output path: {output}. {size_checker(output)}')

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
