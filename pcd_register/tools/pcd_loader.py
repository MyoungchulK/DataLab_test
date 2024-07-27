"""Pcd data loader

This script is designed to load the pcd file and provide the points, colors, 
and normals in a NumPy array format.

    * is_pcd_format - Checks input file is pcd format or not
    * pcd_loader - Class that loading/reading a pcd file
"""

import os
import sys
import numpy as np
import open3d as o3d

def is_pcd_format(data: str) -> bool:
    """Checks input file is pcd format or not.
    This function is made for also using it on main.py

    Parameters
    ----------
    data : str
        The input file path

    Returns
    -------
    bool
        a boolean statement that end state is '.pcd' or not
    """

    # Check whether string has same end with '.pcd'
    is_pcd = data.endswith('.pcd')
 
    return is_pcd

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
    get_pts_in_np()
        Extract the informations from pcd file and wrap with NumPy array
    """

    def __init__(self, data: str, verbose: bool = False):
        """Loads the pcd files. If input file is not pcd format, Use pre-built
        dataset. If there is noting in the input file, exit the script.

        Parameters
        ----------
        data : str
            The input file path
        verbose : bool
            Boolean statement to control the print (default is False)
        """

        self.verbose = verbose
        if self.verbose:
            print(f'Input file: {data}')

        # If the input path is not pcd format, use pre-built dataset.
        # This should not be used in real case. Just for this test.
        if not is_pcd_format(data):
            data = o3d.data.EaglePointCloud().path
            if self.verbose:
                print('Input file is not pcd format! Use EaglePointCloud()')   

        # Load the 3d data 
        self.pcd_dat = o3d.io.read_point_cloud(data)
        if len(self.pcd_dat.points) == 0:
            if self.verbose:
                print('Something is wrong in the input file!')
            sys.exit(1)
    
    def get_pts_in_np(self):
        """Extract the informations from pcd file and wrap with NumPy array"""

        self.pts = np.asarray(self.pcd_dat.points, dtype = float)
        self.col = np.asarray(self.pcd_dat.colors, dtype = float)
        self.nor = np.asarray(self.pcd_dat.normals, dtype = float)
