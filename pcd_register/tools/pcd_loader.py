import os, sys
import numpy as np
import open3d as o3d

def is_pcd_format(data):

    pcd_format = os.path.splitext(data)[1]
 
    return pcd_format == '.pcd'

class pcd_loader:

    def __init__(self, data, verbose = False):

        self.verbose = verbose

        if ~is_pcd_format(data):
            data = o3d.data.EaglePointCloud().path
            if self.verbose:
                print('Input file is not pcd format! Use EaglePointCloud()')   
 
        self.pcd_dat = o3d.io.read_point_cloud(data)
        if len(self.pcd_dat.points) == 0:
            if self.verbose:
                print('Something is wrong in the input file!')
            sys.exit(1)

    def get_pts_in_np(self):

        self.pts = np.asarray(self.pcd_dat.points)
        self.col = np.asarray(self.pcd_dat.colors)
        self.nor = np.asarray(self.pcd_dat.normals)
