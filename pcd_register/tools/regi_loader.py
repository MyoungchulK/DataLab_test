import numpy as np
import open3d as o3d

class regi_loader:

    def __init__(self, 
                 pcd_src,
                 pod_tar,
                 verbose: bool = False, 
                 use_debug: bool = False): 

        self.verbose = verbose
        self.use_debug = use_debug

        # make self for the two pcd files
        self.pcd_src = pcd_src
        self.pcd_tar = pcd_tar

