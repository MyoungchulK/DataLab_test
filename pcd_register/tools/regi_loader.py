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

    def get_voxel_from_bbox(self, pcd, num_bins: float = 64, use_numpy: bool = False):

        # If output of the function is not bounded by open3d format and calculation is simple, We could just use NumPy. 
        # NumPy uses the C for the for loop, It would outcome same speed.
        # It might be more beneficial if we have to do this calculation multiple time.
        if use_numpy:
            pcd_np = np.asarray(pcd.points, dtype = float)
            voxel_size = np.nanmean(np.nanmax(pcd_np, axis=0) - np.nanmin(pcd_np, axis=0)) / num_bins
            del pcd_np
        else
            # Get the position of the point that touchs the bbox
            bbox = pcd.get_axis_aligned_bounding_box()

            # Get the length of the bbox
            bbox_size = bbox.get_extent()

            # Voxel size by dividing the bbox
            voxel_size = np.nanmean(bbox_size) / num_bins
            del bbox, bbox_size

        return voxel_size

    def get_down_sample(self, user_voxel_size = np.nan):

        if np.isnan(user_voxel_size):
            src_vs = self.get_voxel_from_bbox(self.pcd_src)
            tar_vs = self.get_voxel_from_bbox(self.pcd_tar)
        else:
            src_vs = user_voxel_size
            tar_vs = user_voxel_size

        pcd_src_down = self.pcd_src.voxel_down_sample(src_vs)
        pcd_tar_down = self.pcd_tar.voxel_down_sample(tar_vs)








