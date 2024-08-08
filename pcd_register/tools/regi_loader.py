import numpy as np
import open3d as o3d

class regi_loader:

    def __init__(self, 
                 pcd_list,
                 verbose: bool = False, 
                 use_debug: bool = False): 

        self.verbose = verbose
        self.use_debug = use_debug

        # make self for the two pcd files
        self.pcd_list = pcd_list
        self.pcd_list_len = len(self.pcd_list)

    def get_KDTree_params(self, radius, max_nn):
    
        return o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)

    def get_down_samp(self, pcd_idx, voxel_size, radius_down, max_nn_down):
        
        params_down = self.get_KDTree_params(radius_down, max_nn_down) 
        
        pcd_down = self.pcd_list[pcd_idx].voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(params_down)
        del params_down

        return pcd_down

    def get_fpfh(self, radius_fpfh, max_nn_fpfh):

        params_fpfh = self.get_KDTree_params(radius_fpfh, max_nn_fpfh)

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd_indi_down, params_fpfn)
        del params_fpfh

        return pcd_fpfh

    def get_voxel_from_grid(self, pcd_idx, num_bins):

        bbox = self.pcd_list[pcd_idx].get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        voxel_size = np.nanmean(bbox_size) / num_bins
        #pcd_np = np.asarray(self.pcd_list[pcd_idx].points, dtype = float)
        #voxel_size = np.nanmean(np.nanmax(pcd_np, axis=0) \
        #             - np.nanmin(pcd_np, axis=0)) / num_bins
        del bbox, bbox_size

        return voxel_size

    def get_pre_process(
            self, 
            src_idx: int,
            trans_init = np.full((4, 4), np.nan, dtype = float),
            voxel_size: float = np.nan,
            num_bins: float = 64):

        is_nan_init = np.any(np.isnan(trans_init))
        is_nan_voxel = np.isnan(voxel_size)        

        self.pcd_down = []
        self.pcd_fpfh = []
        for idx in tqdm(range(self.pcd_list_len), 
                              disable = ~self.use_debug):
            if idx == src_idx and not is_nan_init:
                self.pcd_list[pcd_idx].transform(trans_init)

            if is_nan_voxel:
                voxel_size = self.get_voxel_from_grid(idx, num_bins)
        
            radius_down = voxel_size * 2
            self.pcd_indi_down = self.get_down_samp(idx, voxel_size, 
                                                    radius_down, 30) 
            self.pcd_down.append(self.pcd_indi_down)          
 
            radius_fpfh = voxel_size * 5
            pcd_indi_fpfh = self.get_fpfh(radius_fpfh, 100)
            self.pcd_fpfh.append(pcd_indi_fpfh)
            







