import numpy as np
import open3d as o3d
from tqdm import tqdm

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
        if self.verbose:
            for idx, pcds in enumerate(self.pcd_list):
                print(f'Input pcd file #{idx}: {pcds}')
        self.pipe_regi = o3d.pipelines.registration
        self.reg_ran_trans = np.full((4, 4), np.nan, dtype=float)

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
            self.pcd_indi_down, params_fpfh)
        del params_fpfh

        return pcd_fpfh

    def get_voxel_from_grid(self, pcd_idx, num_bins):

        bbox = self.pcd_list[pcd_idx].get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        voxel_size = np.nanmean(bbox_size) / num_bins
        
        if self.use_debug:
            self.box_points = bbox.get_box_points()
            self.box_min_max_bound = np.array([bbox.get_min_bound(), 
                                               bbox.get_max_bound()])
        #pcd_np = np.asarray(self.pcd_list[pcd_idx].points, dtype = float)
        #voxel_size = np.nanmean(np.nanmax(pcd_np, axis=0) \
        #             - np.nanmin(pcd_np, axis=0)) / num_bins
        del bbox, bbox_size

        return voxel_size

    def get_pre_process(
            self, 
            src_idx: int,
            trans_init = np.full((4, 4), np.nan, dtype=float),
            voxel_size: float = np.nan,
            num_bins: float = 64,
            rad_down: float = np.nan,
            rad_fpfh: float = np.nan,
            rad_down_fac: float = 2,
            rad_fpfh_fac: float = 5,
            max_nn_down: int = 30,
            max_nn_fpfh: int = 100):

        is_nan_init = np.isnan(np.sum(trans_init))
        is_nan_voxel = np.isnan(voxel_size)
        is_nan_rad_down = np.isnan(rad_down)       
        is_nan_rad_fpfh = np.isnan(rad_fpfh)       
        if self.verbose:
            print(f'Source file index is {src_idx}') 
            if is_nan_init:
                print(f'There is no initial guess for the transformation.')
            else:
                print(f'Trans init array: \n{trans_init}')

        self.pcd_down = []
        self.pcd_fpfh = []
        self.voxels = np.full((self.pcd_list_len), np.nan, dtype=float)
        if self.use_debug:
            self.src_idx = src_idx
            self.trans_init = trans_init
            self.rads = np.full((self.pcd_list_len, 2), np.nan, dtype=float)
            self.max_nns = np.copy(self.rads)
            self.box_pts = np.full((self.pcd_list_len, 8, 3), np.nan, \
                                    dtype=float)
            self.box_min_max = np.full((self.pcd_list_len, 2, 3), np.nan, \
                                        dtype=float)
        for idx in tqdm(range(self.pcd_list_len), disable = ~self.use_debug):
            if idx == src_idx and not is_nan_init:
                self.pcd_list[idx].transform(trans_init)

            if is_nan_voxel:
                voxel_size = self.get_voxel_from_grid(idx, num_bins)
                if self.use_debug:
                    self.box_pts[idx] = self.box_points
                    self.box_min_max[idx] = self.box_min_max_bound       
            self.voxels[idx] = voxel_size
 
            if is_nan_rad_down:
                rad_down = voxel_size * rad_down_fac
            self.pcd_indi_down = self.get_down_samp(idx, voxel_size, 
                                                    rad_down, max_nn_down) 
            self.pcd_down.append(self.pcd_indi_down)          
 
            if is_nan_rad_fpfh:
                rad_fpfh = voxel_size * rad_fpfh_fac 
            pcd_indi_fpfh = self.get_fpfh(rad_fpfh, max_nn_fpfh)
            self.pcd_fpfh.append(pcd_indi_fpfh)
        
            if self.verbose:
                print(f'File #{idx} preporc summary')
                print(f'  Voxel size: {np.round(voxel_size, 2)}')           
                print(f'  Radius for down sampling: {np.round(rad_down, 2)}') 
                print(f'  Radius for fpfh: {np.round(rad_fpfh, 2)}')
                print(f'  Down sampled {self.pcd_indi_down}')
                print(f'  FPFH {pcd_indi_fpfh}')
    
            if self.use_debug:
                self.rads[idx, 0] = rad_down
                self.rads[idx, 1] = rad_fpfh
                self.max_nns[idx, 0] = max_nn_down
                self.max_nns[idx, 1] = max_nn_fpfh
            del voxel_size, rad_down, rad_fpfh 
        self.voxel_avg = np.nanmean(self.voxels)

    def get_ransac_regi(self, 
                        src_idx: int,
                        tar_idx: int,
                        dis_thres: float = np.nan,
                        dis_thres_fac: float = 1.5,
                        mutual_filt: bool = True,
                        ratio: float = 0.9,
                        max_iter: int = 100000,
                        confi: float = 0.999,
                        scaling: bool = False):

        if np.isnan(dis_thres):
            dis_thres = self.voxel_avg * dis_thres_fac

        pt_to_pt = self.pipe_regi.TransformationEstimationPointToPoint(scaling)
        edge_len = self.pipe_regi.CorrespondenceCheckerBasedOnEdgeLength(ratio)
        dis = self.pipe_regi.CorrespondenceCheckerBasedOnDistance(dis_thres)
        criteria = self.pipe_regi.RANSACConvergenceCriteria(max_iter, confi)

        reg_ran = self.pipe_regi.registration_ransac_based_on_feature_matching(
            self.pcd_down[src_idx], self.pcd_down[tar_idx],
            self.pcd_fpfh[src_idx], self.pcd_fpfh[tar_idx],
            mutual_filt, dis_thres, 
            pt_to_pt, 3, [edge_len, dis], criteria)
        self.reg_ran_trans = reg_ran.transformation

        return reg_ran

    def get_icp_regi(self,
                     src_idx: int,
                     tar_idx: int,
                     dis_thres: float = np.nan,
                     dis_thres_fac: float = 0.4,
                     max_iter: int = 2000,
                     trans_init = np.full((4, 4), np.nan, dtype=float),
                     use_p2p: bool = False):

        if np.isnan(dis_thres):
            dis_thres = self.voxel_avg * dis_thres_fac
       
        if not np.isnan(np.sum(trans_init)):
            pass
        elif not np.isnan(np.sum(self.reg_ran_trans)):
            trans_init = self.reg_ran_trans
        else:
            trans_init = np.identity(4, dtype=float)
      
        if use_p2p:
            tans_est = self.pipe_regi.TransformationEstimationPointToPoint()
        else:  
            tans_est = self.pipe_regi.TransformationEstimationPointToPlane()
        criteria = self.pipe_regi.ICPConvergenceCriteria(max_iteration=max_iter)
        
        reg_icp = self.pipe_regi.registration_icp(
            self.pcd_list[src_idx], self.pcd_list[tar_idx], 
            dis_thres, trans_init,
            tans_est, criteria)

        return reg_icp 
















