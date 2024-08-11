import os
import sys
import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path + '/../')
from tools.utility import size_checker

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
                # Check input file information.
                print(f'Input pcd file #{idx}: {pcds}')

        # Make instance for the o3d.pipelines.registration library
        # We could just load this by the import command at the begining.
        self.pipe_regi = o3d.pipelines.registration

        # Place holder for the transformation maxtrix of ransac registration.
        # It will be used for icp registration.
        self.reg_ran_trans = np.full((4, 4), np.nan, dtype=float)

    def get_KDTree_params(
            self, 
            radius: float, 
            max_nn: int
            ) -> o3d.cpu.pybind.geometry.KDTreeSearchParamHybrid:
        """KDTree search parameters for hybrid KNN and radius search.
        It is bit redundant function since it just wrap the
        KDTreeSearchParamHybrid function.       

        Parameters
        ----------
        radius : float
            It will look for the bin within radius. 
        max_nn : int
            The distance for searching the neighboring bin.
 
        Returns
        -------
        kdt_params : o3d.cpu.pybind.geometry.KDTreeSearchParamHybrid
            Contains paramaters in the open3d format.
        """   

        # Set the paramter based on the inputs.
        kdt_params = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn)

        return kdt_params

    def get_down_samp(
            self, 
            pcd_idx: int, 
            voxel_size: float, 
            radius_down: float, 
            max_nn_down: int
            ) -> o3d.geometry.PointCloud:
        """Function to downsample input pointcloud into output pointcloud with 
        a voxel. Normals and colors are averaged if they exist.

        Parameters
        ----------
        pcd_idx : int
            The index of the pcd file list. It will select the file from the list
            based on the index.
        voxel_size : float
            Voxel size to downsample into.
        radius_down : float
            The radius limitation for the KDTree function.
        max_nn_down : int
            The neighboring bin limitation for the KDTree function.

        Returns
        -------
        pcd_down : o3d.geometry.PointCloud
            Contain the information of the 3d points in the open3d format.
        """       

        # Wrap the parameter in the open3d format. 
        params_down = self.get_KDTree_params(radius_down, max_nn_down) 
 
        # Down sample based on the pcd file and vexel size.
        pcd_down = self.pcd_list[pcd_idx].voxel_down_sample(voxel_size)

        # Function to compute the normals of a point cloud. 
        # Normals are oriented with respect to the input point cloud if normals 
        # exist.        
        pcd_down.estimate_normals(params_down)
        del params_down

        return pcd_down

    def get_fpfh(
            self, 
            radius_fpfh: float, 
            max_nn_fpfh: int
            ) -> o3d.pipelines.registration.Feature:
        """Function to compute FPFH feature for a point cloud.

        Parameters
        ----------
        radius_fpfh : float
            The radius limitation for the KDTree function.
        max_nn_fpfh: int
            The neighboring bin limitation for the KDTree function.

        Returns 
        -------
        pcd_fpfh : o3d.pipelines.registration.Feature
            Contains 33 dimension of FPFH feature in the open3d format.
        """

        # Wrap the parameter in the open3d format.
        params_fpfh = self.get_KDTree_params(radius_fpfh, max_nn_fpfh)

        # Calculates the FPFH feature based on the input parameters.
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            self.pcd_indi_down, params_fpfh)
        del params_fpfh

        return pcd_fpfh

    def get_voxel_from_grid(self, pcd_idx: int, num_bins: float) -> float:
        """Calculates voxel size based on the bounding box size and the input 
        binning. It will be only activated if user didn't set the voxel size.

        Parameters
        ----------
        pcd_idx : int
            The index of the pcd file list. It will select the file from the list
            based on the index.
        num_bins : float
            The number of bins that user want to divide the bounding box.

        Returns
        -------
        voxel_size : float
            Returns the voxel size that calculated based on the bounding box grid
            and the input binning.
        """

        # Selects the pcd file and calculate edge points that touching the 
        # bound box.
        bbox = self.pcd_list[pcd_idx].get_axis_aligned_bounding_box()

        # Calculates the size of the bounding box
        bbox_size = bbox.get_extent()

        # Calculates the voxel size by averaging the size of the bins.
        voxel_size = np.nanmean(bbox_size) / num_bins
       
        # Stores the box inforamtion if user want to see the middle step. 
        if self.use_debug:
            self.box_points = bbox.get_box_points() # The box edge points
            # The edge points that touching the bound box.
            self.box_min_max_bound = np.array([bbox.get_min_bound(), 
                                               bbox.get_max_bound()])
        del bbox, bbox_size

        # Since the calcultion is simple, we could just use NumPy to do the job. 
        """
        pcd_np = np.asarray(self.pcd_list[pcd_idx].points, dtype = float)
        voxel_size = np.nanmean(np.nanmax(pcd_np, axis=0) \
                     - np.nanmin(pcd_np, axis=0)) / num_bins
        """

        return voxel_size

    def get_pre_process(
            self, 
            src_idx: int,
            tar_idx: int,
            trans_init: np.ndarray = np.full((4, 4), np.nan, dtype=float),
            voxel_size: float = np.nan,
            num_bins: float = 64,
            rad_down: float = np.nan,
            rad_fpfh: float = np.nan,
            rad_down_fac: float = 2,
            rad_fpfh_fac: float = 5,
            max_nn_down: int = 30,
            max_nn_fpfh: int = 100):
        """Performs the preprocess by calling down sample and fpfh feature 
        functions. It will call the data that stored in the pcd_list. By doing
        this, we will not do the unecessary data copy that can happen during the
        function execution. Since preprocessing is almost identical between the
        source and target file, it is done by the for loop. The only differnce is
        when we process the soruce file, we will apply the initial transformation
        guess. The calculation step is 1) get voxel size, 2) perform down sample,
        and 3) obtain fpfh feature.

        Parameters
        ----------
        src_idx : int        
            The index of the source pcd file from the list. It will select the 
            file from the list based on the index.
        tar_idx : int
            The index of the target pcd file from the list. It will select the
            file from the list based on the index.
        trans_init : np.ndarray
            The initial transformation guess (The default is the nan array).
        voxel_size : float
            The input voxel size by the user (The default is nan). 
        num_bins : float
            The number of bins that user want to divide the bounding box. It will
            be only used if user didn't set the voxel size.
        rad_down : float
            The radius limitation for the KDTree function in the down sampling
            function (The default is nan).
        rad_fpfh : float
            The radius limitation for the KDTree function in the fpfh feature
            function (The default is nan).
        rad_down_fac : flaot
            The radius factor for the down sampling. If user didn't specify the
            radius for the down sampling, It will be calculated by product of
            the radius factor and voxel size (The default is 2).
        rad_fpfh_fac : float
            The radius factor for the fpfh feature. If user didn't specify the 
            radius for fpfh feature, It will be calculated by product of the
            radius factor and voxel size (The default is 5).
        max_nn_down : int
            The neighboring bin limitation for the KDTree function in the down
            sampling function (The default is 30).
        max_nn_fpfh : int
            The neighboring bin limitation for the KDTree function in the fpfh
            feature function (The default is 100).
        """

        # Create the bool statements for choosing conditions.
        # I used NumPy sum for checking whether the array is including the nan.
        # I could use np.any(np.isnan(trans_init)) for the process. But based on
        # the wisdom of the internet, applying np.isnan to all element in the 
        # array is way slower than just sum and then apply np.isnan once. If 
        # array has nan, simple np.sum (not np.nansum) will be consumed by the 
        # even single nan. 
        is_nan_init = np.isnan(np.sum(trans_init))
        is_nan_voxel = np.isnan(voxel_size)
        is_nan_rad_down = np.isnan(rad_down)       
        is_nan_rad_fpfh = np.isnan(rad_fpfh)      

        if self.verbose:
            print(f'Source file index is {src_idx}') 
            # Checks the transformation is set by user or not. 
            if is_nan_init:
                print(f'There is no initial guess for the transformation.')
            else:
                print(f'Trans init array: \n{trans_init}')

        # Begining of the preprocessing. Each voxel size, down sampling, and 
        # fpfh feature calculations will be applied to pcd data one by one.
        # The lists to store the results
        self.pcd_down = []
        self.pcd_fpfh = []

        # Voxel size is stored per pcd file in case user didn't set the vexel
        # size. If it didn't, it will have a different size per pcd file.
        self.voxels = np.full((self.pcd_list_len), np.nan, dtype=float)

        # Stores the secondary information If user want to see what is happening
        # during the calculation.
        if self.use_debug:
            self.src_idx = src_idx # The index of the source pcd file.
            self.trans_init = trans_init # The initial guess for transformation.
            self.rads = np.full((self.pcd_list_len, 2), np.nan, dtype=float)
            self.max_nns = np.copy(self.rads)
            self.box_pts = np.full((self.pcd_list_len, 8, 3), np.nan, \
                                    dtype=float)
            self.box_min_max = np.full((self.pcd_list_len, 2, 3), np.nan, \
                                        dtype=float)
        
        # The tqdm is for user to check the process. In this assignment, we only
        # use two file. But If we do process for multiple files, It will be good
        # to see the progress at the terminal.
        for idx in tqdm(range(self.pcd_list_len), disable = ~self.use_debug):
            
            # Applys the transformation with initial guess to only source file by
            # idx == src_idx condition if user specify it.
            if idx == src_idx and not is_nan_init:
                self.pcd_list[idx].transform(trans_init)

            # Calcualtes voxel size.
            if is_nan_voxel:
                voxel_size = self.get_voxel_from_grid(idx, num_bins)
                if self.use_debug:
                    self.box_pts[idx] = self.box_points
                    self.box_min_max[idx] = self.box_min_max_bound       
            self.voxels[idx] = voxel_size # stores the voxel size.

            # Performs the down sampling based on the radius and max_nn 
            # parameter. 
            if is_nan_rad_down:
                rad_down = voxel_size * rad_down_fac
            self.pcd_indi_down = self.get_down_samp(idx, voxel_size, 
                                                    rad_down, max_nn_down) 
            self.pcd_down.append(self.pcd_indi_down)          

            # Performs the fpfh feature based on the radius and max_nn
            # parameter. 
            if is_nan_rad_fpfh:
                rad_fpfh = voxel_size * rad_fpfh_fac 
            pcd_indi_fpfh = self.get_fpfh(rad_fpfh, max_nn_fpfh)
            self.pcd_fpfh.append(pcd_indi_fpfh)
       
            # print quick summary if user want to see it. 
            if self.verbose:
                print(f'File #{idx} preporc summary')
                print(f'  Voxel size: {np.round(voxel_size, 2)}')           
                print(f'  Radius for down sampling: {np.round(rad_down, 2)}') 
                print(f'  Radius for fpfh: {np.round(rad_fpfh, 2)}')
                print(f'  Down sampled {self.pcd_indi_down}')
                print(f'  FPFH {pcd_indi_fpfh}')
   
            # Stores the parameters if user want to check the middle step. 
            if self.use_debug:
                self.rads[idx, 0] = rad_down
                self.rads[idx, 1] = rad_fpfh
                self.max_nns[idx, 0] = max_nn_down
                self.max_nns[idx, 1] = max_nn_fpfh
            del voxel_size, rad_down, rad_fpfh

        # Calculates the average voxel size for both registration process.
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
        print(pt_to_pt)
        print(edge_len)
        print(dis)
        print(criteria)

        reg_ran = self.pipe_regi.registration_ransac_based_on_feature_matching(
            self.pcd_down[src_idx], self.pcd_down[tar_idx],
            self.pcd_fpfh[src_idx], self.pcd_fpfh[tar_idx],
            mutual_filt, dis_thres, 
            pt_to_pt, 3, [edge_len, dis], criteria)
        self.reg_ran_trans = reg_ran.transformation
        print(reg_ran)

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
        print(tans_est)
        print(criteria)        

        reg_icp = self.pipe_regi.registration_icp(
            self.pcd_list[src_idx], self.pcd_list[tar_idx], 
            dis_thres, trans_init,
            tans_est, criteria)
        print(reg_icp)

        return reg_icp 

def draw_regi_result(
        src_pcd: o3d.geometry.PointCloud,
        tar_pcd: o3d.geometry.PointCloud,
        output: str,
        trans: np.ndarray = np.identity(4, dtype=float),
        width: int = 1920,
        height: int = 1080,
        src_color: list = [1, 0.706, 0],
        tar_color: list = [0, 0.651, 0.929],
        zoom: float = 0.4459,
        front: list = [0.6452, -0.3036, -0.7011],
        lookat: list = [1.9892, 2.0208, 1.8945],
        up: list = [-0.2779, -0.9482, 0.1556],
        verbose: bool = False):
    """Draw the 3d results with transformation. Saves the drawing in the png 
    format. The o3d.visualization.draw_geometries class has a no ability to the 
    save the plot. So, I used o3d.visualization.Visualizer class for saving the
    results into the plot.

    Parameters
    ----------
    src_pcd : o3d.geometry.PointCloud
        The source point cloud. 
    tar_pcd : o3d.geometry.PointCloud
        The target point cloud.
    output : str
        The output path for saving the drawing results.
    trans : np.ndarray
        The transformation maxtrix from the registration results (Default is 4x4
        2d array).
    width : int
        The width of the canvas (Default is 1920).
    height : int
        The height of the canvas (Default is 1080). 
    src_color : list
        The color setting of the source point cloud. (Default is [1, 0.706, 0]).
    tar_color : list
        The color setting of the source point cloud. (Default is 
        [0, 0.651, 0.929]).
    zoom : float
        The zoom of the camera (Default is 0.4459).
    front : list
        The front vector of the camera (Default is [0.6452, -0.3036, -0.7011]).
    lookat : list
        The lookat vector of the camera (Default is [1.9892, 2.0208, 1.8945]).
    up : list
        The up vector of the camera (Default is [-0.2779, -0.9482, 0.1556]).
    verbose : bool
        Boolean statement to control the print (Default is False).
    """

    # deep copy the pcd file so that color and transformation changes are not
    # affected to actual data.
    src_temp = copy.deepcopy(src_pcd)
    tar_temp = copy.deepcopy(tar_pcd)
    src_temp.paint_uniform_color(src_color) # reset the color of the points.
    tar_temp.paint_uniform_color(tar_color)
    src_temp.transform(trans) # transforms the points based on the result.

    # Construct the Visualizer class.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
        
    # Add the points as a geometry.
    vis.add_geometry(src_temp)
    vis.add_geometry(tar_temp)
        
    # Set the point of the view.
    view_control = vis.get_view_control()
    view_control.set_zoom(zoom)
    view_control.set_front(front)
    view_control.set_lookat(lookat)
    view_control.set_up(up)

    # Update the visualization.
    vis.poll_events()
    vis.update_renderer()
        
    # Saves the image in the png format.
    vis.capture_screen_image(output)
        
    # Remove the drawing for the sake of the memory.
    vis.destroy_window()

    # Print the message for saving the plot.
    if verbose:
        print(f'Output path: {output}. {size_checker(output)}')






