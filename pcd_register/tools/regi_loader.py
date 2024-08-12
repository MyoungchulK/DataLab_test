"""registration loader

This script is designed to perform the necessary calculation for the registration
process. It will accpet the pcd files and perform the preprocessing, RANSAC, and
ICP registration. The preprocessing is consist of down sampleing by voxeling and 
extracting feature by FPFH. The class in this script will be called by the script
in the wrappers path and perform the calculation.

    * regi_loader - Class that contains the preprocessing and registration.
    * draw_regi_result - Draw/save the registration results in the png format.
"""

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
    """Designed to perform the preprocessing, RANSAC, and ICP registration.

    ...

    Attributes
    ----------
    verbose : bool
        Boolean statement to control the print (default is False)
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging (default is False)
    pcd_list : list
        The list of the point could data saved in the pcd formats. It is
        saved in the list to be flexible for number of files. But for this
        test, only two pcd files will be stored in the list.
    pcd_list_len : int
        The number of pcd files in the pcd_list  
    pipe_regi : o3d.pipelines.registration
        The instance for the o3d.pipelines.registration library.
        We could just load this by the import command at the begining.
    reg_ran_trans : np.ndarray
        The transformation maxtrix of ransac registration.
        It will be used for icp registration.
    box_points : np.ndarray
        The edge points of the bounding box (available if `use_debug` is True).
    box_min_max_bound : np.adarray
        The edge points that touching the bound box
        (available if `use_debug` is True).
    pcd_down : list
        The down sampled source and target point cloud data in the list.
    pcd_fpfh : list
        The fpfh feature of source and target point cloud data in the list.
    voxels : np.ndarray
        The voxel size per pcd file.
    src_idx : int
        The index of the source pcd file (available if `use_debug` is True).
    trans_init : np.ndarray
        The initial guess for transformation (available if `use_debug` is True).
    rads : np.ndarray
        The radius paramters for preprocessing 
        (available if `use_debug` is True). 
    max_nns : np.ndarray
        The neighboring bin limitation for the KDTree function in preprocessing
        (available if `use_debug` is True).
    box_pts : np.ndarray
        The edge points of the bounding box for all pcd files
        (available if `use_debug` is True).
    box_min_max : np.ndarray
        The edge points that touching the bound box for all pcd files
        (available if `use_debug` is True).
    pcd_indi_down : o3d.geometry.PointCloud 
        The down sample pcd for the fpfh process.
    voxel_avg : float
        The average voxel size for both registration process. 
    reg_ran_fit : float
        The # of inlier correspondences / # of points in source for the RANSAC 
        registration (available if `use_debug` is True).
    reg_ran_rmse : float
        RMSE of all inlier correspondences for the RANSAC registration
        (available if `use_debug` is True).
    reg_ran_corr : np.adarray
        Correspondence set between source and target point cloud for the RANSAC
        registration (available if `use_debug` is True).
    reg_icp_trans : np.adarray
        Transformation matrix for the ICP registration 
        (available if `use_debug` is True).
    reg_icp_fit : float
        The # of inlier correspondences / # of points in source for the ICP 
        registration (available if `use_debug` is True).
    reg_icp_rmse : float
        RMSE of all inlier correspondences for the ICP registration
        (available if `use_debug` is True).
    reg_icp_corr : np.adarray
        Correspondence set between source and target point cloud for the ICP
        registration (available if `use_debug` is True).

    Methods
    -------
    get_KDTree_params(radius: float, max_nn: int
                     ) -> o3d.cpu.pybind.geometry.KDTreeSearchParamHybrid
        KDTree search parameters for hybrid KNN and radius search
    get_down_samp(pcd_idx: int, voxel_size: float, radius_down: float, 
                  max_nn_down: int) -> o3d.geometry.PointCloud
        Down samples input pointcloud into output pointcloud with a voxel.      
    get_fpfh(radius_fpfh: float, max_nn_fpfh: int
            ) -> o3d.pipelines.registration.Feature
        Function to compute FPFH feature for a point cloud.
    get_voxel_from_grid(pcd_idx: int, num_bins: float) -> float
        Calculates voxel size based on the bounding box size and input binning.
    get_pre_process(
        src_idx: int, tar_idx: int, 
        trans_init: np.ndarray = np.full((4, 4), np.nan, dtype=float),
        voxel_size: float = np.nan, num_bins: float = 64, 
        rad_down: float = np.nan, rad_fpfh: float = np.nan,
        rad_down_fac: float = 2, rad_fpfh_fac: float = 5, 
        max_nn_down: int = 30, max_nn_fpfh: int = 100)
        Performs the preprocess by calling down sample and fpfh feature.
    get_ransac_regi(
        src_idx: int, tar_idx: int, dis_thres: float = np.nan, 
        dis_thres_fac: float = 1.5, mutual_filt: bool = True, ratio: float = 0.9,
        max_iter: int = 100000, confi: float = 0.999, scaling: bool = False
        ) -> o3d.pipelines.registration.RegistrationResult
        Performs RANSAC registration. This registration is based on the down 
        sampled dataset and fpfh feature that calculated from preprocessing step.
    get_icp_regi(
        src_idx: int, tar_idx: int, dis_thres: float = np.nan,
        dis_thres_fac: float = 0.4, max_iter: int = 2000, 
        trans_init = np.full((4, 4), np.nan, dtype=float),
        use_p2p: bool = False
        ) -> o3d.pipelines.registration.RegistrationResult
        Performs ICP registration. This registration is based on the 
        transformation matrix that obtained from the RANSAC registration.
    """

    def __init__(self, 
                 pcd_list: list,
                 verbose: bool = False, 
                 use_debug: bool = False): 
        """Initializer for the registration class.

        Parameters
        ----------
        pcd_list : list
            The list of the point could data saved in the pcd formats. It is
            saved in the list to be flexible for number of files. But for this 
            test, only two pcd files will be stored in the list.
        verbose : bool
            Boolean statement to control the print (default is False)
        use_debug : bool
            By changing its to True, use can check and svae the all middle step
            of the calculation. It is useful for the debugging (default is False)
        """

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
        functions. 
        It will call the data that stored in the pcd_list. By doing this, we will
        not do the unecessary data copy that can happen during the function 
        execution. 
        Since preprocessing is almost identical between the source and target 
        file, it is done by the for loop. The only differnce is when we process 
        the soruce file. Then, we will apply the initial transformation guess. 
        The calculation step is 1) get voxel size, 2) perform down sample, and 3)
        obtain fpfh feature.
        There is no good document that explains what is the logical way to set 
        (or how should we set) the voxel size, radius for down sampling and fpfh 
        feature, including the neighboring bins for KDTree process. For now it is
        following the parameters from the examples, but in the future, this 
        should be optimized.

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
                        #ang_thres: float = 2 * np.pi,
                        mutual_filt: bool = True,
                        ratio: float = 0.9,
                        max_iter: int = 100000,
                        confi: float = 0.999,
                        scaling: bool = False
                        ) -> o3d.pipelines.registration.RegistrationResult:
        """Performs RANSAC registration. This registration is based on the down 
        sampled dataset and fpfh feature that calculated from preprocessing step.
        The fidning corresponding target points to the randomly selected source 
        points are done by applying pruning algorithms for rejecting false 
        matches in a early step. The only points that passed all pruning step
        will be used for the registration process. 
        The optimization is done by the hyperparameter of the function which are
        controlling the maximum number of iterations and the confidence 
        probability.
        As I addressed in the preprocessing step, the most of the parameters are
        based on the empirical value. It is challenging to find the proper way to
        set parameters, such as distance threshold, ratio, number of iteration,
        and confidence probability. For now it is following the parameters from 
        the examples, but in the future, this should be optimized.

        Parameters
        ----------
        src_idx : int   
            The index of the source pcd file from the list. It will select the
            file from the list based on the index.
        tar_idx : int
            The index of the target pcd file from the list. It will select the
            file from the list based on the index.        
        dis_thres : float
            The distance threshold for the CorrespondenceCheckerBasedOnDistance()
            class (Default is np.nan). 
        dis_thres_fac : flaot
            The distance threshold factor for the pruning step. If user didn't 
            specify it, It will be calculated by product of the factor and 
            averaged voxel size from the preprocessing (Default is 1.5).
        ang_thres : float
            The angle threshold for the normal related pruning step. It is
            currently disabled (Default is 2 * np.pi).
        mutual_filt : bool
            Enables mutual filter such that the correspondence of the source 
            point’s correspondence is itself(Default is True).
        ratio : float 
            The ratio threshold for the edge length related pruning step 
            (Default is 0.9).
        max_iter : int
            The hyperparameter for the minimization. It is the maximum limitation
            of the number of iterations (Default is 100000). 
        confi : float
            The hyperparameter for the minimization. It is the limitation for
            confidence probability of the fitness(Default is 0.999).
        scaling : bool
            The scaling threshold for the point to point estimation 
            (Default is False).
        
        Returns
        -------
        reg_ran : o3d.pipelines.registration.RegistrationResult
            The results of the RANSAC registration in open3d format
        """

        # Pruning algorithms
        # Checks if the lengths of any two arbitrary edges from each point cloud
        # are similar based on the ratio condition.
        edge_len = self.pipe_regi.CorrespondenceCheckerBasedOnEdgeLength(ratio)

        # Check if aligned point clouds are less than the threshold.
        if np.isnan(dis_thres): # If user didn't set the threshold, use factor.
            dis_thres = self.voxel_avg * dis_thres_fac
        dis = self.pipe_regi.CorrespondenceCheckerBasedOnDistance(dis_thres)

        # Check if two aligned point clouds have less than the angle threshold.
        # I couldn't find the general 'empirical' value for this step.  
        # It is commented out for now.
        #nor = self.pipe_regi.CorrespondenceCheckerBasedOnNormal(ang_thres)

        # Method for estimate a transformation for point to point distance.
        pt_to_pt = self.pipe_regi.TransformationEstimationPointToPoint(scaling)

        # Set the minimization hyperparameters for controlling number of 
        # iteration and confidence probability.
        criteria = self.pipe_regi.RANSACConvergenceCriteria(max_iter, confi)

        # Performs the RANSAC resigtration with the parameters.
        reg_ran = self.pipe_regi.registration_ransac_based_on_feature_matching(
            self.pcd_down[src_idx], self.pcd_down[tar_idx],
            self.pcd_fpfh[src_idx], self.pcd_fpfh[tar_idx],
            mutual_filt, dis_thres, 
            pt_to_pt, 3, [edge_len, dis], criteria)
            #pt_to_pt, 3, [edge_len, dis, nor], criteria)
        del edge_len, dis, pt_to_pt, criteria#, nor

        # Self the transformation matrix for ICP registration.
        self.reg_ran_trans = reg_ran.transformation
        if self.use_debug:
            # The # of inlier correspondences / # of points in source.
            self.reg_ran_fit = reg_ran.fitness
            # RMSE of all inlier correspondences.
            self.reg_ran_rmse = reg_ran.inlier_rmse
            # Correspondence set between source and target point cloud.
            self.reg_ran_corr = reg_ran.correspondence_set

        # print quick summary if user want to see it. 
        if self.verbose:
            print('Ransac registration summary')
            print(reg_ran)
            print('Ransac transformation matrix:')
            print(self.reg_ran_trans)

        return reg_ran

    def get_icp_regi(self,
                     src_idx: int,
                     tar_idx: int,
                     dis_thres: float = np.nan,
                     dis_thres_fac: float = 0.4,
                     max_iter: int = 2000,
                     trans_init = np.full((4, 4), np.nan, dtype=float),
                     use_p2p: bool = False
                     ) -> o3d.pipelines.registration.RegistrationResult:
        """Performs ICP registration. This registration is based on the 
        transformation matrix that obtained from the RANSAC registration. The    
        process is applied to actual point cloud, which is larger dataset, the
        initial guees of the transformation is necessary.
        This registration is following the point-to-plane ICP algorithm. It 
        exploits dot product between the normal vector of the target points and 
        the vector that created by target points and the trasfomred source 
        points. But by make use_p2p to True. User can also use point-to-point 
        ICP algorithm.
        The optimization is done by the hyperparameter of the function which are
        controlling the maximum number of iterations.
        As I addressed in the preprocessing step, the most of the parameters are
        based on the empirical value. It is challenging to find the proper way to
        set parameters, such as distance threshold, and number of iteration.
        For now it is following the parameters from the examples, but in the futu
        re, this should be optimized.

        Parameters
        ----------
        src_idx : int   
            The index of the source pcd file from the list. It will select the
            file from the list based on the index.
        tar_idx : int
            The index of the target pcd file from the list. It will select the
            file from the list based on the index.        
        dis_thres : float
            The distance threshold for the maximum correspondence between data
            (Default is np.nan). 
        dis_thres_fac : flaot
            The distance threshold factor. If user didn't specify it, It will be
            calculated by product of the factor and averaged voxel size from the
            preprocessing (Default is 0.4).
        max_iter : int
            The hyperparameter for the minimization. It is the maximum limitation
            of the number of iterations (Default is 2000).
        trans_init : np.ndarray
            The initial transformation guess (The default is the nan array).
        use_p2p : bool
            The boolean statement for controlling which estimation method user
            will be used (Default is False).

        Returns
        -------
        reg_icp : o3d.pipelines.registration.RegistrationResult
            The results of the ICP registration in open3d format
        """

        # If user didn't set the threshold, use factor and averaged voxel size
        # from the preprocessing.
        if np.isnan(dis_thres):
            dis_thres = self.voxel_avg * dis_thres_fac
       
        # Set the initial guess of transformation matrix.
        # If user didn't set it, it will use the value from ransac and the 
        # identity maxtrix value. 
        if not np.isnan(np.sum(trans_init)):
            pass
        elif not np.isnan(np.sum(self.reg_ran_trans)):
            trans_init = self.reg_ran_trans
        else:
            trans_init = np.identity(4, dtype=float)
     
        # Estimation method. Based on the use_p2p option, use can select. 
        if use_p2p:
            # Estimate a transformation for point to point distance.
            tans_est = self.pipe_regi.TransformationEstimationPointToPoint()
        else:  
            # Estimate a transformation for point to plane distance.
            tans_est = self.pipe_regi.TransformationEstimationPointToPlane()
        
        # Set the minimization hyperparameters for controlling number of 
        # maximum iteration.
        criteria = self.pipe_regi.ICPConvergenceCriteria(max_iteration=max_iter)
    
        # Performs the ICP resigtration with the parameters.
        reg_icp = self.pipe_regi.registration_icp(
            self.pcd_list[src_idx], self.pcd_list[tar_idx], 
            dis_thres, trans_init,
            tans_est, criteria)
        del tans_est, criteria

        if self.use_debug:
            # Transformation matrix.
            self.reg_icp_trans = reg_icp.transformation
            # The # of inlier correspondences / # of points in source.
            self.reg_icp_fit = reg_icp.fitness
            # RMSE of all inlier correspondences.
            self.reg_icp_rmse = reg_icp.inlier_rmse
            # Correspondence set between source and target point cloud.
            self.reg_icp_corr = reg_icp.correspondence_set

        # print quick summary if user want to see it. 
        if self.verbose:
            print('ICP registration summary')
            print(reg_icp)
            print('ICP transformation matrix:')
            print(reg_icp.transformation)

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






