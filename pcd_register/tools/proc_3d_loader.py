"""3d process loader

This script is designed to calculate the covariance matrix, approximate 
curvature, and projection of point to the plane without using the Open3d
package.

All the calculation is done by NumPy functions.

    * proc_3d_loader - Class that contains the calculation by NumPy
"""

import numpy as np

class proc_3d_loader:
    """Designed to calculate the covariance matrix, approximate curvature, 
        and projection of point to the plane without using the Open3d package.

    ...

    Attributes
    ----------
    verbose : bool
        Boolean statement to control the print
    use_debug : bool
        By changing its to True, use can check and svae the all middle step
        of the calculation. It is useful for the debugging    
    use_KDTree : bool
        Boolean statement to control the loading `open3d` package
    pts_nei_cov : ndarray
        The covariance matrix from the centriod
    pts_nei_cen : ndarray
        The value of the centriod (available if `use_debug` is True)
    eig_vec_nei : ndarray
        The egien vecetors from the covariance matrix
    min_idx : int
        The index of the minimum element in the `eig_val_nei` array
    pts_nei_curv : ndarray
        The approximate curvature
    eig_val_nei : ndarray
        The egien values from the covariance matrix
        (available if `use_debug` is True)
    pts_nei_proj : ndarray
        The projected points to the plane
    nor_vec_nei : ndarray
        The normal vector of the point `idx` (available if `use_debug` is True)
    dis_vec_nei : ndarray
        displacement vector (available if `use_debug` is True)
    dis_to_plane : ndarray
        The dot product between the normal vector and the displacement vector
        (available if `use_debug` is True)    
    pts_i : float
        The point with an index `idx`
    rad_all : ndarray
        The radius of all the points (available if `use_debug` is True)
    rad_nei_idx : ndarray
        The index or boolean array that corresponds to the neighboring points
        (available if `use_debug` is True)
    pts_nei : ndarray
        The neighboring points based on the index and the radius    

    Methods
    -------
    get_covar_mtx()
        Get covariance matrix of neighboring points
    get_approx_curv()
        Get approximate curvature at a point `idx`
    get_proj_pts_to_plane()
        Calculates the projected poitns to the plane
    get_3d_process(pts: np.ndarray, idx: int, rad: float)
        Calculates the neighboring points and execute all other functions
    """

    def __init__(self, 
                 verbose: bool = False, 
                 use_debug: bool = False, 
                 use_KDTree: bool = False):
        """Initializer for the 3d process class.

        Parameters
        ----------
        verbose : bool
            Boolean statement to control the print (default is False)
        use_debug : bool
            By changing its to True, use can check and svae the all middle step
            of the calculation. It is useful for the debugging (default is False)
        use_KDTree : bool
            Boolean statement to control the loading `open3d` package
            (default is False)
        """

        self.verbose = verbose
        self.use_debug = use_debug
        self.use_KDTree = use_KDTree
        if self.use_KDTree:
            # Loading `open3d` is placed under `use_KDTree`, since we only need 
            # this lib. if user want to do point selection with KDTree method.
            import open3d as o3d

    def get_covar_mtx(self):
        """Get covariance matrix of neighboring points based on their centriod"""

        # Calculates the centriod of neighboring points.
        pts_nei_cen = np.nanmean(self.pts_nei, axis=0)

        # Calculates covariance matrix from the centriod.
        self.pts_nei_cov = np.cov((self.pts_nei - pts_nei_cen).T)

        if self.use_debug:
            # In case user want to see what was the value of the centriod.
            self.pts_nei_cen = pts_nei_cen
        del pts_nei_cen      
 
    def get_approx_curv(self):
        """Get approximate curvature at a point `idx`"""

        # Egien values and vecetors from the covariance matrix.
        # The egien values will be used for calculating the curvature.
        # The egien vector is contating the normal vector at point `idx`.
        eig_val_nei, self.eig_vec_nei = np.linalg.eigh(self.pts_nei_cov)

        # The index of the smallest element in the `eig_val_nei` array.
        self.min_idx = np.nanargmin(eig_val_nei)

        # The approximate curvature is the smallest egien value divided by the 
        # sum of all egien values
        self.pts_nei_curv = eig_val_nei[self.min_idx] / np.nansum(eig_val_nei)

        if self.use_debug:
            # In case user want to check the egien values.
            self.eig_val_nei = eig_val_nei
        del eig_val_nei

    def get_proj_pts_to_plane(self):
        """Project points within the radius at a point with `idx` to the plane 
        which is perpendicular to the normal at the point `idx`
        """

        # The normal vector of the point `idx` is the vector corresponds to the 
        # smallest egien value.
        # Use the index of the minimum element.
        nor_vec_nei = self.eig_vec_nei[:, self.min_idx]

        # Calculates displacement vector.
        dis_vec_nei = self.pts_nei - self.pts_i

        # The dot product that utilizing NumPy matrix calculation.
        # (# of pts, xyz) * (xyz, 1) = (# of pts, 1).
        # The 1 is automatically removed. So, the output is just '# of pts'.
        dis_to_plane = np.dot(dis_vec_nei, nor_vec_nei)

        # Calculates the projected points by the vector subtraction.
        # To take into account different dimension between two arrays,
        # `np.newaxis` is used to increase the dimension.
        self.pts_nei_proj = self.pts_nei - dis_to_plane[:, np.newaxis]*nor_vec_nei[np.newaxis, :]
        
        if self.use_debug:
            self.nor_vec_nei = nor_vec_nei
            self.dis_vec_nei = dis_vec_nei
            self.dis_to_plane = dis_to_plane 
        del nor_vec_nei, dis_vec_nei, dis_to_plane

    def get_3d_process(self, pts: np.ndarray, idx: int, rad: float):
        """Calculates the neighboring points based on the radius and the point
        with an index. Then, execute other functions to calculate covariance 
        matrix, approximate curvature, and projection of point to the plane. 

        Parameters
        ----------
        pts : ndarray
            The point cloud. It is xyz coodinate of the all points.
        idx : int
            The index for selecting the point.
        rad : float
            The boundary condition to select the neighboring points. 
        """
        
        if self.verbose:
            print(f'Size of total points: {pts.shape}')
            print(f'Selected index: {idx}')
            print(f'Selected radius: {rad}')

        # The point with an index `idx`.
        self.pts_i = pts[idx]

        if self.use_KDTree:
            # Use KDTree class only if `self.use_KDTree` is True.
            kdtree = o3d.geometry.KDTreeFlann(pts)

            # Get the index array that corresponds to the neighboring points.
            rad_nei_idx = kdtree.search_radius_vector_3d(self.pts_i, rad)[1]
            del kdtree
        else:
            # Calculates the radius of all the points.
            rad_all = np.sqrt(np.nansum((pts - self.pts_i) ** 2, axis=1))

            # Get the boolean array based on the radius.
            # The elements that corresponds to the neighboring points
            # would be True.
            rad_nei_idx = rad_all <= rad

            if self.use_debug:
                self.rad_all = rad_all
            if self.verbose: # To give a sense to user about data distribution.
                medi_rad = np.nanmedian(rad_all)
                print(f'The median of all radius is {np.round(medi_rad, 2)}')
            del rad_all

        # The neighboring points based on the index and the radius.
        self.pts_nei = pts[rad_nei_idx]
        if self.verbose:
            print(f'Size of points within radius {rad}: {self.pts_nei.shape}')
        if self.use_debug:
            self.rad_nei_idx = rad_nei_idx
        del rad_nei_idx

        # Based the neighboring points, execute other functions.
        # Calculates the covariance matrix of neighboring points.
        self.get_covar_mtx()

        # Calculates the approximate curvature. 
        self.get_approx_curv()

        # Calculates the projected poitns to the plane.
        self.get_proj_pts_to_plane()
        if not self.use_debug:
            del self.pts_nei, self.eig_vec_nei, self.pts_i, self.min_idx
