import numpy as np
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt

# import container
from . import vessel
from . import measurement
from . import core 
from dataclasses import dataclass   
import tqdm
from typing import List, Tuple, Optional
#from . import plot_utils


__all__ = ["Raytracing", "Ray", "ImageVector"]

class ImageVector(np.ndarray):
    """
    A subclass of numpy.ndarray that represents a 1D array with an associated 2D image shape.
    Provides methods to reshape the array into its 2D image form and flatten it back to 1D.
    """
    def __new__(cls, input_array, im_shape):
        obj = np.asarray(input_array).view(cls)
        if np.prod(im_shape) != obj.size:
            raise ValueError("im_shape does not match the size of input array")
        obj.im_shape = im_shape  # 画像の形状を保存
        return obj

    @property
    def im(self):
        """2次元にリシェイプされた画像を取得"""
        return self.reshape(self.im_shape)

    def flatten(self):
        """1次元に変換"""
        return super().flatten()
    
from typing import cast

@dataclass
class Ray:
    """
    Represents a ray in 3D space with position, direction, length, and other properties.
    Provides methods to generate ray positions in Cartesian and cylindrical coordinates.
    """
    Possition_xyz: np.ndarray
    Direction_xyz: np.ndarray
    Length :np.ndarray|ImageVector = 0.0
    curve_index :np.ndarray|ImageVector = -1
    Cos :np.ndarray|ImageVector = None

    def __post_init__(self):
        self.M = self.Possition_xyz.shape[0]
        self.Direction_xyz = self.Direction_xyz / np.linalg.norm(self.Direction_xyz, axis=1)[:,np.newaxis]

    def generate_xyz(self,Lnum=100, Lmax=None):
        """
        Generate the 3D positions of the ray along its direction.

        Parameters:
        Lnum (int): Number of points to generate along the ray.
        Lmax (float): Maximum length of the ray. Defaults to the ray's length.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of x, y, z positions.
        """
        if Lmax is None:
            Lmax = np.array(self.Length)

        L = np.linspace(0,Lmax,Lnum).T
        #RuntimeWarning を無視
        with np.errstate(invalid='ignore'):
            Lx = self.Possition_xyz[:,0,np.newaxis] + L * self.Direction_xyz[:,0,np.newaxis]
            Ly = self.Possition_xyz[:,1,np.newaxis] + L * self.Direction_xyz[:,1,np.newaxis]
            Lz = self.Possition_xyz[:,2,np.newaxis] + L * self.Direction_xyz[:,2,np.newaxis]

        Lx[np.isnan(Lx)] = np.inf
        Ly[np.isnan(Ly)] = np.inf
        Lz[np.isnan(Lz)] = np.inf

        return Lx,Ly,Lz

    
    def generate_rz(self,Lnum=100, Lmax=None):
        """
        Generate the radial and z positions of the ray in cylindrical coordinates.

        Parameters:
        Lnum (int): Number of points to generate along the ray.
        Lmax (float): Maximum length of the ray. Defaults to the ray's length.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of radial and z positions.
        """
        x,y,z = self.generate_xyz(Lnum, Lmax)

        r = np.sqrt(x**2 + y**2)

        return r,z

    def generate_rphiz(self,Lnum=100, Lmax=None):
        """
        Generate the radial, azimuthal, and z positions of the ray in cylindrical coordinates.

        Parameters:
        Lnum (int): Number of points to generate along the ray.
        Lmax (float): Maximum length of the ray. Defaults to the ray's length.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of radial, azimuthal, and z positions.
        """
        x,y,z = self.generate_xyz(Lnum, Lmax)

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)

        return r,phi,z
    
    def generate_projectionmatrix_from_grid(self,r_grid,z_grid, Lnum=1000):
        """
        Create a projection matrix from a 2D grid.

        Parameters:
        r_grid (np.ndarray): Radial grid points.
        z_grid (np.ndarray): Z grid points.
        Lnum (int): Number of points along the ray.

        Returns:
        np.ndarray: Projection matrix.
        """
        rray,zray = self.generate_rz(Lnum)
        M = rray.shape[0]
        r_grid = r_grid.flatten()
        z_grid = z_grid.flatten()

        H = np.zeros((M,r_grid.size*z_grid.size))

        dL = self.Length / (Lnum-1)


        for i in range(M):
            for j in range(Lnum):
                r = rray[i,j]
                z = zray[i,j]
                if r < r_grid[0] or r > r_grid[-1] or z < z_grid[0] or z > z_grid[-1]:
                    continue
                index1 = np.argmin(np.abs(r_grid - r))
                index2 = np.argmin(np.abs(z_grid - z))
                H[i,r_grid.size*index2 + index1] += 1 

        H = H * dL[:,np.newaxis]
        return H




class Raytracing:
    """
    A class for performing ray tracing in a 3D axisymmetric vessel.
    Handles ray generation, reflection, and interaction with vessel curves.
    """
    def __init__(self, 
        container :vessel.AxisymmetricVessel,
        measurement:measurement.Camera|measurement.MultiCamera,):
        """
        Initialize the Raytracing object with a vessel container and measurement system.

        Parameters:
        container (vessel.AxisymmetricVessel): The vessel containing curves for ray tracing.
        measurement (measurement.Camera | measurement.MultiCamera): The measurement system.
        """
        self.container = container  
        self.ncurves = len(container.Curves)
        self.measurement = measurement
        pass 

    def save_setting(self, filename:str):
        """
        Save the container and measurement settings to a JSON file.

        Parameters:
        filename (str): The name of the file to save the settings.
        """
        # self.containerとself.measurementをjson形式で保存する
        # measurement はcameraかmulti-cameraのどちらかであることを想定

        if not filename.endswith('.json'):
            filename += '.json'
        vessel_dict = self.container.to_dict()
        measurement_dict = self.measurement.to_dict()
        data = {
            'measurement': measurement_dict,
            'container': vessel_dict
        }
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_setting(cls, filename:str):
        """
        Load container and measurement settings from a JSON file.

        Parameters:
        filename (str): The name of the file to load the settings.

        Returns:
        Raytracing: An instance of the Raytracing class.
        """
        # save_settingで保存したjson形式のファイルを読み込む
        # measurement はcameraかmulti-cameraのどちらかであることを想定

        if not filename.endswith('.json'):
            raise ValueError("Filename must end with .json")
        # json.loadで読み込む
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        container_dict = data['container']
        measurement_dict = data['measurement']

        container_obj = vessel.AxisymmetricVessel.from_dict(container_dict)

        if measurement_dict['classname'] == 'MultiCamera':
            measurement_obj = measurement.MultiCamera.from_dict(measurement_dict)
        elif measurement_dict['classname'] == 'Camera2D_xyz':
            measurement_obj = measurement.Camera2D_xyz.from_dict(measurement_dict)
        elif measurement_dict['classname'] == 'Camera2D_rphiz':
            measurement_obj = measurement.Camera2D_rphiz.from_dict(measurement_dict)
        elif measurement_dict['classname'] == 'LineCamera':
            measurement_obj = measurement.LineCamera.from_dict(measurement_dict)
        else:
            raise ValueError("Unknown measurement type")
        return cls(container_obj, measurement_obj)
    
    
    def save_heavy_model(self, filename:str):
        """
        Save the Raytracing object and its settings to a pickle file.

        Parameters:
        filename (str): The name of the file to save the object.
        """
        ext = '.pkl'

        if ext in filename:
            filename = filename.split(ext)[0]

        self.save_setting(filename)
                    
        pd.to_pickle(self, filename+ext)

    @classmethod
    def load_heavy_model(cls, filename:str):
        """
        Load a Raytracing object from a pickle file.

        Parameters:
        filename (str): The name of the file to load the object.

        Returns:
        Raytracing: An instance of the Raytracing class.
        """
        if not filename.endswith('.pkl'):
            raise ValueError("Filename must end with .pkl")
        # pd.read_pickleでクラスを読み込む
        return pd.read_pickle(filename) 
    

    @classmethod
    def load(cls, filename:str):
        """
        Load a Raytracing object from a JSON or pickle file.

        Parameters:
        filename (str): The name of the file to load the object.

        Returns:
        Raytracing: An instance of the Raytracing class.
        """
        if filename.endswith('.json'):
            return cls.load_setting(filename)
        elif filename.endswith('.pkl'):
            return cls.load_heavy_model(filename)


    def main(self,
            nreflections:int = 2,
            pass_through_first:bool = True,
             ):
        """
        Perform the main ray tracing process, including reflections.

        Parameters:
        nreflections (int): Number of reflections to calculate.
        pass_through_first (bool): Whether to pass through the first reflection.
        """
        def main2(O,D,Cos=None):
            
            inf_index = np.isinf(O[:,0])

            O2 = O[~inf_index,:]
            D2 = D[~inf_index,:]
            length_all = np.zeros((O2.shape[0], self.ncurves))

            if inf_index.any():
                print("inf length is detected "+str(inf_index.sum()))

            for i in tqdm.tqdm(range(self.ncurves)):
                curve = self.container.Curves[i]
                length_all[:,i] = core.intersect_rays(curve, O2, D2, tol=1e-5)

            #length_all ncurveに関して最小となるときを選び、またそのときのindexを取得する。ただしＬ=-1hのときは除く
            length_all[length_all<0] = np.inf
            min_index2 = np.argmin(length_all, axis=1)
            min_length2 = np.min(length_all, axis=1)


            min_index = np.zeros((self.measurement.M), dtype=int)
            min_length = np.zeros((self.measurement.M))

            min_index[~inf_index] = min_index2
            min_length[~inf_index] = min_length2

            # self.measurement にim_shapeあるかどうか?
            if hasattr(self.measurement, "im_shape"):
                im_shape = self.measurement.im_shape
                min_length = ImageVector(min_length,im_shape)
                min_index = ImageVector(min_index,im_shape)
                if Cos is not None:
                    Cos = ImageVector(Cos,im_shape)
            else:
                pass 


            return Ray( Possition_xyz = O, Direction_xyz = D, Length = min_length, curve_index = min_index, Cos = Cos)


        def reflection(O,D,curve_index):
            """
            反射の計算
            """

            r = np.sqrt(O[:,0]**2 + O[:,1]**2)
            phi = np.arctan2(O[:,1], O[:,0])
            nr,nz = np.zeros((self.measurement.M)), np.zeros((self.measurement.M))

            for i in range(self.measurement.M):
                curve = self.container.Curves[curve_index[i]]
                n = curve.normal_at(r[i], O[i,2])
                nr[i], nz[i] = n[0], n[1]

            nx = np.cos(phi) * nr
            ny = np.sin(phi) * nr

            #反射ベクトル

            Dn = D[:,0] * nx + D[:,1] * ny + D[:,2] * nz
            Cos = np.abs(Dn) / np.sqrt(nx**2 + ny**2 + nz**2)
            D = D - 2 * Dn[:,np.newaxis] * np.vstack((nx,ny,nz)).T

            # cosineの計算, 法線ベクトルとの内積を計算

            return D,Cos
        

        self.rays:List[Ray] = []

        O,D = self.measurement.generate_ray()

        ray = main2(O,D)
        self.rays.append(ray)
        with np.errstate(invalid='ignore'):
            O = O + 1.00001*ray.Length[:,np.newaxis] * ray.Direction_xyz
        O[np.isnan(O)] = np.inf
        O = np.array(O, dtype=np.float64)

        for i in range(nreflections):
            if i == 0 and pass_through_first:
                Cos = None
                pass 
            else:
                D,Cos = reflection(O,D,ray.curve_index)

            ray = main2(O,D,Cos)
            self.rays.append(ray)
            O = O + 0.99999*ray.Length[:,np.newaxis] * ray.Direction_xyz
            O = np.array(O, dtype=np.float64)

    def plot(self, fig=None,  **kwargs):
        """
        Plot the rays traced in the vessel.

        Parameters:
        fig (matplotlib.figure.Figure): The figure to plot on. If None, a new figure is created.
        kwargs: Additional keyword arguments for the plot.
        """
        n = len(self.rays)


        fig,axs = plt.subplots(1,n,figsize=(2*n,2), sharex=True, sharey=True)

        for i,ray in enumerate(self.rays):
            L = ray.Length

            imshow_cbar_bottom(axs[i], L.im,cbar_title="Length [m]",  **kwargs)






def imshow_cbar_bottom(
    ax:plt.Axes,
    im0:np.ndarray,
    cbar_title=None,
    **kwargs
    ):
    """
    Display an image with a colorbar below it.

    Parameters:
    ax (plt.Axes): The axes to plot on.
    im0 (np.ndarray): The image data to display.
    cbar_title (str): Title for the colorbar.
    kwargs: Additional keyword arguments for imshow.
    """
    import mpl_toolkits.axes_grid1
    im = ax.imshow(im0,**kwargs)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)    
    cax = divider.append_axes("bottom", size="5%", pad='3%')
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    if cbar_title is not None:
        cbar.set_label(cbar_title)
    ax.set_aspect('equal')
