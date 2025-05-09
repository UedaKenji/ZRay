import numpy as np
from dataclasses import dataclass,asdict
from typing import Tuple, List
from abc import ABC, abstractmethod





class Camera(ABC):
    """
    Abstract base class for camera objects.
    Provides methods for generating rays and converting camera data to and from dictionaries.
    """
    @abstractmethod
    def generate_ray(self):
        pass

    def __str__(self):
        # 変数をプロットして、改行する
        return '\n'.join([f'{key}: {value}' for key, value in self.__dict__.items()])
    
    def to_dict(self):
        _dict = asdict(self)
        _dict['classname'] = self.__class__.__name__
        #ndarrayをリストに変換する
        for key, value in _dict.items():
            if isinstance(value, np.ndarray):
                _dict[key] = value.tolist()
        return _dict
        
    @classmethod
    def from_dict(cls, d:dict)-> 'Camera':
        del d['classname']
        return cls(**d)
    
    @property
    @abstractmethod
    def M(self) -> int:
        """
        Returns the number of pixels in the camera.
        """
        pass    

@dataclass
class LineCamera(Camera): #センサーが直線のカメラ
    """
    Represents a line camera with a linear sensor.

    Attributes:
    focal_length (float): Focal length of the camera in meters.
    location (Tuple[float, float, float]): 3D coordinates of the camera's location.
    direction (Tuple[float, float, float]): Direction vector of the camera.
    sensor_size (float): Size of the sensor in meters.
    resolution (int): Number of pixels in the sensor.
    """
    focal_length: float               #: 焦点距離[m]
    location: Tuple[float,float,float] #: カメラの位置（３次元座標）[m]
    direction: Tuple[float,float,float]
    sensor_size: Tuple[float]
    resolution: int = 1

    def __post_init__(self):
        self.location = np.array(self.location)
        self.direction = np.array(self.direction)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.__M        = self.resolution 
        print(self)
        
    
    def generate_ray(self, 
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate rays corresponding to all pixels in the camera's sensor.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of positions and directions for the rays.
        """
        # position は(M,3 ) の配列にブロードキャストする。
        position = np.broadcast_to(self.location, (self.M, 3))


        # センサーの中心座標
        sensor_center = self.location + self.focal_length * self.direction

        # センサーの中心座標から、センサーの水平方向の単位ベクトルを求める
        horizontal = np.cross(self.direction, np.array([0,1,0]))

        if np.linalg.norm(horizontal) == 0:
            horizontal = np.array([-position[0,1], position[0,0], 0])

        horizontal = horizontal / np.linalg.norm(horizontal)
        print(horizontal)

        w = (np.arange(self.M)+0.5) * self.sensor_size / self.M - self.sensor_size / 2

        print(w)

        # センサーの全ピクセルの位置を求める
        sensor_pixel = sensor_center + w[:, np.newaxis] * horizontal

        print(sensor_pixel)

        # センサーピクセルのグローバル座標から、positionを引いて、ray_dirを求める
        ray_dir = sensor_pixel - self.location
        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1)[:,np.newaxis]
        return position, ray_dir
    
    @property
    def M(self) -> int:
        return self.__M



@dataclass
class Camera2D_xyz(Camera):
    """
    Represents a 2D camera in Cartesian coordinates.

    Attributes:
    focal_length (float): Focal length of the camera in meters.
    location (Tuple[float, float, float]): 3D coordinates of the camera's location.
    direction (Tuple[float, float, float]): Direction vector of the camera.
    sensor_size (Tuple[float, float]): Width and height of the sensor in meters.
    resolution (Tuple[int, int]): Number of pixels in the sensor (width, height).
    rotation (float): Rotation angle of the camera in radians.
    """
    focal_length: float               #: 焦点距離[m]
    location: Tuple[float,float,float] #: カメラの位置（３次元座標）[m]
    direction: Tuple[float,float,float]#: カメラの中心軸（正規化済みベクトル）
    sensor_size: Tuple[float, float]  #: センサーサイズ (幅, 高さ) [m 等の単位]
    resolution: Tuple[int, int]       #: 画素数 (pixel_x, pixel_y)
    rotation : float = 0                 #: カメラのねじり回転角度[rad] 
    classname = 'Camera2D_xyz'

    def __post_init__(self):    
        #正規化
        self.location = np.array(self.location)
        self.direction = np.array(self.direction)
        self.direction = self.direction / np.linalg.norm(self.direction)
        self.im_shape = (self.resolution[1], self.resolution[0])    
        self.__M        = self.resolution[0] * self.resolution[1] 
        #print(self)

    



    def generate_ray(self,
            imshape: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:     
        """
        Generate rays corresponding to all pixels in the camera's sensor.

        Parameters:
        imshape (bool): Whether to return the rays in the shape of the image.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of positions and directions for the rays.
        """

        # position は(M,3 ) の配列にブロードキャストする。
        position = np.broadcast_to(self.location, (self.M, 3))


        # センサーの中心座標
        sensor_center = self.location + self.focal_length * self.direction

        # センサーの中心座標から、センサーの水平方向の単位ベクトルを求める
        horizontal = np.cross(self.direction, np.array([0,0,1]))
        if np.linalg.norm(horizontal) == 0:
            horizontal = np.array([-position[0,1], position[0,0], 0])

        horizontal = horizontal / np.linalg.norm(horizontal)

        # センサーの中心座標から、センサーの垂直方向の単位ベクトルを求める
        vertical = np.cross(horizontal, self.direction)
        vertical = vertical / np.linalg.norm(vertical)


        w = (np.arange(self.resolution[0])+0.5) * self.sensor_size[0] / self.resolution[0] - self.sensor_size[0] / 2    
        h = (np.arange(self.resolution[1])+0.5) * self.sensor_size[1] / self.resolution[1] - self.sensor_size[1] / 2

        h[:] = h[::-1]

        W,H = np.meshgrid(w,h,indexing='xy')

        # 回転させる、
        W,H =  W * np.cos(self.rotation) - H * np.sin(self.rotation), W * np.sin(self.rotation) + H * np.cos(self.rotation)

        # センサーの全ピクセルの位置を求める
        sensor_pixel = sensor_center + W[:, :, np.newaxis] * horizontal + H[:, :, np.newaxis] * vertical 

        # センサーピクセルのグローバル座標から、positionを引いて、ray_dirを求める
        ray_dir = sensor_pixel - self.location

        ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1)[:,:,np.newaxis]

        if imshape:
            return position.reshape(-1,3), ray_dir.reshape(self.im_shape[0], self.im_shape[1], 3)
        else:
            return position, ray_dir.reshape(-1,3)
        
    @property
    def M(self) -> int:
        return self.__M
        



@dataclass
class Camera2D_rphiz(Camera):
    """
    Represents a 2D camera in cylindrical coordinates.

    Attributes:
    focal_length (float): Focal length of the camera in meters.
    sensor_size (Tuple[float, float]): Width and height of the sensor in meters.
    resolution (Tuple[int, int]): Number of pixels in the sensor (width, height).
    location (Tuple[float, float, float]): Cylindrical coordinates of the camera's location.
    center_angles (Tuple[float, float]): Horizontal and vertical angles of the camera's center axis in degrees.
    rotation (float): Rotation angle of the camera in radians.
    """
    focal_length: float
    sensor_size: Tuple[float, float]
    resolution: Tuple[int, int]
    location: Tuple[float, float, float]
    center_angles: Tuple[float,float]
    rotation: float=0
    classname = 'Camera2D_rphiz'
    """
    set camera infomation

    Parameters
    ----------
    focal_length: float,
        focal_length [m]
    sensor_size:: Tuple[float,float],
        like as ( Width     [m], Height    [m])
    image_shape  : Tuple[int,int],
        this param is the shape of array associated with image, like as ( num of W, num of H )
    location: Tuple[float, float, float],
        equal to ( Z_cam, Phi_cam, R_cam ) 
    center_angles: Tuple[float,float],
        equal to (horizontal_angle[deg],vertical_angle[deg])
    rotation: float=0,
        the angle of camera rotation [rad]
    Reuturns
    ----------
    None

    """
    def __post_init__(self):
        self.location = np.array(self.location)
        self.im_shape = (self.resolution[1], self.resolution[0])    
        self.__M        = self.resolution[0] * self.resolution[1]
        print(self)
        
    @property
    def M(self) -> int:
        return self.__M


    def generate_ray(self,
            imshape: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:     
        """
        Generate rays corresponding to all pixels in the camera's sensor.

        Parameters:
        imshape (bool): Whether to return the rays in the shape of the image.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of positions and directions for the rays.
        """
        # Camera_rphiの変換して、Camera_xyzに変換して、generate_rayを呼び出す。
        # Camera_rphiの座標を、Camera_xyzの座標に変換する
        location = self.location

        # カメラの中心軸のベクトルを求める
        direction = np.array([-np.cos(np.deg2rad(self.center_angles[1])) * np.cos(np.deg2rad(self.center_angles[0])),
                              np.cos(np.deg2rad(self.center_angles[1])) * np.sin(np.deg2rad(self.center_angles[0])),
                              np.sin(np.deg2rad(self.center_angles[1]))])

        # Camera_xyzに変換する
        cam = Camera2D_xyz(self.focal_length, location, direction, self.sensor_size, self.resolution, self.rotation)
        return cam.generate_ray(imshape=imshape)



class MultiCamera(Camera):
    """
    Represents a collection of multiple cameras.

    Attributes:
    camera_list (List[Camera]): List of camera objects.
    name_list (List[str]): List of names for the cameras.
    """
    def __init__(self, 
        camera_list:List[Camera],
        name_list:List[str] = None,
        ):
        self.camera_list:List[Camera] = camera_list
        self.__M = sum([cam.M for cam in camera_list])

        if name_list is None:
            self.name_list = [f'camera{i}' for i in range(len(camera_list))]
        else:
            if len(name_list) != len(camera_list):
                raise ValueError("name_list must be same length as camera_list")
            self.name_list = name_list
        print(self.name_list)

        self.slice_list = []    
        start = 0
        for cam in camera_list:
            end = start + cam.M
            self.slice_list.append(slice(start, end))
            start = end
        
    @property
    def M(self) -> int:
        return self.__M
    
    def slice(self, arg:int|str=None) -> slice:
        if arg is None:
            _dict = {}
            for i, cam in enumerate(self.camera_list):
                _dict[self.name_list[i]] =  self.slice_list[i]
            return _dict

        elif isinstance(arg, int):
            return self.slice_list[arg]
        elif isinstance(arg, str):
            if arg in self.name_list:
                return self.slice_list[self.name_list.index(arg)]
            else:
                raise ValueError(f"{arg} is not in name_list")
        else:
            raise ValueError("arg must be int or str")
        
    def index(self, arg:int|str) -> np.ndarray:
        res = np.zeros(self.M, dtype=bool)
        slice_ = self.slice(arg)
        res[slice_] = True
        return res
    
    def generate_ray(self):
        """
        Generate rays for all cameras in the collection.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of positions and directions for the rays.
        """
        position = np.empty((self.M, 3))
        ray_dir = np.empty((self.M, 3))
        start = 0
        for cam in self.camera_list:
            end = start + cam.M
            position[start:end], ray_dir[start:end] = cam.generate_ray()
            start = end
        return position, ray_dir
    
    def __str__(self):
        a = []
        for i, cam in enumerate(self.camera_list):
            a.append(f'camera{i}')
            a.append(str(cam))

        return '\n'.join(a)
    
    def to_dict(self):
        return {
            'classname': 'MultiCamera',
            'name_list': self.name_list,
            'camera_list': [cam.to_dict() for cam in self.camera_list]
        }
    
    @classmethod
    def from_dict(cls, d):
        name_list = d['name_list']
        camera_list = []

        for i, cam in enumerate(d['camera_list']):    
            print(f"####### {name_list[i]} #######")

            if cam['classname'] == 'Camera2D_rphiz':
                camera_list.append(Camera2D_rphiz.from_dict(cam))
            elif cam['classname'] == 'Camera2D_xyz':
                camera_list.append(Camera2D_xyz.from_dict(cam))
            elif cam['classname'] == 'LineCamera':
                camera_list.append(LineCamera.from_dict(cam))
            else:
                raise ValueError(f"Unknown camera type: {cam['classname']}")
            
            print('')
        return cls(camera_list=camera_list, name_list=name_list)


