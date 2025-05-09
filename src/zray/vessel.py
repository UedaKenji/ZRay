#!/usr/bin/env python3
import numpy as np
import math
from dataclasses import dataclass,asdict
from typing import List, Tuple, Optional
import dxfgrabber
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


__all__ = ["LineSegment", "CircularArc", "Circle", "AxisymmetricVessel"]  


class RZCurve:
    """RZ平面上の曲線の抽象クラス．
    各派生クラスは intersect_ray，is_point_on_curve，normal_at を実装する．
    """

    def is_point_on_curve(self, r: float, z: float, tol=1e-6) -> bool:
        raise NotImplementedError

    def normal_at(self, r: float, z: float) -> np.ndarray:
        """曲線上の (r, z) における（RZ平面上の）外向き単位法線を返す．"""
        raise NotImplementedError


@dataclass
class LineSegment(RZCurve):
    p0: Tuple[float, float]  #: 始点 (r0, z0)
    p1: Tuple[float, float]  #: 終点 (r1, z1)


    def is_point_on_curve(self, r: float, z: float, tol=1e-6) -> bool:
        r0, z0 = self.p0
        r1, z1 = self.p1
        if abs(z1 - z0) > tol:
            lam = (z - z0) / (z1 - z0)
        else:
            lam = (r - r0) / (r1 - r0) if abs(r1 - r0) > tol else 0.0
        return (-tol <= lam <= 1+tol)

    def normal_at(self, r: float, z: float) -> np.ndarray:
        """
        RZ平面上で，線分の接ベクトル (dr, dz) に対して垂直な方向として，
        (dz, -dr)（正規化したもの）を返す．
        （どちら側が「外側」かは，必要に応じて符号調整してください．）
        """
        dr = self.p1[0] - self.p0[0]
        dz = self.p1[1] - self.p0[1]
        norm = math.hypot(dr, dz)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return np.array([dz/norm, -dr/norm])
    
@dataclass
class CircularArc(RZCurve):
    center: Tuple[float, float]  #: 円の中心 (r_c, z_c)
    radius: float                #: 円の半径
    theta_start: float           #: 円弧開始角（ラジアン）
    theta_end: float             #: 円弧終了角（ラジアン）


    def is_point_on_curve(self, r: float, z: float, tol=1e-6) -> bool:
        r_c, z_c = self.center
        if abs((r - r_c)**2 + (z - z_c)**2 - self.radius**2) > tol:
            return False
        phi = math.atan2(z - z_c, r - r_c) % (2*math.pi)
        theta_start = self.theta_start % (2*math.pi)
        theta_end = self.theta_end % (2*math.pi)
        if theta_start <= theta_end:
            return theta_start - tol <= phi <= theta_end + tol
        else:
            return phi >= theta_start - tol or phi <= theta_end + tol

    def normal_at(self, r: float, z: float) -> np.ndarray:
        """
        円の中心から点への方向 (r - r_c, z - z_c) を正規化したものを返す．
        """
        r_c, z_c = self.center
        n = np.array([r - r_c, z - z_c])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return n / norm

@dataclass
class Circle(RZCurve):
    center: Tuple[float, float]
    radius: float

    def is_point_on_curve(self, r: float, z: float, tol=1e-6) -> bool:
        r_c, z_c = self.center
        return abs((r - r_c)**2 + (z - z_c)**2 - self.radius**2) < tol
    
    def normal_at(self, r: float, z: float) -> np.ndarray:
        r_c, z_c = self.center
        n = np.array([r - r_c, z - z_c])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return n / norm
    
    

@dataclass
class AxisymmetricVessel:
    Curves: List[RZCurve] = None


    def __post_init__(self):
        self.is_closed()
        self.Lines = [curve for curve in self.Curves if isinstance(curve, LineSegment)]
        self.Arcs = [curve for curve in self.Curves if isinstance(curve, CircularArc)]
        self.Circles = [curve for curve in self.Curves if isinstance(curve, Circle)]

    def is_closed(self, tol=1e-6) -> bool:
        """曲線が閉じたループを形成しているかを判定．
           （ここでは，最初の曲線の始点と最後の曲線の終点を全リストアップして、十分近い組み合わせが存在するかを調べる。またその最小値を返す）
        """

        if self.Curves is None:
            return False  
        
        # 曲線の始点と終点をリストアップ
        points = []

        for curve in self.Curves:
            # LineSegment の場合
            if isinstance(curve, LineSegment):
                p0 = curve.p0
                p1 = curve.p1
                points.append(p0)
                points.append(p1)

            # CircularArc の場合
            elif isinstance(curve, CircularArc):
                center = curve.center
                radius = curve.radius
                theta_start = curve.theta_start
                theta_end = curve.theta_end

                p0 = (center[0] + radius * np.cos(np.deg2rad(theta_start)), center[1] + radius * np.sin(np.deg2rad(theta_start)))
                p1 = (center[0] + radius * np.cos(np.deg2rad(theta_end)), center[1]  + radius * np.sin(np.deg2rad(theta_end)))
                points.append(p0)
                points.append(p1)


            # Circle の場合 
            pass

        #各点にたいして、最も近い点を調べて、その時の組み合わせを記録する
        max_min_distance = 0
        for i in range(len(points)):
            
            min_distance = float('inf')
            for j in range( len(points)):

                if i == j:
                    continue
                distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (self.Curves[int(i/2)], self.Curves[int(j/2)])

            if max_min_distance < min_distance:
                max_min_distance = min_distance
                max_closest_pair = closest_pair

        if max_min_distance < tol:
            return True
        else:
            print("this is not closed")
            print("max min distance: ", max_min_distance)
            print("closest pair: ", max_closest_pair)
            return False






        
            

        
    
    def to_dict(self) -> dict:
        """Vessel オブジェクトを辞書形式に変換する．"""

        
        Lines = [curve for curve in self.Curves if isinstance(curve, LineSegment)]
        Arcs = [curve for curve in self.Curves if isinstance(curve, CircularArc)]
        Circles = [curve for curve in self.Curves if isinstance(curve, Circle)]

        data = {
            "lines": [asdict(line) for line in Lines],
            "arcs": [asdict(arc) for arc in Arcs],
            "circles": [asdict(circle) for circle in Circles],
        }
        
        return data
    
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AxisymmetricVessel':
        """辞書形式から Vessel オブジェクトを生成する．"""
        Lines = [LineSegment(**line) for line in data["lines"]]
        Arcs = [CircularArc(**arc) for arc in data["arcs"]]
        Circles = [Circle(**circle) for circle in data["circles"]]
        Curves = Lines + Arcs + Circles
        return cls(Curves)
    
    @classmethod    
    def load_from_dxf(cls, filename: str) -> 'AxisymmetricVessel':
        """DXF ファイルから Container オブジェクトを生成する．"""
        dxf = dxfgrabber.readfile(filename)
        lines = []
        arcs = []
        circles = []

        for entity in dxf.entities:
            
            if entity.dxftype == 'LINE':
                lines.append(LineSegment(p0=(entity.start[0]/1000, entity.start[1]/1000),
                                          p1=(entity.end[0]/1000, entity.end[1]/1000)))
            elif entity.dxftype == 'ARC':
                arcs.append(CircularArc(center=(entity.center[0]/1000, entity.center[1]/1000),
                                        radius=entity.radius/1000,
                                        theta_start=entity.start_angle,
                                        theta_end=entity.end_angle))
            elif entity.dxftype == 'CIRCLE':
                circles.append(Circle(center=(entity.center[0]/1000, entity.center[1]/1000),
                                      radius=entity.radius/1000))
                
        Curves = lines + arcs + circles
                
        return cls(Curves)
    
    def __str__(self):
        #すべてのリストごとに改行して表示
        return "\n".join([str(curve) for curve in self.Curves])
    
    def plot(self,
        ax:plt.Axes = None,
        label:bool=False,   
        **kwargs,
        ):
        if ax is None:
            ax = plt.gca()

            
        default_kwargs = {"linewidth":1, "alpha":1.0, "color":'gray'}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        i = 0
        if 'fontsize' in kwargs:
            fontsize = kwargs.pop('fontsize')  
        else:
            fontsize = plt.rcParams['legend.fontsize']
            
        
        for line in self.Lines:
            ax.plot([line.p0[0], line.p1[0]], [line.p0[1], line.p1[1]], **kwargs)

            i += 1
            
            if label:
                #ax.text(self.all_lines[i].start[0], self.all_lines[i].start[1], "L."+str(i), size = 10, color = "blue")
                ax.text(line.p0[0], line.p0[1], "L."+str(i), color = "blue", fontsize=fontsize)
        

        for arc in self.Arcs:
            patch = patches.Arc((arc.center[0], arc.center[1]), 2*arc.radius, 2*arc.radius,
                                theta1=arc.theta_start, theta2=arc.theta_end, fill=False, **kwargs)
            ax.add_patch(patch)
            i += 1
            if label:
                #x = self.all_arcs[i].center[0]/1000+  self.all_arcs[i].radius/1000*np.cos(np.pi* self.all_arcs[i].end_angle/180)
                #y = self.all_arcs[i].center[1]/1000+  self.all_arcs[i].radius/1000*np.sin(np.pi* self.all_arcs[i].end_angle/180)
                x = arc.center[0] + arc.radius*np.cos(arc.theta_end*np.pi/180)
                y = arc.center[1] + arc.radius*np.sin(arc.theta_end*np.pi/180)
                ax.text(x, y, "A."+str(i), color = "red", fontsize=fontsize)

        for circle in self.Circles:
            ax.add_patch(patches.Circle((circle.center[0], circle.center[1]), circle.radius, fill=False))
            i += 1
            if label:
                x = circle.center[0] + circle.radius
                y = circle.center[1]
                ax.text(x, y, "C."+str(i), color = "green",fontsize=fontsize)

        return ax
    
    
    def detect_grid(
        self,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
        fill_start_point: Tuple[float,float] = None,
        isnt_print: bool = False,
        static : bool = False,
        ):

        """
        Detects and processes a grid based on input radial and axial coordinates, 
        identifying boundaries and filling regions within the grid.
        Parameters:
        -----------
        r_grid : np.ndarray
            1D or 2D array representing the radial coordinates of the grid.
        z_grid : np.ndarray
            1D or 2D array representing the axial coordinates of the grid.
        fill_start_point : Tuple[float, float], optional
            Starting point for the flood fill operation, specified as (r, z). 
            If None, the mean of the grid coordinates is used. Default is None.
        isnt_print : bool, optional
            If True, disables progress bar printing during line and arc detection. 
            Default is False.
        Returns:
        --------
        NaN_factor : np.ndarray
            A 2D array where grid cells inside the detected region are set to 1.0, 
            and cells outside are set to NaN.
        imshow_extent : Tuple[float, float, float, float]
            Extent of the grid for visualization, given as 
            (r_min, r_max, z_min, z_max).
        Notes:
        ------
        - This method processes both lines and arcs defined in the `self.Lines` 
          and `self.Arcs` attributes to determine grid boundaries.
        - The method uses OpenCV's `floodFill` to fill regions within the grid.
        - The `NaN_factor` array can be used for masking or visualization purposes.
        """
        
        import cv2
        from scipy import signal



        if len(r_grid.shape) == 2:
            if abs(r_grid[-1,0]-r_grid[0,0]) < 1e-3:
                r_grid = r_grid[0,:]
            else :
                r_grid = r_grid[:,0]
                
        if len(z_grid.shape) == 2:
            if abs(z_grid[-1,0]-z_grid[0,0]) < 1e-3:
                z_grid = z_grid[0,:]
            else :
                z_grid = z_grid[:,0]

        h,w = z_grid.size,r_grid.size
        R_extend = np.empty(r_grid.size+1,dtype=np.float64)  
        Z_extend = np.empty(z_grid.size+1,dtype=np.float64)
        R_extend[0] =  r_grid[0]  - 0.5* (r_grid[1]-r_grid[0])
        R_extend[-1] = r_grid[-1] + 0.5* (r_grid[-1]-r_grid[-2])
        R_extend[1:-1] = 0.5 * (r_grid[:-1] + r_grid[1:])
        Z_extend[0] =  z_grid[0]  - 0.5* (z_grid[1]-z_grid[0])
        Z_extend[-1] = z_grid[-1] + 0.5* (z_grid[-1]-z_grid[-2])
        Z_extend[1:-1] = 0.5 * (z_grid[:-1] + z_grid[1:])


        RR,ZZ = np.meshgrid(R_extend,Z_extend,indexing='xy')
        R_tr = RR[+1:,+1:]
        Z_tr = ZZ[+1:,+1:]
        R_tl = RR[+1:,:-1]
        Z_tl = ZZ[+1:,:-1]
        R_br = RR[:-1,+1:]
        Z_br = ZZ[:-1,+1:]
        R_bl = RR[:-1,:-1]
        Z_bl = ZZ[:-1,:-1]
        
        list1 = ['top','bottom','left','right']

        is_bound = np.zeros((h,w),np.bool_)

        for i in tqdm(range(len(self.Lines)),desc='Lines detection', disable=isnt_print):
            #R4, Z4 = self.all_lines[i].start[0]/1000, self.all_lines[i].start[1]/1000
            #R3, Z3 = self.all_lines[i].end[0]/1000, self.all_lines[i].end[1]/1000

            R4,Z4 = self.Lines[i].p0[0], self.Lines[i].p0[1]
            R3,Z3 = self.Lines[i].p1[0], self.Lines[i].p1[1]
        
            # if 文からmatch文に変更 python >= 3.10
            for mode in ['top','bottom','left','right']:
                R1,R2,Z1,Z2 = 0,0,0,0 # unbound で怒られたから回避( 意味のない処理 ) 

                match mode:
                    case 'top':
                        R1 = R_tr
                        R2 = R_tl
                        Z1 = Z_tr 
                        Z2 = Z_tl
                    case 'bottom':
                        R1 = R_br
                        R2 = R_bl
                        Z1 = Z_br 
                        Z2 = Z_bl
                    case 'right':
                        R1 = R_tr
                        R2 = R_br
                        Z1 = Z_tr 
                        Z2 = Z_br
                    case 'left':
                        R1 = R_tl
                        R2 = R_bl
                        Z1 = Z_tl
                        Z2 = Z_bl

                D = (R4-R3) * (Z2-Z1) - (R2-R1) * (Z4-Z3)
                W1, W2 = Z3*R4-Z4*R3, Z1*R2 - Z2*R1

                D_is_0 = (D <= 1e-10)

                D += D_is_0 * 1e-10
            
                R_inter = ( (R2-R1) * W1 - (R4-R3) * W2 ) / D
                Z_inter = ( (Z2-Z1) * W1 - (Z4-Z3) * W2 ) / D
                del W1,W2,D
                
                is_in_Rray_range = (R2 - R_inter) * (R1 - R_inter) <= 1.e-8
                is_in_Zray_range = (Z2 - Z_inter) * (Z1 - Z_inter) <= 1.e-8
                is_in_Rfra_range = (R4 - R_inter) * (R3 - R_inter) <= 1.e-8 
                is_in_Zfra_range = (Z4 - Z_inter) * (Z3 - Z_inter) <= 1.e-8
                is_in_range =  np.logical_and(is_in_Rray_range,is_in_Zray_range) * np.logical_and(is_in_Rfra_range,is_in_Zfra_range) 
                # 水平や垂直  な線に対応するため

                is_bound  += is_in_range

    
        for i in tqdm(range(len(self.Arcs)),desc='Arcs  detection',disable=isnt_print):
            #Rc, Zc =(self.all_arcs[i].center[0]/1000, self.all_arcs[i].center[1]/1000)
            #radius = self.all_arcs[i].radius/1000
            #sta_angle, end_angle = self.all_arcs[i].start_angle ,self.all_arcs[i].end_angle 
            Rc, Zc = self.Arcs[i].center[0], self.Arcs[i].center[1]
            radius = self.Arcs[i].radius
            sta_angle, end_angle = self.Arcs[i].theta_start ,self.Arcs[i].theta_end

        
            # if 文からmatch文に変更 python >= 3.10
            for mode in ['top','bottom','left','right']:
                R1,R2,Z1,Z2 = 0,0,0,0 # unbound で怒られたから回避( 意味のない処理 ) 
                
                match mode:
                    case 'top':
                        R1 = R_tr
                        R2 = R_tl
                        Z1 = Z_tr 
                        Z2 = Z_tl
                    case 'bottom':
                        R1 = R_br
                        R2 = R_bl
                        Z1 = Z_br 
                        Z2 = Z_bl
                    case 'right':
                        R1 = R_tr
                        R2 = R_br
                        Z1 = Z_tr 
                        Z2 = Z_br
                    case 'left':
                        R1 = R_tl
                        R2 = R_bl
                        Z1 = Z_tl
                        Z2 = Z_bl

                lR = R2-R1
                lZ = Z2-Z1
                S  = R2*Z1 - R1*Z2      

                D = (lR**2+lZ**2)*radius**2 + 2*lR*lZ*Rc*Zc - 2*(lZ*Rc-lR*Zc)*S - lR**2 *Zc**2 -lZ**2*Rc**2-S**2 #判別式
                exist = D > 0

                Ri1 = (lR**2 *Rc + lR*lZ *Zc - lZ *S + lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のR座標
                Zi1 = (lZ**2 *Zc + lR*lZ *Rc + lR *S + lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# １つ目の交点のZ座標
                Ri2 = (lR**2 *Rc + lR*lZ *Zc - lZ *S - lR * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のR座標
                Zi2 = (lZ**2 *Zc + lR*lZ *Rc + lR *S - lZ * np.sqrt(D*exist) ) / (lR**2 + lZ**2)  * exist# 2つ目の交点のZ座標
                del D, exist

                is_in_ray_range1  = np.logical_and((R2 - Ri1) * (R1 - Ri1) <= 1e-8 ,(Z2 - Zi1) * (Z1- Zi1) <= 1e-7) # 交点1が線分内にあるか判定
                is_in_ray_range2  = np.logical_and((R2 - Ri2) * (R1 - Ri2) <= 1e-8 ,(Z2 - Zi2) * (Z1- Zi2) <= 1e-7) # 交点2が線分内にあるか判定


                cos1 = (Ri1-Rc) / radius
                sin1 = (Zi1-Zc) / radius
                atan = np.arctan2(sin1,cos1)
                theta1 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)
                cos2 = (Ri2-Rc) / radius    
                sin2 = (Zi2-Zc) / radius 
                atan = np.arctan2(sin2,cos2)
                theta2 = np.where(atan > 0, 180/np.pi*atan, 360+180/np.pi*atan)

                del cos1,sin1,atan,cos2,sin2

                is_in_arc1 =  (end_angle - theta1) * (sta_angle - theta1) * (end_angle-sta_angle) <= 1e-7 # 交点1が弧の範囲内あるか判定
                is_in_arc2 =  (end_angle - theta2) * (sta_angle - theta2) * (end_angle-sta_angle) <= 1e-7 # 交点1が弧の範囲内あるか判定

                is_real_intercept1 = is_in_ray_range1 * is_in_arc1
                is_real_intercept2 = is_in_ray_range2 * is_in_arc2
                is_real_intercept  = is_real_intercept1 + is_real_intercept2
                
                is_bound += is_real_intercept


        mask = np.zeros((h,w), np.uint8)
        print('')
        if fill_start_point is None:
            fill_start_point = (r_grid.mean(),z_grid.mean())
            print(f"fill_start_point is None, so use {fill_start_point}")
        else:
            print(f"fill_start_point is {fill_start_point}")

        # 塗りつぶしの開始インデクスを探索
        i_r,i_z = 0,0
        for i in range(r_grid.size-1):
            if (r_grid[i] - fill_start_point[0])*(r_grid[i+1] - fill_start_point[0]  ) <= 1e-8:
                i_r =  i 
                break 
        for i in range(z_grid.size-1):
            if (z_grid[i] - fill_start_point[1])*(z_grid[i+1] - fill_start_point[1]  ) <= 1e-8:
                i_z =  i 
                break 

        #print(i_r,i_z)

        fill =  np.zeros((h,w), np.uint8)
        fill[:,:] = 1 *is_bound[:,:]                        
        mask = np.zeros((h+2, w+2), np.uint8)
        
        cv2.floodFill(fill, mask, (i_r,i_z), 2)


        Is_in = (fill == 2)
        Is_out = (fill == 0)
        mask = signal.convolve2d((fill ==2),np.ones([3,3]),mode='same') > 1.e-5

        NaN_factor = np.where(mask , 1.0, np.nan)
        imshow_extent = (R_extend[0],R_extend[-1],Z_extend[0],Z_extend[-1])

        if static:
            self.fill = fill
            self.NaN_factor = NaN_factor
            self.imshow_extent = imshow_extent
            self.Is_bound = is_bound
            self.Is_In = Is_in
            self.Is_Out = Is_out

        else:
            pass 

        
        return NaN_factor,imshow_extent
    
    def plot_3d_surface(self, ax=None, start_angle=0, end_angle=360, num_points=50,color=None,alpha=0.5):
        """
        Plots a 3D surface by rotating the RZ cross-sectional lines and arcs around the Z-axis.

        Parameters:
        -----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            The 3D axis to plot on. If None, a new figure and axis are created.
        start_angle : float, optional
            The starting angle of rotation in degrees. Default is 0.
        end_angle : float, optional
            The ending angle of rotation in degrees. Default is 360.
        num_points : int, optional
            The number of points to use for the rotation. Default is 100.
        color : str, optional
            The color of the surface. Default is None.
        alpha : float, optional
            The transparency of the surface. Default is 0.5.
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import pyplot as plt
        import numpy as np

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Convert angles to radians
        start_angle_rad = np.radians(start_angle)
        end_angle_rad = np.radians(end_angle)

        # Generate angles for rotation
        theta = np.linspace(start_angle_rad, end_angle_rad, num_points)

        # Plot lines
        for line in self.Lines:
            r = np.array([line.p0[0], line.p1[0]])
            z = np.array([line.p0[1], line.p1[1]])
            r_grid, theta_grid = np.meshgrid(r, theta)
            z_grid, _ = np.meshgrid(z, theta)

            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)
            if color is None:
                _color = 'blue'
            else:
                _color = color
            ax.plot_surface(x, y, z_grid, alpha=alpha, color=_color, edgecolor='none')

        # Plot arcs
        for arc in self.Arcs:
            theta_start = np.radians(arc.theta_start)
            theta_end = np.radians(arc.theta_end)

            if theta_start > theta_end:
                theta_end += 2 * np.pi
            theta_arc = np.linspace(theta_start, theta_end, num_points)

            theta_arc_grid, theta_grid = np.meshgrid(theta_arc, theta)

            r_grid = arc.radius * np.cos(theta_arc_grid) + arc.center[0]
            z_grid = arc.radius * np.sin(theta_arc_grid) + arc.center[1]

            x = r_grid * np.cos(theta_grid)
            y = r_grid * np.sin(theta_grid)

            z = z_grid

            if color is None:
                _color = 'red'
            else:
                _color = color

            ax.plot_surface(x, y, z, alpha=alpha, color=_color, edgecolor='none')




        # Plot circles
        for circle in self.Circles:
            theta_circle = np.linspace(0, 2 * np.pi, num_points)
            r = circle.radius
            z = circle.center[1]

            x = r * np.cos(theta_circle)
            y = r * np.sin(theta_circle)
            z = np.full_like(x, z)

            if color is None:
                _color = 'green'
            else:
                _color = color

            ax.plot(x, y, z, color=_color, alpha=alpha)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratio to be equal
        max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max() / 2.0
        mid_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
        mid_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
        mid_z = (ax.get_zlim()[0] + ax.get_zlim()[1]) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_title('3D Surface Plot of Axisymmetric Vessel') 

        return ax




