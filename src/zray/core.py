import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numba import njit
#import container
from . import vessel

__all__ = ["intersect_rays_line", "intersect_rays_arc"]



def intersect_rays(
        Curve:vessel.RZCurve,    
        Origins:np.ndarray,
        Directions:np.ndarray,
        tol:float=1e-6
    )->np.ndarray:
    
    """
    光線 O+t*D と曲線の交点 t を求める．
    条件を満たす t がない場合は -1.0 を返す．
    Args:
        Curve: container.RZCurve
            曲線
        Origins: np.ndarray
            光線の原点の配列（N,3）
        Directions: np.ndarray
            光線の方向の配列（N,3）
    """

    if isinstance(Curve, vessel.LineSegment):
        return intersect_rays_line(Curve, Origins, Directions)
    elif isinstance(Curve, vessel.CircularArc):
        return intersect_rays_arc(Curve, Origins, Directions, tol)
    else:
        raise ValueError("Invalid curve type")

def intersect_rays_line(
        Line:vessel.LineSegment,
        Origins:np.ndarray, 
        Directions:np.ndarray
    )->np.ndarray:

    """
    光線 O+t*D と線分（p0=(r0,z0)から p1=(r1,z1)）の交点 t を求める．
    条件を満たす t がない場合は -1.0 を返す．
    Args:
        Line: container.LineSegment
            線分
        Origins: np.ndarray
            光線の原点の配列（N,3）
        Directions: np.ndarray
            光線の方向の配列（N,3）
    """
    return _intersect_rays_line(Origins, Directions, Line.p0[0], Line.p0[1], Line.p1[0], Line.p1[1])

def intersect_rays_arc(
        CircularArc:vessel.CircularArc,
        Origins:np.ndarray, 
        Directions:np.ndarray,
        tol:float=1e-6  
    )->np.ndarray:

    """
    光線 O+t*D と円弧の交点 t を求める．
    条件を満たす t がない場合は -1.0 を返す．
    Args:
        CircularArc: container.CircularArc
            円弧
        Origins: np.ndarray
            光線の原点の配列（N,3）
        Directions: np.ndarray
            光線の方向の配列（N,3）
    """

    N = Origins.shape[0]
    coeffs = compute_quartic_coefficients(Origins, Directions, CircularArc.center[0], CircularArc.center[1], CircularArc.radius)
    ts = np.empty(N, dtype=np.float64)
    roots = np.empty((N, 4), dtype=np.complex128)
    for i in range(N):
        roots[i] = np.roots(coeffs[i])
        ts[i] = select_valid_t(roots[i], Origins[i], Directions[i],
                               CircularArc.center[0], CircularArc.center[1], CircularArc.radius,
                               CircularArc.theta_start*np.pi/180, CircularArc.theta_end*np.pi/180, tol)
    return ts





@njit
def _quadratic_smallest_positive(a, b, c, tol=1e-6):
    # a t^2 + b t + c = 0 を解く．
    # a がゼロに近ければ線形解を返す
    if abs(a) < tol:
        if abs(b) < tol:
            return -1.0
        t_lin = -c / b
        return t_lin if t_lin > tol else -1.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return -1.0
    sqrt_disc = math.sqrt(disc)
    sol1 = (-b - sqrt_disc) / (2.0 * a)
    sol2 = (-b + sqrt_disc) / (2.0 * a)
    sol = 1e10  # 大きな値で初期化
    if sol1 > tol and sol1 < sol:
        sol = sol1
    if sol2 > tol and sol2 < sol:
        sol = sol2
    if sol < 1e9:
        return sol
    else:
        return -1.0

@njit
def _intersect_rays_line(origins, directions, r0, z0, r1, z1, tol=1e-6):
    """
    origins, directions は (N,3) の配列．
    各光線について線分との交点 t を計算し、存在しなければ -1.0 を返す．
    """

    def _intersect_ray_single(O, D, r0, z0, r1, z1, tol=1e-6):
        """
        光線 O+t*D と RZ平面上の線分（p0=(r0,z0)から p1=(r1,z1)）から回転して得られる面の交点 t を求める．
        交点が存在すれば t の値を、存在しなければ -1.0 を返す．
        """
        Ox, Oy, Oz = O[0], O[1], O[2]
        Dx, Dy, Dz = D[0], D[1], D[2]
        t_solution = -1.0
        if abs(z1 - z0) > tol:
            A_line = (r1 - r0) / (z1 - z0)
            a_coef = (Dx*Dx + Dy*Dy) - (A_line*Dz)**2
            b_coef = 2.0*(Ox*Dx + Oy*Dy) - 2.0*A_line*Dz*(r0 + A_line*(Oz - z0))
            c_coef = (Ox*Ox + Oy*Oy) - (r0 + A_line*(Oz - z0))**2
            sol = _quadratic_smallest_positive(a_coef, b_coef, c_coef, tol)
            if sol > tol:
                lam = (Oz + sol*Dz - z0) / (z1 - z0)
                if lam >= -tol and lam <= 1.0 + tol:
                    t_solution = sol
        else:
            # z 座標が一定の場合
            if abs(Dz) > tol:
                t = (z0 - Oz) / Dz
                if t > tol:
                    x = Ox + t*Dx
                    y = Oy + t*Dy
                    r = math.sqrt(x*x + y*y)
                    if r >= min(r0, r1) - tol and r <= max(r0, r1) + tol:
                        t_solution = t
        return t_solution
    
    N = origins.shape[0]
    ts = np.empty(N, dtype=np.float64)
    for i in range(N):
        O = origins[i]
        D = directions[i]
        ts[i] = _intersect_ray_single(O, D, r0, z0, r1, z1, tol)
    return ts





# ============================================================================
# (1) 複数光線に対して 4 次方程式の係数を計算する関数（Numba版）
# ============================================================================
@njit
def compute_quartic_coefficients(origins, directions, r_c, z_c, R_val):
    """
    各光線について、以下の 4 次多項式の係数を計算する。
    
      F(t) = coeff4*t^4 + coeff3*t^3 + coeff2*t^2 + coeff1*t + coeff0 = 0
    
    ここで、各係数は以下のように定義される（以前の展開結果に基づく）：
      - まず、光線 O+t*D の (x,y) 成分から
            a2 = Dx^2 + Dy^2
            a1 = 2*(Ox*Dx + Oy*Dy)
            a0 = Ox^2 + Oy^2
      - また、(Oz+t*Dz) と円の z 座標 z_c に関して、以下を定義
            A_Q = a2 + Dz^2
            B_Q = a1 + 2*Dz*(Oz - z_c)
            C_Q = a0 + r_c^2 + (Oz - z_c)^2 - R_val^2
      - そして最終的に
            coeff4 = A_Q^2
            coeff3 = 2*A_Q*B_Q
            coeff2 = B_Q^2 + 2*A_Q*C_Q - 4*r_c^2*a2
            coeff1 = 2*B_Q*C_Q - 4*r_c^2*a1
            coeff0 = C_Q^2 - 4*r_c^2*a0
    """
    N = origins.shape[0]
    coeffs = np.empty((N, 5), dtype=np.float64)
    for i in range(N):
        O = origins[i]
        D = directions[i]
        Ox, Oy, Oz = O[0], O[1], O[2]
        Dx, Dy, Dz = D[0], D[1], D[2]
        a2 = Dx*Dx + Dy*Dy
        a1 = 2.0 * (Ox*Dx + Oy*Dy)
        a0 = Ox*Ox + Oy*Oy

        A_Q = a2 + Dz*Dz
        B_Q = a1 + 2.0 * Dz * (Oz - z_c)
        C_Q = a0 + r_c*r_c + (Oz - z_c)**2 - R_val*R_val

        coeffs[i, 0] = A_Q * A_Q                            # coeff4
        coeffs[i, 1] = 2.0 * A_Q * B_Q                       # coeff3
        coeffs[i, 2] = B_Q * B_Q + 2.0 * A_Q * C_Q - 4.0 * r_c * r_c * a2  # coeff2
        coeffs[i, 3] = 2.0 * B_Q * C_Q - 4.0 * r_c * r_c * a1              # coeff1
        coeffs[i, 4] = C_Q * C_Q - 4.0 * r_c * r_c * a0                      # coeff0
    return coeffs

# ============================================================================
# (2) 各光線について np.roots で解いた後を入力して、条件を満たす最小の t を返す関数
# ============================================================================
@njit
def select_valid_t(roots:np.ndarray, O: np.ndarray, D: np.ndarray,
                    r_c: float, z_c: float, R_val: float,
                    theta_start: float, theta_end: float, tol: float = 1e-6) -> float:
    """
    # 与えられた 4 次方程式の係数 coeff（長さ5 の 1次元配列）に対して np.roots を用い、
    得られた４つの複素数解から、以下の条件を満たすものを選び、最も小さい t を返す。
    
    (a) t > tol
    (b) 解が実数に近い（imaginary part の絶対値が tol 未満）
    (c) 光線 O+t*D により得られる交点の (x,y,z) から r = sqrt(x^2+y^2) を計算し、
        円弧の角度 φ = atan2(z - z_c, r - r_c)（0～2πに正規化）が
        指定された角度範囲 [theta_start, theta_end] 内にある。
    
    条件を満たす解がなければ -1.0 を返す。
    """
    sols = roots  # ４つの複素解（np.roots は複素数配列を返すO
    N = O.shape[0]

    t_valid = 1e10
    for sol in sols:
        if abs(sol.imag) < tol and sol.real > tol:
            t_val = sol.real
            if t_val < t_valid:
                # 交点を計算
                x = O[0] + t_val * D[0]
                y = O[1] + t_val * D[1]
                z = O[2] + t_val * D[2]
                r = math.sqrt(x*x + y*y)
                # 円弧上の角度 φ（0～2πに正規化）
                phi = math.atan2(z - z_c, r - r_c)
                if phi < 0.0:
                    phi += 2.0 * math.pi
                # theta_start, theta_end も 0～2π に正規化
                ts_angle = theta_start if theta_start >= 0.0 else theta_start + 2.0*math.pi
                te_angle = theta_end if theta_end >= 0.0 else theta_end + 2.0*math.pi
                valid = False
                if ts_angle <= te_angle:
                    if phi >= ts_angle - tol and phi <= te_angle + tol:
                        valid = True
                else:
                    # 角度が 0 をまたぐ場合
                    if phi >= ts_angle - tol or phi <= te_angle + tol:
                        valid = True
                if valid:
                    t_valid = t_val
    if t_valid < 1e10:
        return t_valid
    else:
        return -1.0

