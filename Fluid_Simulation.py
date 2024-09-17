
from ast import List
import taichi as ti

ti.init(arch=ti.vulkan)

# 一般物理常数定义
pi = 3.14159265
K = 500     # 状态方程的常数k
viscosity = 1e-6    # 粘性系数
gravity = ti.Vector([0,0,-9.8])

# 粒子物理量定义
mass = 1e-6
n = 100
quad_size = 1.0 / n
density0 = 1
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

x = ti.Vector.field(3,dtype=float,shape=(n,n,n))
v = ti.Vector.field(3,dtype=float,shape=(n,n,n))
density = ti.field(dtype=float,shape=(n,n,n))
pressure = ti.field(dtype=float,shape=(n,n,n))
F = ti.Vector.field(3,dtype=float,shape=(n,n,n))
Neighbor_List = []


# 哈希表定义
Hashing_List = [None] * n**3
Compact_Hashing_List = set()

# 核函数
def kernel_fun(q) -> ti.f32:
    if q>=0 and q<1:
        return (2/3 - q*q + 0.5 * q*q*q)* 3/2/pi * 1/(quad_size**3)
    elif q>=1 and q<2:
        return ((2-q)**3/6) * 3/2/pi * 1/(quad_size**3)
    elif q>=2:
        return 0
    
def grad_kernel_fun(vi: ti.template(), vj: ti.template()) -> ti.Vector:
    q = Vector_Distance(vi,vj) / quad_size
    if q>=0 and q<1:
        return (-2*q + 3/2*q**2) * 3/2/pi/quad_size**5 / q * (vj - vi)
    elif q>=1 and q<2:
        return (-(2-q)**2/2) * 3/2/pi/quad_size**5 / q * (vj - vi)
    elif q>=2:
        return ti.Vector([0,0,0])
    
def Vector_Distance(v1: ti.template(), v2: ti.template()) -> ti.f32:
    diff = v1 - v2
    return diff.norm()
    
# 使用了Z-Order原理的哈希方法进行索引排序
@ti.kernel
def particle_Zindex_Sort():
    Hashing_List.clear
    Compact_Hashing_List.clear
    d = quad_size/n**3 # 尺度缩放比例
    for i,j,k in x:
        r = x[i,j,k]
        # 由公式计算Z曲线原理的哈希值
        Z_curve_index = (  (r[0]/d    * n**2  ) * 73856093 
                         ^ (r[1]/d    * n     ) * 19349663 
                         ^ (r[2]/d            ) * 83492791) % n**3
        index = ti.Vector([i,j,k])
        Hashing_List[Z_curve_index].append(index)
        Compact_Hashing_List.add(Z_curve_index)
        
# 初始化粒子位置和速度
@ti.kernel
def init_particle():
    for i,j,k in x:
        x[i,j,k] = [
            i * quad_size,
            j * quad_size,
            k * quad_size
         ]
        v[i,j,k]=[0,0,0]
        
init_particle()

# 邻域粒子搜索
@ti.kernel
def Neighbor_Search(target_index:ti.template()):
    Neighbor_List.clear
    Z_curve_index = Hashing_List.index([target_index]) 
    # 邻域设定为±1内的哈希箱
    Neighbor_Hashing_List = Hashing_List[Z_curve_index] + Hashing_List[Z_curve_index-1] + Hashing_List[Z_curve_index+1]
    for j in Neighbor_Hashing_List:
        # 2倍quad_size 为核函数有效距离
        distance = Vector_Distance(target_index,j)
        if  distance <= 2 * quad_size and distance > 0:
            i0 = target_index[0]
            j0 = target_index[1]
            k0 = target_index[2]
            Neighbor_List.append(j)



# 使用的算法是最基础的利用状态方程的SPH方法
@ti.kernel
def GetDensityAndPressure():
    for i,j,k in x:
        # 由插值公式计算密度
        i_density = 0
        Neighbor_Search(ti.Vector([i,j,k]))
        for neibor in Neighbor_List:
            q = Vector_Distance(x[i,j,k],neibor) / quad_size
            i_density += mass * kernel_fun(q) # 由插值公式求和计算密度
            
        density[i,j,k] = i_density
        pressure[i,j,k] = K * ((i_density/density0)**7 - 1)  # 通过密度由状态方程计算压强
        
@ti.kernel
def GetForce():
    for i,j,k in x:
        i_grad_pressure = 0
        i_laplace_v = 0
        Neighbor_Search(ti.Vector([i,j,k])) # 在此步骤中需要重新进行邻域搜索，不太合理，还需要进一步改进
        for neibor in Neighbor_List[i,j,k]:
            i1 = neibor[0]
            j1 = neibor[1]
            k1 = neibor[2]
            i_grad_pressure += ti.Vector((pressure[i,j,k]/(density[i,j,k]**2) 
                                          + pressure[i1,j1,k1]/(density[i1,j1,k1]**2)) 
                                         * grad_kernel_fun(x[i,j,k],x[i1,j1,k1]))   # 由插值公式求和计算压强场的梯度
            xij = ti.Vector(x[i,j,k]-x[i1,j1,k1])
            # 由插值公式求和计算速度场在拉普拉斯算符作用后的值
            i_laplace_v += (v[i,j,k].norm() - v[i1,j1,k1].norm) / density[i1,j1,k1] * xij.dot(grad_kernel_fun(x[i,j,k],x[i1,j1,k1])) / (xij.norm **2 + 0.01 * quad_size**2)
            
        # 补足求和时省略的倍率
        i_grad_pressure *= density[i,j,k] * mass
        i_laplace_v *= 2*mass
        
        # 由流体力学公式计算粒子受力
        F_pressure_i = -mass / density[i,j,k] * i_grad_pressure
        F_viscosity_i = mass * viscosity * i_laplace_v
        F[i,j,k] = F_pressure_i + F_viscosity_i + mass * gravity
        
@ti.kernel
def Substep():
    for i,j,k in x:
        v[i,j,k] = v[i,j,k] + dt * F[i,j,k] / mass
        x[i,j,k] = x[i,j,k] + dt * v[i,j,k]


print(kernel_fun(1))