
import taichi as ti
ti.init(arch=ti.vulkan)

# 一般物理常数定义
pi = 3.14159265
k = 500
gravity = ti.Vector([0, -9.8, 0])

# 粒子物理量定义
mass = 1e-6
n = 100
quad_size = 1.0 / n
density0 = 1
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

x = ti.Vector.field(3,dtype=float,shape=(n,n,n))
v = ti.Vector.field(3,dtype=float,shape=(n,n,n))

# 哈希表定义
Hashing_List = [None] * n**3
Compact_Hashing_List = set()

# 核函数
def kernel_fun(q):
    if q>=0 and q<1:
        return (2/3 - q*q + 0.5 * q*q*q)*3/2/pi
    elif q>=1 and q<2:
        return ((2-q)**3/6)*3/2/pi
    elif q>=2:
        return 0
    
@ti.kernel
def Vector_Distance(v1: ti.template(), v2: ti.template()):
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
        Hashing_List[Z_curve_index].append([i,j,k])
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
    Neighbor_List = []
    Z_curve_index = Hashing_List.index([target_index]) 
    # 邻域设定为±1内的哈希箱
    Neighbor_Hashing_List = Hashing_List[Z_curve_index] + Hashing_List[Z_curve_index-1] + Hashing_List[Z_curve_index+1]
    for j in Neighbor_Hashing_List:
        # 2倍quad_size 为核函数有效距离
        distance = Vector_Distance(target_index,j)
        if  distance <= 2 * quad_size and distance > 0:
            Neighbor_List.append(j)
            
    return Neighbor_List


# 使用的算法是最基础的利用状态方程的SPH方法
@ti.kernel
def substep():
    for i,j,k in x:
        Neighbor_List = Neighbor_Search(x[i,j,k])
        # 由插值公式计算密度
        i_density = 0
        for neibor in Neighbor_List:
            q = Vector_Distance(x[i,j,k],neibor) / quad_size
            i_density += mass * kernel_fun(q) 
        
        i_pressure = k * ((i_density/density0)**7 - 1)  # 通过密度由状态方程计算压强
        

        

        
        
    
    
    


    




print(kernel_fun(1))