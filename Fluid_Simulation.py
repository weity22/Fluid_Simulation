
import taichi as ti

ti.init(arch=ti.cpu)

# 一般物理常数定义
pi = 3.14159265
K = 500     # 状态方程的常数k
viscosity = 1e-6    # 粘性系数
gravity = ti.Vector([0,0,-9.8])

# 流体粒子物理量定义
mass = 1e-6
n = 10
quad_size = 1.0 / n
density0 = 1
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

x = ti.Vector.field(3,dtype=float,shape=(n,n,n))
v = ti.Vector.field(3,dtype=float,shape=(n,n,n))
density = ti.field(dtype=float,shape=(n,n,n))
pressure = ti.field(dtype=float,shape=(n,n,n))
F = ti.Vector.field(3,dtype=float,shape=(n,n,n))
Neighbor_Grid_Hashing_List = ti.field(dtype=ti.i32,shape=(n,n,n,125))
Neighbor_Grid_buckets = ti.field(dtype=ti.i32,shape=(n,n,n))
Neighbor_List = ti.Vector.field(3,dtype=ti.i32, shape=(n,n,n,100))
Neighbor_Count = ti.field(dtype=ti.i32,shape=(n,n,n))

W = ti.field(dtype=float,shape=(n,n,n))
delta_W = ti.Vector.field(3,dtype=float,shape=1)

#边界粒子定义

Boundary_x = ti.Vector.field(3,dtype=float,shape=(n,n,8))   # 边界容器尺寸为n*2n*n，拆分为8个n*n的表面（不封顶）
Nei_boundary_List = ti.Vector.field(3,dtype=ti.i32, shape=20)   # 粒子的邻域边界粒子
Nei_boundary_Count = ti.field(dtype=ti.i32,shape=1)

# 哈希表定义
Hashing_List = ti.Vector.field(3,dtype=ti.i32, shape=(n**3, 100))
Z_index_List = ti.field(dtype=ti.i32,shape=(n,n,n))
buckets = ti.field(dtype=ti.i32, shape=n**3)  # 用于存储每个桶当前的元素数量

@ti.func
def Append_To_HashingList(bucket_index, item):
    # 获取当前桶的元素数量
    current_size = buckets[bucket_index]
    # 将新元素添加到当前桶
    Hashing_List[bucket_index, current_size] = item
    # 更新桶的元素数量
    buckets[bucket_index] += 1

ti.func
def Append_To_List(v,item,List,buckets):
    i = v[0]
    j = v[1]
    k = v[2]
    List[i,j,k,buckets[i,j,k]] = item
    buckets[i,j,k] += 1


# 核函数
@ti.func    
def kernel_fun(q,i,j,k):
    if q>=0 and q<1:
        W[i,j,k] = (2/3 - q*q + 0.5 * q*q*q)* 3/2/pi * 1/(quad_size**3)
    elif q>=1 and q<2:
        W[i,j,k] = ((2-q)**3/6) * 3/2/pi * 1/(quad_size**3)
    elif q>=2:
        W[i,j,k] = 0
    
@ti.func
def grad_kernel_fun(vi: ti.template(), vj: ti.template()) -> ti.Vector:
    q = Vector_Distance(vi,vj) / quad_size
    if q>=0 and q<1:
        return (-2*q + 3/2*q**2) * 3/2/pi/quad_size**5 / q * (vj - vi)
    elif q>=1 and q<2:
        return (-(2-q)**2/2) * 3/2/pi/quad_size**5 / q * (vj - vi)
    elif q>=2:
        return ti.Vector([0,0,0])
    
@ti.func
def Vector_Distance(v1: ti.template(), v2: ti.template()) -> ti.f32:
    diff = v1 - v2
    return diff.norm()
    
# 初始化粒子位置和速度
@ti.kernel
def init_particle():
    for i,j,k in x:
        x[i,j,k] = [
            0.5 * quad_size + i * quad_size,
            0.5 * quad_size + j * quad_size,
            0.5 * quad_size + k * quad_size
            ]
        v[i,j,k]=[0,0,0]

@ti.kernel
def init_boundary():
    for i,j,k in Boundary_x:
        if k == 0:
            Boundary_x[i,j,k] = [
                i * quad_size,
                0,
                j * quad_size
                ]
        
        if k == 1 or k == 2:
            Boundary_x[i,j,k] = [
                1.0,
                k-1 + i * quad_size,
                j * quad_size
                ]
        
        if k == 3:
            Boundary_x[i,j,k] = [
                i * quad_size,
                2.0,
                j * quad_size
                ]
            
        if k == 4 or k == 5:
            Boundary_x[i,j,k] = [
                0,
                5-k + i * quad_size,
                j * quad_size
                ]
            
        if k == 6 or k == 7:
            Boundary_x[i,j,k] = [
                i * quad_size,
                k-6 + j * quad_size,
                0
                ]

@ti.kernel
def print_boundary():
    for i,j,k in Boundary_x:
        print('boundary:',ti.Vector([i,j,k]))
        print('x:',Boundary_x[i,j,k])
    

# 使用了Z-Order原理的哈希方法进行索引排序
@ti.func
def HashingList_clear():
    for i,j in Hashing_List:
        Hashing_List[i,j] = ti.Vector([0,0,0])
        
@ti.func
def buckets_clear():
    for i in buckets:
        buckets[i] = 0;
   

@ti.kernel
def particle_Zindex_Sort():
    HashingList_clear()
    buckets_clear()
    # List_clear(Compact_Hashing_List)
    d = quad_size # 尺度缩放比例
    for i,j,k in x:
        r = x[i,j,k]
        # 由公式计算Z曲线原理的哈希值
        Z_curve_index = (  (int((r[0]/d)        ) * 73856093) 
                         ^ (int((r[1]/d)        ) * 19349663) 
                         ^ (int((r[2]/d)        ) * 83492791)   ) % n**3
        #print('Zindex:',Z_curve_index)
        Append_To_HashingList(Z_curve_index,[i,j,k])
        Z_index_List[i,j,k] = Z_curve_index
        # Compact_Hashing_List.add(Z_curve_index)

@ti.kernel
def print_HashingList():
    for i in range(n**3):
        if buckets[i]!=0:
            for j in range(buckets[i]):
                print('Z_index:',i,' conver ',Hashing_List[i,j])
            

# 邻域粒子搜索
@ti.func
def NeighborList_clear():
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for index in range(Neighbor_Count[i,j,k]):
                    Neighbor_List[i,j,k,index] = ti.Vector([0,0,0])
                Neighbor_Count[i,j,k] = 0
                
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for index in range(Neighbor_Grid_buckets[i,j,k]):
                    Neighbor_Grid_Hashing_List[i,j,k,index] = 0
                Neighbor_Grid_buckets[i,j,k] = 0
                
@ti.func
def Append_To_NeighborList(i,j,k,i1,j1,k1):
    Neighbor_List[i,j,k,Neighbor_Count[i,j,k]] = ti.Vector([i1,j1,k1])
    Neighbor_Count[i,j,k] += 1

@ti.func
def Neighbor_Search(i0:ti.i32,j0:ti.i32,k0:ti.i32):
    NeighborList_clear()
    #Z_curve_index = Z_index_List[i0,j0,k0]
    d = quad_size # 尺度缩放比例
    
    r = x[i0,j0,k0]
    # 计算周围125格内，每个格子对应的Z-index，以便后面邻域搜索
    for i in range(-2,3):
        if r[0]/d + i < 0 or r[0]/d + i > n-1:
            continue
        for j in range(-2,3):
            if r[1]/d + j < 0 or r[1]/d + j > 2*(n-1):
                continue
            for k in range(-2,3):
                if r[2]/d + k < 0 or r[2]/d + k > n-1:
                    continue
                r = x[i0,j0,k0]
                Neighbor_Z_curve_index = (   (int((r[0]/d + i)      ) * 73856093) 
                                     ^       (int((r[1]/d + j)      ) * 19349663) 
                                     ^       (int((r[2]/d + k)      ) * 83492791)   ) % n**3
                #print('Neighbor_Z_index = ', Neighbor_Z_curve_index)
                Neighbor_Grid_Hashing_List[i0,j0,k0,Neighbor_Grid_buckets[i0,j0,k0]] = Neighbor_Z_curve_index
                Neighbor_Grid_buckets[i0,j0,k0] += 1
    
    #用上面储存的Z-index查询邻居
    for i in range(Neighbor_Grid_buckets[i0,j0,k0]):
        for j in range(buckets[Neighbor_Grid_Hashing_List[i0,j0,k0,i]]):
            # 2倍quad_size 为核函数有效距离   
            #print('Neighbor_Z_index = ',Neighbor_Grid_Hashing_List[i0,j0,k0,i])
            #print('buckets:',buckets[Neighbor_Grid_Hashing_List[i0,j0,k0,i]]) 
            
            ###获取每一个哈希表上的粒子的index，它们都是可能的邻居，再判断是否为邻居
            i1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][0]
            j1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][1]
            k1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][2]
            #print('PossibleNeighbor:',Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j])
            distance = Vector_Distance(x[i0,j0,k0],x[i1,j1,k1])
            #print('quad_size_distance:',distance/quad_size)
            if  distance <= 2 * quad_size and distance > 0:
                #print('Neighbor:',ti.Vector([i1,j1,k1]))
                Append_To_NeighborList(i0,j0,k0,i1,j1,k1)
            #print('\n')
                
    #print_NeighborList()
    #注释代码为调试用
                
@ti.func
def print_NeighborList(i,j,k):
    for index in range(Neighbor_Count[i,j,k]):
        print('For ',[i,j,k])
        print('Neighbor:',Neighbor_List[i,j,k,index])

## 邻域的边界粒子搜索：

@ti.func
def Nei_boundary_List_clear():
    for i in range(Nei_boundary_Count[0]):
        Nei_boundary_List[i] = ti.Vector([0,0,0])
    
    Nei_boundary_Count[0] = 0

@ti.kernel
def Nei_boundary_Search(i0:ti.i32,j0:ti.i32,k0:ti.i32):
    Nei_boundary_List_clear()
    r = x[i0,j0,k0]
    
    if r[0] <= 2 * quad_size:
        for i in range(n):
            for j in range(n):
                if Vector_Distance(r,Boundary_x[i,j,4]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,4])
                    Nei_boundary_Count[0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,5]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,5])
                    Nei_boundary_Count[0] += 1
                    
    elif r[0] >= 1.0 - 2 * quad_size:
        for i in range(n):
            for j in range(n):
                if Vector_Distance(r,Boundary_x[i,j,1]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,1])
                    Nei_boundary_Count[0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,2]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,2])
                    Nei_boundary_Count[0] += 1
                    
    if r[1] <= 2 * quad_size:
        for i in range(n):
            for j in range(n):
                if Vector_Distance(r,Boundary_x[i,j,0]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,0])
                    Nei_boundary_Count[0] += 1
                    
    elif r[1] >= 2.0 - 2 * quad_size:
        for i in range(n):
            for j in range(n):
                if Vector_Distance(r,Boundary_x[i,j,3]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,3])
                    Nei_boundary_Count[0] += 1
                    
    if r[2] <= 2 * quad_size:
        for i in range(n):
            for j in range(n):
                if Vector_Distance(r,Boundary_x[i,j,6]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,6])
                    Nei_boundary_Count[0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,7]) <= 2 * quad_size:
                    Nei_boundary_List[Nei_boundary_Count[0]] = ti.Vector([i,j,7])
                    Nei_boundary_Count[0] += 1
                   
@ti.kernel
def print_Nei_boundary_List():
    for i in range(Nei_boundary_Count[0]):
        print('No.',i)
        print('Nei_boundary:',Nei_boundary_List[i])

# 使用的算法是最基础的利用状态方程的SPH方法
@ti.kernel
def GetDensityAndPressure():
    
    '''换成下面这段循环可以换为串行
    for l in range(1):
        for i in range(n):
            for j in range(n):
                for k in range(n):
    '''
    for i,j,k in x:
        # 由插值公式计算密度        
        i_density = 0.0
        Neighbor_Search(i,j,k)      #感觉主要是这个函数里面出现了竞争
        print_NeighborList(i,j,k)
        for index in range(Neighbor_Count[i,j,k]):  #每个粒子对应一个邻居表，这个函数返回[i,j,k]对应的粒子的邻居数量
            r = x[
            Neighbor_List[i,j,k,index][0],
            Neighbor_List[i,j,k,index][1],
            Neighbor_List[i,j,k,index][2]
                ]
            q = Vector_Distance(x[i,j,k],r) / quad_size
            kernel_fun(q,i,j,k) #核函数，结果返回到W上
            #if i == 5 and j == 0 and k == 3:
                
            #    print('r = ',r)
            #    print('W = ',W[i,j,k])

        i_density += mass * W[i,j,k]    # 由插值公式求和计算密度
            
        density[i,j,k] = i_density
        pressure[i,j,k] = K * ((i_density/density0)**7 - 1)     # 通过密度由状态方程计算压强
        
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
def print_density(i0:ti.i32,j0:ti.i32,k0:ti.i32):
    print('density in:',[i0,j0,k0])
    print('density = ',density[i0,j0,k0])
        
@ti.kernel
def Substep():
    for i,j,k in x:
        v[i,j,k] = v[i,j,k] + dt * F[i,j,k] / mass
        x[i,j,k] = x[i,j,k] + dt * v[i,j,k]
        
if __name__ == "__main__":
    init_particle()
    init_boundary()
    particle_Zindex_Sort()
    #print_HashingList()
    #Neighbor_Search(55,10,10)
    #print_NeighborList()
    
    #print_boundary()
    #Nei_boundary_Search(0,99,50)
    #print_Nei_boundary_List()
    GetDensityAndPressure()
    print_density(5,0,3)
    