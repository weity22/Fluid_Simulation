
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

# 一般物理常数定义
pi = 3.14159265
K = 3.0e5     # 状态方程的常数k
viscosity = 0.05    # 粘性系数
gravity = ti.Vector([0.0,0.0,-9.8])

# 流体粒子物理量定义
n = 50
L = 2.0
quad_size = L / n
s = 0.68
density0 = 1.0
mass = quad_size**3 * density0

v_m = 5.0
dt = quad_size / v_m / 10

x = ti.Vector.field(3,dtype=float,shape=(n,n,n))
v = ti.Vector.field(3,dtype=float,shape=(n,n,n))
density = ti.field(dtype=float,shape=(n,n,n))
pressure = ti.field(dtype=float,shape=(n,n,n))
a = ti.Vector.field(3,dtype=float,shape=(n,n,n))
Neighbor_Grid_Hashing_List = ti.field(dtype=ti.i32,shape=(n,n,n,27))
Neighbor_Grid_buckets = ti.field(dtype=ti.i32,shape=(n,n,n))
Neighbor_List = ti.Vector.field(3,dtype=ti.i32, shape=(n,n,n,100))
Neighbor_Count = ti.field(dtype=ti.i32,shape=(n,n,n))

W = ti.field(dtype=float,shape=(n,n,n))

#边界粒子定义

Boundary_x = ti.Vector.field(3,dtype=float,shape=(n+2,n+2,14))   # 边界容器尺寸为n*2n*n，拆分为8个n*n的表面（不封顶）
Nei_boundary_List = ti.Vector.field(3,dtype=ti.i32, shape=(n,n,n,20))   # 粒子的邻域边界粒子
Nei_boundary_Count = ti.field(dtype=ti.i32,shape=(n,n,n))

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
        W[i,j,k] = (2/3 - q*q + 0.5 * q*q*q)* 3/2/pi * 1/((quad_size/2.0)**3)
    elif q>=1 and q<2:
        W[i,j,k] = ((2-q)**3/6) * 3/2/pi * 1/((quad_size/2.0)**3)
    elif q>=2:
        W[i,j,k] = 0.0
    
@ti.func
def grad_kernel_fun(vi: ti.template(), vj: ti.template())->ti.Vector:
    q = Vector_Distance(vi,vj) / (quad_size/2)
    delta_W = ti.Vector([0.0,0.0,0.0])
    if q>=0 and q<2:
        delta_W = -45.0 / 8 / pi / (quad_size/2.0)**5 / q * (1 - q/2)**5 * (vi - vj)
        
    elif q>=2:
        delta_W = ti.Vector([0,0,0])
        
    return delta_W
    
@ti.func
def Vector_Distance(v1: ti.template(), v2: ti.template()) -> ti.f32:
    diff = v1 - v2
    return diff.norm()

# 初始化粒子位置和速度
@ti.kernel
def init_particle():
    for i,j,k in x:
        x[i,j,k] = [
            (i+1) * s * quad_size,
            (j+1) * s * quad_size,
            (k+1) * s * quad_size
            ]
        v[i,j,k]=[0.0,0.0,0.0]

@ti.kernel
def init_boundary():
    for i,j,k in Boundary_x:
        if k == 0:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                0.0,
                j * s * quad_size
                ]
        
        if k == 1:
            Boundary_x[i,j,k] = [
                (n+2) * s * quad_size,
                (i+1) * s * quad_size,
                (j+1) * s * quad_size
                ]
            
        if k == 2:
            Boundary_x[i,j,k] = [
                (n+2) * s * quad_size,
                (n+2 + i+1) * s * quad_size,
                (j+1) * s * quad_size
                ]
        
        if k == 3:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                2 * (n+2) * s * quad_size + 1 * s * quad_size,
                j * s * quad_size
                ]
            
        if k == 4:
            Boundary_x[i,j,k] = [
                0.0,
                (n+2 + i+1) * s * quad_size,
                (j+1) * s * quad_size
                ]
            
        if k == 5:
            Boundary_x[i,j,k] = [
                0.0,
                (i+1) * s * quad_size,
                (j+1) * s * quad_size
                ]
            
        if k == 6:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                (j+1) * s * quad_size,
                0.0
                ]
            
        if k == 7:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                (n+2) * s * quad_size + (j+1) * s * quad_size,
                0.0
                ]
            
        if k == 8:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                0.0,
                (n+2) * s * quad_size + j * s * quad_size
                ]
            
        if k == 9:
            Boundary_x[i,j,k] = [
                (n+2) * s * quad_size,
                (i+1) * s * quad_size,
                (n+2) * s * quad_size + (j+1) * s * quad_size
                ]
            
        if k == 10:
            Boundary_x[i,j,k] = [
                (n+2) * s * quad_size,
                (n+2 + i+1) * s * quad_size,
                (n+2) * s * quad_size + (j+1) * s * quad_size
                ]
         
        if k == 11:
            Boundary_x[i,j,k] = [
                i * s * quad_size,
                2 * (n+2) * s * quad_size + 1 * s * quad_size,
                (n+2) * s * quad_size + j * s * quad_size
                ]
            
        if k == 12:
            Boundary_x[i,j,k] = [
                0.0,
                (n+2 + i+1) * s * quad_size,
                (n+2) * s * quad_size + (j+1) * s * quad_size
                ]
            
        if k == 13:
            Boundary_x[i,j,k] = [
                0.0,
                (i+1) * s * quad_size,
                (n+2) * s * quad_size + (j+1) * s * quad_size
                ]

@ti.kernel
def print_boundary():
    for i,j,k in Boundary_x:
        print('boundary:',ti.Vector([i,j,k]))
        print('x:',Boundary_x[i,j,k])
    

# 使用了Z-Order原理的哈希方法进行索引排序
@ti.kernel
def HashingList_clear():
    for i,j in Hashing_List:
        Hashing_List[i,j] = ti.Vector([0,0,0])
        
@ti.kernel
def buckets_clear():
    for i in buckets:
        buckets[i] = 0;
   

@ti.kernel
def particle_Zindex_Sort():
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
@ti.kernel
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
    #Z_curve_index = Z_index_List[i0,j0,k0]
    d = quad_size # 尺度缩放比例
    
    r = x[i0,j0,k0]
    # 计算周围27格内，每个格子对应的Z-index，以便后面邻域搜索
    for i in range(-1,2):
        if r[0]/d + i < 0 or r[0]/d + i > n-1:
            continue
        for j in range(-1,2):
            if r[1]/d + j < 0 or r[1]/d + j > 2*(n-1):
                continue
            for k in range(-1,2):
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
            # quad_size 为核函数有效距离   
            #print('Neighbor_Z_index = ',Neighbor_Grid_Hashing_List[i0,j0,k0,i])
            #print('buckets:',buckets[Neighbor_Grid_Hashing_List[i0,j0,k0,i]]) 
            
            ###获取每一个哈希表上的粒子的index，它们都是可能的邻居，再判断是否为邻居
            i1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][0]
            j1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][1]
            k1 = Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j][2]
            #print('PossibleNeighbor:',Hashing_List[Neighbor_Grid_Hashing_List[i0,j0,k0,i],j])
            distance = Vector_Distance(x[i0,j0,k0],x[i1,j1,k1])
            #print('quad_size_distance:',distance/quad_size)
            if  distance <= 1.1 * quad_size and distance > 0:
                
                #print('Neighbor:',ti.Vector([i1,j1,k1]))
                Append_To_NeighborList(i0,j0,k0,i1,j1,k1)
            #print('\n')
    #注释代码为调试用
                
@ti.func
def print_NeighborList(i,j,k):
    print('For ',[i,j,k])
    for index in range(Neighbor_Count[i,j,k]):
        print('Neighbor:',Neighbor_List[i,j,k,index])

## 邻域的边界粒子搜索：

@ti.kernel
def Nei_boundary_List_clear():
    for i,j,k in Nei_boundary_Count:
        for index in range(Nei_boundary_Count[i,j,k]):
            Nei_boundary_List[i,j,k,index] = ti.Vector([0,0,0])
        Nei_boundary_Count[i,j,k] = 0
    
@ti.func
def Nei_boundary_Search(i0:ti.i32,j0:ti.i32,k0:ti.i32):
    r = x[i0,j0,k0]
    
    if r[0] <= quad_size:
        for i in range(n+2):
            for j in range(n+2):
                if Vector_Distance(r,Boundary_x[i,j,4]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,4])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,5]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,5])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,12]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,12])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,13]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,13])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
    elif r[0] >= L - quad_size:
        for i in range(n+2):
            for j in range(n+2):
                if Vector_Distance(r,Boundary_x[i,j,1]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,1])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,2]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,2])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,9]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,9])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,10]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,10])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
    if r[1] <= quad_size:
        for i in range(n+2):
            for j in range(n+2):
                if Vector_Distance(r,Boundary_x[i,j,0]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,0])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,8]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,8])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    

                    
    elif r[1] >= 2 * (n+2) * quad_size - quad_size:
        for i in range(n+2):
            for j in range(n+2):
                if Vector_Distance(r,Boundary_x[i,j,3]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,3])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,11]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,11])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
    if r[2] <= 2 * quad_size:
        for i in range(n+2):
            for j in range(n+2):
                if Vector_Distance(r,Boundary_x[i,j,6]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,6])
                    Nei_boundary_Count[i0,j0,k0] += 1
                    
                elif Vector_Distance(r,Boundary_x[i,j,7]) <= quad_size:
                    Nei_boundary_List[i0,j0,k0,Nei_boundary_Count[i0,j0,k0]] = ti.Vector([i,j,7])
                    Nei_boundary_Count[i0,j0,k0] += 1
                   
@ti.func
def print_Nei_boundary_List(i0,j0,k0):
    for index in range(Nei_boundary_Count[i0,j0,k0]):
        print('No.',index)
        print('Nei_boundary:',Nei_boundary_List[i0,j0,k0,index])

# 使用的算法是最基础的利用状态方程的SPH方法
@ti.kernel
def GetDensityAndPressure():
    
    for i,j,k in x:
        # 由插值公式计算密度        
        i_density = 0.0
        Neighbor_Search(i,j,k)
        Nei_boundary_Search(i,j,k)
        #if i==10 and j==10 and k==10:
        #    print_NeighborList(i,j,k)
        for index in range(Neighbor_Count[i,j,k]):  #每个粒子对应一个邻居表，这个函数返回[i,j,k]对应的粒子的邻居数量
            r = x[
            Neighbor_List[i,j,k,index][0],
            Neighbor_List[i,j,k,index][1],
            Neighbor_List[i,j,k,index][2]
                ]
            q = Vector_Distance(x[i,j,k],r) / (quad_size/2.0)
            kernel_fun(q,i,j,k) #核函数，结果返回到W上

            i_density += mass * W[i,j,k]    # 由插值公式求和计算密度
        
        for index in range(Nei_boundary_Count[i,j,k]):  #对于靠近边界的粒子，计算密度需要补上修正项
            r = Boundary_x[
            Nei_boundary_List[i,j,k,index][0],
            Nei_boundary_List[i,j,k,index][1],
            Nei_boundary_List[i,j,k,index][2]
                ]
            q = Vector_Distance(x[i,j,k],r) / (quad_size/2.0)
            kernel_fun(q,i,j,k) #核函数，结果返回到W上

            i_density += mass * W[i,j,k]    # 由插值公式求和计算密度
            
        ##if i_density < 1.0:
        ##    i_density = 1.0
        density[i,j,k] = i_density
        pressure[i,j,k] = K * ((i_density/density0)**7 - 1.0)     # 通过密度由状态方程计算压强
        if pressure[i,j,k] < 0.0:
            pressure[i, j, k] = 0.0
        if i == 10 and j == 10 and k == 10: 
           print('density = ',density[i,j,k])
           print('pressure = ',pressure[i,j,k])
        


@ti.kernel
def GetForce():
    for i,j,k in x:
        #i_grad_pressure = ti.Vector([0.0,0.0,0.0])
        #i_laplace_v = 0.0
        #if i == 10 and j == 10 and k == 10: 
        #    print_Nei_boundary_List(i,j,k)
        a_pressure_i = ti.Vector([0.0,0.0,0.0])
        a_viscosity_i = ti.Vector([0.0,0.0,0.0])
        for index in range(Neighbor_Count[i,j,k]):
            i1 = Neighbor_List[i,j,k,index][0]
            j1 = Neighbor_List[i,j,k,index][1]
            k1 = Neighbor_List[i,j,k,index][2]
            delta_W = grad_kernel_fun(x[i,j,k],x[i1,j1,k1])
            #if i == 10 and j == 10 and k == 10: 
            #    print('delta_W1 = ',delta_W)
            # 由插值公式求和计算压强场的梯度
            #i_grad_pressure += (pressure[i,j,k]/density[i,j,k]**2 + 
            #                    pressure[i1,j1,k1]/density[i1,j1,k1]**2) * delta_W
            a_pressure_i += mass * pressure[i1,j1,k1] / (density[i,j,k]*density[i1,j1,k1]) * delta_W
            xij = x[i,j,k]-x[i1,j1,k1]
            vij = v[i,j,k]-v[i1,j1,k1]
            q = Vector_Distance(x[i,j,k],x[i1,j1,k1]) / (quad_size/2.0)
            kernel_fun(q,i,j,k) #核函数，结果返回到W上
            a_viscosity_i += -mass * viscosity * W[i,j,k] * vij / density[i1,j1,k1] / dt
            #xij = x[i,j,k]-x[i1,j1,k1]
            # 由插值公式求和计算速度场在拉普拉斯算符作用后的值
            #i_laplace_v += (v[i,j,k].norm() - v[i1,j1,k1].norm()) / density[i1,j1,k1] * xij.dot(delta_W) / (xij.norm() **2 + 0.01 * quad_size**2)
            
        a_p_FromBoundary = ti.Vector([0.0,0.0,0.0])
        a_v_FromBoundary = ti.Vector([0.0,0.0,0.0])
        for index in range(Nei_boundary_Count[i,j,k]):
            i1 = Nei_boundary_List[i,j,k,index][0]
            j1 = Nei_boundary_List[i,j,k,index][1]
            k1 = Nei_boundary_List[i,j,k,index][2]
            delta_W = grad_kernel_fun(x[i,j,k],Boundary_x[i1,j1,k1])
            if i == 10 and j == 10 and k == 10: 
                print('delta_W2 = ',delta_W)
            #计算边界支持力对粒子的加速度贡献
            a_p_FromBoundary += mass * pressure[i,j,k] / density[i,j,k]**2 * delta_W
            #计算边界粘性力对粒子的加速度贡献
            xij = x[i,j,k]-Boundary_x[i1,j1,k1]
            q = Vector_Distance(x[i,j,k],x[i1,j1,k1]) / (quad_size/2.0)
            kernel_fun(q,i,j,k) #核函数，结果返回到W上
            a_v_FromBoundary += -mass * viscosity * W[i,j,k] * v[i,j,k] / density[i,j,k] / dt

        #补足求和时省略的倍率
        #i_grad_pressure *= density[i,j,k] * mass
        
        #i_laplace_v *= 2*mass
        
        # 由流体力学公式计算粒子加速度
        #a_pressure_i = -1.0 / density[i,j,k] * i_grad_pressure
        #a_viscosity_i = mass * viscosity * i_laplace_v
        a[i,j,k] = a_pressure_i + a_viscosity_i + a_p_FromBoundary + a_v_FromBoundary + gravity
        
        #'''
        if i == 10 and j == 10 and k == 10: 
            print('For ',[i,j,k])
            print('a_pressure_i = ',a_pressure_i)
            print('a_viscosity_i = ',a_viscosity_i)
            print('a_p_FromBoundary = ',a_p_FromBoundary)
            print('a_v_FromBoundary = ',a_v_FromBoundary)
            print('a = ',a[i,j,k])
        #'''
        
        
        

@ti.kernel
def print_density(i0:ti.i32,j0:ti.i32,k0:ti.i32):
    print('density in:',[i0,j0,k0])
    print('density = ',density[i0,j0,k0])
        
@ti.kernel
def Substep():
    for i,j,k in x:
        v[i,j,k] = v[i,j,k] + dt * a[i,j,k]
        v1 = v[i,j,k]
        if v1.norm() > v_m:
            v1 = v1 / v1.norm() * v_m
            v[i,j,k] = v1
        x[i,j,k] = x[i,j,k] + dt * v[i,j,k]
        
    
    
@ti.kernel
def print_info(i:ti.i32,j:ti.i32,k:ti.i32):
    print('a = ',a[i,j,k])
    print('v = ',v[i,j,k])
    print('x = ',x[i,j,k])

@ti.kernel
def test():
    for i,j,k in x:
        Neighbor_Search(i,j,k)

@ti.kernel
def check()->bool:
    bol = True
    for i,j,k in x:
        r = x[i,j,k]
        if r[0]<0.0-1.1 * quad_size or r[0]> (n+1) * quad_size + 1.1 * quad_size:
            print('x',[i,j,k])
            print('=',x[i,j,k])
            bol = False
        if r[1]<0.0-1.1 * quad_size or r[1]>(2 * (n+2)) * quad_size + 1.1 * quad_size:
            print('x',[i,j,k])
            print('=',x[i,j,k])
            bol = False
        if r[2]<0.0-1.1 * quad_size:
            print('x',[i,j,k])
            print('=',x[i,j,k])
            bol = False
    return bol
        
if __name__ == "__main__":
    
    num_vertices = n**3
    series_prefix = "test.ply"
    init_particle()
    init_boundary()
    
    for step in range(1,30000):
        if step % 1 == 0:
            Nei_boundary_List_clear()
            NeighborList_clear()
            HashingList_clear()
            buckets_clear()
            particle_Zindex_Sort()
            GetDensityAndPressure()

        GetForce()
        Substep()
        if step % 200 == 1:
            print("Step:",step)
            print_info(10,10,n-1)

        if step % 200 == 1:
            # 当前只支持通过传递单个 np.array 来添加通道
            # 所以需要转换为 np.ndarray 并且 reshape
            # 记住使用一个临时变量来存储，这样你就不必再转换回来
            np_pos = np.reshape(x.to_numpy(), (num_vertices, 3))
            # 创建一个 PLYWriter 对象
            writer = ti.tools.PLYWriter(num_vertices=num_vertices)

            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
            writer.export_frame_ascii(step//200+1, series_prefix)
            
        #bol = check()
        #if not bol:
        #    print("Failed!")
    