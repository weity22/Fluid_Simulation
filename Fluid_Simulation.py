
import taichi as ti
ti.init(arch=ti.vulkan)

# һ������������
pi = 3.14159265
k = 500
gravity = ti.Vector([0, -9.8, 0])

# ��������������
mass = 1e-6
n = 100
quad_size = 1.0 / n
density0 = 1
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

x = ti.Vector.field(3,dtype=float,shape=(n,n,n))
v = ti.Vector.field(3,dtype=float,shape=(n,n,n))

# ��ϣ����
Hashing_List = [None] * n**3
Compact_Hashing_List = set()

# �˺���
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
    
# ʹ����Z-Orderԭ��Ĺ�ϣ����������������
@ti.kernel
def particle_Zindex_Sort():
    Hashing_List.clear
    Compact_Hashing_List.clear
    d = quad_size/n**3 # �߶����ű���
    for i,j,k in x:
        r = x[i,j,k]
        # �ɹ�ʽ����Z����ԭ��Ĺ�ϣֵ
        Z_curve_index = (  (r[0]/d    * n**2  ) * 73856093 
                         ^ (r[1]/d    * n     ) * 19349663 
                         ^ (r[2]/d            ) * 83492791) % n**3
        Hashing_List[Z_curve_index].append([i,j,k])
        Compact_Hashing_List.add(Z_curve_index)
        
# ��ʼ������λ�ú��ٶ�
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

# ������������
@ti.kernel
def Neighbor_Search(target_index:ti.template()):
    Neighbor_List = []
    Z_curve_index = Hashing_List.index([target_index]) 
    # �����趨Ϊ��1�ڵĹ�ϣ��
    Neighbor_Hashing_List = Hashing_List[Z_curve_index] + Hashing_List[Z_curve_index-1] + Hashing_List[Z_curve_index+1]
    for j in Neighbor_Hashing_List:
        # 2��quad_size Ϊ�˺�����Ч����
        distance = Vector_Distance(target_index,j)
        if  distance <= 2 * quad_size and distance > 0:
            Neighbor_List.append(j)
            
    return Neighbor_List


# ʹ�õ��㷨�������������״̬���̵�SPH����
@ti.kernel
def substep():
    for i,j,k in x:
        Neighbor_List = Neighbor_Search(x[i,j,k])
        # �ɲ�ֵ��ʽ�����ܶ�
        i_density = 0
        for neibor in Neighbor_List:
            q = Vector_Distance(x[i,j,k],neibor) / quad_size
            i_density += mass * kernel_fun(q) 
        
        i_pressure = k * ((i_density/density0)**7 - 1)  # ͨ���ܶ���״̬���̼���ѹǿ
        

        

        
        
    
    
    


    




print(kernel_fun(1))