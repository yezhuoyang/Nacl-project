from numpy import *
from math import *
import pickle
import random
class position:            
    def create(self,positionlist,n,m):
         positionlist=list(positionlist)
         self.xna=positionlist[0:n-1]
         self.xna.insert(0,0)
         self.yna=positionlist[n-1:2*n-2]
         self.yna.insert(0,0)
         self.zna=positionlist[2*n-2:3*n-3]
         self.zna.insert(0,0)
         self.xcl=positionlist[3*n-3:3*n-3+m]
         self.ycl=positionlist[3*n-3+m:3*n-4+2*m]
         self.ycl.insert(0,0)
         self.zcl=positionlist[3*n-4+2*m:3*n+3*m-5]
         self.zcl.insert(0,0)
         self.n=n
         self.m=m
         self.vector=array(positionlist)
    def leastL(self):
         Llist=[]
         for i in range(0,self.n):
            for j in range(0,self.m):
                r=0
                r+=(self.xna[i]-self.xcl[j])**2
                r+=(self.yna[i]-self.ycl[j])**2
                r+=(self.zna[i]-self.zcl[j])**2
                Llist.append(r)
         return max(Llist)
    def center(self):
          self.naxcenter=average(self.xna)
          self.naycenter=average(self.yna)
          self.nazcenter=average(self.zna)
          self.clxcenter=average(self.xcl)
          self.clycenter=average(self.ycl)
          self.clzcenter=average(self.zcl)
          self.naxstd=std(self.xna)
          self.naystd=std(self.yna)
          self.nazstd=std(self.zna)
          self.clxstd=std(self.xcl)
          self.clystd=std(self.ycl)
          self.clzstd=std(self.zcl)
         
r0=0.330                       #距离的单位为A
V0=1.09*10**3                   
gamma=14.3997
eps=10**(-8)                   #规定误差上限
'''
为了避免某些对称的trivial的解出现，将原点设置为某个钠离子的点，并规定第一个氯离子一定在x轴上
'''
'''
数组的归一化
'''
def norm(vector):
    sum=0
    for x in vector:
        sum+=x**2
    sum=sum**0.5
    return sum
    
    
'''
返回钠离子和氯离子的能量，输入值为两个位置向量
'''
def energydiff(position1,position2):             
     r=0
     for i in range(0,3):
         r+=(position1[i]-position2[i])**2
     r=sqrt(r)
     if r==0:
         raise        
     energy=-gamma/r+V0*exp(-r/r0)
     return energy
'''
返回相同离子的能量
'''
def energysame(position1,position2):
     r=0
     for i in range(0,3):
         r+=(position1[i]-position2[i])**2
     r= sqrt(r)
     energy=gamma/r
     return energy
'''
返回位置2对位置1的（相同电荷）力的相反数，实际上就是梯度
'''

def gradsame(position1,position2):
     r=0
     for i in range(0,3):
         r+=(position1[i]-position2[i])**2
     r=sqrt(r)
     Fx=gamma*(position1[0]-position2[0])/r**3
     Fy=gamma*(position1[1]-position2[1])/r**3
     Fz=gamma*(position1[2]-position2[2])/r**3
     return [-Fx,-Fy,-Fz]

'''
 返回不同电荷产生的梯度
'''
def graddiff(position1,position2):
     r=0
     for i in range(0,3):
         r+=(position1[i]-position2[i])**2
     r= sqrt(r)
     Fx=gamma*(position2[0]-position1[0])/r**3+V0/(r0*r)*(position1[0]-position2[0])*exp(-r/r0)
     Fy=gamma*(position2[1]-position1[1])/r**3+V0/(r0*r)*(position1[1]-position2[1])*exp(-r/r0)
     Fz=gamma*(position2[2]-position1[2])/r**3+V0/(r0*r)*(position1[2]-position2[2])*exp(-r/r0)
     return [-Fx,-Fy,-Fz]
''' 
自变量为 3n+3m-5维向量[xna2,..xnan,yna2,....ynan,zna2,....znan,xcl1,..xclm,ycl2,...yclm,zcl2,..zclm]
输入一组个位置类，输出总能量
'''           
def tolenergy(a):
    energy=0
    for i in range(0,a.n):
        for j in range(i+1,a.n):
            energy+=energysame([a.xna[i],a.yna[i],a.zna[i]],[a.xna[j],a.yna[j],a.zna[j]])
    for i in range(0,a.m):
        for j in range(i+1,a.m):
            energy+=energysame([a.xcl[i],a.ycl[i],a.zcl[i]],[a.xcl[j],a.ycl[j],a.zcl[j]])
    for i in range(0,a.n):
        for j in range(0,a.m):
            energy+=energydiff([a.xna[i],a.yna[i],a.zna[i]],[a.xcl[j],a.ycl[j],a.zcl[j]])
    return energy

def tolgrad(a):
    gradxna=[]
    gradyna=[]
    gradzna=[]
    gradxcl=[]
    gradycl=[]
    gradzcl=[]
    for i in range(1,a.n):                     #计算第i个钠离子受到的梯度
        gx=0
        gy=0
        gz=0
        for j in range(0,a.m):
            grad=graddiff([a.xna[i],a.yna[i],a.zna[i]],[a.xcl[j],a.ycl[j],a.zcl[j]])
            gx+=grad[0]
            gy+=grad[1]
            gz+=grad[2]
        for j in range(0,a.n):
            if j==i:
                break
            grad=gradsame([a.xna[i],a.yna[i],a.zna[i]],[a.xna[j],a.yna[j],a.zna[j]])
            gx+=grad[0]
            gy+=grad[1]
            gz+=grad[2]
        gradxna.append(gx)
        gradyna.append(gy)
        gradzna.append(gz)
    for i in range(0,a.m):                 #计算第i个氯离子的梯度
         gx=0
         gy=0
         gz=0
         for j in range(0,a.n):
            grad=graddiff([a.xcl[i],a.ycl[i],a.zcl[i]],[a.xna[j],a.yna[j],a.zna[j]])
            gx+=grad[0]
            gy+=grad[1]
            gz+=grad[2]
         for j in range(0,a.m):
            if j==i:
                break
            grad=gradsame([a.xcl[i],a.ycl[i],a.zcl[i]],[a.xcl[j],a.ycl[j],a.zcl[j]])
            gx+=grad[0]
            gy+=grad[1]
            gz+=grad[2]
         gradxcl.append(gx)
         gradycl.append(gy)
         gradzcl.append(gz)
    gradycl=gradycl[1:]
    gradzcl=gradzcl[1:]
    tolgrad=gradxna+gradyna+gradzna+gradxcl+gradycl+gradzcl
    return array(tolgrad)

'''
输入移动方向，移动距离，和类型，输出移动后的梯度与d1的内积
'''
def grads(s,a,d1):
        b=position()
        position.create(b,a.vector+s*d1,a.n,a.m)         
        return dot(d1,tolgrad(b))

'''
二分求根法法确定使得 h*grad =0的s，一开始的步长取为上一次的步长
'''
def finds(a,d1,laststep):
       sdown=0
       sup=laststep
       while abs(sup-sdown)>0.000001:
            if grads((sup+sdown)/2,a,d1)>0:
                sup=(sup+sdown)/2
            else:
                sdown=(sup+sdown)/2
       return sdown
'''
随机生成一个3n+3m-5维向量
'''
def initial(n,m):
      vector=zeros(3*n+3*m-5)
      for i in range(0,3*n+3*m-5):
          vector[i]=2*(random.random()-0.5)
      return vector
'''
for  n in range(2,6):
    for m in range(n,6):
'''
n=3
m=3
kfinished=0
kt=1
'''
for k in range(1,20):
'''        
step=0
x0=initial(n,m)                                  #随机一个初始向量
a=position()                                                                  
position.create(a,x0,n,m)
g1=tolgrad(a)
d=-g1
s=0.05                                            #第一次查找时的步长
number=0
new=0                                            #重新生成随机向量次数
landa=norm(g1)
b=position()
stay=False
percentage=0
while landa>eps and step<10000:
    stay0=stay
    stay=False
    landa0=landa
    g0=g1
    s=finds(a,d,0.01)
    position.create(b,a.vector+s*d,a.n,a.m)
    a=b       
    g1=tolgrad(b)
    beta=dot(g1,g1)/dot(g0,g0)
    d=-g1+beta*d
    energy=tolenergy(b)
    landa=norm(g1)
    step+=1
    '''
    if landa>=landa0:
        stay=True
    if stay is True and stay0 is True:
        number+=1
    else:
        number=0
    if number>5:                            #超过100次迭代grad的模都没有下降，则重新生成随机向量
        x0=initial(n,m) 
        position.create(a,x0,n,m)
        g1=tolgrad(a)
        d=-g1
        s=0.1 
        number=0
        new+=1
        landa=norm(g1)
    '''    
    step+=1
    percentage+=1/100
    print('能量为:%f，g1的模为%f'%(energy,landa))
'''   
print('第%d 组，进度%f,已经完成%d组'%(kt,percentage,kfinished))
       
 if landa>eps:
        kt+=1
        continue
 else:
    position.center(a)
    kfinished+=1
    kt+=1
    print('n=%d, m=%d,第%d组完成'%(n,m,kfinished))
    F=open('D:\\data\\%d%d\\%d'%(n,m,kfinished),'wb')
    pickle.dump((a.vector,a.naxstd,a.naystd,a.nazstd,a.clxstd,a.clystd,a.clzstd,step),F)
    F.close()
'''

     
    
    
    