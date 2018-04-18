from numpy import *
from math import *
import pickle
import random
'''
[0,n-1]                        xna
[n-1,2*n-2]                    yna
[2*n-2,3*n-4]                  zna
[3*n-4,3*n-4+m]                xcl
[3*n-4+m,3*n-5+2*m]            ycl
[3*n-5+2*m,3*n+3*m-6]          zcl
xna2,xna3,...xnan,yna2,yna3...ynan,zna3,..znan,xcl1,..xclm,ycl2..yclm,
zcl2,...yclm)
'''
class position:            
    def create(self,positionlist,n,m):
         positionlist=list(positionlist)
         self.xna=positionlist[0:n-1]
         self.xna.insert(0,0)
         self.yna=positionlist[n-1:2*n-2]
         self.yna.insert(0,0)
         self.zna=positionlist[2*n-2:3*n-4]
         self.zna.insert(0,0)
         self.zna.insert(0,0)
         self.xcl=positionlist[3*n-4:3*n-4+m]
         self.ycl=positionlist[3*n-4+m:3*n-5+2*m]
         self.ycl.insert(0,0)
         self.zcl=positionlist[3*n-5+2*m:3*n+3*m-6]
         self.zcl.insert(0,0)
         self.n=n
         self.m=m
         self.vector=array(positionlist)
         rlist=[]
         for i in range(0,n):
             r=self.xna[i]**2+self.yna[i]**2+self.zna[i]**2
             r=sqrt(r)
             rlist.append(r)
         for i in range(0,m):
             r=self.xcl[i]**2+self.ycl[i]**2+self.zcl[i]**2
             r=sqrt(r)
             rlist.append(r)   
         self.rlist=rlist
         Llist=[]
         for i in range(0,self.n):
            for j in range(0,self.m):
                r=0
                r+=(self.xna[i]-self.xcl[j])**2
                r+=(self.yna[i]-self.ycl[j])**2
                r+=(self.zna[i]-self.zcl[j])**2
                r=sqrt(r)
                Llist.append(r)
         self.diflist=Llist
         
r0=0.330                       #距离的单位为A
V0=1.09*10**3                   
gamma=14.3997
eps=10**-4                   #规定误差上限
epsstep=0.001               #规定golden method的误差
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
        for x in a.diflist:
                energy+=-gamma/x+V0*exp(-x/r0)
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
        gradzna=gradzna[1:]
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
        if min(b.diflist)<0.1:
          return False
        else:
          return dot(d1,tolgrad(b))
'''
输入移动方向，移动距离，和类型，输出移动后的总能量
'''
def energys(s,a,d1):
        global rbound
        b=position()
        position.create(b,a.vector+s*d1,a.n,a.m)   
        if min(b.diflist)<0.1:
          return False
        if max(b.rlist)>rbound:
          return True
        else:
          return tolenergy(b)

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
Golden section methed 确定使得能量最低的步长，以及相应的能量
'''    
def golfinds(a,d1,maxlen,laststep):
    global rbound
    phi=(1+sqrt(5))/2
    phi1=1/phi
    phi2=1/(phi+1)
    down=0
    top=maxlen
    c=down+phi2*(top-down)
    d=down+phi1*(top-down)
    err=eps+1
    k=0
    while err>epsstep and k<1000:
        f1=energys(c,a,d1)
        f2=energys(d,a,d1)
        if f1 is False or f2 is False:
            b=position()
            s=laststep
            d2=initial(a.n,a.m,s)
            position.create(b,a.vector+d2,a.n,a.m)
            energy=tolenergy(b)
            return s,b,energy,1    
        if f1 is True or f2 is True:
            b=position()
            s=1000000
            d2=initial(a.n,a.m,rbound)
            position.create(b,d2,a.n,a.m)
            energy=tolenergy(b)
            return s,b,energy,1         
        if f1>f2:
            down=c
            c=d
            d=down+phi1*(top-down)
            k+=1
        else:
            top=d
            d=c
            c=down+phi2*(top-down)
            k+=1
    b=position()
    s=(top+down)/2
    position.create(b,a.vector+s*d1,a.n,a.m)
    energy=tolenergy(b)
    return s,b,energy,0  
    
'''
随机生成一个3n+3m-6维向量，length为长度
'''
def initial(n,m,length):
      vector=zeros(3*n+3*m-6)
      for i in range(0,3*n+3*m-6):
          vector[i]=random.random()-0.5
      normv=norm(vector)  
      vector=[length*x/normv for x in vector]
      return vector
'''
for  n in range(1,6):
'''
'''
for n in range(5,6):
    for m in range(n,6):
        #收敛的次数
        kfinished=0
        #跑完总次数
        kt=0    
        for k in range(1,20):    
'''
n=5
m=5
step=0
steplen=n+m
global rbound
rbound=10*(n+m)
x0=initial(n,m,rbound)                                  #随机一个初始向量
a=position()                                                                  
position.create(a,x0,n,m)
g1=tolgrad(a)
d=-g1
s=0.05                                            #第一次查找时的步长
#共轭梯度法连续梯度不下降的次数
number1=0 
#最速降线连续梯度不下降次数
number2=0
#弹开次数
kbound=0
#湮灭次数
kani=0  
#收敛性质不好重新的次数
new=0                                     
landa=norm(g1)
energy=tolenergy(a)
b=position()
stay=False
percentage=0
errornumber=0
while landa>eps and step<10000:   
    s0=s         
    energy0=energy
    stay0=stay
    stay=False
    landa0=landa
    g0=g1
    if number1>3 and number2<3:
        print('最速下降法,number2=%d'%(number2))
        result=golfinds(a,-g1,rbound,s0)
        s,b,energy,error=result
        errornumber+=error
        a=b       
        g1=tolgrad(a)
        landa=norm(g1)
        if landa>=landa0:
           stay=True
           print('梯度未下降，此时stay0为%d,stay为%d'%(stay0,stay))
        if stay is True and stay0 is True:
           number2+=1
           print('number2++')
        else:
           number2=0
           print('n2置0，此时stay0为%d,stay为%d'%(stay0,stay))
        step+=1
        print(energy,landa,s,kbound,kani)  
        continue
    else:
        print('共轭梯度法,number1=%d'%(number1))
        number2=0
        result=golfinds(a,d,steplen,s0)
        s,b,energy,error=result 
        errornumber+=error
        a=b       
        g1=tolgrad(b)
        beta=dot(g1,g1)/dot(g0,g0)
        d=-g1+beta*d
        landa=norm(g1)
        if landa>=landa0:
           stay=True
        if stay is True and stay0 is True:
           number1+=1
        else:
           number1=0
           step+=1
        print(energy,landa,s,kbound,kani)             
'''
        if max(b.rlist)>rbound or b.least<0.01:
            x0=initial(n,m) 
            position.create(a,x0,n,m)
            g1=tolgrad(a)
            d=-g1
            s=0.1 
            number=0
            new+=1
            landa=norm(g1)  
            energy=tolenergy(a) 
            continue
'''

'''
                if energy>=energy0:
                    stay=True
                if stay is True and stay0 is True:
                    number+=1
                else:
                    number=0
                if number>20:                            #超过100次迭代grad的模都没有下降，则重新生成随机向量
                    x0=initial(n,m) 
                    position.create(a,x0,n,m)
                    g1=tolgrad(a)
                    d=-g1
                    s=0.1 
                    number=0
                    new+=1
                    landa=norm(g1)  
                    energy=tolenergy(a)
'''
                
                #kt+=1
                #print('n=%d, m=%d,跑了%d组'%(n,m,kt))
'''    
            if landa>eps:
                continue
            else:
                kfinished+=1
                print('n=%d, m=%d,第%d组完成'%(n,m,kfinished))
                F=open('D:\\data\\%d%d\\%d'%(n,m,kfinished),'wb')
                pickle.dump(a.vector,F)
                F.close()    
'''    
            
        