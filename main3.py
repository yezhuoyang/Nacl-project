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
r0=0.330                       #距离的单位为A
V0=1.09*10**3                   
gamma=14.3997
eps=10**-3                   #规定误差上限
epsstep=0.00001               #规定golden method的误差
class position:            
    global n,m
    def create(self,positionlist):
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
         self.vector=array(positionlist)
    def rnacl(self,i,j):
         r=0
         r+=(self.xna[i-1]-self.xcl[j-1])**2
         r+=(self.yna[i-1]-self.ycl[j-1])**2
         r+=(self.zna[i-1]-self.zcl[j-1])**2
         r=sqrt(r)
         return r
    def rnana(self,i,j):
         r=0        
         r+=(self.xna[i-1]-self.xna[j-1])**2
         r+=(self.yna[i-1]-self.yna[j-1])**2
         r+=(self.zna[i-1]-self.zna[j-1])**2
         r=sqrt(r)
         return r
    def rclcl(self,i,j):
         r=0        
         r+=(self.xcl[i-1]-self.xcl[j-1])**2
         r+=(self.ycl[i-1]-self.ycl[j-1])**2
         r+=(self.zcl[i-1]-self.zcl[j-1])**2
         r=sqrt(r)
         return r
    def grad(self):
          gxna=[]
          gyna=[]
          gzna=[]
          gxcl=[]
          gycl=[]
          gzcl=[]
          for i in range(2,n+1): 
            gxi=0
            gyi=0
            gzi=0
            for j in range(1,m+1):
              rij=position.rnacl(self,i,j)
              gxi+=gamma*(self.xna[i-1]-self.xcl[j-1])/rij**3
              gxi-=V0*exp(-rij/r0)*(self.xna[i-1]-self.xcl[j-1])/(r0*rij)
              gyi+=gamma*(self.yna[i-1]-self.ycl[j-1])/rij**3
              gyi-=V0*exp(-rij/r0)*(self.yna[i-1]-self.ycl[j-1])/(r0*rij)
              gzi+=gamma*(self.zna[i-1]-self.zcl[j-1])/rij**3
              gzi-=V0*exp(-rij/r0)*(self.zna[i-1]-self.zcl[j-1])/(r0*rij)
            for j in range(1,n+1):
              if j!=i:
                  rij=position.rnana(self,i,j)
                  gxi-=gamma*(self.xna[i-1]-self.xna[j-1])/rij**3
                  gyi-=gamma*(self.yna[i-1]-self.yna[j-1])/rij**3
                  gzi-=gamma*(self.zna[i-1]-self.zna[j-1])/rij**3
            gxna.append(gxi)
            gyna.append(gyi)
            if i>2: 
              gzna.append(gzi)
          for i in range(1,m+1): 
            gxi=0
            gyi=0
            gzi=0
            for j in range(1,m+1):
              if j!=i:
                  rij=position.rclcl(self,i,j)
                  gxi-=gamma*(self.xcl[i-1]-self.xcl[j-1])/rij**3
                  gyi-=gamma*(self.ycl[i-1]-self.ycl[j-1])/rij**3
                  gzi-=gamma*(self.zcl[i-1]-self.zcl[j-1])/rij**3
            for j in range(1,n+1):
              rij=position.rnacl(self,j,i)
              gxi+=gamma*(self.xcl[i-1]-self.xna[j-1])/rij**3
              gxi-=V0*exp(-rij/r0)*(self.xcl[i-1]-self.xna[j-1])/(r0*rij)
              gyi+=gamma*(self.ycl[i-1]-self.yna[j-1])/rij**3
              gyi-=V0*exp(-rij/r0)*(self.ycl[i-1]-self.yna[j-1])/(r0*rij)
              gzi+=gamma*(self.zcl[i-1]-self.zna[j-1])/rij**3
              gzi-=V0*exp(-rij/r0)*(self.zcl[i-1]-self.zna[j-1])/(r0*rij)
            gxcl.append(gxi)
            if i>1:
              gycl.append(gyi)
              gzcl.append(gzi)        
          grad=gxna+gyna+gzna+gxcl+gycl+gzcl
          self.grad=array(grad)
          return self.grad
    def energy(self):
        energy=0
        for i in range(1,n+1):
            for j in  range(i+1,n+1):
                    energy+=gamma/position.rnana(self,i,j)
        for i in range(1,m+1):
            for j in  range(i+1,m+1):
                    energy+=gamma/position.rclcl(self,i,j)
        for i in range(1,n+1):
            for j in  range(1,m+1):
                    energy+=(-gamma/position.rnacl(self,i,j)+V0*exp(-position.rnacl(self,i,j)/r0))   
        self.energy=energy            
        return energy 
    def leastr(self):
         rlist=[]
         for i in range(1,n+1):
             for j in range(1,m+1):
                 rlist.append(position.rnacl(self,i,j))
         self.minr=min(rlist)       
         return self.minr
    def rmax(self):
         rlist=[]
         for i in range(0,n):
             rlist.append(sqrt(self.xna[i]**2+self.yna[i]**2+self.zna[i]**2))
         for j in range(0,m):
             rlist.append(sqrt(self.xcl[j]**2+self.ycl[j]**2+self.zcl[j]**2))
         self.rmax=max(rlist)       
         return self.rmax
    def Vxnana(i,j):
         if i!=j:
           rij=position.rnana(self,i,j)
           xi=self.xna[i-1]
           xj=self.xna[j-1]
           val=3*gamma/rij**5*(2*xi*xj-xi**2-xj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,m+1):
               rip=position.rnacl(self,i,p)
               xi=self.xna[i-1]
               xp=self.xcl[p-1]
               val+=(gamma/rip**3-3*gamma*(xi-xp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(xi-xp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,n+1):    
               rip=position.rnana(self,i,p)
               xi=self.xna[i-1]
               xp=self.xna[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(xi**2+xp**2-2*xi*xp)
           return val    
     def Vxnacl(i,j):
          rij=position.rnacl(self,i,j)
          xi=self.xna[i-1]
          xj=self.xcl[j-1]
          val=0
          val+=-gamma/rij**3+V0/r0*1/rij*exp(-rij/r0)+3*gamma/rij**5*(xj-xi)**2
          val+=-V0/r0*exp(-rij/r0)*(1/rij+1/r0)*1/rij**2*(xj-xi)**2
          return val
     def Vynana(i,j):
         if i!=j:
           rij=position.rnana(self,i,j)
           yi=self.yna[i-1]
           yj=self.yna[j-1]
           val=3*gamma/rij**5*(2*yi*yj-yi**2-yj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,m+1):
               rip=position.rnacl(self,i,p)
               yi=self.yna[i-1]
               yp=self.ycl[p-1]
               val+=(gamma/rip**3-3*gamma*(yi-yp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(yi-yp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,n+1):    
               rip=position.rnana(self,i,p)
               yi=self.yna[i-1]
               yp=self.yna[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(yi**2+yp**2-2*yi*yp)
           return val        
     def Vynacl(i,j):
          rij=position.rnacl(self,i,j)
          yi=self.yna[i-1]
          yj=self.ycl[j-1]
          val=0
          val+=-gamma/rij**3+V0/r0*1/rij*exp(-rij/r0)+3*gamma/rij**5*(yj-yi)**2
          val+=-V0/r0*exp(-rij/r0)*(1/rij+1/r0)*1/rij**2*(yj-yi)**2
          return val    
     def Vznana(i,j):
         if i!=j:
           rij=position.rnana(self,i,j)
           zi=self.zna[i-1]
           zj=self.zna[j-1]
           val=3*gamma/rij**5*(2*zi*zj-zi**2-zj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,m+1):
               rip=position.rnacl(self,i,p)
               zi=self.zna[i-1]
               zp=self.zcl[p-1]
               val+=(gamma/rip**3-3*gamma*(zi-zp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(zi-zp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,n+1):    
               rip=position.rnana(self,i,p)
               zi=self.zna[i-1]
               zp=self.zna[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(zi**2+zp**2-2*zi*zp)
           return val  
     def Vznacl(i,j):
          rij=position.rnacl(self,i,j)
          yi=self.zna[i-1]
          yj=self.zcl[j-1]
          val=0
          val+=-gamma/rij**3+V0/r0*1/rij*exp(-rij/r0)+3*gamma/rij**5*(zj-zi)**2
          val+=-V0/r0*exp(-rij/r0)*(1/rij+1/r0)*1/rij**2*(zj-zi)**2
          return val
     def Vxclcl(i,j):
         if i!=j:
           rij=position.rclcl(self,i,j)
           xi=self.xcl[i-1]
           xj=self.xcl[j-1]
           val=3*gamma/rij**5*(2*xi*xj-xi**2-xj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,n+1):
               rip=position.rnacl(self,p,i)
               xi=self.xcl[i-1]
               xp=self.xna[p-1]
               val+=(gamma/rip**3-3*gamma*(xi-xp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(xi-xp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,m+1):    
               rip=position.rclcl(self,i,p)
               xi=self.xcl[i-1]
               xp=self.xcl[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(xi**2+xp**2-2*xi*xp)
           return val     
     def Vxclna(i,j):
          return Vxnacl(j,i)        
     def Vyclcl(i,j):
         if i!=j:
           rij=position.rclcl(self,i,j)
           yi=self.ycl[i-1]
           yj=self.ycl[j-1]
           val=3*gamma/rij**5*(2*yi*yj-yi**2-yj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,n+1):
               rip=position.rnacl(self,p,i)
               yi=self.ycl[i-1]
               yp=self.yna[p-1]
               val+=(gamma/rip**3-3*gamma*(yi-yp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(yi-yp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,m+1):    
               rip=position.rclcl(self,i,p)
               yi=self.ycl[i-1]
               yp=self.ycl[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(yi**2+yp**2-2*yi*xp)
           return val         
     def Vyclna(i,j):
         return Vynacl(j,i)
     def Vzclcl(i,j):
         if i!=j:
           rij=position.rclcl(self,i,j)
           zi=self.zcl[i-1]
           zj=self.zcl[j-1]
           val=3*gamma/rij**5*(2*zi*zj-zi**2-zj**2)+gamma/rij**3
           return val
         else:
           val=0
           for p in range(1,n+1):
               rip=position.rnacl(self,p,i)
               zi=self.zcl[i-1]
               zp=self.zna[p-1]
               val+=(gamma/rip**3-3*gamma*(zi-zp)**2/rij**5)
               val+=(-V0/r0)*exp(rij/r0)/rij
               val+=(V0/r0)*exp(rij/r0)*(zi-zp)**2/rij**2*(1/rij+1/r0)
           for p in range(1,m+1):    
               rip=position.rclcl(self,i,p)
               zi=self.zcl[i-1]
               zp=self.zcl[p-1]
               val+=gamma/rij**3-3*gamma/rij**5*(zi**2+zp**2-2*zi*zp)
           return val         
     def Vzclna(i,j):
          return Vznacl(j,i)
    def hessian(self):
        dim=3*(m+n)-6
        H=zeros([dim,dim])
        Hdiag=zeros([dim,dim])             #用来记录对角线元素
        for i in range(1,n):
             Hdiag[i-1][i-1]=Vxnana(i,i)            
        for i in range(1,n):
             Hdiag[n-2+i][n-2+i]=Vynana(i,i)         
        for i in range(1,n-1):
             Hdiag[2*n-3+i][2*n-3+i]=Vznana(i,i)           
        for i in range(1,m+1):
             Hdiag[3*n-5+i][3*n-5+i]=Vxclcl(i,i)
        for i in range(1,m):
             Hdiag[3*n-5+m+i][3*n-5+m+i]=Vyclcl(i,i)
        for i in range(1,m):
             Hdiag[3*n-6+2*m+i][3*n-6+2*m+i]=Vzclcl(i,i)   
        

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


def energys(s,a,d1):
        global n,m
        b=position()
        position.create(b,a.vector+s*d1)   
        if position.leastr(a)<0.1:
          return False
        else:
          return position.energy(b)
    
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
    err=1+epsstep
    k=0
    while err>epsstep:
        f1=energys(c,a,d1)
        f2=energys(d,a,d1)
        if f1 is False or f2 is False:
            b=position()
            s=laststep
            d2=initial(s)
            position.create(b,a.vector+d2)
            energy=position.energy(b)
            print('ani! s=laststep')
            return s,b,energy                  
        if f1>=f2:
            down=c
            c=d
            d=down+phi1*(top-down)
            k+=1
            err=abs(top-down)/(abs(c)+abs(d))
        else:
            top=d
            d=c
            c=down+phi2*(top-down)
            k+=1
            err=abs(top-down)/(abs(c)+abs(d))
    b=position()
    s=(top+down)/2
    position.create(b,a.vector+s*d1)
    energy=position.energy(b)
    return s,b,energy
'''
backtracking，用来确定步长
'''
def backtrack(a,d1):
       rho=0.9 
       global steplen
       s=steplen
       while energys(s,a,d1)>a.energy+0.0001*s*dot(d1,a.grad):
              s*=rho
       return s     
             
'''
随机生成一个3n+3m-6维向量，length为长度
'''
def initial(length):
      vector=zeros(3*n+3*m-6)
      for i in range(0,3*n+3*m-6):
          vector[i]=random.random()-0.5
      normv=norm(vector)  
      vector=[length*x/normv for x in vector]
      return vector
'''
随机生成一个初始状态，注意与随机生向量的区别
'''    
def initialstate(n,m):
      global rbound
      list=[]
      for i in range(0,3*n+3*m-6):
           ran=rbound*(random.random()-0.5)
           list.append(ran)
      return  array(list)


n=5
m=5
step=0
#返回搜索的最大步长
global steplen
steplen=sqrt((3*n+3*m-6))
global rbound
#粒子离原点的最大距离
rbound=4*(n+m)
x0=initialstate(n,m)                                 #随机一个初始向量
a=position()                                                                  
position.create(a,x0)
g1=position.grad(a)
d=-g1
s=steplen                                            #第一次查找时的步长
#弹开次数
global kbound
kbound=0
#湮灭次数
global kani
kani=0  
#收敛性质不好重新的次数                              
landa=norm(g1)
energy=position.energy(a)
b=position()
stay=False
while landa>eps and step<100000:
    s0=s         
    energy0=energy
    stay0=stay
    stay=False
    landa0=landa
    g0=g1
    result=golfinds(a,d,0.5,s0)
    s,b,energy=result
    decent=s*dot(d,g1)
    b=position()
    position.create(b,a.vector+s*d)
    if position.rmax(b)>rbound:
            a=position()
            x0=initialstate(n,m)
            position.create(a,x0)
            energy=position.energy(a)
            g1=position.grad(a)
            d=-g1
            s=steplen
            landa=norm(g1)
            step+=1
            kbound+=1
            print('bound!')  
            continue   
    a=b                
    g1=position.grad(b)
    beta=dot(g1,g1)/dot(g0,g0)
    d=-g1+beta*d
    landa=norm(g1)
    if energy>=energy0:
        stay=True
    if stay is True and stay0 is True:
        number+=1
    else:
        number=0
    if number>5:                            #超过100次迭代grad的模都没有下降，则重新生成随机向量
        print('new!!')
        x0=initialstate(n,m) 
        position.create(a,x0)
        g1=position.grad(a)
        d=-g1
        s=steplen 
        number=0
        landa=norm(g1)
    step+=1
    print(energy,landa)
if step<100000:
    print('n=%d, m=%d完成'%(n,m))
    F=open('D:\\data\\%d%d'%(n,m),'wb')
    pickle.dump((a.vector),F)
    F.close()

            

        