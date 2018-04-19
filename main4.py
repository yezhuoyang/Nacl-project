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
eps=10**-4                   #规定误差上限
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

def feval(vector):
       a=position()
       position.create(a,vector)
       return position.energy(a)
   
def fevald(vector):
       a=position()
       position.create(a,vector)
       return position.grad(a)

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
           ran=0.5*rbound*(random.random()-0.5)
           list.append(ran)
      return  array(list)


n=3
m=3
step=0
#返回搜索的最大步长
global steplen
steplen=sqrt((3*n+3*m-6))
global rbound
#粒子离原点的最大距离
rbound=6*(n+m)
x0=initialstate(n,m)                                 #随机一个初始向量
a=position()                                                                  
position.create(a,x0)
g1=position.grad(a)
d=-g1
s=steplen                                            #第一次查找时的步长
#弹开次数
lambdad = 10**-8
lambdabar = 0
sigmac = 0.0001
sucess = 1
deltastep = 0 
#Calculate initial gradient
noiter = 0;   
x=initialstate(n,m)                                 #随机一个初始向量  
pv =-fevald(x)
rv = pv;
while norm(rv)>eps:     
   noiter+=1
   position.create(a,x)
   if position.rmax(a)>rbound or position.leastr(a)<0.1:
       lambdad =10**-8
       lambdabar = 0
       sigmac = 0.0001
       sucess = 1
       deltastep = 0 
#Calculate initial gradient
       noiter = 0;   
       x=initialstate(n,m)                                 #随机一个初始向量  
       pv =-fevald(x)
       rv = pv;       
       continue
   print(norm(rv),feval(x))
   if deltastep==0:
      df=fevald(x)
   else:
      df=-rv
   deltastep = 0
   if sucess==1:
      sigma=sigmac/norm(pv)      
      dfplus=fevald(x+sigma*pv)
      stilda=(dfplus-df)/sigma
      delta =dot(pv,stilda)
   delta=delta+(lambdad-lambdabar)*norm(pv)**2
   if delta<=0:
      lambdabar=2*(lambdad-delta/norm(pv)**2)
      delta=-delta+lambdad*norm(pv)**2
      lambdad=lambdabar
#Step size
   mu=dot(pv,rv)
   alpha = mu/delta
   fv=feval(x)
   fvplus=feval(x+alpha*pv)
   delta1=2*delta*(fv-fvplus)/mu**2
   rvold=rv
   pvold=pv
   if delta1>=0:
      deltastep=1
      x1=x+alpha*pv
      rv=-fevald(x1)
      lambdabar=0
      sucess=1
   if (noiter%n)==0:
      pv=rv
   else:
      rdiff=rv-rvold
      beta=dot(rdiff,rv)/dot(rvold,rvold)
      pv=rv+beta*pvold
   if delta1>=0.75:
      lambdad = 0.25*lambdad
   else:
      lambdabar = lambdad
      sucess = 0
      x1=x+alpha*pv
   if delta1<0.25:
      lambdad=lambdad+delta*(1-delta1)/norm(pvold)**2
   x=x1
            

        