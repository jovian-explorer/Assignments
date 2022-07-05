# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 20:44:17 2021

@author: sohini
"""

#Importing necessary packages
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

#defining necessary variables to be used later
c=3*10**8                   #speed of light
ns=5000                     #number of samples
par=2                       #number of parameters
snb=31                      #number of bins in the SN data
z=[]                        #array to store supernova redshift
M=[]                        #array to store distance modulous
sigo=0.01                   #Defining the standard deviation of omgea matter
sigh=0.024                  #defining the standard deviation of hubble parameter
ar=np.empty([ns,par+1])     #empty array to store the parameter values: 1st column holds omega matter, 2nd column holds h and third column holds ln(l)
diff=np.empty(snb)          #temporary variable to store difference between observed distance modulus and theoretical distance modulus
accn=0                      #acceptance number
lpre=0                      #previous likelihood
omnext=0                    #next value of omega matter
hnext=0                     #next value of hubble parameter
lnext=0                     #next likelihood

#defining necessary functions
def eta(a,om):                      #eta
    if om>=0.9999:                  #ignoring unphysical conditions that may arise due to random number generation
        om=0.9999
    elif om<=0.0:
        om=0.000001
    else:
        pass
    s=((1.0-om)/om)**(1.0/3.0)
    n= 2.0*(np.sqrt((s*s*s)+1))*(((a**(-4.0))-(0.1540*s*(a**(-3.0)))+(0.4304*s*s*(a**(-2.0)))+(0.19097*s*s*s*(a**(-1.0)))+(0.066941*s*s*s*s)))**(-1.0/8.0)
    return n

def Dl(z,om):                          #luminosity distance
    eta1=eta(1,om)
    eta2=eta(1/(1+z),om)
    d=(3000.0*(1+z))*(eta1-eta2)
    return d

def mu(z,om,h):                     #distance modulus (theoretical)
    d=Dl(z,om)
    m=25-(5*np.log10(h))+(5*np.log10(d))
    return m

def likelihood(om,h,z,M):                   #natural log likelihood function
    if om<=0.0 or h<=0.0:                   #we know they must be positive so removing unphysical conditions
        l=-1.e100
    else:
        for i in range(snb):
            diff[i]=M[i]-mu(z[i],om,h)
        dt=np.dot(cinv,diff)
        l=-0.5*np.dot(np.transpose(diff),dt)      #taking dot products
    return l                                      #l is ln(L)

#opening both the files and storing their data in lists
Cov=open('jla_mub_covmatrix.txt','r')
data=open('jla_mub_0.txt','r')

#storing the covarient matrix in 'C'
C=np.loadtxt(Cov)
C=np.reshape(C,(snb,snb))
cinv=npl.inv(C)          #inverse of C

#storing redshift and distance modulus as floating points after ignoring the first line(title)
z,M= np.loadtxt('jla_mub_0.txt', unpack=True, usecols=[0,1])

#filling up the first row of the main array: 1st column: omega_m, second column: h, third column: log(likelihood)
ar[0,0]=np.random.uniform()                  #initialising omega_m by using a gaussian normal distribution
ar[0,1]=np.random.uniform(0.738,0.024)       #constraining the hubble parameter as given in the assignment
ar[0,2]=likelihood(ar[0,0], ar[0,1], z, M)   #calculating the likelihood of the first set of parameters

#running the actual sampling loops
for i in range(1,ns):
    lpre=ar[i-1,2]
    omnext=np.random.normal(ar[i-1,0],sigo)
    hnext=np.random.normal(ar[i-1,1],sigh)
    lnext=likelihood(omnext, hnext, z, M)

#Metropolis Hastings Algorithm: will accept if likelihood increases or will pass
    if lnext>=lpre:
        ar[i,0]=omnext
        ar[i,1]=hnext
        ar[i,2]=lnext
        accn=accn+1
        print("Accepted with higher likelihood.")
    else:
        x=np.random.uniform()
        if (lnext-lpre)>np.log(x):
            ar[i,0]=omnext
            ar[i,1]=hnext
            ar[i,2]=lnext
            accn=accn+1
            print("Accepted with lesser likelihood.")
        else:
            ar[i,0]=ar[i-1,0]
            ar[i,1]=ar[i-1,1]
            ar[i,2]=lpre
            print("rejected")

#burnin: Rejecting the first 10% values so as to only select values where the algorithm converges
r=ns//25

#calculating values of statistical importance:
omm=np.mean(ar[r:,0])                 #mean or estimated value of omega matter
hm=np.mean(ar[r:,1])                  #mean or estimated value of h
omsd=np.std(ar[r:,0])                 #standard deviation in omega matter
hsd=np.std(ar[r:,1])                  #standard deviation in h
accr=(accn*100)/ns                    #acceptance ratio

#displaying the above results:
print('Estimated value of omega matter is:'+str(omm))
print('Estimated standard deviation in omega matter is:'+str(omsd))
print('estimated value of scaling factor h is:'+str(hm))
print('Estimated standard deviation in scaling factor is:'+str(hsd))
print('Percentage of samples accepted (acceptance ratio) is:'+str(accr))

#Calculating theoretical values of distance modulus from our estimates
yth=np.empty(snb)
for i in range(snb):
    yth[i]=mu(z[i],omm,hm)

#plotting everything
#1. Plotting the given data and comparing with theoretical data
plt.figure(0)
plt.xlabel('Redshift z')
plt.ylabel('Distance modulus mu')
plt.title('Distance Modulus vs redshift data comparison.')
plt.plot(z,M,c='red',marker='.',label="Given Data")
plt.plot(z,yth,c='blue',marker='x', label="Theoretical Data from estimates obtained")
plt.legend()

#2. plotting the samples
plt.figure(1)
plt.scatter(ar[:,0],ar[:,1],c=ar[:,2])
plt.title("scatter plot")
plt.xlabel("$\Omega_m$")
plt.ylabel("h")
