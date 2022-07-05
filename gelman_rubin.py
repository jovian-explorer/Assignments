# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:18:31 2021

@author: sohini
"""

#importing necessary packages
import numpy as np
import numpy.linalg as npl
#defining necessary variables
c=3*10**8                   #speed of light
ns=250                      #number of samples
par=2                       #number of parameters
snb=31                      #number of bins in the SN data
ar=np.empty([ns,par+1])     #empty array to store the parameter values: 1st column holds omega matter, 2nd column holds h and third column holds ln(l)
z=[]                        #Empty array to store given redshift data
M=[]                        #Empty array to store given distance modulus data
accn=0                      #acceptance number
diff=np.empty(snb)          #temporary variable to store difference between observed distance modulus and theoretical distance modulus
l1=1
#Assuming gaussian distribution and same standard deviations for both omega matter and h
sigo=0.01                   #standard deviation for omega matter
sigh=0.01                   #standard deviation of h

#opening both the files and storing their data in lists
Cov=open('jla_mub_covmatrix.txt','r')
data=open('jla_mub_0.txt','r')

#storing the covarient matrix in 'C' and reshaping it into a 31x31 shape
C=np.loadtxt(Cov)
C=np.reshape(C,(snb,snb))
cinv=npl.inv(C)              #inverse of C

#storing redshift and distance modulus as floating points after ignoring the first line(title)
z,M= np.loadtxt('jla_mub_0.txt', unpack=True, usecols=[0,1])

#defining necessary functions
def eta(a,om):              #eta
    if om>=0.9999:
        om=0.9999
    elif om<=0.0:
        om=0.000001
    else:
        pass
    s=((1.0-om)/om)**(1.0/3.0)
    n= 2.0*(np.sqrt((s*s*s)+1))*(((a**(-4.0))-(0.1540*s*(a**(-3.0)))+(0.4304*s*s*(a**(-2.0)))+(0.19097*s*s*s*(a**(-1.0)))+(0.066941*s*s*s*s)))**(-1.0/8.0)
    return n

def Dl(z,om):               #luminosity distance
    eta1=eta(1,om)
    eta2=eta(1/(1+z),om)
    d=(3000.0*(1+z))*(eta1-eta2)
    return d

def mu(z,om,h):             #distance modulus (theoretical)
    d=Dl(z,om)
    m=25-(5*np.log10(h))+(5*np.log10(d))
    return m

def likelihood(om,h,z,M):   #natural log likelihood function
    if om<=0.0 or h<=0.0:   #we know they must be positive so removing unphysical conditions
        l=-1.e100
    else:
        for i in range(snb):
            diff[i]=M[i]-mu(z[i],om,h)
        dt=np.dot(cinv,diff)
        l=-0.5*np.dot(np.transpose(diff),dt)      #taking dot products
    return l                                      #l is ln(L)

#gelman rubin test
nchain=50                   #length of the chain
omchainm=[]                 #defining some empty arrays for usage later
omchainvar=[]
hchainmean=[]
hchainvar=[]

#running the loops to find the actual ratio
for chain in range(nchain):
    print("Progress: "+str((chain*100)/nchain))         #printing progress percentage
    ar[0,0]=np.random.uniform()                         #initialising values with random  values
    ar[0,1]=np.random.uniform()
    ar[0,2]=likelihood(ar[0,0], ar[0,1], z, M)          #finding likelihood of the first set of (random) values

    for i in range(1,ns):
        lpre=ar[i-1,2]
        omnext=np.random.normal(ar[i-1,0],sigo)
        hnext=np.random.normal(ar[i-1,1],sigh)
        if omnext>0.9999:                               #ignoring unphysical values (omega_m>1) that could be selected due to random number generation
            omnext=0.999
        lnext=likelihood(omnext, hnext, z, M)

        if lnext>=lpre:                                 #testing the posteriors for the current and the new datasets
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

r=ns//25                                       #calculating the number of values to reject (burn in)

omm=np.mean(ar[r:,0])                          #finding the mean of the omega_m
hm=np.mean(ar[r:,1])                           #finding the mean of the hubble parameter values
omvar=np.var(ar[r:,0])                         #finding the variance of the omega m values
hvar=np.var(ar[r:,1])                          #finding the variance of the hubble parameter values

omchainm.append(omm)
hchainmean.append(hm)
omchainvar.append(omvar)
hchainvar.append(hvar)

omsigchsq=np.mean(omchainvar)
omsigmeansq=np.var(omchainm)

ratioom=(((49.0/50.0)*omsigchsq)+(omsigmeansq/50.0))/omsigchsq          #finding the ratio for omega matter

hsigchsq=np.mean(hchainvar)
hsigmeansq=np.var(hchainmean)

ratioh=((((nchain-1.0)/nchain)*hsigchsq)+(hsigmeansq/nchain))/hsigchsq      #finding the ratio for hubble parameter

print("Value of Gelman Rubin convergence ratio for omega using "+str(ns)+" samples is: "+str(ratioom))
print("Value of Gelman Rubin convergence ratio for h using "+str(ns)+" samples is: "+str(ratioh))
