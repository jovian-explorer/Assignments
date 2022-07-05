# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:04:19 2021

@author: sohini
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from scipy.integrate import simps
import matplotlib.cm as cm
from statistics import mean
import copy

#defining some variables
c=3*10**8
ns=5000        #number of samples
par=2           #number of parameters
snb=31          #number of bins in the SN data
z=[]            #array to store supernova redshift
M=[]            #array to store distance modulous
sigo=0.1       #Defining the standard deviation of omgea matter
sigh=0.1      #defining the standard deviation of hubble parameter
ar=np.empty([ns,par+1])     #empty array to store the parameter values: 1st column holds omega matter, 2nd column holds h and third column holds ln(l)
lpns=25                     #leapfrog samples
lpeps=0.0006                #leapfrog epsilon

#opening both the files and storing their data in lists
Cov=open('jla_mub_covmatrix.txt','r')
data=open('jla_mub_0.txt','r')

#storing the covarient matrix in 'C'
l1=1
C=np.loadtxt(Cov)
C=np.reshape(C,(snb,snb))
cinv=npl.inv(C)          #inverse of C

#storing redshift and distance modulus as floating points after ignoring the first line(title)
for line in data:
    a=[0,0]
    a=line.split()
    z.append(float(a[0]))
    M.append(float(a[1]))

#defining necessary functions
def eta(a,om):                      #eta
    if om>=0.9999:
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

diff=np.empty(snb)  #temporary variable to store difference between observed distance modulus and theoretical distance modulus

def likelihood(om,h,z,M):                   #natural log likelihood function
    if om<=0.0 or h<=0.0:            #we know they must be positive so removing unphysical conditions
        l=-1.e100
    else:
        for i in range(snb):
            diff[i]=M[i]-mu(z[i],om,h)
        dt=np.dot(cinv,diff)
        l=-0.5*np.dot(np.transpose(diff),dt)      #taking dot products
    return l    #l is ln(L)


#HMC function to calculate the derivative dU/dx
def der(om,h):
    sampl=5
    ar[0,0]=om
    ar[0,1]=h
    ar[0,2]=likelihood(ar[0,0],ar[0,1],z,M)
    i=1
    for i in range(1,sampl):
        lpre=ar[i-1,2]
        omnext=np.random.normal(ar[i-1,0],sigo)

        if omnext>0.9999:
            omnext=0.99
        else:
            pass

        hnext=np.random.normal(ar[0,1],sigh)
        lnext=likelihood(omnext,hnext,z,M)
        if lnext>lpre:
            ar[i,0]=omnext
            ar[i,1]=hnext
            ar[i,2]=lnext
        else:
            x=np.random.uniform()
            if (lnext-lpre)>np.log(x):
                ar[i,0]=omnext
                ar[i,1]=hnext
                ar[i,2]=lnext
            else:
                ar[i,0]=ar[i-1,0]
                ar[i,1]=ar[i-1,1]
                ar[i,2]=ar[i-1,2]
        logar=ar[:,2]
        gr=-1*mean(np.gradient(logar))
    return gr

#initializing parameter values with a guess
omega=[0.1]
hubble=[0.1]

accn=0


#main HMC loop
for i in range(1,ns):
    m=[0,0]             #mean of 2D gaussian
    covm=[[1,0],[0,1]]      #covariance matrix for the 2D gaussian normal distribution
    ompre=omega[len(omega)-1]
    hpre=hubble[len(hubble)-1]
    uold=-1*likelihood(ompre,hpre,z,M)
    omnext=copy.copy(ompre)
    hnext=copy.copy(hpre)
    gnext=copy.copy(uold)
    v=np.random.multivariate_normal(m,covm)
    hampre=uold+(np.dot(v,v))/2.0

    for j in range(lpns):
        v[0]=v[0]-((lpeps+gnext)/2.0)
        omnext=omnext+(lpeps+v[0])
        gnext=der(omnext,hnext)
        v[0]=v[0]-((lpeps+gnext)/2.0)

    for k in range(1,lpns):
        v[1]=v[1]-((lpeps*gnext)/2.0)
        hnext=hnext+(lpeps*v[1])
        gnext=der(omnext,hnext)
        v[1]=v[1]-((lpeps*gnext)/2.0)
    penew=-1*likelihood(omnext, hnext, z, M)
    v=np.random.multivariate_normal(m, covm)
    hamnext=penew+((np.dot(v,v))/2.0)
    dham=hamnext-hampre
    if dham<0.0:
        omega.append(omnext)
        hubble.append(hnext)
        accn+=1
        print("Accepted. Percentage complete: "+str((100*i)/ns))
    else:
        unew=np.random.uniform(0,1)
        if unew<(np.exp(-dham)):
            omega.append(omnext)
            hubble.append(hnext)
            accn+=1
            print("Accepted. Percentage complete: "+str((100*i)/ns))
        else:
            omega.append(ompre)
            hubble.append(hpre)
            accn+=1
            print("Rejected. Percentage complete: "+str((100*i)/ns))

r=ns//25

omm=np.mean(omega[r:])
hubblem=np.mean(hubble[r:])
omstd=np.std(omega[r:])
hubblestd=np.std(hubble[r:])
accr=(accn*100)/ns

#displaying the above results:
print('Estimated value of omega matter is:'+str(omm))
print('Estimated standard deviation in omega matter is:'+str(omstd))
print('estimated value of scaling factor h is:'+str(hubblem))
print('Estimated standard deviation in scaling factor is:'+str(hubblestd))
print('Percentage of samples accepted (acceptance ratio) is:'+str(accr))

#plotting
plt.scatter(omega,hubble)
