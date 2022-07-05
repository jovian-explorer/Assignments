#Q3: Galaxy sampple problem
import math as m
import matplotlib.pyplot as plt

#Defining probability function
def p(n):
    r=m.comb(30,n)*(0.1**(n+1))*(0.9**(30-n))
    return r

#Defining a range of values of N
N=range(31)
P=[]
fr=[]

#Calculating for each iteration
for n in N:
    P.append(p(n))
    fr.append(n/30)

#Displaying results
plt.plot(fr,P)
plt.title('Probability vs fraction of galaxy clusters with dominant galaxies')
plt.xlabel('Fraction of galaxy clusters with dominant galaxies')
plt.ylabel('Probability')
print('Maximum probability found at n=',P.index(max(P)))