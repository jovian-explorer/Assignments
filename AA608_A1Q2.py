#Q2: Balls in urn problem
import matplotlib.pyplot as mt

#defining the probability function
def p(n):
    p=(2*n**2*(10-n))/1000
    return p

#defining a range of values to test for
N=list(range(11))
P=[]
fr=[]

#calculating for each value
for n in N:
    P.append(p(n))
    fr.append(n/10)

#Displaying results
mt.plot(fr,P)
mt.title('Probability vs Fraction of red balls')
mt.xlabel('Fraction of red balls')
mt.ylabel('Probability')
print('Maximum value of probability found at N=', P.index(max(P)))