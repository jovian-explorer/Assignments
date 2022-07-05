#Q6: Coin tossing simulation
import matplotlib.pyplot as plt
import random as r

#Function to check who's leading
def lead(A,B):
    if A>B:
        l='A'
    elif B>A:
        l='B'
    else:
        l='null'
    return l

#Initializing variables
a=0
b=0
list_lead=[]
diff=[]
ld=' '
n=0
N=range(101)


#Tossing the coin and documenting scores. If result is 0, A wins, if result is 1, B wins
while n<100:
    rd=r.randint(0,2)
    if rd<0.5:
        a=a+1
        b=b-1
    else:
        b=b+1
        a=a-1
    if ld==lead(a,b):
        list_lead.append(0)
    elif ld != 'null':
        list_lead.append(1)
        ld=lead(a,b)
    diff.append(abs(a-b))
    n=n+1
    
#Displaying results
plt.hist(diff,histtype='bar',rwidth=0.5)
plt.title('Score difference turns')
plt.xlabel('Turns')
plt.ylabel('Score difference')
plt.figure(0)
plt.hist(list_lead,histtype='bar',rwidth=0.1)
plt.title('lead change indicator')