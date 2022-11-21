#IMPORTS#
import numpy as np 
import numpy.linalg as LA
import matplotlib.pyplot as plt
import tikzplotlib

###################################SHOOTING METHOD CODE#########################################
def solve(z,a,b,N,k):
    """
        Numerically solving schrodingers equation 

        Parameters:
            z  (float) : the intial y2
            a  (float) : lower boundary
            b  (float) : upper boundary
            N  (int)   : number of grid points
            k  (float) : constant
    """

    y1 = np.zeros(N+1)
    y2 = np.zeros(N+1)

    dx = (b-a)/N

    y1[0] = 0
    y2[0] = z

    for i in range(0,N,1):
        y1[i+1] = y1[i] + dx*y2[i]
        y2[i+1] = y2[i] + k*dx*y1[i]

    return y1

def sign(fa,fc):
    """
        function determines if fa and fc have same sign
    """

    if (fa >0 and fc>0) or (fa<0 and fc<0):
        return True
    else:
        return False

def shooting(z0,z1,k,tol=0.01,n_max=1000000,low=0,high=100,N=1000):
    """
        numerically solves the BVP

        Parameters:
            z0        (float) : lower bound intial y2
            z1        (float) : upper bound intial y2
            tol       (float) : tolerance of method
            n_max     (int)   : maximum number of iterations allowed
            low       (float) : lower boundary
            high      (float) : upper boundary
            N         (int)   : number of grid points
            k         (float) : constant
            eig_state (int)   : the eigen state being solved
    """

    count = 0
    
    a = z0
    b = z1

    fa = solve(a,low,high,N,k)[-1]
    fb = solve(b,low,high,N,k)[-1]

    while count < n_max:

        c = (a+b)/2
        fc = solve(c,low,high,N,k)[-1]

        if fc == 0 or ((b-a)/2) < tol:
            break
        
        count+=1

        if sign(fa,fc) == True:
            a = c
            fa = solve(a,low,high,N,k)[-1]
        else:
            b = c
            fb = solve(b,low,high,N,k)[-1]
    
    psi = solve(c,low,high,N,k)
    return psi

def get_eigen_values(h,M,dx):
    """
        Function returns the respective eignvalues for 
        the particle in a box problem

        Parameters:
            h  (float) : the hamiltonian (??? find out)
            M  (float) : the mass
            dx (float) : the step size
    """

    D = np.zeros((N+1,N+1))

    #setting on boundaries#
    for i in range(1,N,1):
        D[i,i] = -2
        D[i,i-1] = 1
        D[i,i+1] = 1

    D = (-h**2)/(2*M*(dx**2)) * D
    u,v = LA.eig(D)

    #need to remove zeros#
    indices = np.where(u==0)
    u = np.delete(u,indices)
    u = np.sort(u)
    return u


if __name__ == "__main__":
    fig,(ax1,ax2) = plt.subplots(1,2)

    #constants#
    h = 1
    M = 1

    #boundaries#
    a = 0
    b = 100

    #number of points#
    N = 1000
    dx = (b-a)/N

    eign_values = get_eigen_values(h,M,dx)


    psi = []

    z_low = [0,-10,-10,0,-10,0,-10,0,-10,0]
    z_high = [10,0,0,10,0,10,0,10,0,10]

    #getting the wave function plots#
    for i in range (0,10,1):
        z0 = z_low[i]
        z1 = z_high[i]

        E = eign_values[i]

        k = (-2*M*E)/(h**2)

        psi = shooting(z0,z1,k) 
        #shifting graphs so wave function at different eigen states visible#
        norm_psi = psi/(LA.norm(psi)) + 0.25*i
        ax1.plot(norm_psi,label="N = {}".format(i+1))

    #comparing exact eigenvalues to numerical#
    exact = lambda n,L: (n**2)*(np.pi**2)/(2*L**2)

    n_values = np.arange(1,12)

    exact_values = exact(n=n_values,L=1000)

    ax2.plot(exact_values,'r-*',label='exact')
    ax2.plot((eign_values[0:11])/100,'b-',label='numerical')

    plt.legend(loc='best')
    tikzplotlib.save("PIB_fig.tex",axis_height='10cm',axis_width='16cm')
    plt.show()
