#IMPORTS#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def psi0(x, y, x0, y0, sigma=0.5, k=15*np.pi):
    
    """
    	The intial condition of the wave function
	at time t=0 the system is modelled using the the gaussian wave function

	Parameters:
		x,y   (list)  : the grid points in the system
		x0,y0 (float) : where the function is centred at
		
		
    """
    p0 = np.exp(-1/2*((x-x0)**2 + (y-y0)**2)/sigma**2)*np.exp(1j*k*(x-x0))
    return p0 


def Slit(psi, j0, j1, i0, i1, i2, i3):
    
    """
    Makes the value of the wave function 0 at the boundaries of the duble slit

    The indices j0, j1, i0, i1, i2, i3 is used to define the boundaries of the double slit. 
    
	Parameters:
		psi  (array) : an array with the wave values at time t
		j0   (int)   : left edge of the slits
		j1   (int)   : right edge of the slits
		
		i0   (int)   : Lower edge of the lower slit
		i1   (int)   : Upper edge of the lower slit
		i2   (int)   : Lower edge of upper slit
		i3   (int)   : Upper edge of upper slit
   
    """
    
	#checks if psi is an array#
    psi = np.asarray(psi)
    
    # setting values of wave function to zero at boundaries
    psi[0:i3, j0:j1] = 0
    psi[i2:i1,j0:j1] = 0
    psi[i0:,  j0:j1] = 0
    
    return psi

def plot_PSI(psi1,psi2,psi3,psi4,x_values,y_values):
    """
        Creates surface plot of PDE in comparision to exact solution if specified

        Parameters:
            u       (array) : the approximate numerical solution in matrix form
            x_space (list)  : the range of x values
            t_space (list)  : the range of t values 
            u_exact (array) : the exact solution in matrix form
            sol     (bool)  : if True the exact solution and error plot is shown along
                              with approximate solution
    """
    #max value of wave function#
    max_value = np.max(psi1)

    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (30,16)
    #ax = plt.axes(projection = '3d')
    ax1 = fig.add_subplot(2,2, 1, projection='3d')
    #T,X = np.meshgrid(x_values,y_values)
    ax1.plot_surface(y_values,x_values,psi1,cmap='plasma')
    ax1.set_zlim(0, max_value)

    ax2 = fig.add_subplot(2,2, 2, projection='3d')
    #T,X = np.meshgrid(x_values,y_values)
    ax2.plot_surface(y_values,x_values,psi2,cmap='plasma')
    ax2.set_zlim(0, max_value)

    ax3 = fig.add_subplot(2,2, 3, projection='3d')
    #T,X = np.meshgrid(x_values,y_values)
    ax3.plot_surface(y_values,x_values,psi3,cmap='plasma')
    ax3.set_zlim(0, max_value)
    

    ax4 = fig.add_subplot(2,2, 4, projection='3d')
    #T,X = np.meshgrid(x_values,y_values)
    ax4.plot_surface(y_values,x_values,psi4,cmap='plasma')
    ax4.set_zlim(0, max_value)

    plt.show()
    

#Parameters
L = 8 # dimension of rectangle
Dy = 0.05 # space step size.
Dt = Dy**2/4 # time step size.
Nx = int(L/Dy) + 1 # Number of points on the x axis.
Ny = int(L/Dy) + 1 # Number of points on the y axis.
print(Nx)
print(Ny)
Nt = 500 # Number of time steps.
px = -Dt/(2j*Dy**2) 
py = -Dt/(2j*Dy**2)

# Initial position of the center of the Gaussian wave function.
x0 = L/5
y0 = L/2

# Parameters of the double slit.
w = 0.2 # Width of the walls
s = 0.8 # Separation between the edges of the slits.
a = 0.4 # Aperture of the slits.


#location of the boundaries of the double slit#
j0 = int(1/(2*Dy)*(L-w)) # Left edge.
j1 = int(1/(2*Dy)*(L+w)) # Right edge.
i0 = int(1/(2*Dy)*(L+s) + a/Dy) # Lower edge of the lower slit.
i1 = int(1/(2*Dy)*(L+s))        # Upper edge of the lower slit.
i2 = int(1/(2*Dy)*(L-s))        # Lower edge of the upper slit.
i3 = int(1/(2*Dy)*(L-s) - a/Dy) # Upper edge of the upper slit.


#the potential function#
v = np.zeros((Ny,Ny), complex) 

#size of matrix system with unknown values#
Ni = (Nx-2)*(Ny-2)

# setting up matrices for crank nicoleson approach
A = np.zeros((Ni,Ni), complex)
B = np.zeros((Ni,Ni), complex)

#filling in the A and B matrices.
for k in range(Ni):     
    
    # k = (i-1)*(Ny-2) + (j-1)
    i = 1 + k//(Ny-2)
    j = 1 + k%(Ny-2)
    
    # Main central diagonal.
    A[k,k] = 1 + 2*px + 2*py + 1j*Dt/2*v[i,j]
    B[k,k] = 1 - 2*px - 2*py - 1j*Dt/2*v[i,j]
    
    if i != 1: # Lower lone diagonal.
        A[k,(i-2)*(Ny-2)+j-1] = -py 
        B[k,(i-2)*(Ny-2)+j-1] = py
        
    if i != Nx-2: # Upper lone diagonal.
        A[k,i*(Ny-2)+j-1] = -py
        B[k,i*(Ny-2)+j-1] = py
    
    if j != 1: # Lower main diagonal.
        A[k,k-1] = -px 
        B[k,k-1] = px 

    if j != Ny-2: # Upper main diagonal.
        A[k,k+1] = -px
        B[k,k+1] = px

#solving system #
Asp = csc_matrix(A)

x = np.linspace(0, L, Ny-2) # Array of spatial points.
y = np.linspace(0, L, Ny-2) # Array of spatial points.
x, y = np.meshgrid(x, y)
psis = [] # To store the wave function at each time step.

#initialing the wave function#
psi = psi0(x, y, x0, y0) 

#setting boundary conditions#
psi[0,:] = 0
psi[-1,:] = 0
psi[:,0] = 0
psi[:,-1] = 0

#setting values to at slit locations (comment out of standard particle in a box problem)#
psi = Slit(psi, j0, j1, i0, i1, i2, i3) 
psis.append(np.copy(psi))

# We solve the matrix system at each time step in order to obtain the wave function.
for i in range(1,Nt):
    psi_vect = psi.reshape((Ni)) 
    b = np.matmul(B,psi_vect) 
    psi_vect = spsolve(Asp,b)
    psi = psi_vect.reshape((Nx-2,Ny-2))

	#applying boundary conditions on slits (comment out for standard particle in a box problem)#
    psi = Slit(psi, j0, j1, i0, i1, i2, i3) 
    psis.append(np.copy(psi))

# calculating the magnitude of the wave function
mod_psis = [] 
for wavefunc in psis:
    re = np.real(wavefunc) 
    im = np.imag(wavefunc) 
    mod = np.sqrt(re**2 + im**2)
    #mod = re**2 + im**2
    mod_psis.append(mod) 
    
#Deleting unused matrices to free up RAM
del psis
del B
del psi_vect


#plotting at different time values (uncomment to get surface plot of wave function at different time steps)

#psi1 = mod_psis[0]
#psi2 = mod_psis[175]
#psi3 = mod_psis[350]
#psi4 =  mod_psis[499]

#plot_PSI(psi1,psi2,psi3,psi4,x,y)



#CREATING ANIMATION#
fig = plt.figure() # We create the figure.
ax = fig.add_subplot(111, xlim=(0,L), ylim=(0,L)) # We add the subplot to the figure.

img = ax.imshow(mod_psis[0], extent=[0,L,0,L], cmap=plt.get_cmap("plasma"), vmin=0, vmax=np.max(mod_psis), zorder=1, interpolation="none") # Here the modulus of the 2D wave function shall be represented.

#
# We paint the walls of the double slit with rectangles.
wall_bottom = Rectangle((j0*Dy,0),     w, i3*Dy,      color="w", zorder=50) # (x0, y0), width, height
wall_middle = Rectangle((j0*Dy,i2*Dy), w, (i1-i2)*Dy, color="w", zorder=50)
wall_top    = Rectangle((j0*Dy,i0*Dy), w, i3*Dy,      color="w", zorder=50)

# We add the rectangular patches to the plot (comment out for standard particle in box problem).
ax.add_patch(wall_bottom)
ax.add_patch(wall_middle)
ax.add_patch(wall_top)

# We define the animation function for FuncAnimation.

def animate(i):
    
    """
   	creates animations from the images
    """
    
    img.set_data(mod_psis[i]) # Fill img with the modulus data of the wave function.
    img.set_zorder(1)
    
    return img, # We return the result ready to use with blit=True.


anim = FuncAnimation(fig, animate, interval=1, frames=np.arange(0,Nt,2), repeat=False, blit=0) # We generate the animation.

cbar = fig.colorbar(img)
plt.show() # We show the animation finally.

#writer to make animation#
writergif = animation.PillowWriter(fps=30)

#saving animation#
anim.save('slit2.gif',writer=writergif)
