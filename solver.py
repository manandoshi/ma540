import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def u_0(x):
    #return np.sin(np.pi*x)
    #return np.e**(-x*x)
    return np.maximum(0,1-x*x)

def l_boundary(x):
    return np.zeros_like(x)

def r_boundary(x):
    return np.zeros_like(x)

class Solver:
    def __init__(self, dt=5e-4, courant=1e-1, a=0.6, space="F", 
                 time="B", Th=2.0, x_min=-2.0, x_max=2.0, 
                 u0=u_0, l_boundary=l_boundary, r_boundary=r_boundary):
        
        #Setting space and timestep size
        self.dt = dt
        self.dx = dx = abs(a*self.dt)/courant
        
        #Setting wavespeed
        self.a  = a

        #Setting schemes used for time and space discretization
        self.space = space
        self.time  = time

        #Setting domain
        self.x_min = x_min
        self.x_max = x_max
        
        #creating a meshgrid for x and t
        self.x, self.t = np.mgrid[x_min:x_max:1j*(int((x_max-x_min)/dx)+1),0:Th:1j*int(Th/dt + 1)]
        
        #creating matrices for u and du/dx
        self.u = np.zeros_like(self.x)
        self.u_x = np.zeros_like(self.x)

        #setting initial and boundary conditions
        self.u[:,0] = u0(self.x[:,0])
        #self.u[0,:] = l_boundary(self.x[0,:])

    #Function to compute du/dx at timestep i
    def compute_u_x(self, i):
        #Backward difference
        if self.space.lower() == 'b' or self.space.lower() == 'backward':
            self.u_x[:,i] = ( self.u[:,i] - np.roll(self.u[:,i],1) ) / self.dx
            self.u_x[0,i] = 0
        
        #Forward difference
        if self.space.lower() == 'f' or self.space.lower() == 'forward':
            self.u_x[:,i] = (-1*self.u[:,i]+np.roll(self.u[:,i],-1))/self.dx
            self.u_x[-1,i] = 0

        #Central difference
        if self.space.lower() == 'c' or self.space.lower() == 'central':
            self.u_x[:,i] = (np.roll(self.u[:,i],-1) - np.roll(self.u[:,i],1))/(2*self.dx)
            self.u_x[-1,i] = (self.u[-1,i] - self.u[-2,i])/self.dx
            self.u_x[0,i] = (self.u[1,i] - self.u[0,i])/self.dx

    def update_u(self, i):
        #Backward in time
        if self.time.lower() == 'f' or self.time.lower() == 'backward':
            self.u[:,i+1] = self.u[:,i] - self.a*(self.dt)*self.u_x[:,i]

        if self.time.lower() == 'c' or self.time.lower() == 'central':
            if i != 0:
                self.u[:,i+1] = self.u[:,i-1] - 2*self.a*(self.dt)*self.u_x[:,i]
            else:
                self.u[:,i+1] = self.u[:,i] - self.a*(self.dt)*self.u_x[:,i]

        if self.time.lower() == 'lf' or self.time.lower() == 'lax friedrich' or self.time.lower() == 'lax-friedrich' :
            u_t = 0.5*(np.roll(self.u[:,i],1) + np.roll(self.u[:,i],-1))
            u_t[0]  = self.u[0,i]
            u_t[-1] = self.u[-1,i]
            self.u[:,i+1] = u_t - self.a*(self.dt)*self.u_x[:,i]

        if self.time.lower() == 'lw' or self.time.lower() == 'lax wendroff' or self.time.lower() == 'lax-wendroff' :
            c = self.a*self.dt/self.dx

            a_0 = 1-c**2
            u_0 = self.u[:,i]

            a_1 = c + (c**2)/2
            u_1 = np.roll(self.u[:,i], 1)
            
            a_2 = (c**2)/2 - c
            u_2 = np.roll(self.u[:,i], -1)

            self.u[:,i+1] = a_0*u_0 + a_1*u_1 + a_2*u_2
        return

    def solve(self):
        for i, t in enumerate(self.t[0,:-1]):
            self.compute_u_x(i)
            self.update_u(i)
        return

    def plot_contour(self, ax=None, cb=True):
        if ax is None:
            plt.contour(self.x,self.t,self.u,200)
        else:
            cs = ax.contourf(self.x,self.t,self.u,200, vmin=-0.2, vmax=1.2)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$t$')
            if cb:
                cbar = plt.colorbar(cs)
                cbar.ax.set_ylabel('$u$')
            return cs
    
    def plot_3d(self, ax=None):
        if ax is None:
            pass
        else:
            ax.plot_surface(self.x,self.t,self.u)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$t$')

    def animate(self):
        def run(i):
            #t, u = data
            u_line.set_ydata(self.u[:,i])
            #print(t)
            return u_line,

        def init():
            ax.set_ylim(-1.0, 1.0)
            ax.set_xlim(self.x[0,0], self.x[-1,0])
            u_line.set_data(self.x[:,0], self.t[:,0])
            return u_line,

        fig, ax = plt.subplots()
        u_line, = ax.plot(self.x[:,0], self.t[:,0])
        ax.grid()
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u$')

        ani = animation.FuncAnimation(fig, run, np.arange(len(self.t[0,:])), init_func=init,
                                              interval=self.dt*10000, blit=False)

        ani.save('a.avi')

if __name__ == '__main__':
    s = Solver(space='c', time='c', dt=1e-2, courant=0.8, Th=0.5)
    s.solve()
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    s.plot_contour(ax2)
    s.plot_3d(ax1)
    plt.show()
    s.animate()
