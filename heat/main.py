import numpy as np
import matplotlib.pyplot as plt

#Following is the code for theta method to solve the heat equation


def init_f(x):
    return np.exp(-100*x**2)


def construct_matrix(mu, theta, size):
    l = mu*theta
    mat = np.zeros([size,size])
    mat[np.arange(size),np.arange(size)] = 1+2*l
    mat[np.arange(size-1)+1,np.arange(size-1)] = -1*l
    mat[np.arange(size-1),np.arange(size-1)+1] = -1*l
    return mat


def construct_mat(mu,theta,size):
    m1 = construct_matrix(mu,theta,size)
    m2 = construct_matrix(mu,theta-1,size)
    return np.linalg.inv(m1).dot(m2)


def theta_method(theta, mu, num_points, tsteps, init_fn = init_f):
    dx = 2.0/(num_points-1)
    dt = mu*(dx**2)
    x  = np.linspace(-1,1,num_points)
    t  = np.arange(tsteps)*dt
    X,T = np.meshgrid(x,t)
    u_0  = init_f(x)
    
    U  = np.zeros([num_points,tsteps])

    U[:,0] = u_0
    mat = construct_mat(mu, theta, num_points-2)

    for i in range(tsteps-1):
        U[1:-1,i+1] = mat.dot(U[1:-1,i])
    
    return U.T, X, T
    
    
if __name__=='__main__':
    numsteps = 900
    U, X, T = theta_method(1,0.2,100,numsteps)
    plt.contourf(X,T,U,200)
    plt.show()

    fig, axs = plt.subplots(3,3)
    fig.suptitle(r'$\theta = 0.2$  $\frac{\Delta t}{\Delta x^2} = 0.2$')

    for i,ax in enumerate(axs.flatten()):
        ax.plot(X[0,:],U[i*numsteps/9])
        ax.set_title("t = {:.2f}ms".format(T[i*numsteps/9,0]*1000))

    plt.tight_layout()
    plt.show()


