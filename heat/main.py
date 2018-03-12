import numpy as np
import matplotlib.pyplot as plt

#Following is the code for theta method to solve the heat equation


def init_f(x, t=0):
    return np.sqrt(1e-4/(t+1e-4))*np.exp(-x**2/(4*(t+1e-4)))


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
    

def convergence_study(theta):
    num_points = 41
    mu = np.linspace(0.01,0.61,301)
    dx = 2.0/(num_points-1)
    dt = mu*(dx**2)
    errors = np.zeros_like(mu)
    num_steps = (0.01/(mu*dx**2)).astype(int)

    for i,m in enumerate(mu):
        U, X, T     = theta_method(theta, m, num_points,num_steps[i])
        err         = U[-1,:] - init_f(X[0,:],num_steps[i]*dt[i])
        errors[i]   = np.sqrt(np.sum(err**2))

    fig, ax = plt.subplots()
    ax.set_title(r'Convergence plot for $\theta = {}$'.format(theta)+'\n\n\n')
    ax.loglog(mu, errors,'b')
    ax.plot([0.5,0.5],[np.min(errors),np.max(errors)],'r--')
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel('Root mean squared error after 10ms')
    new_t = [xt*(dx**2) for xt in ax.get_xticks()]
    xmin,xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim([xmin*(dx**2), xmax*(dx**2)])
    ax2.loglog(dt, errors,'b')
    ax2.set_xlabel('$dt$')
    #fig.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()


def stability_analysis():
    mu = np.linspace(0,10,100)
    theta = np.linspace(0,1,100)
    mu,theta = np.meshgrid(mu,theta)
    eigval = np.zeros_like(mu)
    for m,t,e in zip(np.nditer(mu), np.nditer(theta), np.nditer(eigval, op_flags=['readwrite'])):
            e[...] = np.max(abs(np.linalg.eigvals(construct_mat(m,t,5))))

    eigval = (eigval>1)+0
    fig, ax = plt.subplots()
    cs = ax.contourf(mu,theta,eigval,1)
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
                for pc in cs.collections]

    plt.legend(proxy, ["Stable (Max eigen value < 1)", "Unstable (Max eigen value > 1)"])

    ax.plot([10,0],[0.5,0.5],'r--')
    ax.plot([0.5,0.5],[1,0],'r--')
    ax.set_xlabel(r'$\mu \left( = \frac{\Delta t}{\Delta x^2}\right)$')
    ax.set_ylabel(r'$\theta$')
    ax.set_title(r"Stability plot for $\theta$ method")
    plt.savefig('eig.png')
    plt.show()

if __name__=='__main__':
    #stability_analysis()
    for theta in np.linspace(0,1,21):
        convergence_study(theta)
    numsteps = 45
    U, X, T = theta_method(1,0.6,100,numsteps)
    plt.contourf(X,T,U,200)

    fig, axs = plt.subplots(3,3)
    fig.suptitle(r'$\theta = 0.2$  $\frac{\Delta t}{\Delta x^2} = 0.2$')

    for i,ax in enumerate(axs.flatten()):
        ax.plot(X[0,:],U[i*numsteps/9],'b-')
        ax.plot(X[0,:],init_f(X[0,:],T[i*numsteps/9,0]),'r--')
        ax.set_title("t = {:.2f}ms".format(T[i*numsteps/9,0]*1000))

    plt.tight_layout()
    plt.show()
