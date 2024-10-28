import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-10

# Flow simulator V, V_fdi, V_cdi = func(x,y)
def uniform(x, y, uc=14.2, vc=0.0):
    """ Uniform flow
    u(x,y) = const
    v(x,y) = const
    """
    assert x.shape == y.shape
    x, y = x-np.mean(x), y-np.mean(y)
    u = np.zeros_like(x)+uc
    v = np.zeros_like(x)+vc
    V = (u,v)
    V_fdi, V_cdi = V, V
    return V, V_fdi, V_cdi   


def solid_rot(x, y, omega=0.06):
    """ Solid body rotation
    u(x,y) = r*omega*(-sin(theta))
    v(x,y) = r*omega*cos(theta)
    """
    assert x.shape == y.shape
    x, y = x-np.mean(x), y-np.mean(y)
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    u, v  = -r*omega*np.sin(theta), r*omega*np.cos(theta)
    V=(u,v)

    # FDI representation of (u,v)
    theta2 = theta+omega
    x2, y2 = r*np.cos(theta2), r*np.sin(theta2)
    V_fdi = (x2-x, y2-y)

    # CDI representation of (u,v)
    coeff = np.tan(omega/2.0)/(omega/2.0)
    V_cdi = (coeff*u,coeff*v)
    return V, V_fdi, V_cdi


def lamb_oseen(x, y, Gamma=5e3, rc=40):
    """ Lamb-Oseen flow
    V_r = 0
    V_theta = (1-exp(r^2/rc^2))*Gamma/(2*pi*r)
    u(x,y) = V_theta*(-sin(theta))
    v(x,y) = V_theta*cos(theta)
    """
    assert x.shape == y.shape
    x, y = x-np.mean(x), y-np.mean(y)
    r = np.sqrt(x**2+y**2)+EPS
    theta = np.arctan2(y,x)
    Amp = Gamma*(1-np.exp(-r**2/rc**2))/(2*np.pi*r)  # circumferential vel
    u, v  = -Amp*np.sin(theta), Amp*np.cos(theta)
    V=(u,v)

    # FDI representation of (u,v)
    omega = Amp/r # angular velocity
    theta2 = theta + omega
    x2, y2 = r*np.cos(theta2), r*np.sin(theta2)
    V_fdi = (x2-x, y2-y)

    # CDI representation of (u,v)
    r2 = np.copy(r)
    for iter in range(10):
        amp = Gamma*(1-np.exp(-r2**2/rc**2))/(2*np.pi*r2)
        r_ = r2*np.cos(amp/(2*r2))
        r2 = r2+(r-r_)
        # print(np.mean(np.abs(r-r_)))
    temp = 2*r2*np.sin(amp/(2*r2))
    V_cdi = -temp*np.sin(theta), temp*np.cos(theta)
    return V, V_fdi, V_cdi


def sin_flow(x, y, a=6, b=128, scale=5):
    assert x.shape == y.shape
    x, y = x-np.mean(x), y-np.mean(y)
    theta = np.arctan(a*np.cos(2*np.pi*x/b)*2*np.pi/b)
    u = scale*np.cos(theta)
    v = scale*np.sin(theta)
    V=(u,v)

    def forward(x1,y1): 
        """ Integrate along the streamline started at (x1,y1),
        Output the final position (x2, y2)
        """
        x2, y2 = np.copy(x1), np.copy(y1)
        N = 1000
        ds = scale/N
        for i in range(N):
            alpha1 = np.arctan(a*np.cos(2*np.pi*x2/b)*2*np.pi/b)
            x2 = x2 + ds*np.cos(alpha1)
            y2 = y2 + ds*np.sin(alpha1)
        return x2, y2

    # FDI representation of (u,v)
    x_, y_ = forward(x,y)
    V_fdi = (x_-x, y_-y)

    # CDI representation of (u,v)
    x1, y1 = x-u/2, y-v/2
    for i in range(10):
        x2, y2 = forward(x1,y1)
        xm, ym = (x1+x2)/2, (y1+y2)/2
        x1 = x1+(x-xm)
        y1 = y1+(y-ym)
    V_cdi = (x2-x1, y2-y1)
    return V, V_fdi, V_cdi


def unit_test():
    x = np.arange(0,256,8)
    x, y  = np.meshgrid(x,x)
    
    for flow in [uniform, solid_rot, lamb_oseen, sin_flow]:
        flow_name = flow.__name__
        
        V, V_fdi, V_cdi = flow(x,y)
        plt.figure(figsize=(15,4))
        plt.subplot(131);plt.quiver(x,y,V[0],V[1]); plt.title(f"Truth({flow_name})")
        plt.subplot(132);plt.quiver(x,y,V_fdi[0],V_fdi[1]); plt.title(f"FDI({flow_name})")
        plt.subplot(133);plt.quiver(x,y,V_cdi[0],V_cdi[1]); plt.title(f"CDI({flow_name})")
    plt.show()


if __name__=='__main__':
    unit_test()