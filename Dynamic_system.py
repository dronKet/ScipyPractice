import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pylab as p
from scipy import integrate
import math


class LotkiVolterra:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


    def dX_dt(self, X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([self.a * X[0] - self.b * X[0] * X[1],
                         -self.c * X[1] + self.d * self.b * X[0] * X[1]])

    def d2X_dt2(self, X, t=0):
        """ Return the Jacobian matrix evaluated in X. """
        return np.array([[self.a - self.b * X[1], -self.b * X[0]],
                         [self.b * self.d * X[1], -self.c + self.b * self.d * X[0]]])

    def X_f1_Func(self):
        return np.array([self.c / (self.d * self.b), self.a / self.b])

    def analysis(self, t, X0):
        X_f0 = np.array([0., 0.])
        self.X_f1=self.X_f1_Func()
        if all(self.dX_dt(X_f0) != np.zeros(2)) and all(self.dX_dt(self.X_f1) != np.zeros(2)):
             return None

        A_f0 = self.d2X_dt2(X_f0)
        A_f1 = self.d2X_dt2(self.X_f1)
        lambda1, lambda2 = np.linalg.eigvals(A_f1)  # >>> (1.22474j, -1.22474j)
        # They are imaginary numbers. The fox and rabbit populations are periodic as follows from further
        # analysis. Their period is given by:
        # print(lambda1, lambda2)
        T_f1 = 2 * np.pi / abs(lambda1)  # >>> 5.130199

        X, infodict = integrate.odeint(self.dX_dt, X0, t, full_output=True)
        # print(X)
        return X


    def plot_x_t(self, name='rabbits_and_foxes_1.png'):
        t = np.linspace(0, 15, 1000)  # time
        X0 = np.array([10, 5])  # initials conditions: 10 rabbits and 5 foxes
        res = self.analysis(t, X0)
        if res is None:
            return None
        rabbits, foxes = res.T
        f1 = p.figure()
        p.plot(t, rabbits, 'r-', label='Rabbits')
        p.plot(t, foxes, 'b-', label='Foxes')
        p.grid()
        p.legend(loc='best')
        p.xlabel('time')
        p.ylabel('population')
        p.title('Evolution of fox and rabbit populations')
        f1.savefig(name)

    def plot_x_t_phase(self, name='rab_and_fox.png'):
        t = np.linspace(0, 15, 1000)  # time
        X0 = np.array([10, 5])  # initials conditions: 10 rabbits and 5 foxes
        res = self.analysis(t, X0)
        if res is None:
            return None
        rabbits, foxes = res.T
        values = np.linspace(0.3, 0.9, 5)  # position of X0 between X_f0 and X_f1
        vcolors = p.cm.autumn_r(np.linspace(0.3, 1., len(values)))  # colors for each trajectory
        print(values)
        f2 = p.figure()

        # -------------------------------------------------------
        # plot trajectories
        for v, col in zip(values, vcolors):
            X0 = v * self.X_f1  # starting point
            X = integrate.odeint(self.dX_dt, X0, t)  # we don't need infodict here
            p.plot(X[:, 0], X[:, 1], lw=3.5 * v, color=col, label='X0=(%.f, %.f)' % (X0[0], X0[1]))

        # -------------------------------------------------------
        # define a grid and compute direction at each point
        ymax = p.ylim(ymin=0)[1]  # get axis limits
        xmax = p.xlim(xmin=0)[1]
        nb_points = 20

        x = np.linspace(0, xmax, nb_points)
        y = np.linspace(0, ymax, nb_points)

        X1, Y1 = np.meshgrid(x, y)  # create a grid
        DX1, DY1 = self.dX_dt([X1, Y1])  # compute growth rate on the gridt
        M = (np.hypot(DX1, DY1))  # Norm of the growth rate
        M[M == 0] = 1.  # Avoid zero division errors
        DX1 /= M  # Normalize each arrows
        DY1 /= M

        # -------------------------------------------------------
        # Drow direction fields, using matplotlib 's quiver function
        # I choose to plot normalized arrows and to use colors to give information on
        # the growth speed
        p.title('Trajectories and direction fields')
        Q = p.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=p.cm.jet)
        p.xlabel('Number of rabbits')
        p.ylabel('Number of foxes')
        p.legend()
        p.grid()
        p.xlim(0, xmax)
        p.ylim(0, ymax)
        f2.savefig('rabbits_and_foxes_2.png')




class LotkiVolterraModified(LotkiVolterra):
    def __init__(self,a,b,c,d,e):
        super().__init__(a,b,c,d)
        self.e = e

    def dX_dt(self, X, t=0):
        """ Return the growth rate of fox and rabbit populations. """
        return np.array([self.a * X[0] - self.b * X[0] * X[1]-self.e*(X[0]**2),
                         -self.c * X[1] + self.d * self.b * X[0] * X[1]])

    def d2X_dt2(self, X, t=0):
        """ Return the Jacobian matrix evaluated in X. """
        return np.array([[self.a - self.b * X[1]-2*self.e*X[0], -self.b * X[0]],
                         [self.b * self.d * X[1], -self.c + self.b * self.d * X[0]]])

    def X_f1_Func(self):
        return np.array([self.c / (self.d * self.b), (self.a-self.e*self.c) / (self.d*(self.b**2))])


if __name__ == "__main__":
    # Definition of parameters
    a = 1.
    b = 0.1
    c = 1.5
    d = 0.75
    e = 0.8
    system = LotkiVolterra(a, b, c, d)
    system.plot_x_t('test.png')
   # plt.show()
    modif = LotkiVolterraModified(a, b, c, d,e )
    modif.plot_x_t('test1.png')
    #plt.show()
    modif.plot_x_t_phase('test.png')
    plt.show()