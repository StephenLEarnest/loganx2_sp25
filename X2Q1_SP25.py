# region imports
import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt


# endregion

# region function definitions
def S(x):
    s = quad(lambda t: np.sin(t ** 2), 0, x)  # the solution for S(x) found using quad
    return s[0]


def Exact(x):
    return 1 / (2.5 - S(x)) + 0.01 * x ** 2  # exact solution for i.v.p. at x


def ODE_System(x, y):
    Y = y[0]  # rename first state variable into convenient name
    Ydot = (Y - 0.01 * x ** 2) ** 2 * np.sin(x ** 2) + 0.02 * x  # calculate derivatives of state variable(s)
    return [Ydot]


def Plot_Result(*args):
    xRange_Num, y_Num, xRange_Xct, y_Xct = args  # unpack args containing plottable arrays for numerical and exact solution

    plt.figure(figsize=(10, 6))

    # Plot exact solution as solid line
    plt.plot(xRange_Xct, y_Xct, '-', label='Exact')

    # Plot numerical solution as upward triangles
    plt.plot(xRange_Num, y_Num, '^', label='Numerical', markersize=8)

    # Set axis limits and labels
    plt.xlim(0.0, 6.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('x')
    plt.ylabel('y')

    # Customize ticks
    plt.tick_params(direction='in', top=True, right=True)
    plt.xticks(np.arange(0, 6.1, 1.0), [f'{x:.1f}' for x in np.arange(0, 6.1, 1.0)])
    plt.yticks(np.arange(0, 1.1, 0.2), [f'{y:.1f}' for y in np.arange(0, 1.1, 0.2)])

    # Add title and legend
    plt.title("IVP:  y'=(y-0.01x^2 )^2 sin(x^2 )+0.02x, y(0)=0.4")
    plt.legend()

    # Display the plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def main():
    """
    This function solves the initial value problem of problem 1 of exam 2, Spring 2023.
    y'=(y-0.01x**2)**2*sin(x**2)+0.02x
    y(0)=0.4
    It then plots the numerical solution and the exact solution according to the formatting criteria
    """
    xRange = np.arange(0, 5.2, 0.2)  # create a numpy array for the x range to evaluate numerical solution (h=0.2)
    xRange_xct = np.linspace(0, 5, 500)  # create a numpy array for the x range for the exact solution
    Y0 = [0.4]  # create initial conditions
    sln = solve_ivp(ODE_System, [0, 5], Y0, t_eval=xRange)  # numerically solve i.v.p. with default RK45 method
    xctSln = np.array([Exact(x) for x in xRange_xct])  # produce array of y values for exact solution
    Plot_Result(xRange, sln.y[0], xRange_xct, xctSln)  # call the plotting function to produce the required plot


# end region

# region function calls
if __name__ == "__main__":
    main()
# end region