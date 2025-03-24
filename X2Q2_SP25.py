from scipy.integrate import solve_ivp
from math import sin
import numpy as np
from matplotlib import pyplot as plt

class circuit():
    def __init__(self, R=20, L=20, C=0.05, A=20, w=20, p=0):
        """
        Initializes the circuit with default or user-defined parameters.
        :param R: Resistance in ohms (Ω)
        :param L: Inductance in henries (H)
        :param C: Capacitance in farads (F)
        :param A: Amplitude of the voltage source (V)
        :param w: Frequency of the voltage source (rad/s)
        :param p: Phase of the voltage source (radians)
        """
        self.R = R
        self.L = L
        self.C = C
        self.A = A
        self.w = w
        self.p = p

    def voltage_source(self, t):
        return self.A * np.sin(self.w * t + self.p)

    def ode_system(self, t, X):
        """
        Defines the system of differential equations for the circuit.
        :param X: The current values of the state variables [i1, i2, vc]
        :param t: The current time
        :return: List of derivatives of state variables
        """
        i1, i2, vc = X
        vi = self.voltage_source(t)
        di1_dt = (vi - self.R * i2 - vc) / self.L
        di2_dt = (i1 - i2) / self.C
        return [di1_dt, di2_dt, i2]

    def simulate(self, t=10, pts=500):
        """
        Simulates the transient behavior of the circuit.
        :param t: Time over which to carry out the simulation in seconds
        :param pts: Number of points in the simulation
        """
        t_eval = np.linspace(0, t, pts)
        y0 = [0, 0, 0]  # Initial conditions for i1, i2, and vc
        sol = solve_ivp(self.ode_system, [0, t], y0, t_eval=t_eval, method='RK45')
        self.t = sol.t
        self.i1 = sol.y[0]
        self.i2 = sol.y[1]
        self.vc = sol.y[2]

    def doPlot(self, ax=None):
        """
        Plots the currents i1, i2, and voltage vc over time.
        :param ax: Matplotlib axis object for GUI integration (optional)
        """
        if ax is None:
            ax = plt.subplot()
            QTPlotting = False  # Using CLI to show the plot
        else:
            QTPlotting = True

        ax.plot(self.t, self.i1, label='$i_1(t)$', linestyle='-', color='black')
        ax.plot(self.t, self.i2, label='$i_2(t)$', linestyle='--', color='black')
        ax.plot(self.t, self.vc, label='$v_c(t)$', linestyle=':', color='black')
        ax.set_xlabel("t (s)")
        ax.set_ylabel("i, v")
        ax.legend()
        ax.grid()

        if not QTPlotting:
            plt.show()


def main():
    """
    For solving the circuit problem interactively.
    """
    goAgain = True
    while goAgain:
        R = float(input("Enter resistance R (Ω) [Default: 20]: ") or 20)
        L = float(input("Enter inductance L (H) [Default: 20]: ") or 20)
        C = float(input("Enter capacitance C (F) [Default: 0.05]: ") or 0.05)
        A = float(input("Enter voltage amplitude (V) [Default: 20]: ") or 20)
        w = float(input("Enter voltage frequency (rad/s) [Default: 20]: ") or 20)
        p = float(input("Enter voltage phase (radians) [Default: 0]: ") or 0)

        Circuit = circuit(R, L, C, A, w, p)
        Circuit.simulate(t=10, pts=500)
        Circuit.doPlot()

        repeat = input("Do you want to run another simulation? (yes/no): ").strip().lower()
        if repeat != 'yes':
            goAgain = False

if __name__ == "__main__":
    main()



