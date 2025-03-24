# region imports
import numpy as np
import math
from scipy.optimize import fsolve
import random as rnd


# endregion

# region class definitions
class UC:  # a units conversion class
    def __init__(self):
        """
        This unit converter class is useful for the pipe network and perhaps other problems.
        The strategy is (number in current units)*(conversion factor)=(number desired units), for instance:
            1(ft)*(self.ft_to_m) = 1/3.28084 (m)
            1(in^2)*(self.in2_to_m2) = 1*(1/(12*3.28084))**2 (m^2)
        """

    # region class constants
    ft_to_m = 1 / 3.28084
    ft2_to_m2 = ft_to_m ** 2
    ft3_to_m3 = ft_to_m ** 3
    ft3_to_L = ft3_to_m3 * 1000
    L_to_ft3 = 1 / ft3_to_L
    in_to_m = ft_to_m / 12
    m_to_in = 1 / in_to_m
    in2_to_m2 = in_to_m ** 2
    m2_to_in2 = 1 / in2_to_m2
    g_SI = 9.80665  # m/s^2
    g_EN = 32.174  # 32.174 ft/s^2
    gc_EN = 32.174  # lbm*ft/lbf*s^2
    gc_SI = 1.0  # kg*m/N*s^2
    lbf_to_kg = 1 / 2.20462
    lbf_to_N = lbf_to_kg * g_SI
    pa_to_psi = (1 / (lbf_to_N)) * in2_to_m2

    # endregion

    @classmethod
    def viscosityEnglishToSI(cls, mu, toSI=True):
        cf = (1 / cls.ft2_to_m2) * (cls.lbf_to_kg) * cls.g_SI
        return mu * cf if toSI else mu / cf

    @classmethod
    def densityEnglishToSI(cls, rho, toSI=True):
        cf = cls.lbf_to_kg / cls.ft3_to_m3
        return rho * cf if toSI else rho / cf

    @classmethod
    def head_to_pressure(cls, h, rho, SI=True):
        if SI:
            cf = rho * cls.g_SI / cls.gc_SI
            return h * cf
        else:
            cf = rho * cls.g_EN / cls.gc_EN * (1 / 12) ** 2
            return h * cf

    @classmethod
    def m_to_psi(cls, h, rho):
        return cls.head_to_pressure(h, rho) * cls.pa_to_psi

    @classmethod
    def psi_to_m(cls, p, rho):
        pa = p / cls.pa_to_psi
        h = pa / (rho * cls.g_SI)
        return h


class Fluid:
    def __init__(self, mu=0.00089, rho=1000, SI=True):
        self.mu = mu if SI else UC.viscosityEnglishToSI(mu)
        self.rho = rho if SI else UC.densityEnglishToSI(rho)
        self.nu = self.mu / self.rho


class Node:
    def __init__(self, Name='a', Pipes=[], ExtFlow=0):
        self.name = Name
        self.pipes = Pipes
        self.extFlow = ExtFlow
        self.QNet = 0
        self.P = 0
        self.oCalculated = False

    def getNetFlowRate(self):
        Qtot = self.extFlow
        for p in self.pipes:
            Qtot += p.getFlowIntoNode(self.name)
        self.QNet = Qtot
        return self.QNet

    def setExtFlow(self, E, SI=True):
        self.extFlow = E if SI else E * UC.ft3_to_L


class Loop:
    def __init__(self, Name='A', Pipes=[]):
        self.name = Name
        self.pipes = Pipes

    def getLoopHeadLoss(self):
        deltaP = 0
        startNode = self.pipes[0].startNode
        for p in self.pipes:
            phl = p.getFlowHeadLoss(startNode)
            deltaP += phl
            startNode = p.endNode if startNode != p.endNode else p.startNode
        return deltaP


class Pipe:
    def __init__(self, Start='A', End='B', L=100, D=200, r=0.00025, fluid=Fluid(), SI=True):
        self.startNode = min(Start.lower(), End.lower())
        self.endNode = max(Start.lower(), End.lower())
        self.length = L if SI else UC.ft_to_m * L
        self.rough = r if SI else UC.ft_to_m * r
        self.fluid = fluid
        self.d = D / 1000.0 if SI else UC.in_to_m * D
        self.relrough = self.rough / self.d
        self.A = np.pi * (self.d / 2) ** 2
        self.Q = 10
        self.vel = self.V()
        self.reynolds = self.Re()
        self.hl = 0

    def V(self):
        self.vel = (self.Q / 1000) / self.A  # Q in L/s to m^3/s, A in m^2
        return self.vel

    def Re(self):
        self.reynolds = self.fluid.rho * self.V() * self.d / self.fluid.mu
        return self.reynolds

    def FrictionFactor(self):
        Re = self.Re()
        rr = self.relrough

        def CB():
            cb = lambda f: 1 / (f ** 0.5) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * f ** 0.5))
            result = fsolve(cb, (0.01))
            return result[0]

        def lam():
            return 64 / Re

        if Re >= 4000:
            return CB()
        if Re <= 2000:
            return lam()
        CBff = CB()
        Lamff = lam()
        mean = (CBff * (Re - 2000) + Lamff * (4000 - Re)) / (4000 - 2000)
        sig_1 = (1 - (Re - 3000) / 1000) * 0.2 * mean
        sig_2 = (1 - (3000 - Re) / 1000) * 0.2 * mean
        sig = sig_1 if Re >= 3000 else sig_2
        return rnd.normalvariate(mean, sig)

    def frictionHeadLoss(self):
        g = 9.81
        ff = self.FrictionFactor()
        self.hl = ff * (self.length / self.d) * (self.vel ** 2) / (2 * g)
        return self.hl

    def getFlowHeadLoss(self, s):
        nTraverse = 1 if s == self.startNode else -1
        nFlow = 1 if self.Q >= 0 else -1
        return nTraverse * nFlow * self.frictionHeadLoss()

    def Name(self):
        return self.startNode + '-' + self.endNode

    def oContainsNode(self, node):
        return self.startNode == node or self.endNode == node

    def printPipeFlowRate(self, SI=True):
        q_units = 'L/s' if SI else 'cfs'
        q = self.Q if SI else self.Q * UC.L_to_ft3
        print(f'The flow in segment {self.Name()} is {q:.2f} ({q_units}) and Re={self.reynolds:.1f}')

    def printPipeHeadLoss(self, SI=True):
        cfd = 1000 if SI else UC.m_to_in
        unitsd = 'mm' if SI else 'in'
        cfL = 1 if SI else UC.m_to_in
        unitsL = 'm' if SI else 'in'
        cfh = cfd
        units_h = unitsd
        print(
            f"head loss in pipe {self.Name()} (L={self.length * cfL:.2f} {unitsL}, d={self.d * cfd:.2f} {unitsd}) is {self.hl * cfh:.2f} {units_h} of water")

    def getFlowIntoNode(self, n):
        if n == self.startNode:
            return -self.Q
        return self.Q


class PipeNetwork:
    def __init__(self, Pipes=[], Loops=[], Nodes=[], fluid=Fluid()):
        self.loops = Loops
        self.nodes = Nodes
        self.Fluid = fluid
        self.pipes = Pipes
        self.pipe_order = [
            ('a', 'b'), ('a', 'h'), ('b', 'c'), ('b', 'e'), ('c', 'd'),
            ('c', 'f'), ('d', 'g'), ('e', 'f'), ('e', 'i'), ('f', 'g'),
            ('g', 'j'), ('h', 'i'), ('i', 'j')
        ]

    def findFlowRates(self):
        N = len(self.nodes) + len(self.loops)
        Q0 = np.full(N, 10)

        def fn(q):
            for n in self.nodes:
                n.P = 0
                n.oCalculated = False
            for i in range(len(self.pipes)):
                self.pipes[i].Q = q[i]
            L = self.getNodeFlowRates()
            L += self.getLoopHeadLosses()
            return L

        FR = fsolve(fn, Q0)
        for i in range(len(self.pipes)):
            self.pipes[i].Q = FR[i]
        return FR

    def getNodeFlowRates(self):
        qNet = [n.getNetFlowRate() for n in self.nodes]
        return qNet

    def getLoopHeadLosses(self):
        lhl = [l.getLoopHeadLoss() for l in self.loops]
        return lhl

    def getNodePressures(self, knownNodeP, knownNode):
        for n in self.nodes:
            n.P = 0.0
            n.oCalculated = False
        for l in self.loops:
            startNode = l.pipes[0].startNode
            n = self.getNode(startNode)
            CurrentP = n.P
            n.oCalculated = True
            for p in l.pipes:
                phl = p.getFlowHeadLoss(startNode)
                CurrentP -= phl
                startNode = p.endNode if startNode != p.endNode else p.startNode
                n = self.getNode(startNode)
                if not n.oCalculated:
                    n.P = CurrentP
                    n.oCalculated = True
        kn = self.getNode(knownNode)
        deltaP = knownNodeP - kn.P
        for n in self.nodes:
            n.P = n.P + deltaP

    def getPipe(self, name):
        for p in self.pipes:
            if name == p.Name():
                return p

    def getNodePipes(self, node):
        l = []
        for p in self.pipes:
            if p.oContainsNode(node):
                l.append(p)
        return l

    def nodeBuilt(self, node):
        for n in self.nodes:
            if n.name == node:
                return True
        return False

    def getNode(self, name):
        for n in self.nodes:
            if n.name == name:
                return n

    def buildNodes(self):
        for p in self.pipes:
            if not self.nodeBuilt(p.startNode):
                self.nodes.append(Node(p.startNode, self.getNodePipes(p.startNode)))
            if not self.nodeBuilt(p.endNode):
                self.nodes.append(Node(p.endNode, self.getNodePipes(p.endNode)))

    def printPipeFlowRates(self, SI=True):
        ordered_pipes = []
        for start, end in self.pipe_order:
            pipe = next(p for p in self.pipes if p.startNode == start and p.endNode == end)
            ordered_pipes.append(pipe)
        for p in ordered_pipes:
            p.printPipeFlowRate(SI=SI)

    def printNetNodeFlows(self, SI=True):
        for n in sorted(self.nodes, key=lambda x: x.name):
            Q = n.QNet if SI else n.QNet * UC.L_to_ft3
            units = 'L/S' if SI else 'cfs'
            print(f'net flow into node {n.name} is {Q:.2f} ({units})')

    def printLoopHeadLoss(self, SI=True):
        cf = UC.m_to_psi(1, self.pipes[0].fluid.rho)
        units = 'm of water' if SI else 'psi'
        for l in self.loops:
            hl = l.getLoopHeadLoss()
            hl = hl if SI else hl * cf
            print(f'head loss for loop {l.name} is {hl:.2f} ({units})')

    def printPipeHeadLoss(self, SI=True):
        ordered_pipes = []
        for start, end in self.pipe_order:
            pipe = next(p for p in self.pipes if p.startNode == start and p.endNode == end)
            ordered_pipes.append(pipe)
        for p in ordered_pipes:
            p.printPipeHeadLoss(SI=SI)

    def printNodePressures(self, SI=True):
        pUnits = 'm of water' if SI else 'psi'
        cf = 1.0 if SI else UC.m_to_psi(1, self.Fluid.rho)
        node_order = ['a', 'b', 'h', 'c', 'e', 'd', 'f', 'g', 'i', 'j']
        for node_name in node_order:
            n = self.getNode(node_name)
            p = n.P * cf
            print(f'Pressure at node {n.name} = {p:.2f} {pUnits}')


# endregion

# region function definitions
def main():
    SIUnits = False
    water = Fluid(mu=20.50e-6, rho=62.3, SI=SIUnits)

    r_CI = 0.00085
    r_CN = 0.003

    PN = PipeNetwork()
    PN.Fluid = water

    PN.pipes.append(Pipe('a', 'b', 1000, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('a', 'h', 1600, 24, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('b', 'c', 500, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('b', 'e', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('c', 'd', 500, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('c', 'f', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('d', 'g', 800, 16, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('e', 'f', 500, 12, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('e', 'i', 800, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('f', 'g', 500, 12, r_CI, water, SI=SIUnits))
    PN.pipes.append(Pipe('g', 'j', 800, 18, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('h', 'i', 1000, 24, r_CN, water, SI=SIUnits))
    PN.pipes.append(Pipe('i', 'j', 1000, 24, r_CN, water, SI=SIUnits))

    PN.buildNodes()

    PN.getNode('h').setExtFlow(10, SI=SIUnits)
    PN.getNode('e').setExtFlow(-3, SI=SIUnits)
    PN.getNode('f').setExtFlow(-5, SI=SIUnits)
    PN.getNode('d').setExtFlow(-2, SI=SIUnits)

    PN.loops.append(
        Loop('A', [PN.getPipe('a-b'), PN.getPipe('b-e'), PN.getPipe('e-i'), PN.getPipe('h-i'), PN.getPipe('a-h')]))
    PN.loops.append(Loop('B', [PN.getPipe('b-c'), PN.getPipe('c-f'), PN.getPipe('e-f'), PN.getPipe('b-e')]))
    PN.loops.append(Loop('C', [PN.getPipe('c-d'), PN.getPipe('d-g'), PN.getPipe('f-g'), PN.getPipe('c-f')]))
    PN.loops.append(
        Loop('D', [PN.getPipe('e-i'), PN.getPipe('i-j'), PN.getPipe('g-j'), PN.getPipe('f-g'), PN.getPipe('e-f')]))

    PN.findFlowRates()
    knownP = UC.psi_to_m(80, water.rho)
    PN.getNodePressures(knownNode='h', knownNodeP=knownP)

    PN.printPipeFlowRates(SI=SIUnits)
    print()
    print('Check node flows:')
    PN.printNetNodeFlows(SI=SIUnits)
    print()
    print('Check loop head loss:')
    PN.printLoopHeadLoss(SI=SIUnits)
    print()
    PN.printPipeHeadLoss(SI=SIUnits)
    print()
    PN.printNodePressures(SI=SIUnits)


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion