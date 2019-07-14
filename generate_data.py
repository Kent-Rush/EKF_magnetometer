from numpy import *
from numpy.linalg import *
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from scipy.integrate import ode

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])

def quat2dcm(E,n):
    #n = scalar part
    #E = vector part
    
    dcm = (2*n**2 - 1)*identity(3) + 2*outer(E,E) - 2*n*crux(E) 

    return dcm

def propagate(t, state, inertia):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    deps = .5*(eta*identity(3) + crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = -inv(inertia)@crux(omega)@inertia@omega

    return hstack([deta, deps, domega])







#Simulate rotating spacecraft
angular_rate = array([1,1,5])*1e-3
inertia = array([[1,.04,.05],
                 [.04, 1.5, .03],
                 [.05, .03, 1.2]])
# inertia = identity(3)
quaternion = array([1,0,0,0])
dt = 1.5
tspan = 150*60

state = hstack([quaternion, angular_rate])

solver = ode(propagate)
solver.set_integrator('lsoda')
solver.set_initial_value(state, 0)
solver.set_f_params(inertia)

print('Simulating')

newstate = []
t = []
percent = 10
while solver.successful() and solver.t < tspan:
    newstate.append(solver.y)
    t.append(solver.t)
    solver.integrate(solver.t + dt)
    if solver.t/tspan*100 > percent:
        print(percent, '% Simulated')
        percent += 10

newstate = vstack(newstate)
t = vstack(t)
save('newstate.npy', newstate )
save('t.npy', t)
save('inertia.npy', inertia)
save('dt.npy', dt)