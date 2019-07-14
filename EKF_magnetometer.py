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

def swap_index(ibar, c):
    if ibar < c:
        return ibar
    else:
        return ibar + 1

def propagate(t, state, inertia):
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    deps = .5*(eta*identity(3) + crux(eps))@omega
    deta = -.5*dot(eps, omega)
    domega = -inv(inertia)@crux(omega)@inertia@omega

    return hstack([deta, deps, domega])


def get_F(omega, dt):

    w1 = omega[0]
    w2 = omega[1]
    w3 = omega[2]
    F = array([[ 0,-w1,-w2,-w3],
               [w1,  0, w3,-w2],
               [w2,-w3,  0, w1],
               [w3, w2,-w1, 0]])

    return F/2*dt + identity(4)


def truncated_F(omega, eps, eta, dt, c):
    quat = hstack([eta, eps])

    w1 = omega[0]
    w2 = omega[1]
    w3 = omega[2]
    F = array([[ 0,-w1,-w2,-w3],
               [w1,  0, w3,-w2],
               [w2,-w3,  0, w1],
               [w3, w2,-w1, 0]])


    F_trunc = zeros((3,3))
    for ibar in range(3):
        for jbar in range(3):
            i = swap_index(ibar, c)
            j = swap_index(jbar, c)
            F_trunc[ibar,jbar] = F[i,j] - quat[j]/quat[c]*F[i,c]

    return .5*F_trunc*dt + identity(3)



def get_B(eps, eta, inertia):
    deps = .5*(eta*identity(3) + crux(eps))
    deta = -.5*eps
    B = vstack([deps, deta])@inv(inertia)
    return B



def get_H(eps, eta, b_eci):
    q1 = eta
    q2 = eps[0]
    q3 = eps[1]
    q4 = eps[2]
    

    b_hat = b_eci/norm(b_eci)
    dCTdq1 = array([[ q1, q4,-q3],
                    [-q4, q1, q2],
                    [ q3,-q2, q1]])

    dCTdq2 = array([[ q2, q3, q4],
                    [ q3,-q2, q1],
                    [ q4,-q1,-q2]])

    dCTdq3 = array([[-q3, q2,-q1],
                    [ q2, q3, q4],
                    [ q1, q4,-q3]])

    dCTdq4 = array([[-q4, q1, q2],
                    [-q1,-q4, q3],
                    [ q2, q3, q4]])

    H = vstack([2*dCTdq1@b_hat, 2*dCTdq2@b_hat, 2*dCTdq3@b_hat, 2*dCTdq4@b_hat]).T

    return H

def truncated_H(eps, eta, b_eci, c):
    quat = hstack([eta, eps])
    H = get_H(eps, eta, b_eci)

    H_trunc = zeros((3,3))
    for i in range(3):
        for jbar in range(3):
            j = swap_index(jbar, c)
            H_trunc[i, jbar] = H[i,j] - quat[j]/quat[c]*H[i,c]

    return H_trunc

def quat_mult(E1, n1, E2, n2):

    # q1 *q2

    E3 = n1*E2 + n2*E1 + crux(E1)@E2
    n3 = n2*n1 - dot(E1, E2)

    return E3, n3

def custom_H(v, E, n):

    qv, qw = quat_mult(v, 0, -E, n)
    qx = qv[0]
    qy = qv[1]
    qz = qv[2]

    return array([[ qx, qw, qz,-qy],
                  [ qy,-qz, qw, qx],
                  [ qz, qy,-qx, qw]])




#Simulate rotating spacecraft
angular_rate = array([1,1,5])*1e-3
inertia = array([[1,.04,.05],
                 [.04, 1.5, .03],
                 [.05, .03, 1.2]])
inertia = identity(3)
quaternion = array([1,0,0,0])
dt = .1
tspan = 60*60

simulate = False
if simulate:
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
else:
    newstate = load('newstate.npy')
    t = load('t.npy')


#Simulate measurements

MAGNETOMETER_NOISE = 1e-3
ANGULAR_RATE_NOISE = 1e-3

b_eci = array([0,0,5])
b_meas = []
w_meas = []
for state in newstate:
    eta = state[0]
    eps = state[1:4]
    omega = state[4:]

    dcm = quat2dcm(eps, eta).T
    b_body = dcm@b_eci
    b_meas.append(b_body + random.normal(0, MAGNETOMETER_NOISE, 3))
    w_meas.append(omega)

b_meas = vstack(b_meas)
w_meas = vstack(w_meas)

#Simulate EKF

#x_posteriori = array(list(Quaternion.random()))[1:]
x_posteriori = array(list(Quaternion(axis = [1,0,0], angle = 2)))
P_posteriori = identity(4)*.5

B = inv(inertia)
Q = ones((4,4))*1e-7
R = diag([MAGNETOMETER_NOISE**2]*3)


print('Simulating EKF')
quat_estimate = []
Ks = []
errors = []
b_preds = []
percent = 10
for mag_b, w_meas, second in zip(b_meas, w_meas, t):


    eta = x_posteriori[0]
    eps = x_posteriori[1:]
    F = get_F( w_meas, dt)
    H = custom_H(b_eci, eps, eta)
    
    # indices = [x for x in range(4) if x != c]
    # x_trunc = x_posteriori[indices]
    B = get_B(eps, eta, inertia)
    Q = B@B.T*1e-6

    state = hstack([x_posteriori, w_meas])
    delta = dt/10
    for i in range(10):
        dstate = propagate(0, state, inertia)
        state += dstate*delta

    x_pred = state[0:4]

    # x_pred = F@x_trunc

    #x_pred = F@x_posteriori

    P_pred = F@P_posteriori@F.T + Q
    y = mag_b - H@x_pred
    errors.append(y)

    S = H@P_pred@H.T + R
    K = P_pred@H.T@inv(S)

    #print(K)
    # q1 = Quaternion(x_pred)
    # q2 = Quaternion(K@y)
    # x_posteriori = array(list(q1*q2))
    x_posteriori = x_pred + K@y
    x_posteriori = x_posteriori/norm(x_posteriori)

    P_posteriori = (identity(4) - K@H)@P_pred

    quat_estimate.append(x_posteriori)
    Ks.append(norm(K))
    b_preds.append(H@x_pred)

    if second/tspan*100 > percent:
        print(percent, '% EKFd')
        percent += 10

quat_estimate = vstack(quat_estimate)
Ks = hstack(Ks)
b_preds = vstack(b_preds)
errors = vstack(errors)

error = []
for q_true, q_est in zip(newstate[:,0:4], quat_estimate):
    error.append(norm(q_true - q_est))


plt.figure()
plt.plot(errors)

#plot results
plt.figure()
plt.plot(t,Ks)
plt.title('norm(K) vs Time')

plt.figure()
plt.plot(t,error)
plt.title('Error vs Time')

quat_actual = newstate[:,0:4]

fig = plt.figure()
ax = plt.axes()
plt.plot(t, quat_actual, 'b')
plt.plot(t, quat_estimate, 'r')
plt.title('Truth vs Estimate')

plt.show()




