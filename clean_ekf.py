from numpy import *
from numpy.linalg import *
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])

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

def get_B(eps, eta, inertia):
    deps = .5*(eta*identity(3) + crux(eps))
    deta = -.5*eps
    B = vstack([deps, deta])@inv(inertia)
    return B


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

def quat2dcm(E,n):
    #n = scalar part
    #E = vector part
    
    dcm = (2*n**2 - 1)*identity(3) + 2*outer(E,E) - 2*n*crux(E) 

    return dcm

newstate = load('newstate.npy')
t = load('t.npy')
inertia = load('inertia.npy')
tspan = t[-1]
dt = load('dt.npy')
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

#x_posteriori = array(list(Quaternion.random()))
x_posteriori = array(list(Quaternion(axis = [1,0,0], angle = 2)))
P_posteriori = identity(4)*1e-5

R = diag([MAGNETOMETER_NOISE**2]*3)

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
    
    B = get_B(eps, eta, inertia)
    Q = B@B.T*1e-7

    state = hstack([x_posteriori, w_meas])
    delta = dt/100
    for i in range(100):
        dstate = propagate(0, state, inertia)
        state += dstate*delta

    x_pred = state[0:4]

    P_pred = F@P_posteriori@F.T + Q
    y = mag_b - H@x_pred
    errors.append(y)

    S = H@P_pred@H.T + R
    K = P_pred@H.T@inv(S)

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
plt.semilogy(t,error)
plt.title('Error vs Time')
plt.xlabel('Time [s]')
plt.ylabel('Residual Error')
plt.savefig('Error vs time.png')

quat_actual = newstate[:,0:4]

fig = plt.figure()
ax = plt.axes()
plt.plot([],[], 'b.')
plt.plot([],[], 'r--')
plt.plot(t, quat_actual, 'b.', label = 'Truth')
plt.plot(t, quat_estimate, 'r--', label = 'Estimate')
plt.title('Truth vs Estimate')
plt.xlabel('Time [s]')
plt.ylabel('Quaternions')
plt.legend(['Truth','Estimate'])
plt.savefig('Truth vs estimate.png')

plt.show()