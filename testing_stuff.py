from numpy import *
from pyquaternion import Quaternion


def quat_mult(E1, n1, E2, n2):

    # q1 *q2

    E3 = n1*E2 + n2*E1 + crux(E1)@E2
    n3 = n2*n1 - dot(E1, E2)

    return E3, n3


def crucify(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    return array([[ qw,-qx,-qy,-qz],
                  [ qx, qw, qz,-qy],
                  [ qy,-qz, qw, qx],
                  [ qz, qy,-qx, qw]])

def crux(A):
    return array([[0, -A[2], A[1]],
                  [A[2], 0, -A[0]],
                  [-A[1],A[0], 0]])


def get_H(v, E, n):

    Eh, nh = quat_mult(v, 0, -E, n)

    return crucify(hstack([nh, Eh]))



q1 = Quaternion.random()

v = array([1,4,3])

actual = q1.rotate(v)

E = array(list(q1))[1:]
n = array(list(q1))[0]
test = get_H(v, E, n)@array(list(q1))


print(actual)
print(test)