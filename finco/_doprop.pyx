import numpy as np
cimport numpy as np
cimport cython
from .time_traj import TimeTrajectory

np.import_array()

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _actual_stuff(complex[:] q, complex[:] p, 
                   complex[:] M_p, complex[:] M_q, complex[:] t_1, 
                   complex[:] V_0, complex[:] V_1, complex[:] V_2, float m):
    
    cdef Py_ssize_t n = q.shape[0]
    res = np.zeros(n * 5, dtype=complex)
    cdef complex[:] res_view = res
    
    cdef Py_ssize_t i

    for i in range(n):
        res_view[i] = p[i] / m * t_1[i]
        res_view[i + n] = -V_1[i] * t_1[i]
        res_view[i + 2*n] = (p[i] ** 2 / (2 * m) - V_0[i]) * t_1[i]
        res_view[i + 3*n] = -V_2[i] * M_q[i] * t_1[i]
        res_view[i + 4*n] = M_p[i] / m * t_1[i]
    
    return res

def _do_step(tau: float, y, t_trajs: TimeTrajectory, V: list, m: float):
    # t = t_trajs.t_0(tau)

    q, p, _, M_p, M_q = y.reshape(5, -1)

    return _actual_stuff(q, p, M_p, M_q, 
                         t_trajs.t_1(tau), V[0](q), V[1](q), V[2](q), m)
