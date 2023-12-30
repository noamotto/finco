import numpy as np
cimport numpy as np
cimport cython
from .time_traj import TimeTrajectory

np.import_array()

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef _actual_stuff(complex[:] q, complex[:] p, 
                   complex[:] M_pp, complex[:] M_pq, complex[:] M_qp, complex[:] M_qq, complex[:] t_1, 
                   complex[:] V_0, complex[:] V_1, complex[:] V_2, float m):
    
    cdef Py_ssize_t n = q.shape[0]
    res = np.zeros(n * 7, dtype=complex)
    cdef complex[:] res_view = res
    
    cdef Py_ssize_t i

    for i in range(n):
        res_view[i] = p[i] / m * t_1[i]                                 # dq/dtau = p/m * dt/dtau
        res_view[i + n] = -V_1[i] * t_1[i]                              # dp/dtau = -V_1(q) * dt/dtau
        res_view[i + 2*n] = (p[i] ** 2 / (2 * m) - V_0[i]) * t_1[i]     # dS/dtau = (p^2/2m - V) * dt/dtau
        res_view[i + 3*n] = -V_2[i] * M_qp[i] * t_1[i]                  # dMpp/dtau = -V_2(q) * Mqp * dt/dtau
        res_view[i + 4*n] = -V_2[i] * M_qq[i] * t_1[i]                  # dMpq/dtau = -V_2(q) * Mqq * dt/dtau
        res_view[i + 5*n] = M_pp[i] / m * t_1[i]                        # dMqp/dtau = Mpp / m * dt/dtau
        res_view[i + 6*n] = M_pq[i] / m * t_1[i]                        # dMqq/dtau = Mpq / m * dt/dtau
    
    return res

def _do_step(tau: float, y, t_trajs: TimeTrajectory, V: list, m: float):
    # t = t_trajs.t_0(tau)

    q, p, _, M_pp, M_pq, M_qp, M_qq = y.reshape(7, -1)

    return _actual_stuff(q, p, M_pp, M_pq, M_qp, M_qq,
                         t_trajs.t_1(tau), V[0](q), V[1](q), V[2](q), m)
