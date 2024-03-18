import numpy as np

def p_change(p, M, k1, k2, k3):
    dpdt = k1 - k2*M*(p / (k3 + p))
    return dpdt

def m_change(p, m, k4, k5):
    dmdt = (k4 * p**2 - k5 * m)
    return dmdt

def M_change(m, M, k6, k7):
    dMdt = k6 * m - k7 * M
    return dMdt

def RungeKuttaLoose(dt, p, m, M, k2, k1, k3, k4, k5, k6, k7):
    p_c1 = dt * p_change(p, M, k1, k2, k3)
    p_c2 = dt * p_change((p + 1/2*p_c1), M, k1, k2, k3)
    p_c3 = dt * p_change((p + 1/2*p_c2), M, k1, k2, k3)
    p_c4 = dt * p_change((p + p_c3), M, k1, k2, k3)

    m_c1 = dt * m_change(p, m, k4, k5)
    m_c2 = dt * m_change(p, (m + 1/2*m_c1), k4, k5)
    m_c3 = dt * m_change(p, (m + 1/2*m_c2), k4, k5)
    m_c4 = dt * m_change(p, (m + m_c3), k4, k5)

    M_c1 = dt * M_change(m, M, k6, k7)
    M_c2 = dt * M_change(m, (M + 1/2*M_c1), k6, k7)
    M_c3 = dt * M_change(m, (M + 1/2*M_c2), k6, k7)
    M_c4 = dt * M_change(m, (M + M_c3), k6, k7)

    p_next = p + 1/6*(p_c1 + 2*p_c2 + 2*p_c3 + p_c4)
    m_next = m + 1/6*(m_c1 + 2*m_c2 + 2*m_c3 + m_c4)
    M_next = M + 1/6*(M_c1 + 2*M_c2 + 2*M_c3 + M_c4)

    return p_next, m_next, M_next