import numpy as np


def k():
    """Return the Boltzmann constant in eV/K."""
    return 8.617333262145e-5


def epsilon_0():
    """Return the permittivity of free space in F/m (Farads per meter)."""
    return 8.854e-12


def h():
    """Return Planck's constant in J.s or eV.s."""
    return 6.62607015e-34


def T_a():
    """Return an array of ambient temperatures ranging from 0.1 to 300 K."""
    return np.linspace(0.1, 300)


def epsilon_si():
    """Return the permittivity of Silicon."""
    return 11.7 * epsilon_0()


def E_g_si_bardin(T):
    """Calculate the bandgap energy of silicon as a function of temperature."""
    return 1.17 + 5.65e-6 * T - 5.11e-7 * T ** 2 - 8.03e-10 * T ** 3 + 2.50e-12 * T ** 4


def m_dh_green_h(T):
    """Calculate the density of states effective mass for holes (Green) as a function of temperature."""
    numerator = 0.444 + 0.361e-2 * T + 0.117e-3 * T ** 2 + 0.126e-5 * T ** 3 + 0.303e-8 * T ** 4
    denominator = 1 + 0.468e-2 * T + 0.229e-3 * T ** 2 + 0.747e-6 * T ** 3 + 0.173e-8 * T ** 4
    return (numerator / denominator) ** (2 / 3)


def m_l_askt_h(T):
    """Calculate the normalized light hole effective mass as a function of temperature."""
    return 0.14615 + 3.7414e-4 * T - 1.8809e-7 * T ** 2


def m_h_askt_h(T):
    """Calculate the normalized heavy hole effective mass as a function of temperature."""
    return 0.51741 + 2.5139e-3 * T - 4.4117e-6 * T ** 2 + 2.6974e-3


def m_so_askt_h(T):
    """Calculate the normalized split-off hole effective mass as a function of temperature."""
    return 0.22775 + 2.963e-4 * T + 2.3872e-7 * T ** 2


def m_dh_askt_h(T):
    """Calculate the density of states effective mass for holes as a function of temperature."""
    m_h = m_h_askt_h(T)
    m_l = m_l_askt_h(T)
    m_so = m_so_askt_h(T)
    return ((m_h ** (3 / 2) + m_l ** (3 / 2) + m_so ** (3 / 2)) ** (2 / 3))


def m_t_askt_e(T):
    """Calculate the transversal effective mass for electrons as a function of temperature."""
    return 0.19049 - 2.0905e-6 * T + 9.8985e-7 * T ** 2 - 2.6798e-9 * T ** 3 + 2.0270e-12 * T ** 4


def m_ce_askt_e(T):
    """Calculate the effective conduction mass for electrons as a function of temperature."""
    m_t = m_t_askt_e(T)
    m_l = 0.9163
    return 1 / ((1 / m_t + 2 / m_l) / 3)


def m_de_askt_e(T):
    """Calculate the electron density-of-states effective mass as a function of temperature."""
    m_t = m_t_askt_e(T)
    m_l = 0.9163
    return (6 * np.sqrt(m_l * (m_t ** 2))) ** (2 / 3)


def N_c(T):
    """Calculate the effective density of states in the conduction band as a function of temperature."""
    m_de = m_de_askt_e(T)
    return (2 * (2 * np.pi * m_de * k() * T) ** (3 / 2)) / (h() ** 2)


def E_c(T):
    """Calculate the conduction band energy as a function of temperature."""
    return E_g_si_bardin(T)


def E_f(T):
    """Calculate the Fermi level energy as a function of temperature."""
    return E_g_si_bardin(T) / 2


def n_0(T):
    """Calculate the electron concentration at equilibrium as a function of temperature."""
    Ec = E_c(T)
    Ef = E_f(T)
    return N_c(T) * (1 / (1 + np.exp((Ec - Ef) / (k() * T))))


def N_v(T):
    """Calculate the effective density of states in the valence band as a function of temperature."""
    m_dh = m_dh_askt_h(T)
    return (2 * (2 * np.pi * m_dh * k() * T) ** (3 / 2)) / (h() ** 2)


def p_0(T):
    """Calculate the hole concentration at equilibrium as a function of temperature."""
    Ec = E_c(T)
    Ef = E_f(T)
    return N_v(T) * (1 - (1 / (1 + np.exp((Ec - Ef) / (k() * T)))))


def n_i(T):
    """Calculate the intrinsic carrier concentration as a function of temperature."""
    return np.sqrt(N_c(T) * N_v(T)) * np.exp(-E_g_si_bardin(T) / (2 * k() * T))


def n_io(T):
    """Calculate the intrinsic carrier concentration for a given temperature using an alternative method."""
    m_t = m_t_askt_e(T)
    m_l = 0.9163
    return 4.82e15 * T ** (3 / 2) * np.sqrt(6 * m_t * np.sqrt(m_l)) * np.exp(-E_g_si_bardin(T) / (2 * k() * T))


def mu_0a_e():
    """Return the parameter mu_0a for electron mobility."""
    return 4195


def mu_0b_e():
    """Return the parameter mu_0b for electron mobility."""
    return 2153


def kappa_a_e():
    """Return the exponent kappa_a for electron mobility."""
    return 1.5


def kappa_b_e():
    """Return the exponent kappa_b for electron mobility."""
    return 3.13


def mu_0a_h():
    """Return the parameter mu_0a for hole mobility."""
    return 2502


def mu_0b_h():
    """Return the parameter mu_0b for hole mobility."""
    return 519


def kappa_a_h():
    """Return the exponent kappa_a for hole mobility."""
    return 1.5


def kappa_b_h():
    """Return the exponent kappa_b for hole mobility."""
    return 3.25


def mu_ps_e(T):
    """Calculate the electron mobility due to phonon scattering as a function of temperature."""
    return 1 / ((mu_0a_e() ** -1) * (T / 300) ** kappa_a_e()) + 1 / ((mu_0b_e() ** -1) * (T / 300) ** kappa_b_e())


def mu_ps_h(T):
    """Calculate the hole mobility due to phonon scattering as a function of temperature."""
    return 1 / ((mu_0a_h() ** -1) * (T / 300) ** kappa_a_h()) + 1 / ((mu_0b_h() ** -1) * (T / 300) ** kappa_b_h())


def mu_min_e(T):
    """Calculate the minimum electron mobility as a function of temperature."""
    return 197.17 - 45.505 * np.log10(T)


def N_ref_e(T):
    """Return the reference concentration N_ref for electrons as a function of temperature."""
    return 1.12e17 * (T / 300) ** 3.2


def kappa_c_e(T):
    """Return the exponent kappa_c for electron mobility as a function of temperature."""
    return 0.72 * (T / 300) ** 0.065


def mu_min_h(T):
    """Calculate the minimum hole mobility as a function of temperature."""
    return 110.90 - 25.597 * np.log10(T)


def N_ref_h(T):
    """Return the reference concentration N_ref for holes as a function of temperature."""
    return 2.23e17 * (T / 300) ** 3.2


def kappa_c_h(T):
    """Return the exponent kappa_c for hole mobility as a function of temperature."""
    return 0.72 * (T / 300) ** 0.065


def mu_psii_e(T, N_I_e):
    """Calculate the phonon scattering and ionized impurity electron scattering mobility."""
    mu_min = mu_min_e(T)
    mu_ps = mu_ps_e(T)
    N_ref = N_ref_e(T)
    kappa_c = kappa_c_e(T)
    return mu_min + (mu_ps - mu_min) / (1 + (N_I_e / N_ref) ** kappa_c)


def mu_psii_h(T, N_I_h):
    """Calculate the phonon scattering and ionized impurity hole scattering mobility."""
    mu_min = mu_min_h(T)
    mu_ps = mu_ps_h(T)
    N_ref = N_ref_h(T)
    kappa_c = kappa_c_h(T)
    return mu_min + (mu_ps - mu_min) / (1 + (N_I_h / N_ref) ** kappa_c)
