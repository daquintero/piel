:py:mod:`piel.materials.silicon`
================================

.. py:module:: piel.materials.silicon


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   piel.materials.silicon.k
   piel.materials.silicon.epsilon_0
   piel.materials.silicon.h
   piel.materials.silicon.T_a
   piel.materials.silicon.epsilon_si
   piel.materials.silicon.E_g_si_bardin
   piel.materials.silicon.m_dh_green_h
   piel.materials.silicon.m_l_askt_h
   piel.materials.silicon.m_h_askt_h
   piel.materials.silicon.m_so_askt_h
   piel.materials.silicon.m_dh_askt_h
   piel.materials.silicon.m_t_askt_e
   piel.materials.silicon.m_ce_askt_e
   piel.materials.silicon.m_de_askt_e
   piel.materials.silicon.N_c
   piel.materials.silicon.E_c
   piel.materials.silicon.E_f
   piel.materials.silicon.n_0
   piel.materials.silicon.N_v
   piel.materials.silicon.p_0
   piel.materials.silicon.n_i
   piel.materials.silicon.n_io
   piel.materials.silicon.mu_0a_e
   piel.materials.silicon.mu_0b_e
   piel.materials.silicon.kappa_a_e
   piel.materials.silicon.kappa_b_e
   piel.materials.silicon.mu_0a_h
   piel.materials.silicon.mu_0b_h
   piel.materials.silicon.kappa_a_h
   piel.materials.silicon.kappa_b_h
   piel.materials.silicon.mu_ps_e
   piel.materials.silicon.mu_ps_h
   piel.materials.silicon.mu_min_e
   piel.materials.silicon.N_ref_e
   piel.materials.silicon.kappa_c_e
   piel.materials.silicon.mu_min_h
   piel.materials.silicon.N_ref_h
   piel.materials.silicon.kappa_c_h
   piel.materials.silicon.mu_psii_e
   piel.materials.silicon.mu_psii_h



.. py:function:: k()

   Return the Boltzmann constant in eV/K.


.. py:function:: epsilon_0()

   Return the permittivity of free space in F/m (Farads per meter).


.. py:function:: h()

   Return Planck's constant in J.s or eV.s.


.. py:function:: T_a()

   Return an array of ambient temperatures ranging from 0.1 to 300 K.


.. py:function:: epsilon_si()

   Return the permittivity of Silicon.


.. py:function:: E_g_si_bardin(T)

   Calculate the bandgap energy of silicon as a function of temperature.


.. py:function:: m_dh_green_h(T)

   Calculate the density of states effective mass for holes (Green) as a function of temperature.


.. py:function:: m_l_askt_h(T)

   Calculate the normalized light hole effective mass as a function of temperature.


.. py:function:: m_h_askt_h(T)

   Calculate the normalized heavy hole effective mass as a function of temperature.


.. py:function:: m_so_askt_h(T)

   Calculate the normalized split-off hole effective mass as a function of temperature.


.. py:function:: m_dh_askt_h(T)

   Calculate the density of states effective mass for holes as a function of temperature.


.. py:function:: m_t_askt_e(T)

   Calculate the transversal effective mass for electrons as a function of temperature.


.. py:function:: m_ce_askt_e(T)

   Calculate the effective conduction mass for electrons as a function of temperature.


.. py:function:: m_de_askt_e(T)

   Calculate the electron density-of-states effective mass as a function of temperature.


.. py:function:: N_c(T)

   Calculate the effective density of states in the conduction band as a function of temperature.


.. py:function:: E_c(T)

   Calculate the conduction band energy as a function of temperature.


.. py:function:: E_f(T)

   Calculate the Fermi level energy as a function of temperature.


.. py:function:: n_0(T)

   Calculate the electron concentration at equilibrium as a function of temperature.


.. py:function:: N_v(T)

   Calculate the effective density of states in the valence band as a function of temperature.


.. py:function:: p_0(T)

   Calculate the hole concentration at equilibrium as a function of temperature.


.. py:function:: n_i(T)

   Calculate the intrinsic carrier concentration as a function of temperature.


.. py:function:: n_io(T)

   Calculate the intrinsic carrier concentration for a given temperature using an alternative method.


.. py:function:: mu_0a_e()

   Return the parameter mu_0a for electron mobility.


.. py:function:: mu_0b_e()

   Return the parameter mu_0b for electron mobility.


.. py:function:: kappa_a_e()

   Return the exponent kappa_a for electron mobility.


.. py:function:: kappa_b_e()

   Return the exponent kappa_b for electron mobility.


.. py:function:: mu_0a_h()

   Return the parameter mu_0a for hole mobility.


.. py:function:: mu_0b_h()

   Return the parameter mu_0b for hole mobility.


.. py:function:: kappa_a_h()

   Return the exponent kappa_a for hole mobility.


.. py:function:: kappa_b_h()

   Return the exponent kappa_b for hole mobility.


.. py:function:: mu_ps_e(T)

   Calculate the electron mobility due to phonon scattering as a function of temperature.


.. py:function:: mu_ps_h(T)

   Calculate the hole mobility due to phonon scattering as a function of temperature.


.. py:function:: mu_min_e(T)

   Calculate the minimum electron mobility as a function of temperature.


.. py:function:: N_ref_e(T)

   Return the reference concentration N_ref for electrons as a function of temperature.


.. py:function:: kappa_c_e(T)

   Return the exponent kappa_c for electron mobility as a function of temperature.


.. py:function:: mu_min_h(T)

   Calculate the minimum hole mobility as a function of temperature.


.. py:function:: N_ref_h(T)

   Return the reference concentration N_ref for holes as a function of temperature.


.. py:function:: kappa_c_h(T)

   Return the exponent kappa_c for hole mobility as a function of temperature.


.. py:function:: mu_psii_e(T, N_I_e)

   Calculate the phonon scattering and ionized impurity electron scattering mobility.


.. py:function:: mu_psii_h(T, N_I_h)

   Calculate the phonon scattering and ionized impurity hole scattering mobility.


