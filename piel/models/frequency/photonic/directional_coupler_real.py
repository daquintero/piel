"""
Translated from https://github.com/flaport/sax or https://github.com/flaport/photontorch/tree/master
"""
# def realistic_directional_coupler(length=12.8e-6, k0=0.2332, n0=0.0208, de1_k0=1.2435, de1_n0=0.1169, de2_k0=5.3022, de2_n0=0.4821, wl0=1.55e-6):
#     wl = torch.tensor(1.55e-6, dtype=torch.float64)
#     dwl = wl - wl0
#     dn = n0 + de1_n0 * dwl + 0.5 * de2_n0 * dwl**2
#     kappa0 = k0 + de1_k0 * dwl + 0.5 * de2_k0 * dwl**2
#     kappa1 = np.pi * dn / wl
#     tau = torch.cos(kappa0 + kappa1 * length)
#     kappa = -torch.sin(kappa0 + kappa1 * length)
#     sdict = sax.reciprocal({
#         ("port0", "port1"): tau,
#         ("port0", "port2"): 1j*kappa,
#         ("port1", "port3"): 1j*kappa,
#         ("port2", "port3"): tau,
#     })
#     return sdict
