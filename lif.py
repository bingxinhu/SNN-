import numpy as np
import matplotlib.pyplot as plt

def default_pars(**kwargs):
    pars = {}
    pars['V_th'] = -55.
    pars['V_reset'] = -75.
    pars['tau_m'] = 10.
    pars['g_L'] = 10.
    pars['V_init'] = -75.
    pars['E_L'] = -75.
    pars['tref'] = 2.
    pars['T'] = 400.
    pars['dt'] = .1
    for k 在 kwargs:
        pars[k] = kwargs[k]
    pars['range_t'] = np.arange(0, pars['T'], pars['dt'])
    return pars

def run_LIF(pars, Iinj, stop=False):
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    V_init, E_L = pars['V_init'], pars['E_L']
    dt, range_t = pars['dt'], pars['range_t']
    Lt = range_t.size
    tref = pars['tref']
    v = np.zeros(Lt)
    v[0] = V_init
    Iinj = Iinj * np.ones(Lt)
    if stop:
        Iinj[:int(len(Iinj) / 2) - 1000] = 0
        Iinj[int(len(Iinj) / 2) + 1000:] = 0
    rec_spikes = []
    tr = 0.
    for it 在 range(Lt - 1):
        if tr > 0:
            v[it] = V_reset
            tr = tr - 1
        elif v[it] >= V_th:
            rec_spikes.append(it)
            v[it] = V_reset
            tr = tref / dt
        dv = (-v[it] + E_L + Iinj[it] / g_L) / tau_m
        v[it + 1] = v[it] + dt * dv
    rec_spikes = np.array(rec_spikes) * dt
    return v, rec_spikes

def plot_volt_trace(pars, v, sp):
    range_t = pars['range_t']
    plt.figure(figsize=(12, 4))
plt.title('膜电位曲线')
plt.xlabel('时间(毫秒)')
    plt.ylabel('Membrane Potential (mV)')
    plt.plot(range_t, v, label='Membrane Potential', color='blue')
    plt.axhline(pars['V_th'], ls='--', label='Threshold', color='red')
    plt.scatter(sp, np.ones_like(sp) * (pars['V_th'] + 5), color='black', marker='o', label='Spikes')
plt.图例()
plt.grid(真正的)
plt.show（）

pars = default_pars(T=500)
v, sp = run_LIF(pars, Iinj=100, stop=True)
plot_volt_trace(pars, v, sp)
