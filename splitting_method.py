# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt

hbar = 1
q = 1
keldysh = 1
omega = 7.35e-2
A0 = -omega / keldysh

class SplittingMethod:
    def __init__(self, x0, x1, dx, T, dt, trecord, imag, psi0, H_p, H_k, H_f_x = None, H_f_t = None):
        self.x0 = x0
        self.x1 = x1
        self.dx = dx
        self.T = T + dt / 2
        self.t = 0
        self.step = 0
        self.dt = dt
        self.trecord = trecord
        self.Ts = np.arange(0, self.T + self.dt, self.dt)
        self.imag = 1 if not imag else -1j

        self.x = np.arange(x0, x1, dx)
        self.p = fftfreq(self.x.size, dx) * np.pi * 2 # Why 2 * np.pi?

        self.prep_fields(H_p, H_f_x, H_f_t, H_k)

        self.psi = psi0(self.x)
        self.psis = []

    def prep_fields(self, H_p, H_f_x, H_f_t, H_k):
        self.expH_p = np.exp(-1j * H_p(self.x) * self.dt / hbar / 2 * self.imag)
        self.field_x = H_f_x(self.x) if H_f_x else np.zeros_like(self.x)
        self.field_t = H_f_t(self.Ts) if H_f_t else np.zeros_like(self.Ts)
        self.expH_f = np.exp(-1j * self.field_x * self.field_t[0] * self.dt / hbar / 2 * self.imag)
        self.expH_k = np.exp(-1j * H_k(self.p) * self.dt / hbar * self.imag)
        
    def do_step(self):
         # Save snapshot
        if np.isclose(self.t % self.trecord, self.trecord) or \
            np.isclose(self.t % self.trecord, 0):
            self.psis.append([self.t,np.copy(self.psi)])
        
        psi1 = self.psi * self.expH_p * self.expH_f
        
        fftd = fft(psi1)
        psi2 = ifft(fftd * self.expH_k)
        
        self.expH_f = np.exp(-1j * self.field_x * self.field_t[self.step + 1] * self.dt / hbar / 2 * self.imag)        
        self.psi = psi2 * self.expH_p * self.expH_f
        
        self.t += self.dt
        self.step += 1
        
    def propagate(self):
        while self.t < self.T:
            self.do_step()

    def show_plots(self, interval, skip = 1):        
        plt.figure()
        plt.xlim(-10,10)
        plt.ylim(-1e-1,1.5)
        psi = plt.plot(self.x , np.abs(self.psis[0][1]))
        # pot = plt.plot(self.x, self.expH_p(self.x, t) + self.H_f(self.x, t))
        for i, plot in enumerate(self.psis[1:]):
            plt.waitforbuttonpress(interval)
            if not i % skip:
                psi[0].set_ydata(np.abs(plot[1]))
                # pot[0].set_ydata(self.H_p(self.x, t) + self.H_f(self.x, t))
                plt.draw()