import numpy as np 
import matplotlib.pyplot as plt

# set up input data
# based on paper "Drawing an elephant with four complex parameters"
# by Jürgen Mayer et al. (2010) DOI:10.1119/1.3254017

def elephant(t):
    y =  50*np.sin(t)+18*np.sin(2*t)-12*np.cos(3*t)+14*np.cos(5*t)
    x = -60*np.cos(t)+30*np.sin(t)  - 8*np.sin(2*t)+10*np.sin(3*t)
    return x/100 + 1j*y/100

# number of points at which to sample elephant
N     = 128
times = np.linspace(0, 2 * np.pi, N+1)

# compute time-series data for elephant
f = elephant(times)

# compute Fourier transform excluding f[-1] == f[0]
fHat = np.fft.fft(f[:-1])

# time at which to evaluate IDFT
t     = 64 

# plane waves with coefficient determined by FFT of input data 
waves   = fHat*np.exp(1j*2*np.pi*t*np.arange(N)/N)/len(fHat)
# returns frequencies corresponding to entries of fHat
# for N even: 0, 1, 2, ..., N/2, -N/2 - 1, ..., -1
freqs   = np.fft.fftfreq(N)
# sort waves by magnitude of frequencies from slow to fast 
waves   = waves[np.argsort(np.abs(freqs))]

# create figure
fig, ax = plt.subplots(dpi=600)
plt.axis("off") 
plt.style.use('dark_background')
plot_colour = '#08F7FE'

# plot elephant 
plt.plot(np.real(f[:t+1]), np.imag(f[:t+1]), c=plot_colour)
# first vector starts at origin
x = 0
y = 0
# do not show vectors shorter than cutoff
cutoff = 1e-2

# sum over plane waves
for i in range(N):
    # determine direction of vectors 
    dx = np.real(waves[i])
    dy = np.imag(waves[i])

    # do not show very short vectors
    if np.abs(waves[i]) > cutoff:
        plt.arrow(x = x, y = y, dx = dx, dy = dy, lw = 0.1, head_width=0.01, head_length=0.02, length_includes_head=True)
        ax.add_patch(plt.Circle((x,y), np.abs(waves[i]), fill = False, lw=0.5))

    # sum up plane waves
    x += dx
    y += dy

plt.savefig("elephant.png")