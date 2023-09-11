import numpy as np 
import scipy
import matplotlib.pyplot as plt 
from matplotlib import animation
import argparse

# parse input parameters
argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument("-z", "--zoom", type=float, help="Zoom into animation by constant zoom factor.", default=1)
argParser.add_argument("-f", "--fps", type=int, help="Set speed of animation in frames per second.", default = 10)
argParser.add_argument("-d", "--dpi", type=int, help="Set resolution of animation in DPI.", default = 150)
argParser.add_argument("-n", "--n", type=int, help="Set resolution of input data.", default = 128)

args = argParser.parse_args()


N      = args.n
zoom   = args.zoom
width  = 2/zoom
height = 2/zoom 
dpi    = args.dpi 
fps    = args.fps 

# set plot style
plt.style.use('dark_background')

# create glowing line effect 
lws    = [0.5, 0.7, 1.0, 1.5, 2, 4, 8, 16]
alphas = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]



# craw an elephant using four complex parameters 

"""
Author: Piotr A. Zolnierczuk (zolnierczukp at ornl dot gov)

Based on a paper by:
Drawing an elephant with four complex parameters
Jurgen Mayer, Khaled Khairy, and Jonathon Howard,
Am. J. Phys. 78, 648 (2010), DOI:10.1119/1.3254017
"""

# elephant parameters
p1, p2, p3, p4 = (50 - 30j, 18 +  8j, 12 - 10j, -14 - 60j )
p5 = 40 + 20j # eyepiece

def fourier(t, C):
    f = np.zeros(t.shape)
    A, B = C.real, C.imag
    for k in range(len(C)):
        f = f + A[k]*np.cos(k*t) + B[k]*np.sin(k*t)
    return f

def elephant(t, p1, p2, p3, p4, p5):
    npar = 6
    Cx = np.zeros((npar,), dtype='complex')
    Cy = np.zeros((npar,), dtype='complex')

    Cx[1] = p1.real*1j
    Cx[2] = p2.real*1j
    Cx[3] = p3.real
    Cx[5] = p4.real

    Cy[1] = p4.imag + p1.imag*1j
    Cy[2] = p2.imag*1j
    Cy[3] = p3.imag*1j

    #x = np.append(fourier(t,Cx), [-p5.imag])
    #y = np.append(fourier(t,Cy), [p5.imag])
    x = fourier(t,Cx)
    y = fourier(t,Cy)


    return x,y

L      = 1 

t = np.linspace(0, L, N+1)

# compute time-series data for elephant 
x, y = elephant(t * 2 * np.pi, p1, p2, p3, p4, p5)

# normalise to [-1,1]x[-1,1]
x /= 100 
y /= 100
f = -x + 1j * y 

# compute Fourier transform excluding f[-1] == f[0]
fHat = np.fft.fftn(f[:-1])

# set up figure
fig = plt.figure(figsize=(4,4), dpi=dpi)
ax = plt.axes(xlim=(-1, -1+width), ylim=(-1, -1+height))

# disables plotting of axes
plt.axis('off')

lines   = []
circles = []
arrows  = []

for lw, alpha in zip(lws, alphas): 
    # plot elephant in cyan 
    line,  = ax.plot([],[], lw = lw, c = '#08F7FE',  alpha = alpha)
    lines.append(line) 

for i in range(N): 
    arrow  = ax.arrow(0, 0, 0, 0, lw=0.1)
    arrows.append(arrow) 

    circle = plt.Circle((0,0), 0, fill = False, lw=0.2, alpha=0.8)
    circles.append(circle)

# initialization function: plot the background of each frame
def init():
    for line in lines: 
        line.set_data([], [])

    for circle in circles:
        ax.add_patch(circle)

    return [*lines, *circles, *arrows] 

# animation function
def animate(time):

    # plot elephant up to time
    fDrawn = f[:time+1]

    for line in lines:
        line.set_data(np.imag(fDrawn), np.real(fDrawn))

    # zoom centered on last arrow drawn
    if zoom > 1: 
        arrowPosition = fDrawn[-1]
        x, y = np.imag(arrowPosition), np.real(arrowPosition)
        ax.set_xlim([x - width /2, x + width /2])
        ax.set_ylim([y - height/2, y + height/2])


    x = 0
    y = 0 

    for i in range(N):
        # start with lowest frequency
        i = N - i - 1

        # plane wave with coefficient determined by FFT of input data 
        c  = fHat[i] * np.exp(1j * 2 * np.pi * time * i/N) / len(fHat)
        re = np.imag(c)
        im = np.real(c)

        arrows[i].set_data(x = x, y = y, dx = re, dy = im)
        circles[i].center = (x,y)
        circles[i].radius = np.abs(c) 

        # sum plane waves
        x += re
        y += im
    return lines


# call the animator
print("Calling animator... ", end="")
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=N+1, blit=True)
print("done!")



# save the animation as gif
print("Save animation... ", end="")
anim.save('animation.gif', fps=fps)
print("done!")
