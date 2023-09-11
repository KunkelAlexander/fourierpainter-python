import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
import argparse
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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

enable_inset = (zoom != 1)

# set plot style
plt.style.use('dark_background')

# create glowing line effect 
lws         = [0.5, 0.7, 1.0, 1.5, 2, 4, 8, 16]
alphas      = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
plot_colour = '#08F7FE'


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
figsize = (4, 4)
xlim    = (-1, 1) 
ylim    = (-1, 1) 

# make space for inset axis
if enable_inset:
    xlim = (-1, 2)
    figsize = (6, 4) 

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = plt.axes(xlim=xlim, ylim=ylim)

# inset zoom axis
if enable_inset: 
    ax_inset = ax.inset_axes([0.66, 0.5, 0.33, 0.47])

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
    ax_inset.set_xticks([]) 
    ax_inset.set_yticks([]) 

# disables plotting of axes
plt.axis('off')

lines         = []
circles       = []
arrows        = []
lines_inset   = []
circles_inset = []
arrows_inset  = []

# plot elephant 
for lw, alpha in zip(lws, alphas): 
    line,        = ax.plot([],[], lw = lw, c = plot_colour,  alpha = alpha)
    lines.append(line)
    
    if enable_inset:
        line_inset,  = ax_inset.plot([],[], lw = lw, c = plot_colour,  alpha = alpha)
        lines_inset.append(line_inset) 

# plot arrows and circles
for i in range(N): 
    arrow       = ax.arrow(0, 0, 0, 0, lw=0.1)
    arrows.append(arrow) 
    circle       = plt.Circle((0,0), 0, fill = False, lw=0.2, alpha=0.8)
    circles.append(circle)

    if enable_inset:
        arrow_inset = ax_inset.arrow(0, 0, 0, 0, lw=0.1)
        arrows_inset.append(arrow_inset) 
        circle_inset = plt.Circle((0,0), 0, fill = False, lw=0.2, alpha=0.8)
        circles_inset.append(circle_inset)

# initialization function: plot the background of each frame
def init():
    for line in lines: 
        line.set_data([], [])

    for circle in circles:
        ax.add_patch(circle)

    if enable_inset:
        for line in lines_inset: 
            line.set_data([], [])

        for circle in circles_inset:
            ax_inset.add_patch(circle)

    return [*lines, *circles, *arrows, *lines_inset, *circles_inset, *arrows_inset] 

# animation function
def animate(time):

    # plot elephant up to time
    fDrawn = f[:time+1]

    for line in lines:
        line.set_data(np.imag(fDrawn), np.real(fDrawn))

    if enable_inset:
        for line in lines_inset:
            line.set_data(np.imag(fDrawn), np.real(fDrawn))

    # zoom centered on last arrow drawn
    arrowPosition = fDrawn[-1]
    x, y = np.imag(arrowPosition), np.real(arrowPosition)
    ax_inset.set_xlim([x - width /2, x + width /2])
    ax_inset.set_ylim([y - height/2, y + height/2])


    x = 0
    y = 0 

    # build list of complex vectors
    indices = np.arange(N)

    # plane waves with coefficient determined by FFT of input data 
    waves   = fHat * np.exp(1j * 2 * np.pi * time * indices/N) / len(fHat)
    # returns frequencies corresponding to entries of fHat
    # 0, 1, 2, ..., N/2, -N/2 - 1, ..., -1
    freqs   = np.fft.fftfreq(N)
    # sort waves by magnitude of frequencies from slow to fast 
    waves   = waves[np.argsort(np.abs(freqs))]

    for i in range(N):
        dx = np.imag(waves[i])
        dy = np.real(waves[i])

        arrows[i].set_data(x = x, y = y, dx = dx, dy = dy)
        circles[i].center = (x,y)
        circles[i].radius = np.abs(waves[i]) 

        if enable_inset:
            arrows_inset[i].set_data(x = x, y = y, dx = dx, dy = dy)
            circles_inset[i].center = (x,y)
            circles_inset[i].radius = np.abs(waves[i]) 

        # sum plane waves
        x += dx
        y += dy
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
