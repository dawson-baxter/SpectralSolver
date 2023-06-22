import tkinter as tk
import pybaselines as pybase
import numpy as np
import sif_parser as sif
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image, ImageFile


from matplotlib.widgets import SpanSelector
from matplotlib.backend_bases import MouseButton
from tkinter import filedialog

def get_single_data():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    data, info = sif.utils.parse(file_path)

    return data[:, 0], data[:, 1], data, info

def internal_truncate(data, xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax += 1
    return data[indmin:indmax, 0], data[indmin:indmax, 1], data[indmin:indmax, :]

def whole_normalize(data):
    return np.divide(data, np.sum(data))

x, y, data, info = get_single_data()

x = 10_000_000 / info['RamanExWavelength'] - 10_000_000 / x
data[:, 0] = x 

x, y, data = internal_truncate(data, 2500, 3900)

signal = np.empty([0, 2])
noise = data

input_click = None
input_key = None

fig, ax = plt.subplots(figsize = (12,8))

ax.plot(x, y, '.', color='blue')
ax.plot(signal[:, 0], signal[:, 1], '.', color = 'red')

ax.set_title("Press Shift to Remove Signal Areas, Press + to Add Areas Back \n Press Enter When Finished")

def array_row_intersection(a,b):
    tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
    return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

def onselect(vmin, vmax):
    global signal
    global noise
    if input_click is MouseButton.LEFT:
        indmin = np.searchsorted(x, vmin, 'left')
        indmax = np.searchsorted(x, vmax, 'right')
        try:
            signal = np.unique(np.vstack((signal, data[indmin:indmax+1, :])), axis=0)
        except ValueError:
            signal = data[indmin:indmax+1]
        noise = np.array(list(set(map(tuple, noise)).difference(set(map(tuple, data[indmin:indmax+1])))))
        ax.clear()
        if np.size(signal) > 0:
            signal = signal[signal[:, 0].argsort()]
            ax.plot(signal[:,0], signal[:, 1], '.', color='red')
        if np.size(noise) > 0:
            noise = noise[noise[:, 0].argsort()]
            ax.plot(noise[:,0], noise[:, 1], '.', color='blue')
        ax.set_title("Remove Areas of Signal\n Press Enter to Finish")
        fig.canvas.draw_idle()
    if input_click is MouseButton.RIGHT:
        indmin, indmax = np.searchsorted(x, (vmin, vmax))
        indmax = min(len(x) - 1, indmax)
        try:
            noise = np.unique(np.vstack((noise, data[indmin:indmax+1, :])), axis=0)
        except ValueError:
            noise = data[indmin:indmax+1]
        signal = np.array(list(set(map(tuple, signal)).difference(set(map(tuple, data[indmin:indmax+1])))))
        ax.clear()
        if np.size(signal)> 0:
            signal = signal[signal[:, 0].argsort()]
            ax.plot(signal[:,0], signal[:, 1], '.', color='red')
        if np.size(noise) > 0:
            noise = noise[noise[:, 0].argsort()]
            ax.plot(noise[:,0], noise[:, 1], '.', color='blue')
        ax.set_title("Add Areas of Background\n Press Enter to Finish")
        fig.canvas.draw_idle()

def on_key_press(event):
    global input_key
    input_key = event.key
    if input_key == 'enter':
        plt.close()
def on_mouse_click(event):
    global input_click
    input_click = event.button
    if input_click is MouseButton.LEFT:
        ax.set_title("Remove Areas of Signal\n Press Enter to Finish")
        fig.canvas.draw_idle()
    if input_click is MouseButton.RIGHT:
        ax.set_title("Add Areas of Background\n Press Enter to Finish")
        fig.canvas.draw_idle()

cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
plt.connect('button_press_event', on_mouse_click)
span = SpanSelector(ax, onselect, 'horizontal', useblit=True, props=dict(facecolor='blue', alpha=0.25), snap_values = x)
plt.show()
fig.canvas.mpl_disconnect(cid)

baseline_fitter = pybase.Baseline(noise[:, 0], assume_sorted = True)

cut_base, params = baseline_fitter.imodpoly(noise[:, 1], poly_order = 3, return_coef = True)

coeffs = params['coef']

baseline_y = np.zeros(np.size(x))

def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

baseline_y = PolyCoefficients(x, coeffs)

corrected_signal = y - baseline_y

corrected_signal = whole_normalize(corrected_signal)

###
# corrected_signal = pybase.smooth.noise_median(corrected_signal, half_window = 6)[0]
#

ax.clear()

plt.plot(x, PolyCoefficients(x, coeffs))
plt.plot(x, y)

fig2 = plt.figure(2)
plt.plot(x, corrected_signal)
plt.show()

fig.canvas.mpl_disconnect(cid)
fig2.canvas.mpl_disconnect(cid)
corrected_data = (np.vstack((x, corrected_signal))).transpose()

def clippable(array, include_x=True):
    if include_x:
        tmp = pd.DataFrame(array)
    else:
        tmp = pd.DataFrame(array[:, 1:])
    tmp.to_clipboard(index=False,header=False)

# clippable(corrected_data, True)
clippable(corrected_data, False)
