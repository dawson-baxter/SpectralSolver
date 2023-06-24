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




xmin = 2000
xmax = 3900

normalize_window = [.35, 1]



def get_multi_data():
    root = tk.Tk()
    root.withdraw()

    datasource = pd.DataFrame(columns=["Capture Info","X Data","Y Data", "Modified Y Data"])
    datasource['Capture Info'] = datasource['Capture Info'].astype(object)
    datasource['X Data'] = datasource['X Data'].astype(object)
    datasource['Y Data'] = datasource['Y Data'].astype(object)
    datasource['Modified Y Data'] = datasource['Modified Y Data'].astype(object)
    file_paths = filedialog.askopenfilenames(filetypes=[("Sif Files", "*.sif")])
    for filecount in range(len(file_paths)):
        tempdata, tempinfo = sif.utils.parse(file_paths[filecount])
        datasource.loc[filecount] = [tempinfo, tempdata[:, 0], tempdata[:, 1], tempdata[:, 1]]
    return datasource

def get_indices(data_x, xmin, xmax):
    indmax = np.searchsorted(data_x, xmax, side = 'right')
    indmin = np.searchsorted(data_x, xmin, side = 'left')
    return indmin, indmax

def truncate(data, indmin, indmax):
    return data[indmin:indmax]

def whole_normalize(data):
    return np.divide(data, np.sum(data))

def window_max_normalize(x_data, data_range):
    indmin = int(data_range[0] * x_data.size)
    indmax = int(data_range[1] * x_data.size)
    window = x_data[indmin:indmax]
    max_val = window.max()
    return x_data / max_val


def wavelength_to_raman(wavelengths, ExcitationWavelength):
    return 10_000_000 / ExcitationWavelength - 10_000_000 / wavelengths

def array_row_intersection(a,b):
    tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
    return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]

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

def PolyCoefficients(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def clippable(array, include_x=True):
    if include_x:
        tmp = pd.DataFrame(array)
    else:
        tmp = pd.DataFrame(array[:, 1])
    tmp.to_clipboard(index=False,header=False)

def onselect(vmin, vmax):
    if input_click is MouseButton.LEFT:
        indmin = np.searchsorted(specs.x, vmin, 'left')
        indmax = np.searchsorted(specs.x, vmax, 'right')
        try:
            specs.signal = np.unique(np.vstack((specs.signal, specs.combdata[indmin:indmax+1, :])), axis=0)
        except ValueError:
            specs.signal = specs.combdata[indmin:indmax+1]
        specs.noise = np.array(list(set(map(tuple, specs.noise)).difference(set(map(tuple, specs.combdata[indmin:indmax+1])))))
        ax.clear()
        if np.size(specs.signal) > 0:
            specs.signal = specs.signal[specs.signal[:, 0].argsort()]
            ax.plot(specs.signal[:,0], specs.signal[:, 1], '.', color='red')
        if np.size(specs.noise) > 0:
            specs.noise = specs.noise[specs.noise[:, 0].argsort()]
            ax.plot(specs.noise[:,0], specs.noise[:, 1], '.', color='blue')
        ax.set_title("Remove Areas of Signal\n Press Enter to Finish")
        fig.canvas.draw_idle()
    if input_click is MouseButton.RIGHT:
        indmin = np.searchsorted(specs.x, vmin, 'left')
        indmax = np.searchsorted(specs.x, vmax, 'right')
        try:
            specs.noise = np.unique(np.vstack((specs.noise, specs.combdata[indmin:indmax+1, :])), axis=0)
        except ValueError:
            specs.noise = specs.combdata[indmin:indmax+1]
        specs.signal = np.array(list(set(map(tuple, specs.signal)).difference(set(map(tuple, specs.combdata[indmin:indmax+1])))))
        ax.clear()
        if np.size(specs.signal)> 0:
            signal = specs.signal[specs.signal[:, 0].argsort()]
            ax.plot(specs.signal[:,0], specs.signal[:, 1], '.', color='red')
        if np.size(specs.noise) > 0:
            specs.noise = specs.noise[specs.noise[:, 0].argsort()]
            ax.plot(specs.noise[:,0], specs.noise[:, 1], '.', color='blue')
        ax.set_title("Add Areas of Background\n Press Enter to Finish")
        fig.canvas.draw_idle()


class Specs:
    def __init__(self, x_, y_, signal_, noise_, combdata_):
        self.x = x_
        self.y = y_
        self.signal = signal_
        self.noise = noise_
        self.combdata = combdata_


data = get_multi_data()

for row in data.index:
    data['X Data'][row] = wavelength_to_raman(data['X Data'][row], data['Capture Info'][row]['RamanExWavelength'])
    indmin, indmax = get_indices(data['X Data'][row], xmin, xmax)
    data['X Data'][row] = truncate(data['X Data'][row], indmin, indmax)
    data['Y Data'][row] = truncate(data['Y Data'][row], indmin, indmax)

    x = data['X Data'][row]
    y = data['Y Data'][row]
    combdata = np.vstack((x, y))
    combdata = combdata.transpose()
    signal = np.empty([0, 2])
    noise = combdata
    specs = Specs(x, y, signal, noise, combdata)
    

    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(x, y, '.', color='blue')
    ax.plot(signal[:, 0], signal[:, 1], '.', color = 'red')
    ax.set_title("Press Shift to Remove Signal Areas, Press + to Add Areas Back \n Press Enter When Finished")

    input_click = None
    input_key = None

    cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.connect('button_press_event', on_mouse_click)
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True, props=dict(facecolor='blue', alpha=0.25), snap_values = x)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    baseline_fitter = pybase.Baseline(specs.noise[:, 0], assume_sorted = True)

    cut_base, params = baseline_fitter.imodpoly(specs.noise[:, 1], poly_order = 3, return_coef = True)

    coeffs = params['coef']

    baseline_y = np.zeros(np.size(specs.x))

    baseline_y = PolyCoefficients(specs.x, coeffs)

    corrected_signal = specs.y - baseline_y

    # corrected_signal = whole_normalize(corrected_signal)
    corrected_signal = window_max_normalize(corrected_signal, normalize_window)

    x = specs.x
    y = specs.y
    noise = specs.noise
    signal = specs.signal

    data['Modified Y Data'][row] = corrected_signal

    ax.clear()

    plt.plot(x, PolyCoefficients(x, coeffs))
    plt.plot(x, y)

    fig2 = plt.figure(2)
    plt.plot(x, corrected_signal)
    plt.show()

    fig.canvas.mpl_disconnect(cid)
    fig2.canvas.mpl_disconnect(cid)
    corrected_data = (np.vstack((x, corrected_signal))).transpose()

average_x = data['X Data'][0]
average_y = np.empty(0)

for item in data['Modified Y Data']:
    if not average_y.size:
        average_y = item
    else:
        average_y = np.vstack((average_y, item))
if average_y.ndim > 1:
    average_y = np.mean(average_y, axis = 0)
average_data = np.vstack((average_x, average_y))

include_x = input('Include x axis data for clipboard? [y/n]: ')
clippable(average_data.transpose(), include_x.lower() == 'y')

for row in data.index:
    print(str(data['Capture Info'][row]['OriginalFilename']))

plt.plot(average_data[0, :], average_data[1,:])
plt.show()
