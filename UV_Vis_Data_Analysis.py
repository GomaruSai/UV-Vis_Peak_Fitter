import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy import optimize, signal
import math
import random

import os

import pandas as pd

from lmfit import models

def Extract_UV_Vis_Data(f, type):
    file = f.split("\n")[1:-1]
    if type == 'tsv':
        l = len(file)
        xvals = [None] * l
        yvals = [None] * l # * lt
        for i, vals in enumerate(file):
            vals = vals.split("\t")[:-1]
            if len(vals) > 2:
                vals = vals[2:]
            xvals[i] = float(vals[0])
            yvals[i] = float(vals[1])

    elif type == 'csv':
        file = file[23:]
        l = len(file)
        xvals = [None] * l
        yvals = [None] * l # * lt
        for i, vals in enumerate(file):
            vals = vals.split(",")[:-1]
            #if len(vals) > 2:
            #    vals = vals[2:]
            xvals[i] = float(vals[0])
            yvals[i] = float(vals[1])
    print(len(yvals))
    return xvals, yvals, f[0]
    # + "_" + filename

#def Analyse_UV_Vis(filename):

def Normalize_UV_Vis_Data(x, y, f):
    if "Absorbance:" not in f:
        y[:] = [yv / max(y) for yv in y]
    return x, y, f

def Plot_UV_Vis_Data(x, y, f):
    plt.plot(x, y)
    plt.title(f.split("_")[-1])
    if "Absorbance:" in f:
        plt.ylabel('Absorbance (OD)')
    else:
        plt.ylabel('Intensity (counts)')
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Wavelength [nm]')
    plt.locator_params(axis='x', nbins=10)
    plt.grid(True)
    plt.show()

def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint('sigma', min=1e-6, max=x_range)
            model.set_param_hint('center', min=x_min, max=x_max)
            model.set_param_hint('height', min=1e-6, max=1.1*y_max)
            model.set_param_hint('amplitude', min=1e-6)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
    return composite_model, params

def lorentzian(x, a, w, c):
    return a * w**2 / ((x - c)**2 + w**2)

def cost(parameters, x, y):
    chi = np.multiply(y, -1)
    for i in range(0, len(parameters), 3):
        chi += lorentzian(x, *parameters[i:i + 3])
    return np.sum(np.power(chi, 2)) / len(x)

def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    #np.random.shuffle(peak_indicies)
    peak_indicies = [x for _, x in reversed(sorted(zip(y[peak_indicies], peak_indicies)))]
    peakvals = []
    for i in peak_indicies:
        peakvals.append(y[i])
    print(peakvals)
    p = []
    for peak_indicie, model_indicie in zip(peak_indicies, model_indicies):
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
            params = {
                'amplitude': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }            
            if 'params' in model:
                spec['model'][model_indicie]['params'].update(params)
            else:
                spec['model'][model_indicie]['params'] = params
        else:
            raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
        p.append(params['amplitude'])
        p.append(params['sigma'])
        p.append(params['center'])
        #print(f"peak {model_indicie}: type: {spec['model'][model_indicie]['type'][:-5]} amplitude: {params['amplitude']:3.3f} sigma: {params['sigma']:3.3f} centre: {params['center']:3.3f}")
    
    result = optimize.minimize(cost, p, args=(x, y))
    print('steps', result.nit, result.fun)
    print('sigma_0', x_range / len(x) * np.min(peak_widths))

    for model_indicie in model_indicies:
        model = spec['model'][model_indicie]
        if model_indicie < len(peak_indicies):
            print(f"peak {model_indicie}: type: {spec['model'][model_indicie]['type'][:-5]} amplitude: {result.x[model_indicie * 3]:3.3f} sigma: {result.x[(model_indicie * 3) + 1]:3.3f} centre: {result.x[(model_indicie * 3) + 2]:3.3f}")
            if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel']:
                params = {
                    'amplitude': result.x[model_indicie * 3],
                    'sigma': result.x[(model_indicie * 3) + 1],
                    'center': result.x[(model_indicie * 3) + 2]
                }
                if 'params' in model:
                    spec['model'][model_indicie]['params'].update(params)
                else:
                    spec['model'][model_indicie]['params'] = params
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')

    return spec, peak_indicies

#def cost(parameters):
#    g_0 = parameters[:3]
#    g_1 = parameters[3:6]
#    return np.sum(np.power(g(x, *g_0) + g(x, *g_1) - y, 2)) / len(x)

#result = optimize.minimize(cost, params)
#print('steps', result.nit, result.fun)
#print(f'g_0: amplitude: {result.x[0]:3.3f} mean: {result.x[1]:3.3f} sigma: {result.x[2]:3.3f}')
#print(f'g_1: amplitude: {result.x[3]:3.3f} mean: {result.x[4]:3.3f} sigma: {result.x[5]:3.3f}')

#def gaussian(x, a, c, s):
#    return a * (1 / (s * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - c) / s)**2)))

#def lorentzian(x, a, c, w):
#    return a * w**2 / ((x - c)**2 + w**2)

#def cost(parameters):
#    a, c, s = parameters
    # y has been calculated in previous snippet
#    return np.sum(np.power(gaussian(xvals, a, c, s) - y, 2)) / len(x)

#result = optimize.minimize(cost, [0, 0, 1])
#print('steps', result.nit, result.fun)
#print(f'amplitude: {result.x[0]:3.3f} mean: {result.x[1]:3.3f} sigma: {result.x[2]:3.3f}')

#dir_path = os.path.dirname(os.path.realpath(__file__))

#dir_path = os.path.abspath('C:/Users/putch/SaiP/')
dir_path = os.path.abspath('C:/Users/putch/Documents/Masters Thesis Work/')#SaiP/0818 Samples/20221006/')


for root, dirs, files in os.walk(dir_path):
    #for dire in dirs:
    for file in files:
        if file.endswith('.txt'):
            fi = open(os.path.abspath(os.path.join(root, file)), 'r').read()
            print(os.path.abspath(os.path.join(root, file)))
            x, y, f = Normalize_UV_Vis_Data(*Extract_UV_Vis_Data(fi, 'tsv'))
            x[:] = [1240.71 / xv for xv in x]
            #Plot_UV_Vis_Data(x, y, f)
            spec = {
                'x': np.array(x),
                'y': np.array(y),
                'model': [
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'}
                ]
            }
            
            #fig, gridspec = output.plot(data_kws={'markersize': 1})
            
            #print(file)
            spec, peaks_found = update_spec_from_peaks(spec, [0, 1, 2, 3], peak_widths=(20, 25))
            fig, ax = plt.subplots()
            ax.plot(spec['x'], spec['y'])
            print(peaks_found)
            for peak in peaks_found[:len(spec['model'])]:
                ax.axvline(x=spec['x'][peak], c='black', linestyle='dotted')

            model, params = generate_model(spec)
            output = model.fit(spec['y'], params, x=spec['x'])
            components = output.eval_components(x=spec['x'])
            print(len(peaks_found))
            for i, model in enumerate(spec['model']):
                ax.plot(spec['x'], components[f'm{i}_'])
            plt.xlabel("Energy [eV]")
            plt.ylabel("Normalised Intensity")
            plt.title(os.path.abspath(os.path.join(root, file)))

            plt.show()

        elif file.endswith('.csv'):
            fi = open(os.path.abspath(os.path.join(root, file)), 'r').read()
            print(os.path.abspath(os.path.join(root, file)))
            x, y, f = Normalize_UV_Vis_Data(*Extract_UV_Vis_Data(fi, 'csv'))
            x[:] = [1240.71 / xv for xv in x]
            #Plot_UV_Vis_Data(x, y, f)
            spec = {
                'x': np.array(x),
                'y': np.array(y),
                'model': [
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'},
                    {'type': 'LorentzianModel'}
                ]
            }
            
            #fig, gridspec = output.plot(data_kws={'markersize': 1})
            
            #print(file)
            spec, peaks_found = update_spec_from_peaks(spec, [0, 1, 2, 3], peak_widths=(20, 25))
            fig, ax = plt.subplots()
            ax.plot(spec['x'], spec['y'])
            print(peaks_found)
            for peak in peaks_found[:len(spec['model'])]:
                ax.axvline(x=spec['x'][peak], c='black', linestyle='dotted')

            model, params = generate_model(spec)
            output = model.fit(spec['y'], params, x=spec['x'])
            components = output.eval_components(x=spec['x'])
            print(len(peaks_found))
            for i, model in enumerate(spec['model']):
                ax.plot(spec['x'], components[f'm{i}_'])
            plt.xlabel("Energy [eV]")
            plt.ylabel("Normalised Intensity")
            plt.title(os.path.abspath(os.path.join(root, file)))

            plt.show()
