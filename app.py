import os
from flask import Flask, render_template, redirect, session, request, url_for, jsonify
import json
from scipy.optimize import curve_fit
from scipy import asarray as exp
import scipy.stats as stats

import array as ar
import math
from math import e, sqrt, pi
import pylab as plt
from numpy import array, linspace, arange, zeros, ceil, amax, amin, argmax, argmin, abs
from numpy import polyfit, polyval, seterr, trunc, mean
from numpy.linalg import norm
from scipy.interpolate import interp1d
import scipy.optimize
import numpy as np
# import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('fermi_edge.html')

@app.route('/', methods=['POST','GET'])
def upload_and_fit():
    if request.form.get('peak'):
        peak = request.form.get('peak')
        area = request.form.get('area')
        fwhm = request.form.get('fwhm')
        global xfit
        global yfit
        global peak_fit
        global area_fit
        global height_fit
        global fwhm_fit
        global chi_square
        (xfit, yfit) = gauss_fit(peak, area, fwhm)
        return render_template('fermi_edge.html')
    uploaded_file = request.files['file']
    print("----",uploaded_file.filename)
    for line in uploaded_file:
        if line.decode("utf-8").startswith('[Data'):
            break
    # Read the rest of the data, using spaces to split. 
    data = [r.decode("utf-8").split() for r in uploaded_file]
    global xdata
    global ydata
    
    xdata = []
    ydata = []
    for d in data:
        if d:
            xdata.append(float(d[0]))
            ydata.append(float(d[1]))
    print(round(xdata[0]-xdata[1], 2))
    print(type(xdata[0]))
    print(xdata)
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return render_template('fermi_edge.html')

@app.route('/data')
def xydata():
   return jsonify({'xdata': xdata, 'ydata':ydata})

def gauss_fit(peak, temp, fwhm):
    xfit = np.array(xdata)
    yfit = np.array(ydata)
    
    kb = 8.6173e-5 # Boltzmann
    
    # xdata = [1.5, 1.45000000000005, 1.39999999999998, 1.35000000000002, 1.29999999999995, 1.25, 1.20000000000005, 1.14999999999998, 1.10000000000002, 1.04999999999995, 1, 0.950000000000045, 0.899999999999977, 0.850000000000023, 0.799999999999955, 0.75, 0.700000000000045, 0.649999999999977, 0.600000000000023, 0.549999999999954, 0.5, 0.450000000000046, 0.399999999999977, 0.350000000000023, 0.299999999999954, 0.25, 0.200000000000045, 0.149999999999977, 0.100000000000023, 0.0499999999999545, 0, -0.0499999999999545, -0.100000000000023, -0.149999999999977, -0.200000000000045, -0.25, -0.299999999999954, -0.350000000000023, -0.399999999999977, -0.450000000000046, -0.5, -0.549999999999954, -0.600000000000023, -0.649999999999977, -0.700000000000045, -0.75, -0.799999999999955, -0.850000000000023, -0.899999999999977, -0.950000000000045, -1, -1.04999999999995, -1.10000000000002, -1.14999999999998, -1.20000000000005, -1.25, -1.29999999999995, -1.35000000000002, -1.39999999999998, -1.45000000000005, -1.5]
    
    def fermi(e,scale,T=300,muf=0):
        return scale*(1.0 / (np.exp((-e - muf)/(kb*T)) + 1)) 
    
    def gaussian(x, mu, sig,scale):
        return scale*(np.exp(-np.power(-x - mu, 2.) / (2 * np.power(sig, 2.))))  
    
    # yfit = np.random.normal((8.07+np.convolve(fermi(xfit,scale=1.05), gaussian(xfit, mu=0, sig=0.3,scale=1.05),mode="same")),10)
    
    
    def convolution(size,sigC,scaleC,xC=xfit):
        return size* np.convolve(gaussian(x=xC,sig=sigC,scale=scaleC,mu=0), fermi(e=xfit,scale=scaleC,T=300,muf=0), mode="same")
    

    popt, pcov = scipy.optimize.curve_fit(convolution, xfit, yfit)
    Opt_size,Opt_sigC,Opt_scaleC = popt
    
    yfit = gaussian(xfit,*popt).tolist()

    # Print out the coefficients determined by the optimization
    print(Opt_size,Opt_sigC,Opt_scaleC)
    # n = len(xfit)      
    # sigma = float(fwhm) / 2.3548
    # amp = float(area) / sigma
    # def gaus(x,a,x0,sigma):
    #     return a*exp(-(x-x0)**2/(2*sigma**2))
    # popt,pcov = curve_fit(gaus,xdata,ydata,p0=[amp,float(peak),sigma])
    # (amp, peak, sigma) = popt
    # yfit = gaus(xfit,*popt).tolist()
    xfit = xfit.tolist()
    return (xfit, yfit)


@app.route('/data_fit')
def data_fit():
    return jsonify({'xfit': xfit,
                    'yfit':yfit})    

if __name__ == '__main__':
    app.run(host=os.environ.get('IP', '0.0.0.0'),
            port=int(os.environ.get('PORT', '3330')),
            debug=True)
