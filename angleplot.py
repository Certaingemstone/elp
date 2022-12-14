import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import Model, Parameters
import csv

filename = r"C:\Users\jade2\Downloads\MIT\FA2022\JLab\pendulum\Kiran.csv"

def damped_sine(t, a, w, b, phi, c):
    return (a*np.sin(w*t + phi))*np.exp(-b*t) + c

def fit_decay_sine(x, y):
    params = Parameters()
    # Tune these to your observed data
    params.add('a', value=700, min=0)
    params.add('w', value=0.9, min=0)
    params.add('b', value=0.001, min=0)
    params.add('phi', value=0)
    params.add('c', value=935)
    model = Model(damped_sine)
    out = model.fit(y, params, t=x)
    return out

if __name__ == "__main__":
    # Read data
    times = []
    rhos = []
    with open(filename, newline='') as file:
        reader = csv.reader(file, delimiter=' ', quotechar='|')
        for row in reader:
            times.append(float(row[0]))
            rhos.append(float(row[1]))
    times = np.array(times)
    rhos = np.array(rhos)
    # Fit
    result = fit_decay_sine(times, rhos)
    # Plot
    print(result.fit_report())
    plt.plot(times, rhos, label='data')
    plt.plot(times, result.best_fit, '-', label='best fit')
    plt.xlabel("Time (s)")
    plt.ylabel("Position (px)")
    plt.legend()
    plt.show()