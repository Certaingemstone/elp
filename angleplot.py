import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit.models import SineModel
import csv

filename = r"C:\Users\jade2\Downloads\MIT\FA2022\JLab\pendulum\C0425.csv"

def fit_sine(x, y):
    model = lmfit.SineModel()
    # model parameters: amplitude, frequency, shift
    # functional form: f(x; A, phi, f) = A sin(fx + phi)
    params = model.guess(y, x=x)
    out = mod.fit(y, params, x=x)
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
    # Fit
    fit = fit_sine(times, rhos)
    # Plot
    print(fit.fit_report())
    plt.plot(times, rhos)
    plt.show()