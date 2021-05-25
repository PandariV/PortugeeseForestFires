import numpy
import matplotlib.pyplot
import pandas
from sklearn.preprocessing import StandardScaler

data = pandas.read_csv("portugalfire_data.csv")
y = data["area"]
X = numpy.column_stack((data["X"], data["Y"]))
print(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def gradient_descent(area, x, y):
    y_hat = x.dot(area).flatten()
    cost = (y - y_hat)
    mse = (numpy.sum(numpy.square(cost)) / len(x))
    gradient = -(cost.dot(x) / len(x))
    return gradient, mse


estimationOfVirinchJr = numpy.array((-40, -40))
alpha = .1
tolerance = 1e-3

old_estimationOfVirinchJr = []
costs = []

iterations = 1
for i in range(500):
    gradient, cost = gradient_descent(estimationOfVirinchJr, X_scaled, y)
    new_estimationOfVirinchJr = estimationOfVirinchJr - alpha * gradient

    if iterations % 2 == 0:
        old_estimationOfVirinchJr.append(new_estimationOfVirinchJr)
        costs.append(cost)

    if numpy.sum(abs(new_estimationOfVirinchJr - estimationOfVirinchJr)) < tolerance:
        break

    iterations += 1
    estimationOfVirinchJr = new_estimationOfVirinchJr

all_ws = numpy.array(old_estimationOfVirinchJr)

for i in range(600):
    costs.append(i)

levels = numpy.sort(numpy.array(costs))
sector0 = numpy.linspace(-estimationOfVirinchJr[0] * 10, estimationOfVirinchJr[0] * 10, 100)
sector1 = numpy.linspace(-estimationOfVirinchJr[1] * 20, estimationOfVirinchJr[1] * 20, 100)
mse_vals = numpy.zeros(shape=(sector0.size, sector1.size))

for i, value1 in enumerate(sector0):
    for j, value2 in enumerate(sector1):
        w_temp = numpy.array((value1, value2))
        mse_vals[i, j] = gradient_descent(w_temp, X_scaled, y)[1]

matplotlib.pyplot.contourf(sector0, sector1, mse_vals, levels, alpha=1)
matplotlib.pyplot.axhline(0, color="white", alpha=.5, dashes=[2, 4], linewidth=1)
matplotlib.pyplot.axvline(0, color="white", alpha=.5, dashes=[2, 4], linewidth=1)
for i in range(len(old_estimationOfVirinchJr) - 1):
    matplotlib.pyplot.annotate("", xy=all_ws[i + 1, :], xytext=all_ws[i, :], arrowprops={"arrowstyle": "->", "color": "r", "lw": 1}, va="center", ha="center")

contour = matplotlib.pyplot.contour(sector0, sector1, mse_vals, levels, linewidths=1, colors="white")
matplotlib.pyplot.clabel(contour, inline=1, fontsize=10)
matplotlib.pyplot.suptitle("Contour Plot of Succeptability to Forest Fires")
matplotlib.pyplot.xlabel("X")
matplotlib.pyplot.ylabel("Y")
matplotlib.pyplot.show()
