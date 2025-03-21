from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from utils import featurize
from function_approx import SimpleNNApprox
import torch

# put the path to the model you want here
PATH_NAME = "results/K25H40_trial1/model.pth"
TIME_UPPER = 20

device = torch.device("cpu")

QFunctionApprox = SimpleNNApprox(learning_rate=0.01)
QFunctionApprox.model.load_state_dict(torch.load(PATH_NAME))
QFunctionApprox.model.eval()
QFunctionApprox.model.to(device)

def f(t, p):
    x = featurize(time = t, price = p).to(device)
    #output = max(0, torch.max(QFunctionApprox.forward(x).detach()).numpy())
    output = torch.max(QFunctionApprox.forward(x).detach()).numpy()
    return output

t = np.linspace(1, TIME_UPPER, 20)
p = np.linspace(10, 40, 30)

T, P = np.meshgrid(t, p)

Z = np.array([[
    f(x, y)
for x in t]for y in p])


ax = plt.axes(projection='3d')
ax.plot_surface(T, P, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.view_init(20, 290)
ax.contour3D(T, P, Z, 20, cmap='binary')
ax.set_xlabel('Time')
ax.set_ylabel('Share Price')
ax.set_zlabel('Value')

ax.set_box_aspect((1, 1, 0.5))

plt.savefig("graphics/20_1401.png",dpi = 300, bbox_inches='tight', pad_inches = 0.5)

plt.show()