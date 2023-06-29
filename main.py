import torch
import torch.optim as optim

def get_model(method, x, y, max_step):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    mdl = torch.nn.Linear(1, 1)
    cr = torch.nn.MSELoss()
    opt = method(mdl.parameters(), lr=1.e-2)
    sm_var = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=1.e-3)
    prv = float('inf')
    it = 0
    
    for epoch in range(max_step):
        it += 1
        opt.zero_grad()
        pred = mdl(x.view(-1, 1))
        diff = cr(pred, y)
        diff.backward()
        opt.step()
        sm_var.step()
        if diff.item() < prv:
            prv = diff.item()
        else:
            break

    return mdl, it
//====
from scipy.optimize import least_squares, minimize
def pf(x, cf):
    deg = len(cf) - 1
    return np.polyval(cf, x)
def start_pred(deg):
    return np.zeros(deg + 1)
def inn(cf, x, y):
    predicted_y = pf(x, cf)
    rs = y - predicted_y
    mse = np.mean(rs ** 2)
    return mse
def polyMinReg(x, y, deg):
    gs = start_pred(deg)
    res = minimize(inn, gs, args=(x, y))
    fitted_coeffs = res.x
    return fitted_coeffs
def pMin(self, x_in, y_in, deg):
    return polyMinReg(x_in, y_in, deg)
def inn(cf, x, y):
    predicted_y = pf(cf, x)
    rs = y - predicted_y
    return rs
def polyLeastSqrReg(x, y, deg):
    gs = start_pred(deg)
    res = least_squares(inn, gs, args=(x, y))
    fitted_coeffs = res.x
    return fitted_coeffs
def pls(self, x_in, y_in, deg):
    return polyLeastSqrReg(x_in, y_in, deg)
s_o = ScipyOptimizer()
cf_ls = s_o.pls(x, y, deg)
cf_min = s_o.pMin(x, y, deg)
//==

def obj(cf, poly_deg, X, y):
    y_pred = np.polyval(np.flip(cf), X)
    rs = y - y_pred
    return np.sum(rs ** 2)
    
def jacobian(cf, poly_deg, X, y):
    jacob = np.zeros((sampl, feat))
    for i in range(sampl):
        for j in range(feat):
            jacob[i][j] = -X[i] ** (feat - j - 1)
    return jacob

def get_coef(cf, poly_deg, X, y):
    sampl = X.shape[0]
    feat = poly_deg + 1
    cf = np.zeros(feat)

    gs = np.zeros(feat)
    res = minimize(obj, gs, method='Powell', jac=jacobian)
    cf = res.x
    return cf

def predict(cf, poly_deg, X):
    y_pred = np.polyval(np.flip(cf), X)
    return y_pred

md = PowellDoglegPolynomialRegression(poly_deg=poly_deg)
cf = None
s = time.time()
get_coef(cf, poly_deg, x, y)
frst = time.time() - s
reg = Regression()
s = time.time()
cf, iters = reg.powell_dog_leg_regression(x, y, poly_deg)
two = time.time() - s
//======
def sciPyGrad(f, s):
    return approx_fprime(s, f, epsilon=0.0001)
def pyTorchGrad(f, s):
    x, y = torch.tensor(s[0], requires_grad=True), torch.tensor(s[1], requires_grad=True)
    output = f(x, y)
    output.backward()
    return x.grad, y.grad
def gradient_by_hand(f, s, h=0.00001):
    return (f([s[0] + h, s[1]]) - f([s[0] - h, s[1]])) / (2 * h), (f([s[0], s[1] + h]) - f([s[0], s[1] - h])) / (2 * h)


//==
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

# Функция Розенброка
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Нелинейные ограничения
def nonlinear_constraint(x):
    return x[0]**2 + 2*x[1]**2 - 4

# Создание объекта NonlinearConstraint
nonlinear_constraint = NonlinearConstraint(nonlinear_constraint, -np.inf, 0)

# Начальное приближение
x0 = np.array([-1.5, 1.5])
path = []
def callback_func(xk):
    path.append(xk)

# Минимизация с использованием нелинейных ограничений
result_nonlinear = minimize(rosenbrock, x0, method='L-BFGS-B', constraints=[nonlinear_constraint], callback=callback_func)
print("Минимум с нелинейными ограничениями:")
print(result_nonlinear)

# Отображение графика с линиями уровня и областями ограничений
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y])

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=50)

# Отображение области ограничений
plt.fill_between(x, np.sqrt(2 - x**2/2), color='cyan', alpha=0.4, label='Нелинейные ограничения')

# Отображение минимума и пути оптимизации
plt.plot(1,1, 'r*', label='Min')
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'o-', color="green", label="Trust-Constr")

plt.xlabel('x')
plt.ylabel('y')
plt.title('Функция Розенброка с ограничениями')
plt.legend()
plt.grid(True)
plt.show()
