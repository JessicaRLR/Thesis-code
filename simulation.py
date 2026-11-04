import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =========================================================
# Funciones mu
# =========================================================

def fun_mu_1(X, params):
    """
    Calcula

    mu_i = sum_{k>=1} pi_k [(A_i)^{k+1} - (A_{i-1})^{k+1} - x_i]

    donde A_i = x_0 + ... + x_i

    Parámetros
    ----------
    X: Vector de frecuencias/types (array).
    params: Vector pi_k (array)

    Returns
    -------
    mu : np.array
    """

    pi_k = params

    K = len(X)

    A = np.cumsum(X)
    A_prev = np.zeros(K)
    A_prev[1:] = A[:-1]

    mu = np.zeros(K)

    for i in range(K):

        suma = 0

        for k in range(1, len(pi_k) + 1):
            termino = (A[i]**(k+1)- A_prev[i]**(k+1)- X[i])
            suma += pi_k[k-1] * termino

        mu[i] = suma

    return mu


def fun_mu_2(X, params):
    """
    Calcula

    mu_i = x_i (x_{i-1} - x_{i+1})

    usando indices modulo 3.

    Parámetros
    ----------
    X: Vector de frecuencias/types (array)
    params: No se usa.

    Returns
    -------
    mu : np.array
    """

    mu = np.zeros(3)

    for i in range(3):

        x_prev = X[i-1]
        x_next = X[(i+1) % 3]

        mu[i] = X[i] * (x_prev - x_next)

    return mu


def fun_mu_3(X, params):
    """
    Calcula

    mu_i(x) = x_i (1 - x_i - 2 sum_{j != i} p_ij x_j)

    Parámetros
    ----------
    X : Vector de frecuencias/types (array).
    params : Matriz P = (p_ij) (matrix (K x K))

    Returns
    -------
    mu : np.array
    """

    P = params
    X = np.array(X)
    K = len(X)
    mu = np.zeros(K)

    for i in range(K):

        sum_acum = 0

        for j in range(K):
            if j != i:
                sum_acum += P[i][j] * X[j]

        mu[i] = X[i]*( 1 - X[i] - 2*sum_acum)

    return mu


# =========================================================
# Sistema determinista
# =========================================================

def deterministic_system(_, X, kappa, params, mu_function):
    """
    Sistema determinista

    dX/dt = kappa * mu(X)

    Parámetros
    ----------
    X: Vector de frecuencias/types (array).
    kappa: Intensidad de selección (float).
    params: Parametros d ela función mu_function.
    mu_function: Función mu a utilizar (function).

    Returns
    -------
    dX/dt : np.array
    """

    mu = mu_function(X, params)

    return kappa * mu


def simulate_deterministic(X0, kappa, params, T, mu_function, dt_save=0.01):
    """
    Simula el sistema determinista.

    Parámetros
    ----------
    X0: Condición inicial (array)
    kappa: Intensidad de selección (float).
    params: Parámetros de la función mu_function.
    T: Tiempo final (float).
    mu_function: Función mu a utilizar (function).
    dt_save: Tamaño de paso (float).

    Returns
    -------
    t: Tiempos (array).
    X: Trayectoria simulada (np.array).

    """

    t_span = (0, T)

    t_eval = np.arange(0,T + dt_save,dt_save)

    sol = solve_ivp(
        deterministic_system,
        t_span,
        X0,
        args=(kappa, params, mu_function),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )

    return sol.t, sol.y.T


# =========================================================
# Simulación
# =========================================================

K = 3
kappa = 0.4
T = 10.0
X0 = np.array([0.2, 0.4, 0.4])



# ---------------------------------------------------------
# Elegir dinámica
# ---------------------------------------------------------

#mu_function = fun_mu_2

#params = None


#mu_function = fun_mu_1
#params = [0.4, 0.3, 0.2, 0.1]



mu_function = fun_mu_3
#
params = np.array([
    [0, 0.5, 0],  # p_11, p_12, p_13 (1 neutral con 2, 1 le gana a 3)
    [0.5, 0, 1],  # p_21, p_22, p_23 (2 neutral con 1, 2 pierde con 3)
    [1, 0, 0]   # p_31, p_32, p_33 (3 pierde con 1, 3 le gana a 2)
])
#params = np.array([
#    [0,   0.7, 0.4],
#    [0.3, 0,   0.8],
#    [0.6, 0.2, 0]])


#params = np.array([
#    [0,   1/2, 1/2],
#    [1/2, 0,   1/2],
#    [1/2, 1/2, 0]
#])

# ---------------------------------------------------------
# Simular
# ---------------------------------------------------------

t, X = simulate_deterministic(X0, kappa, params, T, mu_function)


# =========================================================
# Gráfica temporal
# =========================================================

plt.figure(figsize=(14, 5))


mu_names = {
    fun_mu_1: "Transitive ordering",
    fun_mu_2: "Rock-paper-scissors dynamics",
    fun_mu_3: "Logistic competition"
}

colores = ['red','blue','green','orange','purple','cyan','magenta','brown']

etiquetas = [ f'Allele {i+1}' for i in range(K) ]

for i in range(K):
    plt.plot(t, X[:, i], color=colores[i], label=etiquetas[i], linewidth=2)

plt.xlabel('Time', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'{mu_names[mu_function]}\n' f'$\\kappa={kappa}$', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# =========================================================
# Diagrama de fase
# =========================================================

x1 = np.linspace(0.01, 0.99, 20)
x2 = np.linspace(0.01, 0.99, 20)

X1, X2 = np.meshgrid(x1, x2)

U = np.zeros(X1.shape)
V = np.zeros(X2.shape)

for i in range(X1.shape[0]):

    for j in range(X1.shape[1]):
        x3 = 1 - X1[i, j] - X2[i, j]

        if x3 > 0:
            X_vec = np.array([X1[i, j], X2[i, j], x3])
            dX = deterministic_system(0, X_vec, kappa, params, mu_function)
            U[i, j] = dX[0]
            V[i, j] = dX[1]


plt.figure(figsize=(10, 8))

plt.quiver( X1, X2, U, V, alpha=0.6)

plt.xlabel(r'$x_1$', fontsize=12)
plt.ylabel(r'$x_2$', fontsize=12)
plt.title(f'Phase diagram ($\\kappa={kappa}$)\n'r'($x_3 = 1 - x_1 - x_2$)')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.show()