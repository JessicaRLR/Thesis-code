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

        mu[i] = X[i]*(2*sum_acum -( 1 - X[i] ) )

    return mu



def fun_mu_4(X, params):
    """
    Calcula
    
    mu_i = 2 x_i [ sum_j x_j^2 - x_i ]
    
    Parámetros
    ----------
    X: Vector de frecuencias/types (array).
    params: No se usa.
    
    Returns
    -------
    mu : np.array
    """
    X = np.array(X)
    suma_cuadrados = np.sum(X**2)
    mu_aprox = 2 * X * (suma_cuadrados - X)
    

    mu = mu_aprox - np.mean(mu_aprox)
    
    return mu



def fun_mu_5(X, params):
    """
    Calcula
    
    mu_i = x_i [x_i - sum_{j=1}^K x_j^2]
    
    Parámetros
    ----------
    X: Vector de frecuencias/types (array).
    params: No se usa.
    
    Returns
    -------
    mu : np.array
    """
    X = np.array(X)
    suma_cuadrados = np.sum(X**2)
    
    mu = X * (X - suma_cuadrados)
    
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

    X = np.maximum(X, 0)  # Eliminar negativos
    X = X / X.sum()       # Normalizar a 1

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
# Configuration of the simulation 
# =========================================================

# Parameters 
K = 3
kappa = 0.4
sigma = 0.01
T = 60.0
X0 = np.array([0.6, 0.35, 0.05])

# ---------------------------------------------------------
# Elegir función
# ---------------------------------------------------------
function= 4

# ---------------------------------------------------------
# Configuración según la función elegida
# ---------------------------------------------------------

if function == 1:
    mu_function = fun_mu_1
    params = [0.6, 0.3, 0.1]
    
elif function == 2:
    mu_function = fun_mu_2
    params = None
    
elif function == 3:
    mu_function = fun_mu_3
    params = np.array([
        [0, 1, 0.2],   # p_11, p_12, p_13 
        [0, 0, 1],   # p_21, p_22, p_23 
        [0.8, 0, 0]      # p_31, p_32, p_33 
    ])
    # params = np.array([
    #     [0,   0.7, 0.4],
    #     [0.3, 0,   0.8],
    #     [0.6, 0.2, 0]
    # ])
    # params = np.array([
    #     [0,   1/2, 1/2],
    #     [1/2, 0,   1/2],
    #     [1/2, 1/2, 0]
    # ])
elif function == 4:
    mu_function = fun_mu_4
    params = None
elif function == 5:
    mu_function = fun_mu_5
    params = None   

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
    fun_mu_3: "Logistic competition",
    fun_mu_4: "Negative frequency-dependent",
    fun_mu_5: "Positive frequency-dependent"
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
# Diagrama de fase (solo para 3 especies)
# =========================================================

if K == 3:
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

    # Trayectoria del punto inicial
    x1_traj = X[:, 0]
    x2_traj = X[:, 1]
    plt.plot(x1_traj, x2_traj, 'k-', linewidth=2.5, label='Trayectoria', alpha=0.8, color= 'blue')
    
    # Punto inicial
    plt.plot(X0[0], X0[1], 'ko', markersize=10, label='Punto inicial')
    
    # Punto final
    plt.plot(X[-1, 0], X[-1, 1], 'k*', markersize=12, label='Punto final')
    # plt.plot(5/13,3/13, 'ro', markersize=6, label='Punto crítico')
    plt.xlabel(r'$x_1$', fontsize=12)
    plt.ylabel(r'$x_2$', fontsize=12)
    plt.title(f'Phase diagram ($\\kappa={kappa}$)\n'r'($x_3 = 1 - x_1 - x_2$)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()







# =========================================================
# Simulación E-M con Saltos
# =========================================================


def zeta(x):
    x = np.asarray(x, dtype=float)
    K = len(x)

    Z = np.zeros((K, K))

    sum = np.cumsum(x)

    for i in range(K):

        # diagonal
        denom = 1.0 if i == 0 else 1.0 - sum[i-1]

        numer = x[i] * (1.0 - sum[i])

        Z[i, i] = np.sqrt(max(numer / denom, 0.0))

        # parte inferior
        for j in range(i):

            a = 1.0 if j == 0 else 1.0 - sum[j-1]
            b = 1.0 - sum[j]

            value = -x[i] * np.sqrt(x[j] / (a * b))

            Z[i, j] = value

    return Z

def jump(x, z):

    u = np.random.uniform(0, 1)
    cumsum = np.cumsum(x)
    
    i = np.searchsorted(cumsum, u)

    e = np.zeros_like(x)
    e[i] = 1.0
    jump_vec = z * (e - x)  
    
    return jump_vec


def truncated_exponential(lam):
    u = np.random.rand()
    return -np.log(1 - u*(1 - np.exp(-lam))) / lam
    

        
    


def euler_maruyama_with_jumps(mu, params, x0, kappa, sigma, T, N):

    dt = T / N

    x0 = np.asarray(x0, dtype=float)
    K = len(x0)

    X = np.zeros((N + 1, K))
    X[0] = x0
    count = 0
    for n in range(N):

        x = X[n]



        x = np.maximum(x, 0)

        s = np.sum(x)
        if s > 0:
            x /= s



        # --- Drift ---
        drift = kappa * mu(x, params) * dt

        # --- Difusión Browniana ---
        dW = np.sqrt(dt) * np.random.randn(K)
        diffusion = np.sqrt(sigma) * zeta(x) @ dW

        x_new = x + drift + diffusion
        x_new = np.maximum(x_new, 0)

        sm = np.sum(x_new)
        if sm > 0:
            x_new /= sm

        # --- Saltos ---
        # Numero de saltos en el intervalo [t, t+dt]: Poisson(lambda_total * dt)
        lam = 0.01 #parametro de la exponencial
        lambda_total = (1 - np.exp(-lam))*10

        # numero de saltos
        num_jumps = np.random.poisson(lambda_total * dt)
        
        if num_jumps!= 0:
            count += 1
            print(n*dt)
        jumps = 1
        if jumps == 1:
            for _ in range(num_jumps):

                z = truncated_exponential(lam)
                jump_increment = jump(x, z)

                x_new = x_new + jump_increment

        X[n+1] = x_new

    print('cantidad de saltos: ', count)
    return X

# =========================================================
# Ejemplo de uso
# =========================================================

T = 1000
N = 500000

# Ejecutar simulación con saltos
X = euler_maruyama_with_jumps(
    mu=mu_function,
    params=params,
    x0=X0,
    kappa=kappa,
    sigma=sigma,
    T=T,
    N=N
)

t = np.linspace(0, T, N + 1)

colores = ['red','blue','green']
for i in range(K):
    plt.plot(t, X[:, i], label=f"$x_{i+1}$", color=colores[i])

plt.xlabel("t")
plt.ylabel("frequency")
plt.legend()
plt.title(f"Simulation with jumps ")
plt.show()