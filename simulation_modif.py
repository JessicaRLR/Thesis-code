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
T = 100.0
center = 1
X0 = (0.4, 0.35, 0.25)
#X0 = (0.6, 0.35, 0.05)
#X0 = np.array([0.4, 0.3, 0.3])
#X0 = np.array([ 0.2, 0.79, 0.01])
#X0 = np.array([0.6, 0.39, 0.01])
#X0 = (0.05, 0.65, 0.3)
# ---------------------------------------------------------
# Elegir función
# ---------------------------------------------------------
function= 5

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
        [0, 0, 1],   # p_11, p_12, p_13 
        [1, 0, 0],   # p_21, p_22, p_23 
        [0, 1, 0]      # p_31, p_32, p_33 
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

    plt.figure(figsize=(9, 8))

    plt.quiver( X1, X2, U, V, alpha=0.6)

    # Trayectoria del punto inicial
    x1_traj = X[:, 0]
    x2_traj = X[:, 1]
    plt.plot(x1_traj, x2_traj, 'k-', linewidth=2.5, label='Trayectoria', alpha=0.8, color= 'blue')

    if function == 3 and center ==1 :
        a23 = 2*params[1, 2] - 1
        a13 = 2*params[0,2] - 1
        a12 = 2*params[0, 1] - 1
        D = a23 - a13 + a12
        cent = np.array([a23,-a13 , a12]/D)
        plt.plot(cent[0], cent[1], 'ko', markersize=10)      

            
    # Punto inicial
    plt.plot(X0[0], X0[1], 'ko', markersize=10, label='Punto inicial', color = 'green')
    # Punto final
    plt.plot(X[-1, 0], X[-1, 1], 'k*', markersize=12, label='Punto final', color = 'red')


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
    z = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i < j:
                z[i,j] = 0
            elif i == j:

                sum_i = np.sum(x[:i+1])
                sum_i_menos_1 = np.sum(x[:i])

                num = x[i] *(1- sum_i)
                den = 1 - sum_i_menos_1
                z[i,j] = np.sqrt(max(0, num/den)) if den > 1e-12 else 0
            else:
                sum_j = np.sum(x[:j+1])
                sum_j_menos_1 = np.sum(x[:j])
                num = x[j]
                den = (1 - sum_j_menos_1)*(1 - sum_j)

                z[i,j] = -x[i]*np.sqrt(max(0, num/den)) if den > 1e-12 else 0
    return z


def truncated_exponential(lam):
    u = np.random.rand()
    return -np.log(1 - u*(1 - np.exp(-lam))) / lam
    


def euler_maruyama_with_jumps(mu, params, x0, kappa, sigma, T, N):

    dt = T / N

    x0 = np.asarray(x0, dtype=float)
    K = len(x0)

    X = np.zeros((N + 1, K))
    X[0] = x0

    jump_times = []
    for n in range(N):

        x_current = X[n].copy()

        # --- Deriva ---
        drift = kappa * mu(x_current, params) * dt

        # --- Difusión Browniana ---
        dW = np.random.normal(0, np.sqrt(dt), K)
        diffusion = np.sqrt(sigma) * zeta(x_current) @ dW


        # --- Saltos ---
        # Numero de saltos en el intervalo [t, t+dt]: Poisson(lambda_total * dt)
        lam = 0.05 #parametro de la exponencial
        lambda_total = (1 - np.exp(-lam))
        # creo que multiplicar * 10 no es necesario

        num_jumps = np.random.poisson(lambda_total * dt)
        x_temp = x_current + drift + diffusion

        for _ in range(num_jumps):

            u = np.random.uniform(0,1)
            z = truncated_exponential(lam)

            # Guardamos el tiempo donde ocurrió el salto
            jump_times.append(n * dt)

            # Encontrar el indice i
            cumsum = np.cumsum(x_temp)
            i = np.searchsorted(cumsum, u)
            
            # actualizar salto
            x_temp = x_temp - z*x_temp
            x_temp[i] += z
        
        # ---- Actualizacion -----#

        # proyeccion al simplex
        x_temp = np.maximum(x_temp, 0)
        x_temp = x_temp/np.sum(x_temp)

        X[n+1] = x_temp

    print("Cantidad de intervalos con saltos:", len(jump_times))
    return X, jump_times

# =========================================================
# Ejemplo de uso
# =========================================================

T = 1000
N = 500000

# Ejecutar simulación con saltos
X, jump_times = euler_maruyama_with_jumps(
    mu=mu_function,
    params=params,
    x0=X0,
    kappa=kappa,
    sigma=sigma,
    T=T,
    N=N
)

# Tiempo
t = np.linspace(0, T, N + 1)


plt.figure(figsize=(12, 5))

# Grafica
colores = ['red','blue','green']
for i in range(K):
    plt.plot(t, X[:, i], label=f"$x_{i+1}$", color=colores[i])

# Donde ocurrieron los saltos
bool = True
for tj in jump_times:
    if bool:
        plt.axvline(x = tj, color = 'black', linestyle = '--', alpha = 0.3, label = 'jump')
        bool = False
    plt.axvline(x = tj, color = 'black', linestyle = '--', alpha = 0.3)


plt.xlabel("t")
plt.ylabel("frequency")
plt.legend()
plt.title(f"Simulation with jumps ")
plt.show()

