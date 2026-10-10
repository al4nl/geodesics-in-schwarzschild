import sympy as sp
import numpy as np
import cupy as cp

import matplotlib
matplotlib.use('TkAgg')  # para forzar que se genere la figura, por alguna razon mi pc no lo hace solo.

import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# === Calculo simbolico ===

# 1. Símbolos y métrica
t, r, theta, phi = sp.symbols('t r theta phi')
M = sp.symbols('M')
coords = [t, r, theta, phi]
n = 4   # dimensiones

g = sp.Matrix([
    [-(1 - 2*M/r), 0, 0, 0],
    [0, 1/(1 - 2*M/r), 0, 0],
    [0, 0, r**2, 0],
    [0, 0, 0, r**2 * sp.sin(theta)**2]
])
g_inv = g.inv()

# 2. Símbolos de Christoffel
Gamma = [[[0]*n for _ in range(n)] for _ in range(n)]
for l in range(n):
    for m in range(n):
        for k in range(n):
            term = sum(g_inv[l, j] * (
                sp.diff(g[j, m], coords[k]) + sp.diff(g[j, k], coords[m]) - sp.diff(g[m, k], coords[j])
            ) for j in range(n))
            Gamma[l][m][k] = sp.simplify(0.5 * term)

# 3. Lambdificar Gamma
'''
    Este paso es para que las variables simbolicas de SymPy se convierta en
    una funcion que se pueda usar en los otros paquetes.
    CuPy necesita de este paso.
'''
Gamma_funcs = [[[None]*n for _ in range(n)] for _ in range(n)]
for mu in range(n):
    for nu in range(n):
        for rho in range(n):
            Gamma_funcs[mu][nu][rho] = lambdify((t, r, theta, phi, M), Gamma[mu][nu][rho], modules='numpy')


# === Calculo numerico ===
# 1. Condiciones iniciales
'''
    Definimos las condiciones iniciales, y todas esas cosas que las involucra (como
    en las componentes de la metrica), y las para a variables tipo CuPy para que la gpu use CUDA.
'''
def generar_condiciones_iniciales(alpha, beta, observer_pos, M_val):
    t0, r0, theta0, phi0 = observer_pos
    A = 1 - 2 * M_val / r0  # f(r) del espacio de Schwarzschild
    
    dphi0 = alpha / r0  # Ver bien que hace alpha y beta aqui
    dtheta0 = beta / r0
    dr0 = -1.0  # cte, velocidad de la luz, NO TOCAR.

    # Convertimos a cupy explícitamente
    t0_cp = cp.array(t0)
    r0_cp = cp.array(r0)
    theta0_cp = cp.array(theta0)
    phi0_cp = cp.array(phi0)
    dr0_cp = cp.array(dr0)
    dphi0_cp = cp.array(dphi0)
    dtheta0_cp = cp.array(dtheta0)
    
    dt0 = cp.sqrt((dr0_cp**2)/A**2 + (r0**2 * dtheta0_cp**2 + r0**2 * np.sin(theta0)**2 * dphi0_cp**2)/A)
    # despeja el paso del tiempo inicial desde el ds^2, recuerda que para lineas de mundo tipo luz =0
    return cp.stack([t0_cp, r0_cp, theta0_cp, phi0_cp, dt0, dr0_cp, dtheta0_cp, dphi0_cp])



# 2. Ecuaciones diferenciales
''' 
    Aqui tenemos el set de ecuaciones diferenciales, le entregamos una lista
    con los valores de cara coordenada y su derivadas y[t,r,theta,phi,dt,dr,dtheta,dphi]
'''
def rhs_gpu(y, M_val, Gamma_funcs):
    x = y[:4]   # reconocemos la parte de coordenadas
    v = y[4:]   # reconocemos la parte de las velocidades
    dx = v      # Ec. Dif. para bajar el grado a la Ec. Dif. de grado 2 (la original)
    dv = cp.zeros(4)    # preparamos cosa para guardar las Ec. Dif.
    x_cpu = cp.asnumpy(x)
    # Desglosamos la expresion tensorial
    for mu in range(4):
        acc = 0.0
        for nu in range(4):
            for rho in range(4):
                Gamma_val = Gamma_funcs[mu][nu][rho](*x_cpu, M_val)
                acc -= Gamma_val * v[nu] * v[rho]
        dv[mu] = acc
    return cp.concatenate([dx, dv])


# 3. Integrador RK4
def rk4_integrate_ray(y0, M_val, Gamma_funcs, event_horizon, h=0.2, N_steps=500):
    traj = cp.zeros((N_steps+1, 8))     # guarda la aproximacion despues de todos los pasos
    y = y0.copy()                       # esta guarda todo, incluido cada paso intermedio
    traj[0] = y                         # el valor inicial es el mismo en los dos
    for i in range(N_steps):
        k1 = rhs_gpu(y, M_val, Gamma_funcs)
        k2 = rhs_gpu(y + 0.5*h*k1, M_val, Gamma_funcs)
        k3 = rhs_gpu(y + 0.5*h*k2, M_val, Gamma_funcs)
        k4 = rhs_gpu(y + h*k3, M_val, Gamma_funcs)
        y += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        traj[i+1] = y                   # guardamos el importante
        # Justo en el horizonte de eventos salen infinitos, cortamos antes de que pase eso
        if y[1] < event_horizon + h:
            print(f"Rayo cayó al horizonte de eventos en el paso {i}")
            break
    return traj[:i+2] # devuelve la funcion que resuelve al Ec. Dif




# 4. Parámetros
M_val = 1.0                                 # Masa del black hole
observer_pos = [0.0, 10.0, np.pi/2, 0.0]    # Posicion fuente de luz (t, r, theta, phi)

alphas = np.linspace(0.5, 1.0, 3)           # ancho (rayo inicial, rayo final, cantidad de rayos)
betas = np.linspace(0.0, 0.5, 2)           # alto (rayo inicial, rayo final, cantidad de rayos)

# 5. Parámetros del horizonte
R_horizon = 2 * M_val
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
# pasamos a cartesianas para la animacion
hx = R_horizon * np.outer(np.cos(u), np.sin(v))
hy = R_horizon * np.outer(np.sin(u), np.sin(v))
hz = R_horizon * np.outer(np.ones_like(u), np.cos(v))

# 6. Metodo numerico

trayectorias = []       # las geodesicas nulas
normas_nulas = []       # Guardamos las normas (deben conservarse el cero)
for alpha in alphas:    # recorremos cada rayo disparado
    for beta in betas:
        y0 = generar_condiciones_iniciales(alpha, beta, observer_pos, M_val)
        trayectoria_gpu = rk4_integrate_ray(y0, M_val, Gamma_funcs, R_horizon)
        trayectoria = cp.asnumpy(trayectoria_gpu)

        # tomamos las velocidades y posiciones para el calculo de la norma
        v_mu = trayectoria[:, 4:]  
        x_mu = trayectoria[:, :4]  

        norma = []
        for i in range(len(trayectoria)):
            # separamos los vectores x_mu y v_mu por comodidad
            t_i, r_i, th_i, ph_i = x_mu[i]
            dt_i, dr_i, dth_i, dph_i = v_mu[i]
            
            # Evaluar componentes de la métrica
            A = 1 - 2*M_val/r_i
            g_tt = -A
            g_rr = 1/A
            g_thth = r_i**2
            g_phph = r_i**2 * np.sin(th_i)**2
            
            # Calcular norma del cuadrivector
            norm_i = (g_tt * dt_i**2 + 
                      g_rr * dr_i**2 + 
                      g_thth * dth_i**2 + 
                      g_phph * dph_i**2)
            norma.append(norm_i)
        
        normas_nulas.append(norma)

        # separamos las componentes espaciales de las geodesicas
        r_vals = trayectoria[:, 1]
        theta_vals = trayectoria[:, 2]
        phi_vals = trayectoria[:, 3]

        # Conversión a coordenadas cartesianas para la animacion
        x_vals = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        y_vals = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        z_vals = r_vals * np.cos(theta_vals)

        trayectorias.append((x_vals, y_vals, z_vals))

print(f"Cargando animación...")

# 2. Crear figura y ejes 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(hx, hy, hz, color='black', alpha=0.6, edgecolor='none')

fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-6, 6)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.title.set_color('white')
ax.tick_params(colors='white')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Geodésicas nulas en Schwarzschild')

# 3. Inicializar líneas
lines = [ax.plot([], [], [], color='orange', lw=1)[0] for _ in trayectorias]

# 4. Función de actualización
def update(frame):
    for line, (x, y, z) in zip(lines, trayectorias):
        if frame < len(x):
            line.set_data(x[:frame], y[:frame])
            line.set_3d_properties(z[:frame])
    return lines

# Calcular frames máximo
max_frames = max(len(traj[0]) for traj in trayectorias)

# 5. Animar
ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=True)

# Guardar como GIF
ani.save('schwarzschild_geodesics.gif', writer='pillow', fps=30)

plt.show()


# === Norma nula ===
plt.figure(figsize=(8,5))
for norma in normas_nulas:
    plt.plot(norma, label='‖v‖²', alpha=0.7)

plt.axhline(0, color='white', ls='--', lw=1)
plt.xlabel("Paso de integración")
plt.ylabel(r"$g_{\mu\nu} \dot{x}^\mu \dot{x}^\nu$")
plt.title("Norma del cuadrivector")
plt.grid(True)
plt.tight_layout()
plt.ylim(-1e-5, 1e-5)  
plt.show()


