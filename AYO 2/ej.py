import time
import math
import random
import itertools
from multiprocessing import Pool

try:
    import numpy as np
except ImportError as exc:
    raise ImportError("Instala la dependencia 'numpy' ejecutando: pip install numpy") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("Instala la dependencia 'matplotlib' ejecutando: pip install matplotlib") from exc

plt.style.use('default')


# ==========================================
# EJERCICIO 1: Aceleración Paralela (Parallel Speedup)
# ==========================================
def worker_pi_samples_pure(num_samples):
    count = 0
    for _ in range(num_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            count += 1
    return count

def estimate_pi(num_processes, total_samples=20_000_000):
    samples_per_process = total_samples // num_processes
    start_time = time.perf_counter()
    
    if num_processes == 1:
        inside = worker_pi_samples_pure(total_samples)
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(worker_pi_samples_pure, [samples_per_process] * num_processes)
            inside = sum(results)
            
    elapsed_time = time.perf_counter() - start_time
    pi_val = 4 * inside / total_samples
    return elapsed_time, pi_val

def ejecutar_ejercicio_1():
    print("\n--- Ejercicio 1: Parallel Speedup ---")
    procesos = [1, 2, 4, 8]
    tiempos = []
    
    for p in procesos:
        t, pi_est = estimate_pi(p)
        tiempos.append(t)
        print(f"Procesos: {p} | Tiempo: {t:.4f}s | Pi estimado: {pi_est:.4f}")
        
    t1 = tiempos[0]
    speedup = [t1 / tp for tp in tiempos]
    
    plt.figure(figsize=(7, 5))
    plt.plot(procesos, speedup, 'o-', label='Speedup Real', linewidth=2)
    plt.plot(procesos, procesos, '--', label='Speedup Ideal (y=x)', color='red')
    plt.xlabel('Número de Procesos')
    plt.ylabel('Speedup ($T_1 / T_p$)')
    plt.title('Ejercicio 1: Aceleración Paralela (Parallel Speedup)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('ejercicio1_speedup.png')
    plt.close()


# ==========================================
# EJERCICIO 2: Greedy Knapsack
# ==========================================
def knapsack_greedy_by_ratio(items, capacity):
    sorted_items = sorted(items, key=lambda x: x[2]/x[1], reverse=True)
    selected, total_weight, total_value = [], 0, 0
    for item in sorted_items:
        if total_weight + item[1] <= capacity:
            selected.append(item[0])
            total_weight += item[1]
            total_value += item[2]
    return selected, total_weight, total_value

def knapsack_greedy_by_min_weight(items, capacity):
    sorted_items = sorted(items, key=lambda x: x[1])
    selected, total_weight, total_value = [], 0, 0
    for item in sorted_items:
        if total_weight + item[1] <= capacity:
            selected.append(item[0])
            total_weight += item[1]
            total_value += item[2]
    return selected, total_weight, total_value

def ejecutar_ejercicio_2():
    print("\n--- Ejercicio 2: Greedy Knapsack ---")
    capacity = 10
    items = [
        ('Obj A', 1, 1),
        ('Obj B', 2, 1),
        ('Obj C', 9, 90)
    ]
    
    sel_ratio, w_ratio, v_ratio = knapsack_greedy_by_ratio(items, capacity)
    sel_minw, w_minw, v_minw = knapsack_greedy_by_min_weight(items, capacity)
    
    print(f"Valor/Peso -> Seleccionados: {sel_ratio}, Peso Usado: {w_ratio}, Valor: {v_ratio}")
    print(f"Menor Peso -> Seleccionados: {sel_minw}, Peso Usado: {w_minw}, Valor: {v_minw}")
    
    metodos = ['Valor / Peso', 'Menor Peso']
    valores = [v_ratio, v_minw]
    pesos = [w_ratio, w_minw]
    
    x = np.arange(len(metodos))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, valores, width, label='Valor Total Obt.', color='skyblue')
    ax.bar(x + width/2, pesos, width, label='Capacidad Usada', color='salmon')
    ax.set_ylabel('Métricas')
    ax.set_title('Ejercicio 2: Comparación Criterios Voraces (Knapsack)')
    ax.set_xticks(x)
    ax.set_xticklabels(metodos)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ejercicio2_knapsack.png')
    plt.close()


def calcular_distancia_matriz(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def costo_tour(tour, dist_matrix):
    return sum(dist_matrix[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def nearest_neighbor_tour(dist_matrix, start_node=0):
    n = len(dist_matrix)
    unvisited = set(range(n))
    unvisited.remove(start_node)
    tour = [start_node]
    current = start_node
    while unvisited:
        nxt = min(unvisited, key=lambda x: dist_matrix[current][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


# ==========================================
# EJERCICIO 3: Búsqueda Local (2-opt TSP)
# ==========================================
def two_opt_local_search(tour, dist_matrix):
    best_tour = tour[:]
    best_cost = costo_tour(best_tour, dist_matrix)
    improvements = 0
    n = len(tour)
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                new_cost = costo_tour(new_tour, dist_matrix)
                if new_cost < best_cost - 1e-6:
                    best_tour = new_tour
                    best_cost = new_cost
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
    return best_tour, best_cost, improvements

def ejecutar_ejercicio_3(coords, dist_matrix):
    print("\n--- Ejercicio 3: 2-opt Local Search ---")
    n = len(coords)
    num_trials = 30
    
    imp_random, cost_random = [], []
    imp_nn, cost_nn = [], []
    
    for _ in range(num_trials):
        rand_tour = list(range(n))
        random.shuffle(rand_tour)
        _, c_rand, k_rand = two_opt_local_search(rand_tour, dist_matrix)
        imp_random.append(k_rand)
        cost_random.append(c_rand)
        
        start_node = random.randint(0, n - 1)
        nn_tour = nearest_neighbor_tour(dist_matrix, start_node)
        _, c_nn, k_nn = two_opt_local_search(nn_tour, dist_matrix)
        imp_nn.append(k_nn)
        cost_nn.append(c_nn)
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].boxplot([imp_random, imp_nn])
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['Aleatorio', 'Vecino Más Cercano'])
    axes[0].set_ylabel('Número de Mejoras')
    axes[0].set_title('Mejoras requeridas hasta converger')
    
    axes[1].boxplot([cost_random, cost_nn])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Aleatorio', 'Vecino Más Cercano'])
    axes[1].set_ylabel('Costo de Ruta Final')
    axes[1].set_title('Calidad del Óptimo Local')
    
    plt.suptitle('Ejercicio 3: Búsqueda Local (2-opt)')
    plt.tight_layout()
    plt.savefig('ejercicio3_2opt.png')
    plt.close()


def simulated_annealing(dist_matrix, schedule='exp', t0=100.0, tf=0.1, alpha=0.995, beta=0.1):
    n = len(dist_matrix)
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_cost = costo_tour(current_tour, dist_matrix)
    best_tour, best_cost = current_tour[:], current_cost
    
    t = t0
    history_cost = []
    start_time = time.perf_counter()
    
    while t > tf:
        i, j = sorted(random.sample(range(n), 2))
        neighbor_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
        neighbor_cost = costo_tour(neighbor_tour, dist_matrix)
        
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / t):
            current_tour = neighbor_tour
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_tour = current_tour
                best_cost = current_cost
                
        history_cost.append(current_cost)
        
        if schedule == 'exp':
            t *= alpha
        elif schedule == 'linear':
            t -= beta
            
    elapsed_time = time.perf_counter() - start_time
    return best_tour, best_cost, elapsed_time, history_cost

def ejecutar_ejercicio_4(dist_matrix):
    print("\n--- Ejercicio 4: Simulated Annealing ---")
    _, c_exp, t_exp, hist_exp = simulated_annealing(dist_matrix, schedule='exp', t0=100, tf=0.1, alpha=0.99)
    _, c_lin, t_lin, hist_lin = simulated_annealing(dist_matrix, schedule='linear', t0=100, tf=0.1, beta=0.05)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist_exp, label='Enfriamiento Exponencial', alpha=0.8)
    axes[0].plot(hist_lin, label='Enfriamiento Lineal', alpha=0.8)
    axes[0].set_xlabel('Iteración')
    axes[0].set_ylabel('Costo Actual')
    axes[0].set_title('Curvas de Convergencia')
    axes[0].legend()
    
    schemes = ['Exponencial', 'Lineal']
    times = [t_exp, t_lin]
    costs = [c_exp, c_lin]
    
    axes[1].scatter(times, costs, color=['purple', 'green'], s=100)
    for i, txt in enumerate(schemes):
        axes[1].annotate(txt, (times[i], costs[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    axes[1].set_xlabel('Tiempo de Ejecución (s)')
    axes[1].set_ylabel('Costo Final')
    axes[1].set_title('Tiempo vs. Calidad')
    
    plt.suptitle('Ejercicio 4: Simulated Annealing')
    plt.tight_layout()
    plt.savefig('ejercicio4_sa.png')
    plt.close()


# ==========================================
# EJERCICIO 5: Razón de Aproximación (MST)
# ==========================================
def tsp_optimo_exacto(dist_matrix):
    n = len(dist_matrix)
    nodes = list(range(1, n))
    min_cost = float('inf')
    for perm in itertools.permutations(nodes):
        tour = [0] + list(perm)
        c = costo_tour(tour, dist_matrix)
        if c < min_cost:
            min_cost = c
    return min_cost

def mst_approximation_tsp(dist_matrix):
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst = minimum_spanning_tree(dist_matrix).toarray()
    
    n = len(dist_matrix)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mst[i, j] > 0 or mst[j, i] > 0:
                adj[i].append(j)
                
    visited = []
    stack = [0]
    seen = set()
    while stack:
        u = stack.pop()
        if u not in seen:
            seen.add(u)
            visited.append(u)
            for v in reversed(adj[u]):
                if v not in seen:
                    stack.append(v)
                    
    cost = costo_tour(visited, dist_matrix)
    return visited, cost

def ejecutar_ejercicio_5():
    print("\n--- Ejercicio 5: MST Approximation ---")
    num_instances = 30
    num_cities = 9
    
    ratios = []
    costos_mst = []
    costos_mst_2opt = []
    costos_local_search = []
    costos_sa = []
    
    for _ in range(num_instances):
        coords = np.random.rand(num_cities, 2)
        dist_mat = calcular_distancia_matriz(coords)
        
        c_opt = tsp_optimo_exacto(dist_mat)
        tour_mst, c_mst = mst_approximation_tsp(dist_mat)
        _, c_mst_2opt, _ = two_opt_local_search(tour_mst, dist_mat)
        
        rand_tour = list(range(num_cities))
        random.shuffle(rand_tour)
        _, c_ls, _ = two_opt_local_search(rand_tour, dist_mat)
        
        _, c_sa, _, _ = simulated_annealing(dist_mat, schedule='exp', t0=50, tf=0.1, alpha=0.95)
        
        ratios.append(c_mst / c_opt)
        costos_mst.append(c_mst)
        costos_mst_2opt.append(c_mst_2opt)
        costos_local_search.append(c_ls)
        costos_sa.append(c_sa)
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(ratios, bins=10, color='teal', edgecolor='black', alpha=0.7)
    axes[0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='Límite Teórico (2.0)')
    axes[0].set_xlabel('Razón de Aproximación ($C_{obtenido} / C_{óptimo}$)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de la Razón de Aproximación')
    axes[0].legend()
    
    axes[1].boxplot([costos_mst, costos_mst_2opt, costos_local_search, costos_sa])
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xticklabels(['MST Puro', 'MST + 2-opt', 'Búsqueda Local', 'Recocido Simulado'])
    axes[1].set_ylabel('Costo de la Ruta')
    axes[1].set_title('Comparativa de Métodos')
    
    plt.suptitle('Ejercicio 5: Razón de Aproximación basado en MST')
    plt.tight_layout()
    plt.savefig('ejercicio5_mst.png')
    plt.close()


# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == '__main__':
    ejecutar_ejercicio_1()
    ejecutar_ejercicio_2()
    
    np.random.seed(42)
    random.seed(42)
    coords_tsp = np.random.rand(20, 2)
    dist_matrix_tsp = calcular_distancia_matriz(coords_tsp)
    
    ejecutar_ejercicio_3(coords_tsp, dist_matrix_tsp)
    ejecutar_ejercicio_4(dist_matrix_tsp)
    ejecutar_ejercicio_5()
    
    print("\nProceso finalizado.")