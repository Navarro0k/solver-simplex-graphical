import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def generate_explanation_steps(vertices, best_value, c1, c2, optimization_type):
    """
    Genera y formatea los pasos explicativos del método gráfico.
    Retorna una lista de strings donde cada elemento es un paso.
    """
    pasos = []
    
    # Paso 1
    texto1 = "1. Paso analizar Gráfico y vértices.\n"
    texto1 += "-" * 40 + "\n"
    texto1 += f"Vértices encontrados:\n{vertices}\n\n"
    pasos.append(texto1)
    
    # Paso 2
    texto2 = "2. Calcular valores en Z por cada vértice.\n"
    texto2 += "-" * 40 + "\n"
    for x, y in vertices:
        z = c1 * x + c2 * y
        texto2 += f"Z en ({x}, {y}) = {c1}({x}) + {c2}({y}) = {z:.2f}\n"
    texto2 += "\n"
    pasos.append(texto2)
    
    # Paso 3
    z_opt, x_opt, y_opt = best_value
    texto3 = "3. Resultado final.\n"
    texto3 += "-" * 40 + "\n"
    texto3 += f"El vértice óptimo para {optimization_type.upper()} es ({x_opt}, {y_opt})\n"
    texto3 += f"Con un valor final de Z = {z_opt:.2f}\n"
    pasos.append(texto3)
    
    return pasos

def find_vertices(restrictions):
    """Calcula los vértices válidos de la región factible."""
    lineas = []
    for a, b, operador, c in restrictions:
        lineas.append((a, b, c))

    lineas.append((1, 0, 0))
    lineas.append((0, 1, 0))

    vertices = set()

    for (a1, b1, c1), (a2, b2, c2) in combinations(lineas, 2):
        try:
            x, y = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])
        except np.linalg.LinAlgError:
            continue

        x = round(x, 4)
        y = round(y, 4)

        if x < 0 or y < 0:
            continue

        punto_valido = True

        for a, b, operador, c in restrictions:
            valor = round(a * x + b * y, 4)
            if operador == "<=":
                if valor > c:
                    punto_valido = False
                    break
            else:
                if valor < c:
                    punto_valido = False
                    break

        if punto_valido:
            vertices.add((x, y))

    return list(vertices)

def calculate_optimal(vertices, c1, c2, optimization_type):
    """Evalúa la función objetivo en los vértices para encontrar el óptimo."""
    if not vertices:
        raise ValueError("No existe una región factible con estas restricciones")

    values_z = [(c1 * vx + c2 * vy, vx, vy) for vx, vy in vertices]

    if optimization_type == "max":
        return max(values_z)
    elif optimization_type == "min":
        return min(values_z)
    else:
        raise ValueError("El tipo de optimización debe ser 'max' o 'min'")

def plot_solution(restrictions, vertices, best_value, optimization_type):
    """Genera la figura de Matplotlib y la retorna para ser incrustada en la GUI."""
    best_z, best_x, best_y = best_value

    max_x = max([v[0] for v in vertices] + [1]) * 1.2
    max_y = max([v[1] for v in vertices] + [1]) * 1.2

    # Cambiamos a la creación explícita de Figure y Axes
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.linspace(0, max_x, 400)
    X, Y = np.meshgrid(x, np.linspace(0, max_y, 400))
    feasible_region = np.ones(X.shape, dtype=bool)

    for a, b, operador, c in restrictions:
        valores_evaluados = a * X + b * Y
        if b != 0:
            y_recta = (c - a * x) / b
            ax.plot(x, y_recta, label=f"{a}x₁ + {b}x₂ {operador} {c}")
        else:
            ax.axvline(c / a, label=f"{a}x₁ {operador} {c}")

        if operador == "<=":
            feasible_region &= valores_evaluados <= c
        else:
            feasible_region &= valores_evaluados >= c

    ax.contourf(X, Y, feasible_region, levels=[0.5, 1], colors=["lightgreen"], alpha=0.5)

    for vx, vy in vertices:
        ax.plot(vx, vy, marker="o", color="red", markersize=6, zorder=5) 
        # Convertimos los valores a float puro para formatear correctamente
        ax.text(vx + (max_x*0.02), vy + (max_y*0.02), f"({float(vx):.1f}, {float(vy):.1f})", fontweight='bold', color='darkred', fontsize=9)

    ax.plot(best_x, best_y, marker="*", color="gold", markeredgecolor="black", markersize=15,
             label=f"Óptimo: Z = {best_z:.2f}", zorder=6)

    ax.set_xlim(-0.5, max_x)
    ax.set_ylim(-0.5, max_y)
    ax.set_xlabel("Variable x₁")
    ax.set_ylabel("Variable x₂")
    ax.axhline(0, color="black", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Método Gráfico ({optimization_type.upper()})")
    
    fig.tight_layout()
    return fig # Retornamos la figura sin llamar a plt.show()

def graphic_solver(restrictions, c1, c2, optimization_type="max"):
    """Función orquestadora"""
    vertices = find_vertices(restrictions)
    best_value = calculate_optimal(vertices, c1, c2, optimization_type)
    
    fig = plot_solution(restrictions, vertices, best_value, optimization_type)
    
    return vertices, best_value, fig




