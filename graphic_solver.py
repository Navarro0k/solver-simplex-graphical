import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def find_vertices(restrictions):
    """Calcula los vértices válidos de la región factible."""

    # se trata como una línea a*x + b*y = c
    lineas = []
    for a, b, operador, c in restrictions:
        lineas.append((a, b, c))

    # ejes como líneas adicionales
    lineas.append((1, 0, 0))  # eje Y (x = 0)
    lineas.append((0, 1, 0))  # eje X (y = 0)

    vertices = set()

    for (a1, b1, c1), (a2, b2, c2) in combinations(lineas, 2):

        # Intentamos resolver el sistema de 2 ecuaciones (intersección de las líneas)
        try:
            x, y = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])
        except np.linalg.LinAlgError:
            continue  # las líneas son paralelas, no hay intersección

        x = round(x, 4)
        y = round(y, 4)

        # Descartamos puntos fuera del primer cuadrante
        if x < 0 or y < 0:
            continue

        # Verificamos que el punto cumpla TODAS las restricciones originales
        punto_valido = True

        for a, b, operador, c in restrictions:
            valor = round(a * x + b * y, 4)

            if operador == "<=":
                if valor > c:
                    punto_valido = False
                    break
            else:  # operador == ">="
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

    # Lista por comprensión para evaluar Z más rápido
    values_z = [(c1 * vx + c2 * vy, vx, vy) for vx, vy in vertices]

    if optimization_type == "max":
        return max(values_z)
    elif optimization_type == "min":
        return min(values_z)
    else:
        raise ValueError("El tipo de optimización debe ser 'max' o 'min'")

def plot_solution(restrictions, vertices, best_value, optimization_type):
    """Se encarga exclusivamente del renderizado del gráfico de Matplotlib."""
    best_z, best_x, best_y = best_value

    # Escala dinámica: 20% más allá del vértice más lejano
    max_x = max([v[0] for v in vertices] + [1]) * 1.2
    max_y = max([v[1] for v in vertices] + [1]) * 1.2

    x = np.linspace(0, max_x, 400)
    X, Y = np.meshgrid(x, np.linspace(0, max_y, 400))
    feasible_region = np.ones(X.shape, dtype=bool)

    # Procesar restricciones:
    for a, b, operador, c in restrictions:
        valores_evaluados = a * X + b * Y

        if b != 0:
            # a*x + b*y = c  ->  y = (c - a*x) / b
            y_recta = (c - a * x) / b
            plt.plot(x, y_recta, label=f"{a}x + {b}y {operador} {c}")
        else:
            # x = c/a
            plt.axvline(c / a, label=f"{a}x {operador} {c}")

        if operador == "<=":
            feasible_region &= valores_evaluados <= c
        else:  # ">="
            feasible_region &= valores_evaluados >= c

    # Dibujar componentes
    plt.contourf(X, Y, feasible_region, levels=[0.5, 1], colors=["lightgreen"], alpha=0.5)

    for vx, vy in vertices:
        plt.plot(vx, vy, marker="o", color="red", markersize=6, zorder=5) 
        plt.text(vx + (max_x*0.02), vy + (max_y*0.02), f"({vx}, {vy})", fontweight='bold', color='darkred', fontsize=9)

    plt.plot(best_x, best_y, marker="*", color="gold", markeredgecolor="black", markersize=15,
             label=f"Óptimo: Z = {best_z:.2f}", zorder=6)

    # Configuración de ejes y leyenda
    plt.xlim(-0.5, max_x)
    plt.ylim(-0.5, max_y)
    plt.xlabel("Variable X")
    plt.ylabel("Variable Y")
    plt.axhline(0, color="black", linewidth=1.5)
    plt.axvline(0, color="black", linewidth=1.5)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f"Método Gráfico ({optimization_type.upper()})")
    plt.tight_layout()
    plt.show()

def graphic_solver(restrictions, c1, c2, optimization_type="max"):
    """Función orquestadora que une todas las piezas."""
    vertices = find_vertices(restrictions)
    best_value = calculate_optimal(vertices, c1, c2, optimization_type)
    
    plot_solution(restrictions, vertices, best_value, optimization_type)
    
    return vertices, best_value

if __name__ == "__main__":
    mis_restricciones = [
        (2, 1, "<=", 8),
        (1, 2, "<=", 7),
        (0, 1, ">=", 3)
    ]
    
    print("Generando gráfico y calculando vértices...")
    puntos_encontrados, resultado = graphic_solver(mis_restricciones, c1=3, c2=5, optimization_type="max")
    
    print(f"Los vértices calculados son: {puntos_encontrados}")
    print(f"Resultado óptimo: Z = {resultado[0]:.2f} en ({resultado[1]}, {resultado[2]})")