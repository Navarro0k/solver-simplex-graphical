import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class GraphicSolver:
    """
    Encapsula el estado y la lógica del método gráfico de Programación Lineal.
    """
    def __init__(self, restrictions, c1, c2, optimization_type="max"):
        self.restrictions = restrictions
        self.c1 = c1
        self.c2 = c2
        self.optimization_type = optimization_type
        
        self.vertices = []
        self.best_value = None

    def resolver(self):
        """Orquesta el cálculo de la región factible y el punto óptimo."""
        self._find_vertices()
        self._calculate_optimal()

    def _find_vertices(self):
        """Calcula los vértices válidos de la región factible."""
        lineas = [(a, b, c) for a, b, _, c in self.restrictions]
        lineas.extend([(1, 0, 0), (0, 1, 0)])

        vertices_set = set()

        for (a1, b1, c1), (a2, b2, c2) in combinations(lineas, 2):
            try:
                x, y = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])
            except np.linalg.LinAlgError:
                continue

            x, y = round(x, 4), round(y, 4)
            if x < 0 or y < 0:
                continue

            punto_valido = True
            for a, b, operador, c in self.restrictions:
                valor = round(a * x + b * y, 4)
                if operador == "<=" and valor > c:
                    punto_valido = False
                    break
                elif operador == ">=" and valor < c:
                    punto_valido = False
                    break

            if punto_valido:
                vertices_set.add((x, y))

        self.vertices = list(vertices_set)

    def _calculate_optimal(self):
        """Evalúa la función objetivo en los vértices para encontrar el óptimo."""
        if not self.vertices:
            raise ValueError("No existe una región factible con estas restricciones")

        values_z = [(self.c1 * vx + self.c2 * vy, vx, vy) for vx, vy in self.vertices]

        if self.optimization_type == "max":
            self.best_value = max(values_z)
        elif self.optimization_type == "min":
            self.best_value = min(values_z)
        else:
            raise ValueError("El tipo de optimización debe ser 'max' o 'min'")

    def plot_solution(self):
        """Genera la figura de Matplotlib y la retorna."""
        best_z, best_x, best_y = self.best_value

        max_x = max([v[0] for v in self.vertices] + [1]) * 1.2
        max_y = max([v[1] for v in self.vertices] + [1]) * 1.2

        fig, ax = plt.subplots(figsize=(6, 5))

        x_vals = np.linspace(0, max_x, 400)
        X, Y = np.meshgrid(x_vals, np.linspace(0, max_y, 400))
        feasible_region = np.ones(X.shape, dtype=bool)

        for a, b, operador, c in self.restrictions:
            valores_evaluados = a * X + b * Y
            if b != 0:
                y_recta = (c - a * x_vals) / b
                ax.plot(x_vals, y_recta, label=f"{a}x₁ + {b}x₂ {operador} {c}")
            else:
                ax.axvline(c / a, label=f"{a}x₁ {operador} {c}")

            if operador == "<=":
                feasible_region &= (valores_evaluados <= c)
            else:
                feasible_region &= (valores_evaluados >= c)

        ax.contourf(X, Y, feasible_region, levels=[0.5, 1], colors=["lightgreen"], alpha=0.5)

        for vx, vy in self.vertices:
            ax.plot(vx, vy, marker="o", color="red", markersize=6, zorder=5) 
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
        ax.set_title(f"Método Gráfico ({self.optimization_type.upper()})")
        
        fig.tight_layout()
        return fig

    def generate_steps(self):
        """Genera y formatea los pasos explicativos del método gráfico."""
        pasos = []
        
        texto1 = "1. Analizar gráfico y vértices.\n" + "-" * 40 + "\n"
        texto1 += f"Vértices encontrados:\n{self.vertices}\n\n"
        pasos.append(texto1)
        
        texto2 = "2. Calcular valores en Z por cada vértice.\n" + "-" * 40 + "\n"
        for x, y in self.vertices:
            z = self.c1 * x + self.c2 * y
            texto2 += f"Z en ({x}, {y}) = {self.c1}({x}) + {self.c2}({y}) = {z:.2f}\n"
        texto2 += "\n"
        pasos.append(texto2)
        
        z_opt, x_opt, y_opt = self.best_value
        texto3 = "3. Resultado final.\n" + "-" * 40 + "\n"
        texto3 += f"El vértice óptimo para {self.optimization_type.upper()} es ({x_opt}, {y_opt})\n"
        texto3 += f"Con un valor final de Z = {z_opt:.2f}\n"
        pasos.append(texto3)
        
        return pasos