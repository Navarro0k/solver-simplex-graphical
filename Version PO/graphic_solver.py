import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from problem import LinearProblem

class GraphicSolver:
    """
    Encapsula el estado y la lógica del método gráfico de Programación Lineal.
    Recibe una instancia de LinearProblem (clase entidad).
    """
    def __init__(self, problem: LinearProblem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("GraphicSolver requiere una instancia de LinearProblem")

        if len(problem.objective_coefficients) > 2:
            raise ValueError("El método gráfico solo admite problemas de hasta 2 variables")

        self.problem = problem
        self.optimization_type = problem.optimization_type
        self.vertices = []
        self.best_value = None

        # Extraer los coeficientes de la función objetivo (c1, c2).
        self.c1 = problem.objective_coefficients[0]
        self.c2 = problem.objective_coefficients[1]

        # Convertir al formato (a, b, operador, rhs) que usa el método gráfico.
        self.restrictions = []
        for coeficientes, operador, rhs in problem.restrictions:
            a = coeficientes[0]
            b = coeficientes[1] if len(coeficientes) > 1 else 0
            self.restrictions.append((a, b, operador, rhs))

    def resolver(self):
        """Orquesta el cálculo de la región factible y el punto óptimo."""
        self._find_vertices()
        self._calculate_optimal()

    def _find_vertices(self):
        """
        Calcula los vértices válidos de la región factible para el método gráfico.
        Encuentra las intersecciones de todas las rectas y filtra los puntos 
        que no cumplen con el sistema de inecuaciones.
        """
        ecuaciones_rectas = []
        
        for a, b, operador, c in self.restrictions:
            ecuaciones_rectas.append((a, b, c))

        # Añadimos las restricciones de no negatividad (x >= 0, y >= 0)
        ecuaciones_rectas.append((1, 0, 0))
        ecuaciones_rectas.append((0, 1, 0))

        vertices_validos = set()
        pares_de_rectas = combinations(ecuaciones_rectas, 2)

        for recta_1, recta_2 in pares_de_rectas:
            a1, b1, c1 = recta_1
            a2, b2, c2 = recta_2

            try:
                matriz_coeficientes = [[a1, b1], [a2, b2]]
                terminos_independientes = [c1, c2]
                x, y = np.linalg.solve(matriz_coeficientes, terminos_independientes)
            except np.linalg.LinAlgError:
                continue

            x = round(x, 4)
            y = round(y, 4)

            if x < 0 or y < 0:
                continue

            cumple_todas_las_restricciones = True

            for a, b, operador, c in self.restrictions:
                valor_evaluado = round((a * x) + (b * y), 4)

                if operador == "<=":
                    if valor_evaluado > c:
                        cumple_todas_las_restricciones = False
                        break
                elif operador == ">=":
                    if valor_evaluado < c:
                        cumple_todas_las_restricciones = False
                        break
                elif operador == "=":
                    if valor_evaluado != c:
                        cumple_todas_las_restricciones = False
                        break

            if cumple_todas_las_restricciones:
                vertices_validos.add((x, y))

        self.vertices = list(vertices_validos)

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

    def _plot_solution(self):
        """Genera la figura de Matplotlib (Método interno)."""
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

    def obtener_pasos_ui(self):
        """
        Empaqueta la solución del método gráfico en un formato estandarizado
        (texto, figura, titulo) reemplazando la antigua lógica suelta.
        """
        pasos = []
        figura_grafico = self._plot_solution()
        
        # Paso 1: Vértices
        texto1 = "1. Analizar gráfico y vértices.\n" + "-" * 40 + "\n"
        
        # Formateamos los números de numpy a texto limpio con 2 decimales
        vertices_limpios = []
        for x, y in self.vertices:
            vertices_limpios.append(f"({float(x):.2f}, {float(y):.2f})")
            
        texto1 += f"Vértices encontrados:\n[{', '.join(vertices_limpios)}]\n\n"
        pasos.append({
            "texto": texto1,
            "figura": figura_grafico, # Solo enviamos la figura en el primer paso
            "titulo": f"Método Gráfico ({self.optimization_type.upper()})"
        })
        
        # Paso 2: Evaluación
        texto2 = "2. Calcular valores en Z por cada vértice.\n" + "-" * 40 + "\n"
        for x, y in self.vertices:
            z = self.c1 * x + self.c2 * y
            texto2 += f"Z en ({x}, {y}) = {self.c1}({x}) + {self.c2}({y}) = {z:.2f}\n"
        pasos.append({
            "texto": texto2,
            "figura": None,
            "titulo": "Evaluación de Vértices"
        })
        
        # Paso 3: Conclusión
        z_opt, x_opt, y_opt = self.best_value
        texto3 = "3. Resultado final.\n" + "-" * 40 + "\n"
        texto3 += f"El vértice óptimo para {self.optimization_type.upper()} es ({x_opt}, {y_opt})\n"
        texto3 += f"Con un valor final de Z = {z_opt:.2f}\n"
        pasos.append({
            "texto": texto3,
            "figura": None,
            "titulo": "Solución Óptima"
        })
        
        return pasos