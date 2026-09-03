#simplex_solver.py

import numpy as np
import matplotlib.pyplot as plt

from problem import LinearProblem

class SimplexModel:
    """
    Encapsula el estado y las operaciones del método Simplex de Dos Fases.
    Recibe una instancia de LinearProblem (clase entidad).
    """
    def __init__(self, problem: LinearProblem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("SimplexModel requiere una instancia de LinearProblem")

        self.problem = problem
        restrictions = problem.restrictions
        objective_coefficients = problem.objective_coefficients

        self.num_restricciones = len(restrictions)
        self.num_variables = len(objective_coefficients)
        self.optimization_type = problem.optimization_type
        self.objective_coefficients = objective_coefficients

        self.num_artificiales = sum(1 for _, op, _ in restrictions if op == ">=")
        self.indice_inicio_artificiales = self.num_variables + self.num_restricciones
        total_columnas = self.indice_inicio_artificiales + self.num_artificiales + 1

        self.tableau = np.zeros((self.num_restricciones + 1, total_columnas))
        self.variables_basicas = [0] * self.num_restricciones
        self.historial = []
        
        self._construir_matriz_y_nombres(restrictions)

    def _construir_matriz_y_nombres(self, restrictions):
        """
        Asigna los coeficientes iniciales, holguras, excesos y variables artificiales 
        a la tabla Simplex.
        """
        self.nombres = []
        for i in range(self.num_variables):
            self.nombres.append(f"x{i+1}")
            
        nombres_holgura = []
        nombres_artificiales = []
        columna_artificial_actual = self.indice_inicio_artificiales

        for indice_restriccion, (coeficientes, operador, termino_independiente) in enumerate(restrictions):
            
            # Si el término independiente es negativo, se multiplica la restricción por -1
            if termino_independiente < 0:
                coeficientes = [-c for c in coeficientes]
                termino_independiente = -termino_independiente
                
                if operador == "<=":
                    operador = ">="
                elif operador == ">=":
                    operador = "<="

            fila_matriz = indice_restriccion + 1
            
            # Asignar los pesos de las variables (aij) y la constante (bi)
            self.tableau[fila_matriz, :len(coeficientes)] = coeficientes
            self.tableau[fila_matriz, -1] = termino_independiente

            columna_holgura_exceso = self.num_variables + indice_restriccion
            
            if operador == "<=":
                # Restricción <= : Se suma una variable de holgura (s)
                self.tableau[fila_matriz, columna_holgura_exceso] = 1
                self.variables_basicas[indice_restriccion] = columna_holgura_exceso
                
                nombres_holgura.append(f"s{indice_restriccion + 1}")
                
            elif operador == ">=":
                # Restricción >= : Se resta exceso (e) y se suma artificial (a)
                self.tableau[fila_matriz, columna_holgura_exceso] = -1 
                self.tableau[fila_matriz, columna_artificial_actual] = 1 
                self.variables_basicas[indice_restriccion] = columna_artificial_actual
                
                columna_artificial_actual += 1
                
                nombres_holgura.append(f"e{indice_restriccion + 1}")
                nombres_artificiales.append(f"a{indice_restriccion + 1}")

        self.nombres.extend(nombres_holgura)
        self.nombres.extend(nombres_artificiales)
        self.nombres.append("RHS")
    
    def _encontrar_columna_pivote(self, fase):
        """Busca el coeficiente más negativo en la función objetivo."""
        fila_z = self.tableau[0, :-1].copy()
        
        # En fase 2 bloqueamos matemáticamente que las variables artificiales regresen
        if fase == 2 and self.num_artificiales > 0:
            fila_z[self.indice_inicio_artificiales:] = np.inf

        if np.all(fila_z >= 0):
            return None
        return int(np.argmin(fila_z))

    def _encontrar_fila_pivote(self, columna_pivote):
        """Aplica la prueba de la razón mínima para determinar la variable que sale."""
        coeficiente_restriccion = self.tableau[1:, -1]
        valores_columna = self.tableau[1:, columna_pivote]

        razones = []
        for i in range(self.num_restricciones):
            if valores_columna[i] > 0:
                razones.append(coeficiente_restriccion[i] / valores_columna[i])
            else:
                razones.append(np.inf)

        if min(razones) == np.inf:
            raise ValueError("El problema no está acotado (solución infinita).")

        return int(np.argmin(razones)) + 1

    def _ejecutar_pivote(self, fila_pivote, columna_pivote):
        self.variables_basicas[fila_pivote - 1] = columna_pivote
        self.tableau[fila_pivote, :] /= self.tableau[fila_pivote, columna_pivote]
        for i in range(self.tableau.shape[0]):
            if i != fila_pivote:
                self.tableau[i, :] -= self.tableau[i, columna_pivote] * self.tableau[fila_pivote, :]

    def _guardar_estado(self, fase, pivote=None):
        """Guarda la instantánea actual para la interfaz gráfica, corrigiendo el signo en MIN."""
        matriz_historial = np.round(self.tableau.copy(), 4)
        if self.optimization_type == "min" and fase == 2:
            matriz_historial[0, -1] *= -1

        self.historial.append({
            "matriz": matriz_historial,
            "pivote": pivote,
            "fase": fase,
            "basicas": list(self.variables_basicas),
            "nombres": self.nombres
        })

    def _fase1(self):
        for i, col_basica in enumerate(self.variables_basicas):
            if col_basica >= self.indice_inicio_artificiales:
                self.tableau[0, col_basica] = 1
                self.tableau[0, :] -= self.tableau[i + 1, :]

        while True:
            col = self._encontrar_columna_pivote(fase=1)
            if col is None:
                break

            fila = self._encontrar_fila_pivote(col)
            self._guardar_estado(fase=1, pivote=(fila, col))
            self._ejecutar_pivote(fila, col)

        if not np.isclose(self.tableau[0, -1], 0, atol=1e-7):
            raise ValueError("El problema no tiene región factible.")

    def _fase2(self):
        """Optimiza la función objetivo original."""
        self.tableau[0, :] = 0
        if self.optimization_type == "max":
            self.tableau[0, :self.num_variables] = -np.array(self.objective_coefficients)
        else:
            self.tableau[0, :self.num_variables] = np.array(self.objective_coefficients)

        for i in range(self.num_restricciones):
            col = self.variables_basicas[i]
            coeficiente_z = self.tableau[0, col]
            if coeficiente_z != 0:
                self.tableau[0, :] -= coeficiente_z * self.tableau[i + 1, :]

        while True:
            col = self._encontrar_columna_pivote(fase=2)
            if col is None:
                break

            fila = self._encontrar_fila_pivote(col)
            self._guardar_estado(fase=2, pivote=(fila, col))
            self._ejecutar_pivote(fila, col)

    def resolver(self):
        """Orquesta las fases del Simplex y retorna el historial completo."""
        if self.num_artificiales > 0:
            self._fase1()

        self._fase2()
        self._guardar_estado(fase=2, pivote=None)
        return self.historial

    def obtener_pasos_ui(self):
        """
        Empaqueta el historial del Simplex generando los textos y las figuras internamente.
        Reemplaza la antigua función suelta 'generateSimplexSteps'.
        """
        pasos_estandarizados = []
        
        for indice, paso in enumerate(self.historial):
            matriz, pivote, fase = paso["matriz"], paso["pivote"], paso["fase"]
            basicas, nombres = paso["basicas"], paso["nombres"]
            
            # --- Generación dinámica del texto explicativo ---
            texto = f"Iteración {indice} — Fase {fase}\n" + "-" * 40 + "\n"
            
            if fase == 2 and indice > 0 and self.historial[indice-1]["fase"] == 1:
                texto += "¡Transición a Fase 2!\nLas variables artificiales se excluyen del análisis.\n\n"
                
            if not pivote:
                texto += "Solución óptima alcanzada.\n"
            else:
                fila, col = pivote
                texto += f"Entra: {nombres[col]} (columna {col})\n"
                texto += f"Sale: {nombres[basicas[fila - 1]]} (Fila {fila})\n"
                texto += f"Elemento Pivote = {matriz[fila, col]:.4f}\n"
            
            # --- Generación de la tabla gráfica ---
            figura = graficar_tableau(matriz, pivote, fase, basicas, nombres)
            
            pasos_estandarizados.append({
                "texto": texto,
                "figura": figura,
                "titulo": f"Método Simplex - Fase {fase}"
            })
            
        return pasos_estandarizados



# Función auxiliar de diseño gráfico 

def graficar_tableau(matriz, pivote, fase, basicas, nombres):
    indices_mantener = [i for i, nom in enumerate(nombres) if not (fase == 2 and nom.startswith('a'))]
    etiquetas_columnas = [nombres[i] for i in indices_mantener]
    
    matriz_filtrada = matriz[:, indices_mantener]
    etiquetas_filas = ["Z"] + [nombres[b] for b in basicas]
    texto_celdas = [[f"{v:.2f}" for v in fila] for fila in matriz_filtrada]

    fig, ax = plt.subplots(figsize=(min(1.3 * len(etiquetas_columnas), 10), 0.6 * matriz_filtrada.shape[0] + 1.5))
    ax.axis("off")

    tabla = ax.table(cellText=texto_celdas, colLabels=etiquetas_columnas, rowLabels=etiquetas_filas, cellLoc="center", loc="center")
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    if pivote:
        fila_pivote, columna_pivote = pivote
        if columna_pivote in indices_mantener:
            celda = tabla[(fila_pivote + 1, indices_mantener.index(columna_pivote))]
            celda.set_facecolor("#ffd54f")
            celda.set_edgecolor("black")
            celda.set_linewidth(2)
    else:
        indice_rhs = len(etiquetas_columnas) - 1
        for i, etiqueta in enumerate(etiquetas_filas):
            if etiqueta.startswith("x") or etiqueta == "Z":
                tabla[(i + 1, -1)].set_facecolor("#a1f0a3")
                tabla[(i + 1, indice_rhs)].set_facecolor("#a1f0a3")

    ax.set_title(f"Fase {fase} — {'Solución óptima' if not pivote else 'Elemento pivote resaltado'}", fontsize=12, fontweight="bold", color="#1a3c6e")
    fig.tight_layout()
    return fig


#graphic_solver.py
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
        # Rectas a intersectar: las de las restricciones + los ejes x=0, y=0
        rectas = [(a, b, c) for a, b, _operador, c in self.restrictions]
        rectas.append((1, 0, 0))  # eje: x = 0
        rectas.append((0, 1, 0))  # eje: y = 0

        vertices_validos = set()

        # 2. Intersectar cada par de rectas para obtener posibles vértices.
        for (a1, b1, c1), (a2, b2, c2) in combinations(rectas, 2):

            # Resolver el sistema 2x2: a1*x + b1*y = c1 ; a2*x + b2*y = c2
            try:
                x, y = np.linalg.solve([[a1, b1], [a2, b2]], [c1, c2])
            except np.linalg.LinAlgError:
                continue  

            x, y = round(x, 4), round(y, 4)

            # Descartar puntos fuera del primer cuadrante.
            if x < 0 or y < 0:
                continue

            es_vertice_valido = True

            for a, b, operador, c in self.restrictions:
                valor_evaluado = round((a * x) + (b * y), 4)

                incumple = (
                    (operador == "<=" and valor_evaluado > c) or
                    (operador == ">=" and valor_evaluado < c) or
                    (operador == "=" and valor_evaluado != c)
                )

                if incumple:
                    es_vertice_valido = False
                    break

            if es_vertice_valido:
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

#main.py

"""
Interfaz gráfica para el solucionador de Programación Lineal.
Carga por defecto el problema:
    Max Z = 2x₁ + x₂
    s.a.
        x₁ ≥ 3          ->  1x₁ + 0x₂ ≥ 3
        x₂ - 2x₁ ≥ 0    -> -2x₁ + 1x₂ ≥ 0
        40x₁ + 30x₂ ≤ 600
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --- Importación de las Clases ---
# Ya no necesitamos importar funciones de dibujo del simplex, 
# las clases se encargan de todo internamente.
from problem import LinearProblem
from graphic_solver import GraphicSolver
from simplex_solver import SimplexModel

# Constantes de Estilo
COLOR_FONDO, COLOR_TARJETA, COLOR_PRIMARIO = "#f4f6f8", "#ffffff", "#1a3c6e"
COLOR_ACENTO, COLOR_AVISO = "#2f80ed", "#b3541e"
FUENTES = {
    "base": ("Segoe UI", 10),
    "titulo": ("Segoe UI", 17, "bold"),
    "subtitulo": ("Segoe UI", 10),
    "seccion": ("Segoe UI", 11, "bold")
}
SUBINDICES = ["₁", "₂", "₃", "₄", "₅", "₆"]
MAX_VARIABLES = 6


class AplicacionProgramacionLineal:
    def __init__(self, root):
        self.root = root
        self.root.title("Solucionador de Programación Lineal")
        self.root.geometry("1200x800")
        self.root.minsize(1050, 700)
        self.root.configure(bg=COLOR_FONDO)

        # Variables de estado
        self.num_variables = tk.IntVar(value=2)
        self.metodo_solucion = tk.StringVar(value="grafico")
        self.tipo_optimizacion = tk.StringVar(value="max")
        
        self.entradas_objetivo = []
        self.filas_restricciones = []
        
        # Nueva variable estandarizada para los pasos de cualquier solver
        self.pasos_ui = []
        self.paso_actual = 0
        self.problema_actual = None

        self._configurar_estilos()
        self._construir_interfaz()
        self._regenerar_formulario(cargar_defecto=True)

    def _configurar_estilos(self):
        estilo = ttk.Style()
        estilo.theme_use("clam")
        
        configs = {
            "TFrame": {"background": COLOR_FONDO},
            "Tarjeta.TFrame": {"background": COLOR_TARJETA},
            "TLabel": {"background": COLOR_FONDO, "font": FUENTES["base"]},
            "Tarjeta.TLabel": {"background": COLOR_TARJETA, "font": FUENTES["base"]},
            "Titulo.TLabel": {"font": FUENTES["titulo"], "foreground": COLOR_PRIMARIO},
            "Subtitulo.TLabel": {"font": FUENTES["subtitulo"], "foreground": "#5a6472"},
            "Seccion.TLabel": {"background": COLOR_TARJETA, "font": FUENTES["seccion"], "foreground": COLOR_PRIMARIO},
            "Aviso.TLabel": {"background": COLOR_TARJETA, "font": ("Segoe UI", 9, "italic"), "foreground": COLOR_AVISO},
            "TButton": {"font": FUENTES["base"], "padding": 6},
            "Resolver.TButton": {"font": ("Segoe UI", 12, "bold"), "padding": 12},
            "Siguiente.TButton": {"font": ("Segoe UI", 10, "bold"), "padding": 8},
            "TRadiobutton": {"background": COLOR_TARJETA, "font": FUENTES["base"]},
        }
        for nombre, configuracion in configs.items():
            estilo.configure(nombre, **configuracion)

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self.root, style="TFrame")
        contenedor.pack(fill="both", expand=True, padx=20, pady=20)
        contenedor.columnconfigure(0, weight=4)
        contenedor.columnconfigure(1, weight=6)
        contenedor.rowconfigure(0, weight=1)

        self._construir_panel_izquierdo(contenedor)
        self._construir_panel_derecho(contenedor)

    def _construir_panel_izquierdo(self, padre):
        panel = ttk.Frame(padre, style="TFrame")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(panel, text="Solucionador de Programación Lineal", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(panel, text="Configura el modelo y visualiza la solución paso a paso.", style="Subtitulo.TLabel").pack(anchor="w", pady=(5, 15))

        scroll_container = ttk.Frame(panel, style="TFrame")
        scroll_container.pack(fill="both", expand=True)
        self.frame_formulario = self._crear_scroll(scroll_container)

        self._crear_seccion_variables()
        self._crear_seccion_objetivo()
        self._crear_seccion_restricciones()
        self._crear_seccion_metodo()

        ttk.Button(panel, text="Resolver Modelo", style="Resolver.TButton", command=self._resolver).pack(fill="x", pady=(15, 0))

    def _crear_scroll(self, padre):
        canvas = tk.Canvas(padre, bg=COLOR_FONDO, highlightthickness=0)
        scroll = ttk.Scrollbar(padre, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas, style="TFrame")
        
        win_id = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(yscrollcommand=scroll.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        return frame

    def _crear_tarjeta(self, titulo):
        tarjeta = ttk.Frame(self.frame_formulario, style="Tarjeta.TFrame", padding=16)
        tarjeta.pack(fill="x", pady=(0, 14))
        ttk.Label(tarjeta, text=titulo, style="Seccion.TLabel").pack(anchor="w", pady=(0, 6))
        return tarjeta

    def _crear_seccion_variables(self):
        tarjeta = self._crear_tarjeta("1. Variables de decisión")
        ttk.Label(tarjeta, text="Cantidad de variables:", style="Tarjeta.TLabel").pack(side="left")
        ttk.Spinbox(tarjeta, from_=1, to=MAX_VARIABLES, width=5, textvariable=self.num_variables, 
                    justify="center", command=self._regenerar_formulario).pack(side="left", padx=12)

    def _crear_seccion_objetivo(self):
        self.tarjeta_objetivo = self._crear_tarjeta("2. Función objetivo")

        frame_fila_obj = ttk.Frame(self.tarjeta_objetivo, style="Tarjeta.TFrame")
        frame_fila_obj.pack(fill="x", pady=(0, 10))
        ttk.Label(frame_fila_obj, text="Z =", style="Tarjeta.TLabel").pack(side="left", padx=(0, 6))
        self.frame_obj_vars = self._crear_fila_desplazable(frame_fila_obj)

        frame_tipo = ttk.Frame(self.tarjeta_objetivo, style="Tarjeta.TFrame")
        frame_tipo.pack(fill="x")
        ttk.Label(frame_tipo, text="Optimización:", style="Tarjeta.TLabel").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(frame_tipo, text="Maximizar", value="max", variable=self.tipo_optimizacion).pack(side="left")
        ttk.Radiobutton(frame_tipo, text="Minimizar", value="min", variable=self.tipo_optimizacion).pack(side="left", padx=(10, 0))

    def _crear_seccion_restricciones(self):
        self.tarjeta_restricciones = self._crear_tarjeta("3. Restricciones")
        self.frame_filas_rest = ttk.Frame(self.tarjeta_restricciones, style="Tarjeta.TFrame")
        self.frame_filas_rest.pack(fill="x", pady=(0, 6))
        ttk.Button(self.tarjeta_restricciones, text="+ Agregar restricción", command=self._agregar_fila_restriccion).pack(anchor="w")

    def _crear_seccion_metodo(self):
        tarjeta = self._crear_tarjeta("4. Método de solución")
        self.radio_grafico = ttk.Radiobutton(tarjeta, text="Método Gráfico", value="grafico", variable=self.metodo_solucion)
        self.radio_grafico.pack(side="left")
        self.radio_simplex = ttk.Radiobutton(tarjeta, text="Método Simplex", value="simplex", variable=self.metodo_solucion)
        self.radio_simplex.pack(side="left", padx=(20, 0))
        self.etiqueta_aviso_metodo = ttk.Label(tarjeta, text="", style="Aviso.TLabel")
        self.etiqueta_aviso_metodo.pack(side="left", padx=(15, 0))

    def _crear_entradas_variables(self, contenedor, valores_defecto=None):
        entradas = []
        cantidad = self.num_variables.get()
        
        for i in range(cantidad):
            entrada = ttk.Entry(contenedor, width=6, justify="center")
            
            # Reemplazo de operador ternario por if clásico
            valor = "1"
            if valores_defecto is not None:
                if i < len(valores_defecto):
                    valor = str(valores_defecto[i])
                    
            entrada.insert(0, valor)
            entrada.pack(side="left")
            ttk.Label(contenedor, text=f"x{SUBINDICES[i]}", style="Tarjeta.TLabel").pack(side="left", padx=(3, 4))
            
            if i < cantidad - 1:
                ttk.Label(contenedor, text="+", style="Tarjeta.TLabel").pack(side="left", padx=(0, 4))
                
            entradas.append(entrada)
            
        return entradas

    def _crear_fila_desplazable(self, padre, alto=34):
        contenedor = ttk.Frame(padre, style="Tarjeta.TFrame")
        contenedor.pack(side="left", fill="both", expand=True)

        canvas = tk.Canvas(contenedor, bg=COLOR_TARJETA, highlightthickness=0, height=alto)
        canvas.pack(side="top", fill="both", expand=True)

        frame_interno = ttk.Frame(canvas, style="Tarjeta.TFrame")
        id_ventana = canvas.create_window((0, 0), window=frame_interno, anchor="nw")

        slider = ttk.Scale(contenedor, orient="horizontal", from_=0, to=1,
                            command=lambda v: canvas.xview_moveto(float(v)))

        def _actualizar_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(id_ventana, height=frame_interno.winfo_reqheight())
            ancho_contenido = frame_interno.winfo_reqwidth()
            ancho_visible = canvas.winfo_width()
            if ancho_visible > 1 and ancho_contenido > ancho_visible:
                if not slider.winfo_ismapped():
                    slider.pack(side="top", fill="x", pady=(3, 0))
            else:
                if slider.winfo_ismapped():
                    slider.pack_forget()
                canvas.xview_moveto(0)

        frame_interno.bind("<Configure>", _actualizar_scroll)
        canvas.bind("<Configure>", _actualizar_scroll)
        canvas.bind("<Shift-MouseWheel>", lambda e: canvas.xview_scroll(int(-e.delta / 60), "units"))

        return frame_interno

    def _regenerar_formulario(self, cargar_defecto=False):
        for widget in self.frame_obj_vars.winfo_children(): 
            widget.destroy()
        for fila in self.filas_restricciones: 
            fila["frame"].destroy()
            
        self.filas_restricciones.clear()

        if cargar_defecto and self.num_variables.get() == 2:
            self.tipo_optimizacion.set("max")
            self.entradas_objetivo = self._crear_entradas_variables(self.frame_obj_vars, valores_defecto=[2, 1])
            self._agregar_fila_restriccion(coefs=[1, 0], signo=">=", cte_val=3)
            self._agregar_fila_restriccion(coefs=[-2, 1], signo=">=", cte_val=0)
            self._agregar_fila_restriccion(coefs=[40, 30], signo="<=", cte_val=600)
        else:
            self.entradas_objetivo = self._crear_entradas_variables(self.frame_obj_vars)
            self._agregar_fila_restriccion()
        
        es_multivariable = self.num_variables.get() > 2
        
        # Reemplazo de operadores ternarios en la configuración visual
        estado_radio = "normal"
        metodo_seleccionado = self.metodo_solucion.get()
        texto_aviso = ""
        
        if es_multivariable:
            estado_radio = "disabled"
            metodo_seleccionado = "simplex"
            texto_aviso = "Solo Simplex (>2 variables)."
            
        self.radio_grafico.configure(state=estado_radio)
        self.metodo_solucion.set(metodo_seleccionado)
        self.etiqueta_aviso_metodo.configure(text=texto_aviso)
            
        self._limpiar_resultados()

    def _agregar_fila_restriccion(self, coefs=None, signo="<=", cte_val=0):
        frame = ttk.Frame(self.frame_filas_rest, style="Tarjeta.TFrame")
        frame.pack(fill="x", pady=3)

        combo = ttk.Combobox(frame, values=["<=", ">="], width=4, state="readonly", justify="center")
        cte = ttk.Entry(frame, width=8, justify="center")
        fila = {"frame": frame, "coeficientes": [], "signo": combo, "constante": cte}
        boton_quitar = ttk.Button(frame, text="Quitar", command=lambda: self._quitar_fila_restriccion(fila))

        boton_quitar.pack(side="right", padx=(6, 0))
        cte.pack(side="right", padx=(0, 10))
        combo.pack(side="right", padx=(5, 10))
        combo.set(signo)
        cte.insert(0, str(cte_val))

        frame_vars = self._crear_fila_desplazable(frame)
        fila["coeficientes"] = self._crear_entradas_variables(frame_vars, valores_defecto=coefs)

        self.filas_restricciones.append(fila)

    def _quitar_fila_restriccion(self, fila):
        if len(self.filas_restricciones) > 1:
            fila["frame"].destroy()
            self.filas_restricciones.remove(fila)
        else:
            messagebox.showinfo("Aviso", "Debe existir al menos una restricción.")

    def _construir_panel_derecho(self, padre):
        panel = ttk.Frame(padre, style="TFrame")
        panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=7)
        panel.rowconfigure(1, weight=3)

        self.panel_grafico = ttk.Frame(panel, style="Tarjeta.TFrame")
        self.panel_grafico.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.lbl_grafico_vacio = ttk.Label(self.panel_grafico, text="El gráfico se generará aquí.", style="Subtitulo.TLabel")
        self.lbl_grafico_vacio.place(relx=0.5, rely=0.5, anchor="center")

        panel_exp = ttk.Frame(panel, style="Tarjeta.TFrame", padding=15)
        panel_exp.grid(row=1, column=0, sticky="nsew")

        frame_encabezado_exp = ttk.Frame(panel_exp, style="Tarjeta.TFrame")
        frame_encabezado_exp.pack(fill="x", pady=(0, 5))
        ttk.Label(frame_encabezado_exp, text="Explicación Paso a Paso", style="Seccion.TLabel").pack(side="left")
        self.etiqueta_fase = ttk.Label(frame_encabezado_exp, text="", style="Seccion.TLabel")
        self.etiqueta_fase.pack(side="right")
        
        self.texto_explicacion = tk.Text(panel_exp, height=6, font=("Consolas", 11), bg=COLOR_FONDO, relief="flat", wrap="word", state="disabled")
        self.texto_explicacion.pack(fill="both", expand=True, pady=5)
        
        self.boton_siguiente = ttk.Button(panel_exp, text="Siguiente Paso ➡", style="Siguiente.TButton", command=self._avanzar_paso, state="disabled")
        self.boton_siguiente.pack(fill="x", pady=(10, 0))

    def _leer_datos(self):
        c_obj = [float(e.get()) for e in self.entradas_objetivo]
        rest = [([float(e.get()) for e in f["coeficientes"]], f["signo"].get(), float(f["constante"].get())) for f in self.filas_restricciones]
        return c_obj, rest

    def _resolver(self):
        try:
            c_obj, rest = self._leer_datos()
        except ValueError:
            return messagebox.showerror("Error", "Revisa que todos los campos tengan números válidos.")

        try:
            problema = LinearProblem(c_obj, rest, self.tipo_optimizacion.get())
        except ValueError as e:
            return messagebox.showerror("Error", str(e))

        self._limpiar_resultados()

        try:
            # responden al mismo contrato (resolver -> obtener_pasos_ui)
            if self.metodo_solucion.get() == "grafico":
                modelo = GraphicSolver(problema)
            else:
                modelo = SimplexModel(problema)
                
            modelo.resolver()
            self.pasos_ui = modelo.obtener_pasos_ui()

        except ValueError as e:
            self.lbl_grafico_vacio.place(relx=0.5, rely=0.5, anchor="center")
            return messagebox.showerror("Sin solución", str(e))
        
        self.boton_siguiente.configure(state="normal")
        self._avanzar_paso()

    def _avanzar_paso(self):
        # Protegemos contra desbordamientos
        if self.paso_actual >= len(self.pasos_ui):
            return

        # Obtenemos el diccionario con (texto, figura, titulo)
        paso = self.pasos_ui[self.paso_actual]

        # 1. Título
        self.etiqueta_fase.configure(text=paso["titulo"])

        # 2. Dibujo en Matplotlib (solo si el paso incluye figura nueva)
        if paso["figura"] is not None:
            self._renderizar_canvas(paso["figura"])

        # 3. Texto en consola
        self._mostrar_texto_simple(paso["texto"])

        # Avanzar puntero
        self.paso_actual += 1
        
        # Deshabilitar botón si llegamos al final
        if self.paso_actual >= len(self.pasos_ui):
            self.boton_siguiente.configure(state="disabled")

    def _renderizar_canvas(self, fig):
        self.lbl_grafico_vacio.place_forget()
        for widget in self.panel_grafico.winfo_children():
            if widget != self.lbl_grafico_vacio:
                widget.destroy()

        canvas_grafico = FigureCanvasTkAgg(fig, master=self.panel_grafico)
        canvas_grafico.draw()
        canvas_grafico.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def _limpiar_resultados(self):
        for widget in self.panel_grafico.winfo_children():
            if widget != self.lbl_grafico_vacio: 
                widget.destroy()
        
        self.paso_actual = 0
        self.pasos_ui = []
        self.problema_actual = None
        
        self.etiqueta_fase.configure(text="")
        self.boton_siguiente.configure(state="disabled")
        self._mostrar_texto_simple("")
        self.lbl_grafico_vacio.place(relx=0.5, rely=0.5, anchor="center")

    def _mostrar_texto_simple(self, texto):
        self.texto_explicacion.configure(state="normal")
        self.texto_explicacion.delete("1.0", tk.END)
        
        if texto: 
            self.texto_explicacion.insert(tk.END, texto)
            
        self.texto_explicacion.see(tk.END)
        self.texto_explicacion.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    AplicacionProgramacionLineal(root)
    root.mainloop()
