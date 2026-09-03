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