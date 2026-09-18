import numpy as np
import matplotlib.pyplot as plt

from problem import LinearProblem


class SimplexModel:
    """
    Encapsula el estado y las operaciones del método Simplex COMÚN (estándar,
    de una sola fase, sin variables artificiales).

    Reglas aplicadas (según el procedimiento pedido):

    Paso 1 - Preparación del modelo:
        - Toda restricción con '>=' se multiplica por -1. Esto la convierte
          en '<=', pero el Lado Derecho (CR) puede volverse negativo.
        - Como todas las restricciones quedan en '<=', se agrega una
          variable de holgura (+s_n) a cada una para armar la tabla inicial.
          NO se usan variables de exceso ni variables artificiales.

    Paso 2 - Variable que entra (columna pivote):
        - Maximización: coeficiente más negativo de la fila Z.
        - Minimización: se sigue el mismo criterio (coeficiente más
          negativo) sobre la fila Z construida con los coeficientes
          originales, igual que en la convención clásica de tablas Simplex.

    Paso 3 - Variable que sale (fila pivote) - Prueba de la razón especial:
        - Si el CR (RHS) de la fila es negativo -> se ignora esa fila.
        - Si el elemento de la Columna Pivote es negativo (o cero) -> se
          ignora esa fila.
        - Si el CR es 0, solo se divide entre valores positivos de la
          Columna Pivote (0 / + = 0).
        - Entre todos los cocientes válidos, se elige el de menor valor
          absoluto. Esa es la Fila Pivote y la intersección es el Elemento
          Pivote.

    Paso 4 - Gauss-Jordan:
        - El elemento pivote se convierte en 1 (dividiendo toda su fila).
        - El resto de la Columna Pivote se convierte en 0 (sumas/restas
          entre filas).

    Paso 5 - Parada:
        - Se repiten los pasos 2 a 4 hasta que la fila Z no tenga
          coeficientes que mejoren la función objetivo (no queden
          negativos).
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

        # Sin variables artificiales: columnas = variables + holguras (una por
        # restricción) + RHS.
        total_columnas = self.num_variables + self.num_restricciones + 1

        self.tableau = np.zeros((self.num_restricciones + 1, total_columnas))
        self.variables_basicas = [0] * self.num_restricciones
        self.historial = []

        self._construir_matriz_y_nombres(restrictions)
        self._inicializar_fila_z()

    def _construir_matriz_y_nombres(self, restrictions):
        """
        Convierte cualquier restricción '>=' multiplicándola por -1 (paso 1)
        y agrega una variable de holgura normal a cada restricción, sin
        importar el signo resultante del CR.
        """
        self.nombres = [f"x{i + 1}" for i in range(self.num_variables)]
        nombres_holgura = []

        for indice_restriccion, (coeficientes, operador, termino_independiente) in enumerate(restrictions):
            coeficientes = list(coeficientes)

            # --- Paso 1: si es '>=' se multiplica toda la inecuación por -1 ---
            if operador == ">=":
                coeficientes = [-c for c in coeficientes]
                termino_independiente = -termino_independiente
                operador = "<="
            elif operador not in ("<=", "="):
                raise ValueError(f"Operador de restricción no soportado: {operador}")

            fila_matriz = indice_restriccion + 1

            # Coeficientes de las variables originales (aij) y CR (bi)
            self.tableau[fila_matriz, :len(coeficientes)] = coeficientes
            self.tableau[fila_matriz, -1] = termino_independiente

            # --- Se agrega SIEMPRE una variable de holgura normal (+s_n) ---
            columna_holgura = self.num_variables + indice_restriccion
            self.tableau[fila_matriz, columna_holgura] = 1
            self.variables_basicas[indice_restriccion] = columna_holgura

            nombres_holgura.append(f"s{indice_restriccion + 1}")

        self.nombres.extend(nombres_holgura)
        self.nombres.append("RHS")

    def _inicializar_fila_z(self):
        """Construye la fila Z inicial a partir de la función objetivo."""
        if self.optimization_type == "max":
            self.tableau[0, :self.num_variables] = -np.array(self.objective_coefficients)
        else:
            self.tableau[0, :self.num_variables] = np.array(self.objective_coefficients)

    def _encontrar_columna_pivote(self):
        """Selecciona el coeficiente más negativo de la fila Z (variable que entra)."""
        fila_z = self.tableau[0, :-1]
        if np.all(fila_z >= 0):
            return None
        return int(np.argmin(fila_z))

    def _encontrar_fila_pivote(self, columna_pivote):
        """
        Prueba de la razón especial del método Simplex común:
          - CR negativo  -> se ignora la fila.
          - Elemento de la columna pivote negativo (o cero) -> se ignora la fila.
          - CR == 0      -> solo se divide entre elementos positivos (0/+ = 0).
          - Entre los cocientes válidos, se elige el de menor valor absoluto.
        """
        cr = self.tableau[1:, -1]
        columna = self.tableau[1:, columna_pivote]

        cocientes_validos = []
        for i in range(self.num_restricciones):
            valor_cr = cr[i]
            valor_columna = columna[i]

            if valor_cr < 0:
                continue  # CR negativo -> se ignora la fila
            if valor_columna <= 0:
                continue  # elemento negativo (o cero) en la columna -> se ignora

            cociente = valor_cr / valor_columna
            cocientes_validos.append((i, cociente))

        if not cocientes_validos:
            raise ValueError(
                "No hay fila pivote válida (problema no acotado, o se requieren "
                "variables artificiales / método de dos fases para esta tabla)."
            )

        indice_fila, _ = min(cocientes_validos, key=lambda par: abs(par[1]))
        return indice_fila + 1

    def _ejecutar_pivote(self, fila_pivote, columna_pivote):
        """Paso 4: Reducción de Gauss-Jordan."""
        self.variables_basicas[fila_pivote - 1] = columna_pivote
        self.tableau[fila_pivote, :] /= self.tableau[fila_pivote, columna_pivote]
        for i in range(self.tableau.shape[0]):
            if i != fila_pivote:
                self.tableau[i, :] -= self.tableau[i, columna_pivote] * self.tableau[fila_pivote, :]

    def _guardar_estado(self, pivote=None):
        """Guarda la instantánea actual para la interfaz gráfica."""
        matriz_historial = np.round(self.tableau.copy(), 4)
        if self.optimization_type == "min":
            matriz_historial[0, -1] *= -1

        self.historial.append({
            "matriz": matriz_historial,
            "pivote": pivote,
            "basicas": list(self.variables_basicas),
            "nombres": self.nombres
        })

    def resolver(self):
        """Orquesta el método Simplex común (una sola fase) y retorna el historial."""
        while True:
            col = self._encontrar_columna_pivote()
            if col is None:
                break

            fila = self._encontrar_fila_pivote(col)
            self._guardar_estado(pivote=(fila, col))
            self._ejecutar_pivote(fila, col)

        self._guardar_estado(pivote=None)
        return self.historial

    def obtener_pasos_ui(self):
        """
        Empaqueta el historial del Simplex generando los textos y las figuras
        internamente, con el mismo contrato (texto, figura, titulo) que el
        resto de los solvers de la aplicación.
        """
        pasos_estandarizados = []

        for indice, paso in enumerate(self.historial):
            matriz, pivote = paso["matriz"], paso["pivote"]
            basicas, nombres = paso["basicas"], paso["nombres"]

            texto = f"Iteración {indice} — Método Simplex Común\n" + "-" * 40 + "\n"

            if not pivote:
                texto += "Solución óptima alcanzada.\n"
                if np.any(matriz[1:, -1] < -1e-7):
                    texto += (
                        "Advertencia: aún hay valores negativos en el CR. "
                        "El método Simplex común no garantiza una solución "
                        "factible cuando hay restricciones '>='; para esos "
                        "casos se recomienda el método de Dos Fases.\n"
                    )
            else:
                fila, col = pivote
                texto += f"Entra: {nombres[col]} (columna {col})\n"
                texto += f"Sale: {nombres[basicas[fila - 1]]} (Fila {fila})\n"
                texto += f"Elemento Pivote = {matriz[fila, col]:.4f}\n"

            figura = graficar_tableau(matriz, pivote, basicas, nombres)

            pasos_estandarizados.append({
                "texto": texto,
                "figura": figura,
                "titulo": "Método Simplex"
            })

        return pasos_estandarizados


# Función auxiliar de diseño gráfico

def graficar_tableau(matriz, pivote, basicas, nombres):
    etiquetas_columnas = nombres
    etiquetas_filas = ["Z"] + [nombres[b] for b in basicas]
    texto_celdas = [[f"{v:.2f}" for v in fila] for fila in matriz]

    fig, ax = plt.subplots(figsize=(min(1.3 * len(etiquetas_columnas), 10), 0.6 * matriz.shape[0] + 1.5))
    ax.axis("off")

    tabla = ax.table(cellText=texto_celdas, colLabels=etiquetas_columnas, rowLabels=etiquetas_filas, cellLoc="center", loc="center")
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    if pivote:
        fila_pivote, columna_pivote = pivote
        celda = tabla[(fila_pivote + 1, columna_pivote)]
        celda.set_facecolor("#ffd54f")
        celda.set_edgecolor("black")
        celda.set_linewidth(2)
    else:
        indice_rhs = len(etiquetas_columnas) - 1
        for i, etiqueta in enumerate(etiquetas_filas):
            if etiqueta.startswith("x") or etiqueta == "Z":
                tabla[(i + 1, indice_rhs)].set_facecolor("#a1f0a3")

    ax.set_title(f"Método Simplex — {'Solución óptima' if not pivote else 'Elemento pivote resaltado'}", fontsize=12, fontweight="bold", color="#1a3c6e")
    fig.tight_layout()
    return fig
