import numpy as np
import matplotlib.pyplot as plt

from problem import LinearProblem


class SimplexModel:
    def __init__(self, problem: LinearProblem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("SimplexModel requiere una instancia de LinearProblem")

        if problem.optimization_type != "max":
            raise ValueError(
                "El método Simplex común solo admite maximización. "
                "Usa el método de Dos Fases para problemas de Min."
            )

        self.problem = problem
        restrictions = problem.restrictions
        objective_coefficients = problem.objective_coefficients

        self.num_restricciones = len(restrictions)
        self.num_variables = len(objective_coefficients)
        self.objective_coefficients = objective_coefficients

        total_columnas = self.num_variables + self.num_restricciones + 1

        self.tableau = np.zeros((self.num_restricciones + 1, total_columnas))
        self.variables_basicas = [0] * self.num_restricciones
        self.historial = []

        self._construir_matriz_y_nombres(restrictions)
        self._inicializar_fila_z()

    def _construir_matriz_y_nombres(self, restrictions):
        self.nombres = [f"x{i + 1}" for i in range(self.num_variables)]
        nombres_holgura = []

        for indice_restriccion, (coeficientes, operador, termino_independiente) in enumerate(restrictions):
            coeficientes = list(coeficientes)

            if operador == ">=":
                coeficientes = [-c for c in coeficientes]
                termino_independiente = -termino_independiente
                operador = "<="
            elif operador != "<=":
                raise ValueError(f"Operador de restricción no soportado: {operador}")

            fila_matriz = indice_restriccion + 1

            self.tableau[fila_matriz, :len(coeficientes)] = coeficientes
            self.tableau[fila_matriz, -1] = termino_independiente

            columna_holgura = self.num_variables + indice_restriccion
            self.tableau[fila_matriz, columna_holgura] = 1
            self.variables_basicas[indice_restriccion] = columna_holgura

            nombres_holgura.append(f"s{indice_restriccion + 1}")

        self.nombres.extend(nombres_holgura)
        self.nombres.append("RHS")

    def _inicializar_fila_z(self):
        self.tableau[0, :self.num_variables] = -np.array(self.objective_coefficients)

    def _encontrar_columna_pivote(self):
        fila_z = self.tableau[0, :-1]

        if np.all(fila_z >= 0):
            return None

        return int(np.argmin(fila_z))

    def _encontrar_fila_pivote(self, columna_pivote):
        columna_cr = self.tableau[1:, -1]
        columna_pivote_valores = self.tableau[1:, columna_pivote]

        cocientes_validos = []
        for i in range(self.num_restricciones):
            valor_cr = columna_cr[i]
            valor_columna = columna_pivote_valores[i]

            if valor_cr < 0:
                continue

            if valor_columna <= 0:
                continue

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
        self.variables_basicas[fila_pivote - 1] = columna_pivote

        self.tableau[fila_pivote, :] /= self.tableau[fila_pivote, columna_pivote]

        for i in range(self.tableau.shape[0]):
            if i != fila_pivote:
                self.tableau[i, :] -= self.tableau[i, columna_pivote] * self.tableau[fila_pivote, :]

    def _guardar_estado(self, pivote=None):
        matriz_historial = np.round(self.tableau.copy(), 4)

        self.historial.append({
            "matriz": matriz_historial,
            "pivote": pivote,
            "basicas": list(self.variables_basicas),
            "nombres": self.nombres
        })

    def resolver(self):
        while True:
            columna_pivote = self._encontrar_columna_pivote()
            if columna_pivote is None:
                break

            fila_pivote = self._encontrar_fila_pivote(columna_pivote)
            self._guardar_estado(pivote=(fila_pivote, columna_pivote))
            self._ejecutar_pivote(fila_pivote, columna_pivote)

        self._guardar_estado(pivote=None)
        return self.historial

    def obtener_pasos_ui(self):
        pasos_estandarizados = []

        for indice, paso in enumerate(self.historial):
            matriz = paso["matriz"]
            pivote = paso["pivote"]
            basicas = paso["basicas"]
            nombres = paso["nombres"]

            texto = f"Iteración {indice} — Método Simplex Común\n" + "-" * 40 + "\n"

            if not pivote:
                texto += "Solución óptima alcanzada.\n"

            else:
                fila, columna = pivote
                texto += f"Entra: {nombres[columna]} (columna {columna})\n"
                texto += f"Sale: {nombres[basicas[fila - 1]]} (Fila {fila})\n"
                texto += f"Elemento Pivote = {matriz[fila, columna]:.4f}\n"

            figura = graficar_tableau(matriz, pivote, basicas, nombres)

            pasos_estandarizados.append({
                "texto": texto,
                "figura": figura,
                "titulo": "Método Simplex"
            })

        return pasos_estandarizados


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

    ax.set_title(
        f"Método Simplex — {'Solución óptima' if not pivote else 'Elemento pivote resaltado'}",
        fontsize=12, fontweight="bold", color="#1a3c6e"
    )
    fig.tight_layout()
    return fig