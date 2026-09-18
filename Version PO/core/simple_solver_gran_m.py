import numpy as np
import matplotlib.pyplot as plt
from model.problem import LinearProblem

def _formatear_par(constante, coeficiente_m):
    if abs(constante - round(constante)) < 1e-9:
        constante = int(round(constante))
    if abs(coeficiente_m - round(coeficiente_m)) < 1e-9:
        coeficiente_m = int(round(coeficiente_m))

    if coeficiente_m == 0:
        return f"{constante:g}" if isinstance(constante, float) else str(constante)

    termino_m = "M" if abs(coeficiente_m) == 1 else f"{abs(coeficiente_m):g}M"
    signo_m = "-" if coeficiente_m < 0 else "+"

    if constante == 0:
        return f"-{termino_m}" if coeficiente_m < 0 else termino_m

    return f"{constante:g} {signo_m} {termino_m}"

class SimplexModelGranM:
    def __init__(self, problem: LinearProblem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("Requiere una instancia de LinearProblem")

        self.problem = problem
        self.objective_coefficients = problem.objective_coefficients
        self.optimization_type = problem.optimization_type
        restrictions = problem.restrictions

        self.num_variables = len(self.objective_coefficients)
        self.num_restricciones = len(restrictions)

        self.num_s = sum(1 for _, op, _ in restrictions if op in ("<=", ">="))
        self.num_a = sum(1 for _, op, _ in restrictions if op in (">=", "="))

        total_columnas = self.num_variables + self.num_s + self.num_a + 1

        self.tableau = np.zeros((self.num_restricciones, total_columnas))
        self.variables_basicas = [0] * self.num_restricciones
        self.historial = []
        self.nombres = []

        self.costos_const = [0.0] * (total_columnas - 1)
        self.costos_mcoef = [0.0] * (total_columnas - 1)

        self._construir_modelo(restrictions)

    def _construir_modelo(self, restrictions):
        for i in range(self.num_variables):
            self.nombres.append(f"x{i + 1}")
            self.costos_const[i] = self.objective_coefficients[i]

        for i in range(self.num_s):
            self.nombres.append(f"s{i + 1}")

        for i in range(self.num_a):
            self.nombres.append(f"A{i + 1}")

        self.nombres.append("RHS")

        indice_s = self.num_variables
        indice_a = self.num_variables + self.num_s
        signo_m = 1 if self.optimization_type != "max" else -1

        for i in range(self.num_a):
            self.costos_mcoef[indice_a + i] = signo_m

        for fila, (coeficientes, operador, termino) in enumerate(restrictions):
            coeficientes = list(coeficientes)
            if termino < 0:
                coeficientes = [-c for c in coeficientes]
                termino = -termino
                if operador == "<=":
                    operador = ">="
                elif operador == ">=":
                    operador = "<="

            for j in range(self.num_variables):
                self.tableau[fila, j] = coeficientes[j]
            self.tableau[fila, -1] = termino

            if operador == "<=":
                self.tableau[fila, indice_s] = 1
                self.variables_basicas[fila] = indice_s
                indice_s += 1
            elif operador == ">=":
                self.tableau[fila, indice_s] = -1
                self.tableau[fila, indice_a] = 1
                self.variables_basicas[fila] = indice_a
                indice_s += 1
                indice_a += 1
            elif operador == "=":
                self.tableau[fila, indice_a] = 1
                self.variables_basicas[fila] = indice_a
                indice_a += 1

    def _calcular_zj_y_diferencia(self):
        filas = self.tableau.shape[0]
        columnas = self.tableau.shape[1]
        
        zj_const = [0.0] * columnas
        zj_mcoef = [0.0] * columnas

        for j in range(columnas):
            suma_const = 0.0
            suma_mcoef = 0.0
            for i in range(filas):
                columna_basica = self.variables_basicas[i]
                suma_const += self.costos_const[columna_basica] * self.tableau[i, j]
                suma_mcoef += self.costos_mcoef[columna_basica] * self.tableau[i, j]
            zj_const[j] = suma_const
            zj_mcoef[j] = suma_mcoef

        diferencia_const = [0.0] * (columnas - 1)
        diferencia_mcoef = [0.0] * (columnas - 1)

        for j in range(columnas - 1):
            diferencia_const[j] = self.costos_const[j] - zj_const[j]
            diferencia_mcoef[j] = self.costos_mcoef[j] - zj_mcoef[j]

        return zj_const, zj_mcoef, diferencia_const, diferencia_mcoef

    def _encontrar_columna_pivote(self, diferencia_const, diferencia_mcoef):
        factor = 1 if self.optimization_type != "max" else -1
        mejor_columna = None
        mejor_valor_m = None
        mejor_valor_const = None

        for j in range(len(diferencia_const)):
            valor_m = factor * diferencia_mcoef[j]
            valor_const = factor * diferencia_const[j]
            
            es_menor = False
            if mejor_columna is None:
                es_menor = True
            else:
                if valor_m < mejor_valor_m:
                    es_menor = True
                elif valor_m == mejor_valor_m and valor_const < mejor_valor_const:
                    es_menor = True

            if es_menor:
                mejor_valor_m = valor_m
                mejor_valor_const = valor_const
                mejor_columna = j

        if mejor_valor_m > 1e-9 or (abs(mejor_valor_m) <= 1e-9 and mejor_valor_const >= -1e-9):
            return None

        return mejor_columna

    def _encontrar_fila_pivote(self, columna_pivote):
        mejor_fila = -1
        menor_razon = float('inf')

        for i in range(self.num_restricciones):
            valor_columna = self.tableau[i, columna_pivote]
            if valor_columna > 1e-9:
                razon = self.tableau[i, -1] / valor_columna
                if razon < menor_razon:
                    menor_razon = razon
                    mejor_fila = i

        if mejor_fila == -1:
            raise ValueError("El problema no está acotado (solución infinita).")

        return mejor_fila

    def _ejecutar_pivote(self, fila_pivote, columna_pivote):
        self.variables_basicas[fila_pivote] = columna_pivote
        pivote = self.tableau[fila_pivote, columna_pivote]
        
        for j in range(self.tableau.shape[1]):
            self.tableau[fila_pivote, j] /= pivote
            
        for i in range(self.tableau.shape[0]):
            if i != fila_pivote:
                factor = self.tableau[i, columna_pivote]
                for j in range(self.tableau.shape[1]):
                    self.tableau[i, j] -= factor * self.tableau[fila_pivote, j]

    def _guardar_estado(self, pivote=None):
        self.historial.append({
            "matriz": self.tableau.copy(),
            "pivote": pivote,
            "basicas": list(self.variables_basicas),
            "nombres": self.nombres,
        })

    def resolver(self):
        while True:
            _, _, diferencia_const, diferencia_mcoef = self._calcular_zj_y_diferencia()
            columna_pivote = self._encontrar_columna_pivote(diferencia_const, diferencia_mcoef)

            if columna_pivote is None:
                break

            fila_pivote = self._encontrar_fila_pivote(columna_pivote)
            self._guardar_estado(pivote=(fila_pivote, columna_pivote))
            self._ejecutar_pivote(fila_pivote, columna_pivote)

        self._guardar_estado(pivote=None)

        for i in range(self.num_restricciones):
            columna_basica = self.variables_basicas[i]
            if columna_basica >= self.num_variables + self.num_s and self.tableau[i, -1] > 1e-7:
                raise ValueError("El problema no tiene región factible.")

        return self.historial

    def obtener_pasos_ui(self):
        pasos_estandarizados = []
        for indice, paso in enumerate(self.historial):
            matriz, pivote = paso["matriz"], paso["pivote"]
            basicas, nombres = paso["basicas"], paso["nombres"]

            texto = f"Iteración {indice} — Método de la Gran M\n" + "-" * 40 + "\n"
            if not pivote:
                texto += "Solución óptima alcanzada.\n"
            else:
                fila, columna = pivote
                texto += f"Entra: {nombres[columna]} (columna {columna})\n"
                texto += f"Sale: {nombres[basicas[fila]]} (Fila {fila + 1})\n"
                texto += f"Elemento Pivote = {matriz[fila, columna]:.4f}\n"

            figura = graficar_tableau(matriz, pivote, basicas, nombres, self.costos_const, self.costos_mcoef)
            pasos_estandarizados.append({
                "texto": texto,
                "figura": figura,
                "titulo": "Método Gran M",
            })
        return pasos_estandarizados

def graficar_tableau(matriz, pivote, basicas, nombres, costos_const, costos_mcoef):
    filas = matriz.shape[0]
    columnas = matriz.shape[1]
    
    zj_const = [0.0] * columnas
    zj_mcoef = [0.0] * columnas
    for j in range(columnas):
        suma_const = 0.0
        suma_mcoef = 0.0
        for i in range(filas):
            columna_basica = basicas[i]
            suma_const += costos_const[columna_basica] * matriz[i, j]
            suma_mcoef += costos_mcoef[columna_basica] * matriz[i, j]
        zj_const[j] = suma_const
        zj_mcoef[j] = suma_mcoef

    diferencia_const = [0.0] * (columnas - 1)
    diferencia_mcoef = [0.0] * (columnas - 1)
    for j in range(columnas - 1):
        diferencia_const[j] = costos_const[j] - zj_const[j]
        diferencia_mcoef[j] = costos_mcoef[j] - zj_mcoef[j]

    num_columnas_variables = len(nombres) - 1

    fila_cj = ["", "Cj"] + [_formatear_par(costos_const[j], costos_mcoef[j]) for j in range(num_columnas_variables)] + [""]
    
    filas_vb = []
    for i in range(filas):
        columna_basica = basicas[i]
        etiqueta_cb = _formatear_par(costos_const[columna_basica], costos_mcoef[columna_basica])
        fila = [etiqueta_cb, nombres[columna_basica]]
        for j in range(columnas - 1):
            fila.append(f"{matriz[i, j]:.2f}")
        fila.append(f"{matriz[i, -1]:.2f}")
        filas_vb.append(fila)

    fila_zj = ["", "Zj"] + [_formatear_par(zj_const[j], zj_mcoef[j]) for j in range(num_columnas_variables)]
    fila_zj.append(_formatear_par(zj_const[-1], zj_mcoef[-1]))

    fila_cz = ["", "Cj - Zj"] + [_formatear_par(diferencia_const[j], diferencia_mcoef[j]) for j in range(num_columnas_variables)]
    fila_cz.append("")

    texto_celdas = [fila_cj] + filas_vb + [fila_zj, fila_cz]
    encabezados = ["Cb", "VB"] + nombres

    fig, ax = plt.subplots(figsize=(min(1.3 * len(encabezados), 11), 0.6 * len(texto_celdas) + 2))
    ax.axis("off")

    tabla = ax.table(cellText=texto_celdas, colLabels=encabezados, cellLoc="center", loc="center")
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    if pivote:
        fila_pivote, columna_pivote = pivote
        celda = tabla[(fila_pivote + 2, columna_pivote + 2)]
        celda.set_facecolor("#ffd54f")
        celda.set_edgecolor("black")
        celda.set_linewidth(2)
    else:
        for i in range(2, len(filas_vb) + 2):
            tabla[(i, 1)].set_facecolor("#a1f0a3")
            tabla[(i, len(encabezados) - 1)].set_facecolor("#a1f0a3")
            
        fila_zj_idx = len(filas_vb) + 2
        tabla[(fila_zj_idx, 1)].set_facecolor("#a1f0a3")
        tabla[(fila_zj_idx, len(encabezados) - 1)].set_facecolor("#a1f0a3")

    ax.set_title(
        f"Método Gran M — {'Solución óptima' if not pivote else 'Elemento pivote resaltado'}",
        fontsize=12, fontweight="bold", color="#1a3c6e",
    )
    fig.tight_layout()
    return fig