import numpy as np
import matplotlib.pyplot as plt

from model.problem import LinearProblem


def _numero_a_texto(numero):
    numero = round(numero, 4)
    if numero == int(numero):
        return str(int(numero))
    return str(numero)


def _costo_a_texto(parte_numero, parte_de_M):
    parte_numero = round(parte_numero, 4)
    parte_de_M = round(parte_de_M, 4)

    if parte_de_M == 0:
        return _numero_a_texto(parte_numero)

    if parte_de_M == 1:
        termino_M = "M"
    elif parte_de_M == -1:
        termino_M = "-M"
    else:
        termino_M = f"{_numero_a_texto(parte_de_M)}M"

    if parte_numero == 0:
        return termino_M

    if parte_de_M > 0:
        return f"{_numero_a_texto(parte_numero)} + {termino_M}"
    return f"{_numero_a_texto(parte_numero)} - {termino_M.replace('-', '')}"


class SimplexModelGranM:
    def __init__(self, problem: LinearProblem):
        if not isinstance(problem, LinearProblem):
            raise TypeError("SimplexModelGranM requiere una instancia de LinearProblem")

        self.problem = problem
        self.objective_coefficients = problem.objective_coefficients
        self.optimization_type = problem.optimization_type
        restrictions = problem.restrictions

        self.num_variables = len(self.objective_coefficients)
        self.num_restricciones = len(restrictions)

        self.num_holgura = sum(1 for _, op, _ in restrictions if op in ("<=", ">="))
        self.num_artificiales = sum(1 for _, op, _ in restrictions if op in (">="))

        total_columnas = self.num_variables + self.num_holgura + self.num_artificiales + 1

        self.tableau = np.zeros((self.num_restricciones, total_columnas))
        self.variables_basicas = [0] * self.num_restricciones
        self.historial = []
        self.nombres = []

        self.costo_numero = [0.0] * (total_columnas - 1)
        self.costo_M = [0.0] * (total_columnas - 1)

        self._armar_tabla_inicial(restrictions)

    def _armar_tabla_inicial(self, restrictions):
        for i in range(self.num_variables):
            self.nombres.append(f"x{i + 1}")
            self.costo_numero[i] = self.objective_coefficients[i]

        for i in range(self.num_holgura):
            self.nombres.append(f"s{i + 1}")

        for i in range(self.num_artificiales):
            self.nombres.append(f"A{i + 1}")

        self.nombres.append("RHS")

        signo_de_M = 1 if self.optimization_type != "max" else -1
        columna_holgura = self.num_variables
        columna_artificial = self.num_variables + self.num_holgura

        for i in range(self.num_artificiales):
            self.costo_M[columna_artificial + i] = signo_de_M

        for fila, (coeficientes, operador, termino_independiente) in enumerate(restrictions):
            coeficientes = list(coeficientes)

            if termino_independiente < 0:
                coeficientes = [-c for c in coeficientes]
                termino_independiente = -termino_independiente
                if operador == "<=":
                    operador = ">="
                elif operador == ">=":
                    operador = "<="

            for j in range(self.num_variables):
                self.tableau[fila, j] = coeficientes[j]
            self.tableau[fila, -1] = termino_independiente

            if operador == "<=":
                self.tableau[fila, columna_holgura] = 1
                self.variables_basicas[fila] = columna_holgura
                columna_holgura += 1

            elif operador == ">=":
                self.tableau[fila, columna_holgura] = -1
                self.tableau[fila, columna_artificial] = 1
                self.variables_basicas[fila] = columna_artificial
                columna_holgura += 1
                columna_artificial += 1

            else:
                raise ValueError(f"Operador de restricción no soportado: {operador}")

    def _calcular_zj(self, matriz, basicas):
        num_filas, num_columnas = matriz.shape
        zj_numero = [0.0] * num_columnas
        zj_M = [0.0] * num_columnas

        for columna in range(num_columnas):
            suma_numero = 0.0
            suma_M = 0.0
            for fila in range(num_filas):
                variable_basica = basicas[fila]
                suma_numero += self.costo_numero[variable_basica] * matriz[fila, columna]
                suma_M += self.costo_M[variable_basica] * matriz[fila, columna]
            zj_numero[columna] = suma_numero
            zj_M[columna] = suma_M

        return zj_numero, zj_M

    def _calcular_cj_menos_zj(self, zj_numero, zj_M):
        num_columnas = len(zj_numero) - 1
        diferencia_numero = [0.0] * num_columnas
        diferencia_M = [0.0] * num_columnas

        for columna in range(num_columnas):
            diferencia_numero[columna] = self.costo_numero[columna] - zj_numero[columna]
            diferencia_M[columna] = self.costo_M[columna] - zj_M[columna]

        return diferencia_numero, diferencia_M

    def _elegir_columna_pivote(self, diferencia_numero, diferencia_M):
        factor = 1 if self.optimization_type != "max" else -1

        columna_elegida = None
        mejor_parte_M = None
        mejor_parte_numero = None

        for columna in range(len(diferencia_numero)):
            valor_M = factor * diferencia_M[columna]
            valor_numero = factor * diferencia_numero[columna]

            es_mejor = columna_elegida is None
            if not es_mejor:
                if valor_M < mejor_parte_M:
                    es_mejor = True
                elif valor_M == mejor_parte_M and valor_numero < mejor_parte_numero:
                    es_mejor = True

            if es_mejor:
                columna_elegida = columna
                mejor_parte_M = valor_M
                mejor_parte_numero = valor_numero

        if mejor_parte_M > 0 or (mejor_parte_M == 0 and mejor_parte_numero >= 0):
            return None

        return columna_elegida

    def _elegir_fila_pivote(self, columna_pivote):
        fila_elegida = None
        menor_razon = None

        for fila in range(self.num_restricciones):
            valor_columna = self.tableau[fila, columna_pivote]
            if valor_columna > 0:
                razon = self.tableau[fila, -1] / valor_columna
                if menor_razon is None or razon < menor_razon:
                    menor_razon = razon
                    fila_elegida = fila

        if fila_elegida is None:
            raise ValueError("El problema no está acotado (solución infinita).")

        return fila_elegida

    def _pivotear(self, fila_pivote, columna_pivote):
        self.variables_basicas[fila_pivote] = columna_pivote

        valor_pivote = self.tableau[fila_pivote, columna_pivote]
        for columna in range(self.tableau.shape[1]):
            self.tableau[fila_pivote, columna] /= valor_pivote

        for fila in range(self.tableau.shape[0]):
            if fila != fila_pivote:
                factor = self.tableau[fila, columna_pivote]
                for columna in range(self.tableau.shape[1]):
                    self.tableau[fila, columna] -= factor * self.tableau[fila_pivote, columna]

    def _guardar_estado(self, pivote=None):
        self.historial.append({
            "matriz": self.tableau.copy(),
            "pivote": pivote,
            "basicas": list(self.variables_basicas),
            "nombres": self.nombres,
        })

    def resolver(self):
        while True:
            zj_numero, zj_M = self._calcular_zj(self.tableau, self.variables_basicas)
            diferencia_numero, diferencia_M = self._calcular_cj_menos_zj(zj_numero, zj_M)

            columna_pivote = self._elegir_columna_pivote(diferencia_numero, diferencia_M)
            if columna_pivote is None:
                break

            fila_pivote = self._elegir_fila_pivote(columna_pivote)
            self._guardar_estado(pivote=(fila_pivote, columna_pivote))
            self._pivotear(fila_pivote, columna_pivote)

        self._guardar_estado(pivote=None)

        primera_columna_artificial = self.num_variables + self.num_holgura
        for fila in range(self.num_restricciones):
            variable_basica = self.variables_basicas[fila]
            if variable_basica >= primera_columna_artificial and self.tableau[fila, -1] > 0:
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

            figura = graficar_tableau(matriz, pivote, basicas, nombres, self.costo_numero, self.costo_M)

            pasos_estandarizados.append({
                "texto": texto,
                "figura": figura,
                "titulo": "Método Gran M",
            })

        return pasos_estandarizados


def graficar_tableau(matriz, pivote, basicas, nombres, costo_numero, costo_M):
    num_filas, num_columnas = matriz.shape

    zj_numero = [0.0] * num_columnas
    zj_M = [0.0] * num_columnas
    for columna in range(num_columnas):
        suma_numero = 0.0
        suma_M = 0.0
        for fila in range(num_filas):
            variable_basica = basicas[fila]
            suma_numero += costo_numero[variable_basica] * matriz[fila, columna]
            suma_M += costo_M[variable_basica] * matriz[fila, columna]
        zj_numero[columna] = suma_numero
        zj_M[columna] = suma_M

    num_columnas_variables = num_columnas - 1
    diferencia_numero = [costo_numero[j] - zj_numero[j] for j in range(num_columnas_variables)]
    diferencia_M = [costo_M[j] - zj_M[j] for j in range(num_columnas_variables)]

    fila_cj = ["", "Cj"]
    for j in range(num_columnas_variables):
        fila_cj.append(_costo_a_texto(costo_numero[j], costo_M[j]))
    fila_cj.append("")

    filas_vb = []
    for fila in range(num_filas):
        variable_basica = basicas[fila]
        etiqueta_cb = _costo_a_texto(costo_numero[variable_basica], costo_M[variable_basica])
        fila_tabla = [etiqueta_cb, nombres[variable_basica]]
        for columna in range(num_columnas - 1):
            fila_tabla.append(f"{matriz[fila, columna]:.2f}")
        fila_tabla.append(f"{matriz[fila, -1]:.2f}")
        filas_vb.append(fila_tabla)

    fila_zj = ["", "Zj"]
    for j in range(num_columnas_variables):
        fila_zj.append(_costo_a_texto(zj_numero[j], zj_M[j]))
    fila_zj.append(_costo_a_texto(zj_numero[-1], zj_M[-1]))

    fila_cz = ["", "Cj - Zj"]
    for j in range(num_columnas_variables):
        fila_cz.append(_costo_a_texto(diferencia_numero[j], diferencia_M[j]))
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
        for fila in range(len(filas_vb)):
            fila_tabla_indice = fila + 2
            tabla[(fila_tabla_indice, 1)].set_facecolor("#a1f0a3")
            tabla[(fila_tabla_indice, len(encabezados) - 1)].set_facecolor("#a1f0a3")

    ax.set_title(
        f"Método Gran M — {'Solución óptima' if not pivote else 'Elemento pivote resaltado'}",
        fontsize=12, fontweight="bold", color="#1a3c6e",
    )
    fig.tight_layout()
    return fig