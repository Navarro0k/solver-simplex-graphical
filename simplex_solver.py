import numpy as np
import matplotlib.pyplot as plt


def encontrar_columna_pivote(tableau, indice_inicio_artificiales=None):
    """
    Variable que entra: la de coeficiente más negativo en la fila Z. None si ya es óptimo.
    """
    fila_z = tableau[0, :-1].copy()
    if indice_inicio_artificiales is not None:
        fila_z[indice_inicio_artificiales:] = np.inf

    if np.all(fila_z >= 0):
        return None
    return int(np.argmin(fila_z))


def encontrar_fila_pivote(tableau, columna_pivote, num_restricciones):
    """Variable que sale: prueba del cociente mínimo (regla de razón mínima)."""
    lado_derecho = tableau[1:, -1]
    valores_columna_pivote = tableau[1:, columna_pivote]

    razones = []
    for indice_restriccion in range(num_restricciones):
        if valores_columna_pivote[indice_restriccion] > 0:
            razones.append(lado_derecho[indice_restriccion] / valores_columna_pivote[indice_restriccion])
        else:
            razones.append(np.inf)

    if min(razones) == np.inf:
        raise ValueError("El problema no está acotado (solución infinita).")

    return int(np.argmin(razones)) + 1  # +1 porque la fila 0 es la fila Z


def fase1(tableau, variables_basicas, num_restricciones, indice_inicio_artificiales, historial):
    """
    Minimiza la suma de variables artificiales para encontrar una solución básica factible.
    """
    for indice_restriccion, columna_var_basica in enumerate(variables_basicas):
        if columna_var_basica >= indice_inicio_artificiales:
            tableau[0, columna_var_basica] = 1

    for indice_restriccion, columna_var_basica in enumerate(variables_basicas):
        if columna_var_basica >= indice_inicio_artificiales:
            tableau[0, :] -= tableau[indice_restriccion + 1, :]

    while True:
        columna_pivote = encontrar_columna_pivote(tableau)
        if columna_pivote is None:
            break

        fila_pivote = encontrar_fila_pivote(tableau, columna_pivote, num_restricciones)

        historial.append({
            "matriz": np.round(tableau.copy(), 4),
            "pivote": (fila_pivote, columna_pivote),
            "fase": 1,
            "basicas": list(variables_basicas) # Guardado de variables básicas
        })

        variables_basicas[fila_pivote - 1] = columna_pivote
        tableau[fila_pivote, :] /= tableau[fila_pivote, columna_pivote]
        for indice_fila_tableau in range(tableau.shape[0]):
            if indice_fila_tableau != fila_pivote:
                tableau[indice_fila_tableau, :] -= (
                    tableau[indice_fila_tableau, columna_pivote] * tableau[fila_pivote, :]
                )

    if tableau[0, -1] != 0:
        raise ValueError("El problema no tiene región factible.")


def fase2(tableau, variables_basicas, num_restricciones, objective_coefficients,
          optimization_type, indice_inicio_artificiales, historial):
    """
    Optimiza la función objetivo original, partiendo de la solución factible de Fase 1.
    """
    num_variables_decision = len(objective_coefficients)

    tableau[0, :] = 0
    if optimization_type == "max":
        tableau[0, :num_variables_decision] = -np.array(objective_coefficients)
    else:
        tableau[0, :num_variables_decision] = np.array(objective_coefficients)

    for indice_restriccion in range(num_restricciones):
        columna_var_basica = variables_basicas[indice_restriccion]
        coeficiente_en_z = tableau[0, columna_var_basica]
        if coeficiente_en_z != 0:
            tableau[0, :] -= coeficiente_en_z * tableau[indice_restriccion + 1, :]

    while True:
        columna_pivote = encontrar_columna_pivote(tableau, indice_inicio_artificiales)
        if columna_pivote is None:
            break

        fila_pivote = encontrar_fila_pivote(tableau, columna_pivote, num_restricciones)

        historial.append({
            "matriz": np.round(tableau.copy(), 4),
            "pivote": (fila_pivote, columna_pivote),
            "fase": 2,
            "basicas": list(variables_basicas) # Guardado de variables básicas
        })

        variables_basicas[fila_pivote - 1] = columna_pivote
        tableau[fila_pivote, :] /= tableau[fila_pivote, columna_pivote]
        for indice_fila_tableau in range(tableau.shape[0]):
            if indice_fila_tableau != fila_pivote:
                tableau[indice_fila_tableau, :] -= (
                    tableau[indice_fila_tableau, columna_pivote] * tableau[fila_pivote, :]
                )


def simplex_solver(restrictions, objective_coefficients, optimization_type="max", iteration=0):
    """
    Resuelve un modelo de Programación Lineal usando el método Simplex de Dos Fases.
    """
    num_restricciones = len(restrictions)
    num_variables_decision = len(objective_coefficients)

    num_artificiales = sum(1 for _, operador, _ in restrictions if operador == ">=")
    total_columnas = num_variables_decision + num_restricciones + num_artificiales + 1
    indice_inicio_artificiales = num_variables_decision + num_restricciones

    tableau = np.zeros((num_restricciones + 1, total_columnas))
    variables_basicas = [0] * num_restricciones
    siguiente_columna_artificial = indice_inicio_artificiales

    for indice_restriccion, (coeficientes, operador, lado_derecho) in enumerate(restrictions):
        if lado_derecho < 0:
            coeficientes = [-c for c in coeficientes]
            lado_derecho = -lado_derecho
            operador = ">=" if operador == "<=" else "<="

        fila_tableau = indice_restriccion + 1
        tableau[fila_tableau, :len(coeficientes)] = coeficientes
        tableau[fila_tableau, -1] = lado_derecho

        columna_holgura_o_exceso = num_variables_decision + indice_restriccion
        if operador == "<=":
            tableau[fila_tableau, columna_holgura_o_exceso] = 1
            variables_basicas[indice_restriccion] = columna_holgura_o_exceso
        elif operador == ">=":
            tableau[fila_tableau, columna_holgura_o_exceso] = -1 
            tableau[fila_tableau, siguiente_columna_artificial] = 1 
            variables_basicas[indice_restriccion] = siguiente_columna_artificial
            siguiente_columna_artificial += 1

    historial = []

    if num_artificiales > 0:
        fase1(tableau, variables_basicas, num_restricciones, indice_inicio_artificiales, historial)

    fase2(tableau, variables_basicas, num_restricciones, objective_coefficients,
          optimization_type, indice_inicio_artificiales, historial)

    historial.append({
        "matriz": np.round(tableau.copy(), 4),
        "pivote": None,
        "fase": 2,
        "basicas": list(variables_basicas) # Guardado final
    })

    if optimization_type == "min":
        for paso in historial:
            if paso["fase"] == 2:
                paso["matriz"][0, -1] *= -1

    estado = historial[iteration] if iteration < len(historial) else historial[-1]
    return estado["matriz"], estado["pivote"], estado["fase"], estado["basicas"]


def getColumnNames(restrictions, numDecisionVars):
    """
    Asigna la nomenclatura estándar a las columnas de la matriz.
    """
    names = [f"x{i+1}" for i in range(numDecisionVars)]
    artificials = []
    
    for row, (_, operator, _) in enumerate(restrictions):
        if operator == "<=":
            names.append(f"s{row+1}") 
        elif operator == ">=":
            names.append(f"e{row+1}") 
            artificials.append(f"a{row+1}") 
            
    names.extend(artificials)
    names.append("RHS")
    return names


def graficar_tableau(matriz, pivote, fase, restrictions, objective_coefficients, basicas):
    """
    Genera una figura de Matplotlib con el tableau filtrando variables artificiales en Fase 2
    y nombrando correctamente las filas. Resalta resultados finales si es óptimo.
    """
    num_variables_decision = len(objective_coefficients)
    nombres_todas_columnas = getColumnNames(restrictions, num_variables_decision)

    indices_columnas_mantener = []
    etiquetas_columnas = []

    # Filtrar columnas artificiales visualmente en la fase 2
    for i, nombre in enumerate(nombres_todas_columnas):
        if fase == 2 and nombre.startswith('a'):
            continue
        indices_columnas_mantener.append(i)
        etiquetas_columnas.append(nombre)

    matriz_filtrada = matriz[:, indices_columnas_mantener]
    
    # Nombrar las filas según la variable que las ocupa
    etiquetas_filas = ["Z"] + [nombres_todas_columnas[b] for b in basicas]
    texto_celdas = [[f"{valor:.2f}" for valor in fila] for fila in matriz_filtrada]

    ancho = min(1.3 * len(etiquetas_columnas), 10)
    alto = 0.6 * matriz_filtrada.shape[0] + 1.5
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.axis("off")

    tabla = ax.table(
        cellText=texto_celdas,
        colLabels=etiquetas_columnas,
        rowLabels=etiquetas_filas,
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1, 1.6)

    if pivote is not None:
        # Resaltar la celda pivote mientras el algoritmo se ejecuta
        fila_pivote, columna_pivote = pivote
        if columna_pivote in indices_columnas_mantener:
            col_index = indices_columnas_mantener.index(columna_pivote)
            celda_pivote = tabla[(fila_pivote + 1, col_index)]
            celda_pivote.set_facecolor("#ffd54f")
            celda_pivote.set_edgecolor("black")
            celda_pivote.set_linewidth(2)
    else:
        # Es la iteración final: Resaltar la fila Z, variables de decisión (x) y su RHS
        indice_columna_rhs = len(etiquetas_columnas) - 1
        for i, etiqueta in enumerate(etiquetas_filas):
            # Condición actualizada: si la fila empieza con "x" o es exactamente "Z"
            if etiqueta.startswith("x") or etiqueta == "Z":
                fila_tabla = i + 1  # +1 porque la fila 0 son los encabezados de columna
                
                # Pintar la celda de la etiqueta (el índice -1 corresponde a los rowLabels)
                celda_etiqueta = tabla[(fila_tabla, -1)]
                celda_etiqueta.set_facecolor("#a1f0a3")  # Verde claro
                
                # Pintar la celda del RHS
                celda_rhs = tabla[(fila_tabla, indice_columna_rhs)]
                celda_rhs.set_facecolor("#a1f0a3")

    estado_texto = "Solución óptima" if pivote is None else "Elemento pivote resaltado"
    ax.set_title(f"Fase {fase} — {estado_texto}", fontsize=12, fontweight="bold", color="#1a3c6e")

    fig.tight_layout()
    return fig


def generateSimplexSteps(history, restrictions, objectiveCoefficients):
    """
    Genera y formatea los pasos explicativos del método Simplex de Dos Fases
    nombrando correctamente la variable que sale de la matriz.
    """
    numDecisionVars = len(objectiveCoefficients)
    columnNames = getColumnNames(restrictions, numDecisionVars)
    
    textos = []
    # Se añade la 4ta variable (basicas) extraída del historial
    for indice, (matriz, pivote, fase, basicas) in enumerate(history):
        texto = f"Iteración {indice} — Fase {fase}\n" + "-" * 40 + "\n"
        
        if fase == 2 and indice > 0 and history[indice-1][2] == 1:
            texto += "¡Transición a Fase 2!\n"
            texto += "Teoría aplicada: Las variables artificiales (a) han cumplido su propósito de encontrar una solución factible básica y se eliminan (ignoran) del análisis.\n\n"
            
        if pivote is None:
            texto += "La fila Z ya no tiene coeficientes negativos (o positivos si es minimización).\n"
            texto += "¡Solución óptima alcanzada!\n"
        else:
            filaPivote, columnaPivote = pivote
            varEntra = columnNames[columnaPivote]
            varSale = columnNames[basicas[filaPivote - 1]] # Identificación de la variable que sale
            
            texto += f"Variable que entra: {varEntra} (columna {columnaPivote})\n"
            texto += f"Variable que sale: {varSale} (Fila {filaPivote}, por regla de la razón mínima)\n"
            texto += f"Elemento pivote = {matriz[filaPivote, columnaPivote]:.4f}\n"
            
        textos.append(texto)
        
    return textos