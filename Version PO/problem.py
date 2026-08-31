class LinearProblem:
    """
    Clase Entidad: guarda los datos de un problema de Programación Lineal.

    """

    def __init__(self, objective_coefficients, restrictions, optimization_type="max"):
        self.objective_coefficients = objective_coefficients
        self.restrictions = restrictions
        self.optimization_type = optimization_type

