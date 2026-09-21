import tkinter as tk
from ui.main_view import MainView

from model.problem import LinearProblem
from core.graphic_solver import GraphicSolver
from core.simplex_solver import SimplexModel
from core.simple_solver_gran_m import SimplexModelGranM

class ControladorProgramacionLineal:
    """
    Controlador (MVC): Actúa como intermediario entre la Vista (UI) 
    y el Modelo (LinearProblem + Solvers).
    """
    def __init__(self, root):
        self.view = MainView(root, self.resolver_problema)

    def resolver_problema(self, c_obj, rest, opt_type, metodo):
        # 1. Crear el problema lineal
        try:
            problema = LinearProblem(c_obj, rest, opt_type)
        except ValueError as e:
            self.view.mostrar_error("Error", str(e))
            return

        # 2. Seleccionar y ejecutar el algoritmo adecuado
        try:
            if metodo == "grafico":
                modelo = GraphicSolver(problema)
            elif metodo == "simplex_comun":
                modelo = SimplexModel(problema)
            elif metodo == "simplex_gran_m":
                modelo = SimplexModelGranM(problema)
            else:
                self.view.mostrar_error("Error", "Método no reconocido.")
                return

            modelo.resolver()
            pasos = modelo.obtener_pasos_ui()
            
            # 3. Enviar los pasos de vuelta a la vista
            self.view.cargar_pasos(pasos)

        except ValueError as e:
            # Captura errores matemáticos (ej. región no acotada, sin solución)
            self.view.mostrar_error("Sin solución", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ControladorProgramacionLineal(root)
    root.mainloop()