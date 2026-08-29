"""
Interfaz gráfica para el solucionador de Programación Lineal (Versión Final).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from graphic_solver import graphic_solver, simplex_solver, generate_explanation_steps

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
        self.textos_pasos = []
        self.paso_actual = 0

        self._configurar_estilos()
        self._construir_interfaz()
        self._regenerar_formulario()

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

    # =========================================================
    # CONSTRUCCIÓN DEL PANEL IZQUIERDO (Formulario)
    # =========================================================
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
        self.frame_obj_vars = ttk.Frame(self.tarjeta_objetivo, style="Tarjeta.TFrame")
        self.frame_obj_vars.pack(fill="x", pady=(0, 10))
        
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

    def _crear_entradas_variables(self, contenedor):
        entradas = []
        cantidad = self.num_variables.get()
        for i in range(cantidad):
            entrada = ttk.Entry(contenedor, width=6, justify="center")
            entrada.insert(0, "1")
            entrada.pack(side="left")
            ttk.Label(contenedor, text=f"x{SUBINDICES[i]}", style="Tarjeta.TLabel").pack(side="left", padx=(3, 4))
            if i < cantidad - 1:
                ttk.Label(contenedor, text="+", style="Tarjeta.TLabel").pack(side="left", padx=(0, 4))
            entradas.append(entrada)
        return entradas

    def _regenerar_formulario(self):
        for widget in self.frame_obj_vars.winfo_children(): widget.destroy()
        ttk.Label(self.frame_obj_vars, text="Z =", style="Tarjeta.TLabel").pack(side="left", padx=(0, 6))
        self.entradas_objetivo = self._crear_entradas_variables(self.frame_obj_vars)

        for fila in self.filas_restricciones: fila["frame"].destroy()
        self.filas_restricciones.clear()
        self._agregar_fila_restriccion()
        
        es_multivariable = self.num_variables.get() > 2
        self.radio_grafico.configure(state="disabled" if es_multivariable else "normal")
        self.metodo_solucion.set("simplex" if es_multivariable else self.metodo_solucion.get())
        self.etiqueta_aviso_metodo.configure(text="Solo Simplex (>2 variables)." if es_multivariable else "")
            
        self._limpiar_resultados()

    def _agregar_fila_restriccion(self):
        frame = ttk.Frame(self.frame_filas_rest, style="Tarjeta.TFrame")
        frame.pack(fill="x", pady=3)

        entradas = self._crear_entradas_variables(frame)
        
        combo = ttk.Combobox(frame, values=["<=", ">="], width=4, state="readonly", justify="center")
        combo.set("<=")
        combo.pack(side="left", padx=(5, 10))

        cte = ttk.Entry(frame, width=8, justify="center")
        cte.insert(0, "0")
        cte.pack(side="left", padx=(0, 10))

        fila = {"frame": frame, "coeficientes": entradas, "signo": combo, "constante": cte}
        ttk.Button(frame, text="Quitar", command=lambda: self._quitar_fila_restriccion(fila)).pack(side="left")
        self.filas_restricciones.append(fila)

    def _quitar_fila_restriccion(self, fila):
        if len(self.filas_restricciones) > 1:
            fila["frame"].destroy()
            self.filas_restricciones.remove(fila)
        else:
            messagebox.showinfo("Aviso", "Debe existir al menos una restricción.")

    # =========================================================
    # CONSTRUCCIÓN DEL PANEL DERECHO (Resultados)
    # =========================================================
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
        ttk.Label(panel_exp, text="Explicación Paso a Paso", style="Seccion.TLabel").pack(anchor="w", pady=(0, 5))
        
        self.texto_explicacion = tk.Text(panel_exp, height=6, font=("Consolas", 11), bg=COLOR_FONDO, relief="flat", wrap="word", state="disabled")
        self.texto_explicacion.pack(fill="both", expand=True, pady=5)
        
        self.boton_siguiente = ttk.Button(panel_exp, text="Siguiente Paso ➡", style="Siguiente.TButton", command=self._avanzar_paso, state="disabled")
        self.boton_siguiente.pack(fill="x", pady=(10, 0))

    # =========================================================
    # LÓGICA DE SOLUCIÓN Y PASOS
    # =========================================================
    def _leer_datos(self):
        c_obj = [float(e.get()) for e in self.entradas_objetivo]
        rest = [( [float(e.get()) for e in f["coeficientes"]], f["signo"].get(), float(f["constante"].get()) ) for f in self.filas_restricciones]
        return c_obj, rest

    def _resolver(self):
        try:
            c_obj, rest = self._leer_datos()
        except ValueError:
            return messagebox.showerror("Error", "Revisa que todos los campos tengan números válidos.")

        self._limpiar_resultados()
        
        if self.metodo_solucion.get() == "grafico":
            self._procesar_grafico(c_obj, rest)
        else:
            self._mostrar_texto_simple("El método Simplex aún está en desarrollo.")

    def _procesar_grafico(self, c_obj, rest):
        c1 = c_obj[0]
        if len(c_obj) > 1:
            c2 = c_obj[1]
        else:
            c2 = 0
            
        rest_2d = []
        for r in rest:
            coeficientes = r[0]
            operador = r[1]
            constante = r[2]
            
            a = coeficientes[0]
            if len(coeficientes) > 1:
                b = coeficientes[1]
            else:
                b = 0
                
            rest_2d.append((a, b, operador, constante))

        try:
            vertices_raw, mejor_val_raw, fig = graphic_solver(rest_2d, c1, c2, self.tipo_optimizacion.get())
        except ValueError as e:
            self.lbl_grafico_vacio.place(relx=0.5, rely=0.5, anchor="center")
            return messagebox.showerror("Sin solución", str(e))
        
        vertices = [(float(x), float(y)) for x, y in vertices_raw]
        mejor_val = (float(mejor_val_raw[0]), float(mejor_val_raw[1]), float(mejor_val_raw[2]))

        self._renderizar_canvas(fig)
        self.textos_pasos = generate_explanation_steps(vertices, mejor_val, c1, c2, self.tipo_optimizacion.get())
        
        self.boton_siguiente.configure(state="normal")
        self._avanzar_paso()

    def _renderizar_canvas(self, fig):
        self.lbl_grafico_vacio.place_forget()
        canvas_grafico = FigureCanvasTkAgg(fig, master=self.panel_grafico)
        canvas_grafico.draw()
        canvas_grafico.get_tk_widget().pack(fill="both", expand=True)

    def _avanzar_paso(self):
        self.texto_explicacion.configure(state="normal")
        if self.paso_actual < len(self.textos_pasos):
            self.texto_explicacion.insert(tk.END, self.textos_pasos[self.paso_actual])
            self.paso_actual += 1

        if self.paso_actual >= len(self.textos_pasos):
            self.boton_siguiente.configure(state="disabled")

        self.texto_explicacion.see(tk.END)
        self.texto_explicacion.configure(state="disabled")

    def _limpiar_resultados(self):
        for widget in self.panel_grafico.winfo_children():
            if widget != self.lbl_grafico_vacio: widget.destroy()
        
        self.paso_actual = 0
        self.textos_pasos.clear()
        self.boton_siguiente.configure(state="disabled")
        self._mostrar_texto_simple("")
        self.lbl_grafico_vacio.place(relx=0.5, rely=0.5, anchor="center")

    def _mostrar_texto_simple(self, texto):
        self.texto_explicacion.configure(state="normal")
        self.texto_explicacion.delete("1.0", tk.END)
        if texto: self.texto_explicacion.insert(tk.END, texto)
        self.texto_explicacion.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    AplicacionProgramacionLineal(root)
    root.mainloop()