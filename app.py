from tkinter import *
from tkinter import ttk, filedialog, font, messagebox
from PIL import ImageTk, Image
from read_img import read_dicom_file, read_jpg_file
from integrator import predict

class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        # Configuración de ventana
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Fuente en negrita
        bold_font = font.Font(weight="bold")

        # Etiquetas
        ttk.Label(self.root, text="Imagen Radiográfica", font=bold_font).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold_font).place(x=545, y=65)
        ttk.Label(self.root, text="Resultado:", font=bold_font).place(x=500, y=350)
        ttk.Label(self.root, text="Cédula Paciente:", font=bold_font).place(x=65, y=350)
        ttk.Label(self.root, text="Probabilidad:", font=bold_font).place(x=500, y=400)
        ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold_font,
        ).place(x=122, y=25)

        # Variables de texto para resultados
        self.ID = StringVar()
        self.result = StringVar()

        # Campos de entrada
        self.entry_patient_id = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.entry_patient_id.place(x=200, y=350)

        # Textos para mostrar imágenes y resultados
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img1.place(x=65, y=90)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text_img2.place(x=500, y=90)
        self.text_result = Text(self.root, width=12, height=1)
        self.text_result.place(x=610, y=350)
        self.text_probability = Text(self.root, width=12, height=1)
        self.text_probability.place(x=610, y=400)

        # Botones
        self.button_predict = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button_predict.place(x=220, y=460)
        ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file).place(x=70, y=460)
        ttk.Button(self.root, text="Borrar", command=self.clear_all).place(x=670, y=460)
        ttk.Button(self.root, text="PDF", command=self.create_pdf).place(x=520, y=460)
        ttk.Button(self.root, text="Guardar", command=self.save_results_csv).place(x=370, y=460)

        # Inicializar variables
        self.array = None
        self.report_id = 0  # ID para los reportes PDF

        self.root.mainloop()

    def load_img_file(self):
        """Carga un archivo de imagen DICOM o JPG y lo muestra en el widget."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar Imagen",
            filetypes=(("DICOM", "*.dcm"), ("JPEG", "*.jpeg"), ("JPG", "*.jpg"), ("PNG", "*.png")),
        )
        if not filepath:
            return

        try:
            if filepath.lower().endswith(".dcm"):
                self.array, img2show = read_dicom_file(filepath)
            elif filepath.lower().endswith((".jpg", ".jpeg", ".png")):
                self.array, img2show = read_jpg_file(filepath)
            else:
                raise ValueError("Formato no válido. Usa archivos DICOM, JPG, JPEG o PNG.")
            
            img_resized = img2show.resize((250, 250), Image.Resampling.LANCZOS)
            self.img1 = ImageTk.PhotoImage(img_resized)
            self.text_img1.image_create(END, image=self.img1)
            self.button_predict["state"] = "enabled"
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    def run_model(self):
        """Ejecuta el modelo para predecir clase, probabilidad y generar Grad-CAM."""
        if self.array is None:
            messagebox.showwarning("Advertencia", "Carga una imagen antes de predecir.")
            return

        label, probability, heatmap = predict(self.array)
        self.text_result.delete(1.0, END)
        self.text_result.insert(END, label)
        self.text_probability.delete(1.0, END)
        self.text_probability.insert(END, f"{probability:.2f}%")

        heatmap_img = Image.fromarray(heatmap).resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(heatmap_img)
        self.text_img2.image_create(END, image=self.img2)

    def save_results_csv(self):
        """Guarda los resultados en un archivo CSV."""
        try:
            with open("historial.csv", "a") as csvfile:
                csvfile.write(f"{self.ID.get()}, {self.text_result.get(1.0, END).strip()}, {self.text_probability.get(1.0, END).strip()}\n")
            messagebox.showinfo("Éxito", "Resultados guardados en historial.csv")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo: {e}")

    def create_pdf(self):
        """Crea un reporte en formato PDF del análisis actual."""
        try:
            from tkcap import CAP

            cap = CAP(self.root)
            img_path = f"Reporte_{self.report_id}.jpg"
            cap.capture(img_path)

            img = Image.open(img_path).convert("RGB")
            pdf_path = f"Reporte_{self.report_id}.pdf"
            img.save(pdf_path)
            self.report_id += 1
            messagebox.showinfo("Éxito", f"PDF generado: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el PDF: {e}")

    def clear_all(self):
        """Limpia todos los campos e imágenes de la interfaz."""
        self.entry_patient_id.delete(0, END)
        self.text_img1.delete(1.0, END)
        self.text_img2.delete(1.0, END)
        self.text_result.delete(1.0, END)
        self.text_probability.delete(1.0, END)
        self.button_predict["state"] = "disabled"
        self.array = None
        messagebox.showinfo("Éxito", "Todos los datos fueron borrados.")

if __name__ == "__main__":
    App()
