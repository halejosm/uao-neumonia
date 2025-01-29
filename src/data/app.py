import csv
from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import tkcap
import tensorflow as tf

# Importaciones de módulos personalizados
from data.read_img import read_image_file
from models.integrator import predict_image

# Configuración global de TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


# Clase para la lógica de predicción
class Predictor:
    def __init__(self):
        self.image_array = None
        self.label = ""
        self.probability = 0.0
        self.heatmap = None

    def load_image(self, filepath):
        """
        Carga la imagen desde un archivo y devuelve la imagen procesada.

        Args:
            filepath: La ruta del archivo de la imagen a cargar.

        Returns:
            img_to_show: Imagen procesada lista para mostrar.
        """
        self.image_array, img_to_show = read_image_file(filepath)
        return img_to_show

    def run_model(self):
        """
        Ejecuta el modelo de predicción sobre la imagen cargada.

        Returns:
            tuple: Etiqueta predicha, probabilidad de la predicción, y el heatmap.
        """
        self.label, self.probability, self.heatmap = predict_image(self.image_array)
        return self.label, self.probability, self.heatmap


# Clase para manejar el almacenamiento
class StorageHandler:
    def __init__(self):
        self.csv_file = "src/reports/historial.csv"
        self.report_id = 0

    def save_to_csv(self, patient_id, label, probability):
        """
        Guarda los resultados en un archivo CSV.

        Args:
            patient_id: Identificación del paciente.
            label: Etiqueta predicha por el modelo.
            probability: Probabilidad de la predicción.
        """
        with open(self.csv_file, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([patient_id, label, f"{probability:.2f}%"])
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def generate_pdf(self, root):
        """
        Genera un PDF del estado actual de la interfaz.

        Args:
            root: La raíz de la interfaz gráfica de usuario.
        """
        cap = tkcap.CAP(root)
        report_name = f"src/reports/Reporte{self.report_id}.jpg"
        img = cap.capture(report_name)
        img = Image.open(report_name).convert("RGB")
        pdf_path = f"src/reports/Reporte{self.report_id}.pdf"
        img.save(pdf_path)
        self.report_id += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")


# Clase principal para la interfaz gráfica
class App:
    def __init__(self):
        self.root = Tk()
        self.predictor = Predictor()
        self.storage = StorageHandler()
        self.setup_ui()

    def setup_ui(self):
        """
        Configura la interfaz gráfica y los componentes.
        """
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("900x560")
        self.root.resizable(0, 0)

        # Fuentes
        fonti = font.Font(weight="bold")

        # Configuración de etiquetas
        self.setup_labels(fonti)

        # Variables de control
        self.id_var = StringVar()
        self.result_var = StringVar()

        # Configuración de entradas y botones
        self.setup_inputs_and_buttons()

        # Elementos auxiliares
        self.image_1 = None
        self.image_2 = None

        self.root.mainloop()

    def setup_labels(self, fonti):
        """
        Configura y coloca las etiquetas.

        Args:
            fonti: La fuente utilizada para las etiquetas.
        """
        labels = [
            ("Imagen Radiográfica", (110, 65)),
            ("Imagen con Heatmap", (545, 65)),
            ("Resultado:", (500, 350)),
            ("Cédula Paciente:", (65, 400)),
            ("SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", (122, 25)),
            ("Probabilidad:", (500, 400)),
        ]
        for text, position in labels:
            label = ttk.Label(self.root, text=text, font=fonti)
            label.place(x=position[0], y=position[1])

    def setup_inputs_and_buttons(self):
        """
        Configura entradas y botones de la interfaz.
        """
        # Entradas de texto
        self.id_entry = ttk.Entry(self.root, textvariable=self.id_var, width=10)
        self.img_display_1 = Text(self.root, width=31, height=15)
        self.img_display_2 = Text(self.root, width=31, height=15)
        self.result_display = Text(self.root)
        self.probability_display = Text(self.root)

        # Botones
        self.predict_button = ttk.Button(self.root, text="Predecir", state="disabled", command=self.predict)
        self.load_button = ttk.Button(self.root, text="Cargar Imagen", command=self.load_image)
        self.clear_button = ttk.Button(self.root, text="Borrar", command=self.clear)
        self.pdf_button = ttk.Button(self.root, text="PDF", command=self.generate_pdf)
        self.save_button = ttk.Button(self.root, text="Guardar", command=self.save_results)

        # Posicionamiento
        self.id_entry.place(x=250, y=400)
        self.img_display_1.place(x=65, y=90)
        self.img_display_2.place(x=500, y=90)
        self.result_display.place(x=610, y=350, width=90, height=30)
        self.probability_display.place(x=660, y=400, width=90, height=30)
        self.predict_button.place(x=220, y=460)
        self.load_button.place(x=70, y=460)
        self.clear_button.place(x=670, y=460)
        self.pdf_button.place(x=520, y=460)
        self.save_button.place(x=370, y=460)

        # Foco inicial
        self.id_entry.focus_set()

    # Métodos de eventos
    def load_image(self):
        """Carga una imagen desde un archivo."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Seleccionar imagen",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )
        if filepath:
            try:
                img_to_show = self.predictor.load_image(filepath)
                self.image_1 = img_to_show.resize((250, 250), Image.Resampling.LANCZOS)
                self.image_1 = ImageTk.PhotoImage(self.image_1)
                self.img_display_1.image_create(END, image=self.image_1)
                self.predict_button["state"] = "enabled"
            except ValueError as e:
                showinfo(title="Error", message=str(e))

    def predict(self):
        """Realiza la predicción sobre la imagen cargada."""
        label, proba, heatmap = self.predictor.run_model()
        self.image_2 = Image.fromarray(heatmap).resize((250, 250), Image.Resampling.LANCZOS)
        self.image_2 = ImageTk.PhotoImage(self.image_2)
        self.img_display_2.image_create(END, image=self.image_2)
        self.result_display.insert(END, label)
        self.probability_display.insert(END, f"{proba:.2f}%")

    def save_results(self):
        """Guarda los resultados en un archivo CSV."""
        self.storage.save_to_csv(self.id_var.get(), self.predictor.label, self.predictor.probability)

    def generate_pdf(self):
        """Genera un PDF del estado actual de la aplicación."""
        self.storage.generate_pdf(self.root)

    def clear(self):
        """Limpia todos los campos de entrada y visualización."""
        if askokcancel(title="Confirmación", message="¿Borrar todos los datos?", icon=WARNING):
            self.id_entry.delete(0, END)
            self.result_display.delete(1.0, END)
            self.probability_display.delete(1.0, END)
            self.img_display_1.delete(1.0, END)
            self.img_display_2.delete(1.0, END)
            showinfo(title="Borrar", message="Los datos se borraron con éxito.")


def main():
    App()


if __name__ == "__main__":
    main()