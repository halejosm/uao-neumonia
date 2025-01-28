# Importaciones necesarias
import csv
from tkinter import *
from tkinter import ttk, font, filedialog
from tkinter.messagebox import askokcancel, showinfo, WARNING
from PIL import ImageTk, Image
import tkcap
import tensorflow as tf
from tensorflow.keras import backend as K  # type: ignore

# Importaciones de módulos personalizados
from read_img import read_image_file
from integrator import predict_image

# Configuración global de TensorFlow
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

class App:
    def __init__(self):
        """Inicialización de la aplicación gráfica."""
        self.root = Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(0, 0)

        # Configuración de fuentes
        fonti = font.Font(weight="bold")

        # Configuración de etiquetas
        self.setup_labels(fonti)

        # Variables de control
        self.ID = StringVar()
        self.result = StringVar()

        # Configuración de entradas de texto y botones
        self.setup_inputs_and_buttons()

        # Elementos adicionales
        self.array = None
        self.reportID = 0  # Número de identificación para PDF

        # Inicia el bucle de la interfaz gráfica
        self.root.mainloop()

    def setup_labels(self, fonti):
        """Configura y coloca las etiquetas en la ventana."""
        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root, text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA", font=fonti
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        # Posicionamiento de las etiquetas
        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)

    def setup_inputs_and_buttons(self):
        """Configura las entradas de texto y botones."""
        # Entradas de texto
        self.text1 = ttk.Entry(self.root, textvariable=self.ID, width=10)
        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        # Botones
        self.button1 = ttk.Button(self.root, text="Predecir", state="disabled", command=self.run_model)
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)

        # Posicionamiento
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)

        # Foco inicial en el campo de ID
        self.text1.focus_set()

    # Métodos principales de la aplicación
    def load_img_file(self):
        """Carga un archivo de imagen y lo muestra en la interfaz."""
        filepath = filedialog.askopenfilename(
            initialdir="/",
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("JPG", "*.jpg"),
                ("PNG", "*.png"),
            ),
        )
        if filepath:
            try:
                self.array, img2show = read_image_file(filepath)
                self.img1 = img2show.resize((250, 250), Image.Resampling.LANCZOS)
                self.img1 = ImageTk.PhotoImage(self.img1)
                self.text_img1.image_create(END, image=self.img1)
                self.button1["state"] = "enabled"
            except ValueError as e:
                showinfo(title="Error", message=str(e))

    def run_model(self):
        """Ejecuta el modelo de predicción y muestra los resultados."""
        self.label, self.proba, self.heatmap = predict_image(self.array)
        self.img2 = Image.fromarray(self.heatmap)
        self.img2 = self.img2.resize((250, 250), Image.Resampling.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.image_create(END, image=self.img2)
        self.text2.insert(END, self.label)
        self.text3.insert(END, "{:.2f}".format(self.proba) + "%")

    def save_results_csv(self):
        """Guarda los resultados en un archivo CSV."""
        with open("historial.csv", "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([self.text1.get(), self.label, f"{self.proba:.2f}%"])
            showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self):
        """Genera un PDF del reporte actual."""
        cap = tkcap.CAP(self.root)
        report_name = f"Reporte{self.reportID}.jpg"
        img = cap.capture(report_name)
        img = Image.open(report_name).convert("RGB")
        pdf_path = f"Reporte{self.reportID}.pdf"
        img.save(pdf_path)
        self.reportID += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self):
        """Borra todos los datos ingresados en la interfaz."""
        if askokcancel(title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING):
            self.text1.delete(0, END)
            self.text2.delete(1.0, END)
            self.text3.delete(1.0, END)
            self.text_img1.delete(1.0, END)
            self.text_img2.delete(1.0, END)
            showinfo(title="Borrar", message="Los datos se borraron con éxito.")

def main():
    App()

if __name__ == "__main__":
    main()
