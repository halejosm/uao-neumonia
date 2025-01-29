import csv
from tkinter.messagebox import showinfo
from PIL import Image
import tkcap

class StorageHandler:
    def __init__(self):
        self.csv_file = "historial.csv"
        self.report_id = 0

    def save_to_csv(self, patient_id, label, probability):
        """Guarda los resultados en un archivo CSV."""
        with open(self.csv_file, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([patient_id, label, f"{probability:.2f}%"])
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def generate_pdf(self, root):
        """Genera un PDF del estado actual de la interfaz."""
        cap = tkcap.CAP(root)
        report_name = f"Reporte{self.report_id}.jpg"
        img = cap.capture(report_name)
        img = Image.open(report_name).convert("RGB")
        pdf_path = f"Reporte{self.report_id}.pdf"
        img.save(pdf_path)
        self.report_id += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")