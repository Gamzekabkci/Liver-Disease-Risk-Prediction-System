import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# En iyi modeli ve scaler'ı yükle
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict():
    try:
        # Kullanıcıdan değerleri al
        age = int(entry_age.get())
        gender = entry_gender.get()
        tb = float(entry_tb.get())
        db = float(entry_db.get())
        ap = float(entry_ap.get())
        aa = float(entry_aa.get())
        asat = float(entry_asat.get())
        tp = float(entry_tp.get())
        alb = float(entry_alb.get())
        agr = float(entry_agr.get())

        # Cinsiyet dönüştürme
        gender_val = 1 if gender.lower() == "male" else 0

        # Yeni veri vektörü
        X_new = np.array([[age, gender_val, tb, db, ap, aa, asat, tp, alb, agr]])
        X_new_scaled = scaler.transform(X_new)

        # Tahmin
        prob = model.predict_proba(X_new_scaled)[0][1]
        pred = model.predict(X_new_scaled)[0]
        sonuc = "Hasta" if pred == 1 else "Hasta değil"

        messagebox.showinfo("Tahmin Sonucu", f"%{prob*100:.2f} olasılıkla {sonuc}")

    except Exception as e:
        messagebox.showerror("Hata", str(e))

# Tkinter arayüzü
root = tk.Tk()
root.title("Karaciğer Hastalığı Tahmini")

labels = ["Yaş", "Cinsiyet (Male/Female)", "Total Bilirubin", "Direct Bilirubin",
          "Alkaline Phosphotase", "Alanine Aminotransferase", "Aspartate Aminotransferase",
          "Total Proteins", "Albumin", "Albumin and Globulin Ratio"]

entries = []
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=5, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

(entry_age, entry_gender, entry_tb, entry_db, entry_ap,
 entry_aa, entry_asat, entry_tp, entry_alb, entry_agr) = entries

tk.Button(root, text="Tahmin Et", command=predict).grid(row=len(labels), column=0, columnspan=2, pady=10)

root.mainloop()