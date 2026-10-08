# statistical_tests.py
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar(y_true, y_pred_model1, y_pred_model2, model1_name="Model1", model2_name="Model2"):
    """
    McNemar testi: iki sınıflandırıcının hata oranlarını karşılaştırır.
    """
    # Tablo için değerler
    b01 = np.sum((y_pred_model1 == y_true) & (y_pred_model2 != y_true))  # model1 doğru, model2 yanlış
    b10 = np.sum((y_pred_model1 != y_true) & (y_pred_model2 == y_true))  # model1 yanlış, model2 doğru

    table = [[0, b01],
             [b10, 0]]

    result = mcnemar(table, exact=True)
    print(f"McNemar Testi ({model1_name} vs {model2_name})")
    print("Chi2 istatistiği:", result.statistic)
    print("p-değeri:", result.pvalue)

    if result.pvalue < 0.05:
        print("→ İstatistiksel olarak anlamlı fark var.")
    else:
        print("→ İstatistiksel olarak anlamlı fark yok.")