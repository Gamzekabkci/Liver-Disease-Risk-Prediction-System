import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath):
    # 1. Veri setini oku (header yok!)
    column_names = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
        'Alkaline_Phosphotase', 'Alanine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Proteins',
        'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset'
    ]
    df = pd.read_csv(filepath, header=None, names=column_names)

    # 2. Eksik değerleri temizle
    df = df.dropna()

    # 3. Hedef değişken
    y = df['Dataset'].replace({1: 1, 2: 0})  # 1 = Hasta, 0 = Hasta değil

    # 4. Özellikler
    X = df.drop('Dataset', axis=1)

    # 5. Kategorik değişkenleri dönüştür (Gender: Male/Female)
    if 'Gender' in X.columns:
        le = LabelEncoder()
        X['Gender'] = le.fit_transform(X['Gender'])

    # 6. Eğitim/Test ayırma
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 7. Ölçekleme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
