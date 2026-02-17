"""Módulo para la gestión y preprocesamiento del dataset de radiación."""

import pandas as pd
import torch
from torch.utils import data
from sklearn import model_selection


class RadiationDataset(data.Dataset):
  """Clase Dataset de PyTorch para datos de fragilización por radiación.

  Esta clase gestiona la carga de datos desde un archivo CSV, permite la
  división en subconjuntos y ofrece un método para aplicar preprocesado a
  todo el conjunto de datos basándose en el entrenamiento.
  """

  def __init__(self, csv_path):
    """Inicializa el dataset cargando el archivo CSV.

    Args:
      csv_path: Cadena con la ruta al archivo CSV del dataset.
    """
    self.data = pd.read_csv(csv_path)

    # Convertimos categorías a códigos numéricos para permitir cálculos.
    if 'Product_Form' in self.data.columns:
      self.data['Product_Form'] = (
          self.data['Product_Form'].astype('category').cat.codes
      )

  def __len__(self):
    """Devuelve la cantidad total de registros en el dataset."""
    return len(self.data)

  def __getitem__(self, idx):
    """Devuelve un registro del dataset convertido a tensor.

    Args:
      idx: Índice de la muestra a obtener.

    Returns:
      Un tensor de PyTorch con los datos de la fila solicitada.
    """
    # Se obtienen los valores de la fila y se transforman a tensor de PyTorch.
    sample = self.data.iloc[idx].values
    return torch.tensor(sample, dtype=torch.float32)

  def split_data(self, test_factor, val_factor=None):
    """Divide el dataset en subconjuntos de train, val y test.

    Los subconjuntos devueltos son objetos 'Subset', lo que significa que
    apuntan directamente a la memoria de este objeto dataset. Si se modifican
    los valores en 'self.data', los subconjuntos reflejarán el cambio.

    Args:
      test_factor: Proporción del conjunto total para test (entre 0 y 1).
      val_factor: Proporción opcional para el conjunto de validación.

    Returns:
      Una tupla con los objetos Subset: (train, val, test) o (train, test).
    """
    indices = list(range(len(self)))

    if val_factor:
      # Primero separamos el conjunto de test.
      train_val_idx, test_idx = model_selection.train_test_split(
          indices, test_size=test_factor, random_state=42
      )
      # Ajustamos el factor de validación sobre el remanente.
      adj_val_factor = val_factor / (1 - test_factor)
      train_idx, val_idx = model_selection.train_test_split(
          train_val_idx, test_size=adj_val_factor, random_state=42
      )
      return (data.Subset(self, train_idx),
              data.Subset(self, val_idx),
              data.Subset(self, test_idx))

    # División simple en entrenamiento y test.
    train_idx, test_idx = model_selection.train_test_split(
        indices, test_size=test_factor, random_state=42
    )
    return data.Subset(self, train_idx), data.Subset(self, test_idx)

  def preprocess(self, train_set, preprocessor):
    """Entrena un preprocesador con el conjunto de train y escala todo el dataset.

    Este método debe ser llamado manualmente por el usuario tras crear el objeto.
    Al modificar el DataFrame interno 'self.data', todos los objetos Subset
    creados anteriormente se verán afectados automáticamente.

    Args:
      train_set: El objeto Subset que representa los datos de entrenamiento.
      preprocessor: Objeto con métodos fit/transform (ej. StandardScaler).
    """
    # Extraemos los índices del subconjunto de entrenamiento.
    train_indices = train_set.indices
    train_data = self.data.iloc[train_indices]

    # Entrenamos el preprocesador con los datos de entrenamiento.
    preprocessor.fit(train_data)

    # Transformamos el dataset completo para mantener la coherencia.
    transformed_values = preprocessor.transform(self.data)
    
    # Sobrescribimos el DataFrame interno con los nuevos valores.
    self.data.iloc[:, :] = transformed_values


# Ejemplo de uso tal como lo solicitaste:
if __name__ == "__main__":
  from sklearn.preprocessing import StandardScaler

  # 1. Crear el objeto de la clase.
  dataset = RadiationDataset('df_plotter_cm2.csv')

  # 2. Dividir en conjuntos (ej. 70% train, 15% val, 15% test).
  # Los conjuntos devueltos son 'vistas' del dataset original.
  train, val, test = dataset.split_data(test_factor=0.15, val_factor=0.15)

  # 3. El usuario activa el preprocesado manualmente.
  scaler = StandardScaler()
  dataset.preprocess(train, scaler)

  # Ahora 'train', 'val' y 'test' devuelven valores normalizados.
  print(f"Muestra de train normalizada: {train[0]}")
  print(f"Muestra de test normalizada: {test[0]}")