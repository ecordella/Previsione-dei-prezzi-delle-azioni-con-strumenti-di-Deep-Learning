"""
Titolo: Previsione dei Prezzi delle Azioni con PyTorch
Autore: Enrico Cordella
Data: 22/04/2025

## Descrizione: 
Questo programma utilizza PyTorch per costruire un modello LSTM (Long Short-Term Memory) per la previsione dei prezzi delle azioni.
Il modello è addestrato su dati storici delle azioni scaricati da Yahoo Finance.
Il modello LSTM è una rete neurale ricorrente (RNN) progettata per lavorare con sequenze di dati e può catturare le dipendenze temporali nei prezzi delle azioni.
Il programma include la creazione di un dataset, la definizione del modello LSTM, l'addestramento del modello e la previsione dei prezzi futuri delle azioni.

## Requisiti:
 - Python 3.x
 - Librerie: pandas, numpy, random, matplotlib, yfinance, torch, sklearn


## Funzionalità
 1. Scarica i dati storici delle azioni da Yahoo Finance.
 2. Prepara i dati per l'addestramento e il test del modello LSTM.
 3. Definisce un modello di RNN personalizzato.
 4. Addestra il modello sui dati storici delle azioni.
 5. Prevede i prezzi futuri delle azioni.
 6. Visualizza i risultati delle previsioni in un grafico e salva il grafico in un file.
"""

import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler

# Per leggere i dati finanziari
import yfinance as yf

# Per gestire la generazione delle date
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


SEED = 42


def plot_grafico_finale(tk_azione, df_dataset, y_prev, salva=False):

    """
    Funzione per disegnare il grafico finale delle previsioni
    :param df_dataset: dataframe con i dati di produzione
    :param y_prev: previsioni
    :param salva: se True salva il grafico in un file
    :return: None
    """

    # Recupero del nome dell'azienda associata al ticker
    nome_azienda = yf.Ticker(tk_azione).info['longName']

    # Disegno del grafico finale delle previsioni
    train = df_dataset[:train_index]
    prod = df_dataset[train_index:]

    # Le previsioni sono messe insieme ai valori reali di produzione
    pd.options.mode.copy_on_write = True
    prod['Previsioni'] = y_prev

    fig = plt.figure(figsize=(12,9))
    plt.title(nome_azienda, fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Prezzo', fontsize=12)

    # Gestione degli assi del grafico
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

    # Disegno finale
    plt.plot(train[tk_azione])
    plt.plot(prod[[tk_azione, 'Previsioni']])
    plt.legend(['Training', 'Reali', 'Previsioni'], fontsize=12)

    if salva:
        # Se non esiste, crea la cartella "img"
        os.makedirs("img", exist_ok=True)

        # Percorso per salvare l'immagine
        nome_file= f'PyTorch_Previsioni_{tk_azione}.png'        
        path_file = os.path.join("img", nome_file)

        fig.savefig(path_file, dpi=300, bbox_inches='tight')
        print('Plot grafico finale: grafico salvato come "' + nome_file + '"')
    else:
        plt.show()

def crea_dataset(dataset, lookback):

    """
    Funzione per la creazione dei dataset per il training e il test
    :param dataset: array numpy con i dati delle azioni
    :param lookback: numero di giorni da utilizzare per le previsioni
    :return: X_tensor e y_tensor, i dataset di input e output
    """

    X,y = [], []

    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)

    X_tensor = torch.tensor(np.array(X)).float().to(device)
    y_tensor = torch.tensor(np.array(y)).float().to(device)
    return X_tensor, y_tensor



class RNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = n_layers

        self.fcIn = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fcOut = nn.Linear(hidden_dim, output_dim)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):

        # Inizializzazione dell'hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializzione del cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch

        x = self.fcIn(x) 
        #x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x, _ = self.lstm(x)
        x = self.fcOut(x) 

        return x


if __name__ == "__main__":

    # Imposta il seed per la riproducibilità
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    # Imposta la modalità di esecuzione
    # Se 'GRAFICA' il programma disegna il grafico finale delle previsioni
    MODALITA = 'GRAFICA' 
    verbose = True

    # Permette di utilizzare l'acceleratore grafico, se disponibile
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'


    # Si fissa un ticker su cui lavorare
    tk_azione = 'AAPL' # Apple Inc.

    # Definizione del periodo di analisi
    inizio = datetime(2020, 1, 1)
    fine = datetime(2025, 3, 31)

    # Numero di giorni da utilizzare per le previsioni
    GG_PREVISIONE = 50

    # Recupero delle informazioni finanziarie da utilizzare per le previsioni
    df_dati_by_ticker = yf.download(tk_azione, start=inizio, end=fine)

    # Gestione dei prezzi di chiusura
    df_dataset = df_dati_by_ticker['Close']
    np_dataset = df_dataset.values
    dataset_len = len(np_dataset)

    # train_index è l'indice per la costruzione del dataset di training
    train_index = dataset_len - GG_PREVISIONE

    # Scaling dei dati mediante MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(np_dataset.reshape(-1, 1))

    # Creazione del training set a partire dagli scaled_data di lunghezza pari al train_index
    train_data = scaled_data[0:int(train_index), :]
    # Creazione del Dataset di Test: un nuovo array  che contiene i valori scalati nell'ultimo periodo
    test_data = scaled_data[train_index - GG_PREVISIONE: , :]

    X_train, y_train = crea_dataset(train_data, lookback=GG_PREVISIONE)
    X_test, y_test = crea_dataset(test_data, lookback=GG_PREVISIONE)

    # Parametri di configurazione del modello LSTM
    input_dim = 1
    hidden_dim = 32
    output_dim = 1
    n_layers = 2
    batch_dim = 1
    num_epoche = 5


    model = RNN_LSTM_Model(input_dim, hidden_dim, output_dim, n_layers).to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_dim)
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_dim)

    print("Inizio addestramento...")
    model.train()
    for epoch in range(num_epoche):

        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoca {epoch+1}/{num_epoche}", unit="batch", ncols=80, smoothing=0.1)):
    
            # Azzeramento del gradiente
            optimizer.zero_grad()

            # Forward pass
            # Calcolo delle previsioni del modello
            y_pred = model(X_batch)

            # Calcolo della loss
            # La loss è calcolata come la differenza tra le previsioni e i valori reali
            loss = loss_fn(y_pred, y_batch)
            
            # Backpropagation
            # Calcolo dei gradienti della funzione di perdita rispetto ai parametri del modello
            loss.backward()

            # Aggiornamento dei pesi
            optimizer.step()

        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_train.cpu()).detach().numpy())            

        print(f"Epoca {epoch+1}/{num_epoche} - Loss: {train_rmse:.4f}")

    print("Addestramento completato!")

    # Fase di validazione
    print("Inizio Predizione...")    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_test.cpu()).detach().numpy())            
        print("Test Loss: %.4f" % test_rmse)
    print("Predizione completata!")    
    # Convert y_pred to a NumPy array
    y_pred_numpy = y_pred.detach().cpu().numpy()


    # Poiché il modello è stato addestrato su dati scalati, anche le previsioni sono scalate.
    # Con questa istruzione si utilizza lo scaler per riconvertire le previsioni ai loro valori originali, non scalati.
    y_prev = scaler.inverse_transform(y_pred_numpy[0].reshape(-1, 1))

    y_real = np_dataset[train_index:, :]
    test_rmse = np.sqrt(np.mean(((y_prev - y_real) ** 2)))
    print("REAL RMSE: %.4f" % float(test_rmse))

    # Se la modalità è 'GRAFICA', si disegna il grafico finale delle previsioni
    if (MODALITA == 'GRAFICA'):
        plot_grafico_finale(tk_azione, df_dataset, y_prev, salva=verbose)
