"""
Titolo: Previsione dei Prezzi delle Azioni con Keras
Autore: Enrico Cordella
Data: 22/04/2025

## Descrizione: 
Previsione dei prezzi delle azioni utilizzando un modello LSTM (Long Short-Term Memory) con Keras.
Questo programma scarica i dati storici delle azioni da Yahoo Finance e utilizza un modello LSTM per prevedere i prezzi futuri.
Il modello viene addestrato sui dati storici e le previsioni vengono visualizzate in un grafico.
Le previsioni sono effettuate su un periodo di 50 giorni.
Il modello è configurato per utilizzare 128 unità LSTM nel primo strato, 64 unità LSTM nel secondo strato e 32 unità Dense nel terzo strato.
Il modello finale ha un'unità Dense con un singolo neurone per la previsione del prezzo.
Il modello viene addestrato per 2 epoche con un batch size di 1.
L'ottimizzatore utilizzato è 'adam' e la metrica di valutazione è l'errore quadratico medio (MSE).


## Requisiti:
 - Python 3.x
 - Librerie: pandas, numpy, random, matplotlib, yfinance, tensorflow, keras


## Funzionalità
 1. Scarica i dati storici delle azioni da Yahoo Finance.
 2. Prepara i dati per l'addestramento del modello LSTM.
 3. Costruisce e addestra un modello LSTM per la previsione dei prezzi delle azioni.
 4. Effettua previsioni sui prezzi futuri delle azioni.
 5. Calcola l'errore quadratico medio (RMSE) tra le previsioni e i valori reali.    
 6. Visualizza i risultati delle previsioni in un grafico e lo salva in un file immagine.
"""

import pandas as pd
import numpy as np
import os
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.preprocessing import MinMaxScaler

# Per leggere i dati finanziari
import yfinance as yf

# Per gestire la generazione delle date
from datetime import datetime

# Import di modelli e layers di Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input


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

    fig = plt.figure(figsize=(16,10))
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
        nome_file= f'Keras_Previsioni_{tk_azione}.png'        
        path_file = os.path.join("img", nome_file)
        fig.savefig(path_file, dpi=300, bbox_inches='tight')
        print('Plot grafico finale: grafico salvato come "' + nome_file + '"')
    else:
        plt.show()


if __name__ == "__main__":


    # Imposta il seme per la riproducibilità
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # Modalità di esecuzione del programma
    # Se 'GRAFICA', il programma disegna il grafico finale delle previsioni
    MODALITA = 'GRAFICA' 
    verbose = True

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

    # Suddivisione dei dati in x_train e y_train
    x_train = []
    y_train = []

    # Costruzione dei dataset di x_train e y_train
    for i in range(GG_PREVISIONE, train_index):
        # In ogni iterazione, si estrae una sequenza di 50 valori consecutivi dall'array train_data,
        # Questi 50 valori rappresentano le feature di input del modello.
        x_train.append(train_data[i - GG_PREVISIONE : i, 0])

        # Il target value è rappresentato dall'iesimo campione
        y_train.append(train_data[i, 0])

    # Conversione delle liste in numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape dei dati da mandare in pasto al modello
    # La forma finale di x_train è: numero di campioni, intervalli temporali, numero di feature
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Parametri di configuazione del modello LSTM
    n_unita_LSTM1 = 128
    n_unita_LSTM2 = 64
    n_unita_Dense = 32
    n_unita_Dense_finale = 1
    n_epoche = 2
    n_batch = 1
    ottimizzatore='adam'
    metrica='mean_squared_error'


    # Costruzione del modello sequenziale inizialmente vuoto
    # I livelli verranno aggiunti successivamente
    model = Sequential()

    # Aggiunta di un livello di tipo Input per definire la tipologia dei dati passati:
    # - x_train.shape[1]: rappresenta il numero di intervalli temporali (giorni) utilizzati per la previsione
    # - 1: indica la singola caratteristica (prezzo di chiusura).
    input_shape = (x_train.shape[1], 1)
    model.add(Input(shape=input_shape)) 

    # Con questa istruzione si aggiunge il primo livello di tipo LSTM al modello.
    # I parametri sono i seguenti:
    # - 128: specifica il numero di unità di memoria (neuroni) all'interno del livello LSTM
    # - activation: funzione di attivazione "tanh"
    # - recurrent_activation: funzione di attivazione ricorrente "sigmoid"
    # - return_sequences=True: indica che a seguire ci sarà un altro livello LSTM (generare una sequenza in output)
    model.add(LSTM(n_unita_LSTM1, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))

    # Si aggiunge un secondo livello LSTM:
    # - 64 neuroni nel livello
    # - activation: funzione di attivazione "tanh"
    # - recurrent_activation: funzione di attivazione ricorrente "sigmoid"
    # - return_sequences=False: per indicare che si tratta dell'ultimo livello LSTM e si desidera un singolo output.
    model.add(LSTM(n_unita_LSTM2, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))

    # Si aggiunge uno strato Dense (fully connected) con 32 neuroni per una elaborazione intermedia
    model.add(Dense(n_unita_Dense, activation='relu'))
    # La funzione di attivazione 'relu' (Rectified Linear Unit) è comunemente usata nei modelli di deep learning

    # Si aggiunge il livello Dense (fully connected) finale con un singolo neurone.
    # Questo livello restituisce il prezzo previsto delle azioni.
    model.add(Dense(n_unita_Dense_finale))

    # Configurazione del modello per il training:
    # - optimizer='adam': Specifica l'algoritmo di ottimizzazione utilizzato per aggiornare i pesi del modello durante l'addestramento.
    #     'adam' è una scelta diffusa per la sua efficienza.
    # - loss='mean_squared_error': Questa è la funzione utilizzata per valutare le prestazioni del modello.
    #     L'obiettivo è minimizzare l'errore quadratico medio tra i prezzi delle azioni previsti e quelli effettivi.
    model.compile(optimizer=ottimizzatore, loss=metrica)


    # Addestra il modello
    # - x_train: contiene i dati di addestramento in input (sequenze dei prezzi azionari passati)
    # - y_train: contiene i valori target per l'addestramento (i prezzi azionari effettivi)
    # - batch_size=1: Il numero di campioni elaborati prima dell'aggiornamento dei pesi del modello
    # - epochs=1: Il numero di volte in cui l'intero set di dati di addestramento viene passato attraverso il modello
    print("Inizio addestramento...")
    model.fit(x_train, y_train, batch_size=n_batch, epochs=n_epoche)
    print("Addestramento completato!")

    # Creazione del Dataset di Test: un nuovo array  che contiene i valori scalati nell'ultimo periodo
    test_data = scaled_data[train_index - GG_PREVISIONE: , :]

    # Creazione dei dataset x_test and y_test
    x_test = []
    y_test = []

    for i in range(GG_PREVISIONE, len(test_data)):
        x_test.append(test_data[i-GG_PREVISIONE:i, 0])
        y_test.append(test_data[i, 0])        

    # Converte i dati in un numpy array
    x_test = np.array(x_test)

    # Reshape dei dati per convertire x_test in un array NumPy e quindi adattarli nel formato tridimensionale richiesto dal modello LSTM.
    # La forma è: numero di campioni, intervalli temporali, numero di feature
    # In questo caso il numero di feature è 1 perché utilizziamo solo il prezzo di chiusura come input.
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Previsioni dal modello a partire dai dati x_test passati al modello LSTM
    # Le previsioni rappresentano i prezzi delle azioni future
    print("Inizio Predizione...")        
    y_prev = model.predict(x_test)
    test_rmse = np.sqrt(np.mean(((y_prev - y_test) ** 2)))
    print("Test Loss: %.4f" % test_rmse)
    print("Predizione completata!")    

    # Poiché il modello è stato addestrato su dati scalati, anche le previsioni sono scalate.
    # Con questa istruzione si utilizza lo scaler per riconvertire le previsioni ai loro valori originali, non scalati.
    y_prev = scaler.inverse_transform(y_prev)

    # Usiamo la metrica dell'errore quadratico medio (RMSE) per la valutazione del nostro modello.
    # Misura la differenza media tra i prezzi delle azioni previsti (previsioni) e i prezzi delle azioni effettivi (y_test).
    # In generale, più basso è l'RMSE, migliore è l'accuratezza del modello
    y_real = np_dataset[train_index:, :]
    test_rmse = np.sqrt(np.mean(((y_prev - y_real) ** 2)))
    print("REAL RMSE: %.4f" % float(test_rmse))
    
    # Se la modalità è 'GRAFICA', si disegna il grafico finale delle previsioni
    if (MODALITA == 'GRAFICA'):
        plot_grafico_finale(tk_azione, df_dataset, y_prev, salva=verbose)

