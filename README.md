# Previsione dei prezzi delle azioni con strumenti di Deep Learning
Progetto realizzato al termine dello "Short Master in Intelligenza Artificiale e Machine Learning - A.A. 2024/25" del Politecnico di Bari.

Questo lavoro è stato realizzato nel corso del mese di aprile 2025 e ha preso in esame l'andamento dei prezzi delle azioni fino al 31 marzo 2025 ovvero fino al penultimo giorno prima dell'introduzione, da parte del presidente degli Stati Uniti Donald Trump, dei dazi tariffari che hanno sconvolto tutte le principali borse mondiali con un impatto anomalo ed eccezionale su tutti gli strumenti finanziari.

Il presente elaborato ha una valenza esclusivamente sperimentale ed ha il solo scopo di dimostrare le competenze acquisite: non ha alcuna validità dal punto di vista dell'analisi finanziaria, né è da prendere in considerazione per attività di trading e/o per investimenti in borsa.

Nel suo complesso il lavoro si compone di 3 programmi, scritti in Python, presenti nella cartella *src*:
1. [**AnalisiMercatoAzionario.py**](#1-Analisi-del-Mercato-Azionario): analizza i dati di mercato azionario per 4 diverse aziende quotate sul mercato azionario NASDAQ (è aperto anche a titoli quotati su Borsa Italiana). Utilizza la libreria yfinance per scaricare i dati storici delle azioni e calcolare i rendimenti e i rischi associati. Il programma permette di visualizzare i grafici dei prezzi e dei volumi delle azioni, calcolare e visualizzare la distribuzione dei rendimenti giornalieri e identificare il miglior titolo in base all'Indice di Sharpe.

2. [**PrevisionePrezzi_Keras.py**](#2-previsione-prezzi-con-keras): scarica i dati storici di un'azione (Apple nella fattispecie) da Yahoo Finance e utilizza la libreria Keras di TensorFlow per implementare un modello LSTM per prevedere i prezzi futuri dell'azione. Il modello viene addestrato sui dati storici e le previsioni vengono visualizzate in un grafico insieme ai valori reali del periodo di previsione.

3. [**PrevisionePrezzy_PyTorch.py**](#3-previsione-prezzi-con-pytorch): utilizza PyTorch per costruire un modello LSTM (Long Short-Term Memory) per la previsione dei prezzi delle azioni. Anche questo modello è addestrato su dati storici delle azioni scaricati da Yahoo Finance.


## 1. Analisi del Mercato Azionario
Per recuperare le informazioni finanziarie relative alle azioni quotate sui principali mercati internazionali è stata usata la libreria yFinance, strumento open source che utilizza le API messe a disposizione da Yahoo! ai soli fini didattici e di ricerca.

Grazie a questa libreria, avendo a disposizione una grande profondità temporale e la massima ampiezza sulle informazioni finanziarie di tutti i mercati mondiali, abbiamo scelto di mettere a confronto quattro aziende e di scaricare i prezzi relativi alle loro azioni nel periodo che va dal 1 gennaio 2020 al 31 marzo 2025.

Per poter recuperare prezzi e volumi di un'azione è fondamentale conoscerne il ticker ovvero un codice alfanumerico univoco che identifica una particolare azione o un altro titolo negoziato in borsa.
Per il nostro progetto abbiamo scelto di analizzare le seguenti 4 aziende americane cosiddette ***Big Tech***:

- **AAPL**: [Apple Inc.](https://www.apple.com/)
- **GOOG**: [Alphabet Inc.](https://www.google.com/)
- **MSFT**: [Microsoft Corporation](https://www.microsoft.com/)
- **AMZN**: [Amazon Inc.](https://www.amazon.com/)

Tramite la libreria yfinance, recupera, per tutte e 4 le azioni interessate, le informazioni principali sull'andamento dei prezzi rispettivamente di:

- Close: chiusura serale
- High: massimo di giornata
- Low: minimo di giornata
- Open: apertura mattutina
- Volume: volumi in termini di numero di azioni scambiate

Tramite la libreria matplotlib, a partire dai dati scaricati allo step precedente, viene creata una griglia 2x2 all'interno della quale verrà disegnato, per ogni azione, un grafico composto da due sezioni: in quella superiore saranno riportati l'andamento dei prezzi e le medie mobili a 50, 100 e 200 giorni; in quella inferiore, invece, saranno riportati i volumi delle azioni scambiate nel corso di ogni singola giornata di contrattazione.


![Andamento Prezzi e Volumi delle azioni analizzate](/img/PrezziVolumi.png)

Quella delle medie mobili è la famiglia degli indicatori tecnici più utilizzata perché consentono di smussare le fluttuazioni erratiche dei prezzi e dare un'indicazione approssimativa dell'andamento dei prezzi.

Dopo aver fatto l'analisi visiva dell'andamento delle azioni è stato approfondito il rischio di mercato delle azioni prese in esame e, quindi, l'andamento delle variazioni giornaliere di ogni singolo titolo.
Tracciando istogrammi e curve di distribuzione normale si comprendono più facilmente i rendimenti tipici delle singole azioni (media) ed il relativo rischio associato (indicato dalla deviazione standard).

![Frequenze Rendimenti giornalieri](/img/FrequenzeRendimenti.png)

Dallo studio visivo dei rendimenti giornalieri delle 4 azioni, si nota facilmente che mediamente tutte e 4 producono un ritorno giornaliero degli investimenti intorno allo 0,1%, ma con un rischio abbastanza diversificato.

Per valutare e confrontare in maniera scientifica la performance di diversi investimenti, tenendo conto del loro livello di rischio, si utilizza l'indice di Sharpe ovvero una misura del rapporto rischio-rendimento (rispetto al tasso privo di rischio) per unità di rischio assunto. 
Per calcolarlo, si utilizza la seguente formula:

$$
IS = (Rp - Rf) / σp
$$

Dove:

&nbsp; &nbsp; Rp = rendimento dell'investimento o del portafoglio

&nbsp; &nbsp; Rf = rendimento del tasso privo di rischio (lo abbiamo impostato a 0)

&nbsp; &nbsp; σp = deviazione standard del rendimento dell'investimento (volatilità)

Dal calcolo matematico degli indici di Sharpe per le 4 azioni analizzate si ottengono i seguenti valori:
 - AAPL: 0.042
 - MSFT: 0.035
 - GOOGL: 0.031
 - AMZN: 0.026

Il programma genera, infine, un grafico relativo alla matrice Rischio-Rendimento di Markovitz, mettendo in evidenza il titolo APPLE poichè presenta il miglior rapporto tra rendimento e rischio.

![Matrice Rischio Rendimento](/img/MatriceRischioRendimento.png)



## 2. Previsione prezzi con Keras
Dopo aver determinato il titolo con il rendimento più conveniente, viene stabilito di considerare i soli prezzi di chiusura delle azioni APPLE dal 1 gennaio 2020 fino al 31 marzo 2025 e di fissare a 50 giorni il periodo di osservazione per la determinazione delle sequenze di input.

Per analizzare e modellare i dati sui prezzi delle azioni, i dati vengono scalati mediante l'estimatore *MinMaxScaler* della libreria *ski-kit learn*: rispetto ad altri scaler, come lo *StandardScaler*, che presuppone una distribuzione normale, il MinMaxScaler è la scelta più adatta perché preserva la forma della distribuzione originale ed è meno sensibile ai valori anomali.

```python
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(np_dataset.reshape(-1, 1))
```

La scaling dei dati è fondamentale per gli LSTM per evitare problemi con gli aggiornamenti del gradiente e per garantire che tutte le feature contribuiscano equamente al processo di apprendimento. I dati scalati rientrano in un intervallo specifico (da 0 a 1), rendendoli più facili da interpretare e utilizzare dal modello.

La preparazione del set di dati si completa con la creazione degli x_train e y_train da mandare in input alla nostra rete neurale ricorrente (RNN) di tipo LSTM: l'idea è quella di creare un set di dati in cui ogni campione di input (feature) è costituito da una sequenza dei 50 intervalli temporali precedenti (GG_PREVISIONE) e la label corrispondente è il valore dell'intervallo temporale immediatamente successivo.

```python

    # Determino l'indice per la costruzione del dataset di training
    train_index = dataset_len - GG_PREVISIONE

    # Creazione del training set a partire dagli scaled_data di lunghezza pari al train_index
    train_data = scaled_data[0:int(train_index), :]

    # ASuddivisione dei dati in x_train e y_train
    x_train = []
    y_train = []

    # Il ciclo itera sui dati di training partendo da GG_PREVISIONE fino alla fine dei dati di training.
    # Il modello utilizza i 50 campioni temporali precedenti come input per prevedere l'i-esimo valore.
    for i in range(GG_PREVISIONE, train_index):

        # Per ogni iterazione, si estrae una sequenza di 50 valori consecutivi dall'array train_data,
        # Questi 50 valori rappresentano le feature di input del modello.
        x_train.append(train_data[i - GG_PREVISIONE : i, 0])

        # Il target value è rappresentato dall'iesimo campione
        y_train.append(train_data[i, 0])

    # Conversione delle liste in numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape dei dati da mandare in pasto al modello
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
```

La scelta del modello di Deep Learning utilizzato per fare previsioni sui prezzi delle azioni è ricaduta sulla RNN (Recurrent Neural Network) di tipo LSTM (Long Short Term Memory) poiché esse sono progettate per catturare le dipendenze a lungo termine nei dati sequenziali: i prezzi delle azioni, infatti, sono influenzati da una serie di fattori storici e presentano una natura temporale, rendendo cruciale l'analisi delle relazioni tra i dati passati e futuri.
Quanto alle LSTM, nello specifico,  queste superano i limiti delle RNN tradizionali che soffrono del problema del "vanish and exploding gradient" grazie alla loro struttura interna composta da celle di memoria e meccanismi di gate (input, forget e output): questi meccanismi consentono alle LSTM di mantenere informazioni rilevanti per periodi di tempo più lunghi e di ignorare quelle meno significative, migliorando così la capacità di modellare serie temporali complesse come i prezzi azionari.

La libreria utilizzata in questo programma è stata Keras di TensorFlow.


```python

    model = Sequential()

    # Aggiunta di un livello di tipo Input per definire la tipologia dei dati passati in input
    input_shape = (x_train.shape[1], 1)
    model.add(Input(shape=input_shape)) 

    # Aggiunta di un livello di tipo LSTM con 128 neuroni
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))

    # Aggiunta di un ulteriore livello di tipo LSTM con 64 unità
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))

    # Aggiunta di uno strato Dense (32 neuroni fully connected) per una elaborazione intermedia di tipo RELU
    model.add(Dense(32, activation='relu'))

    # Aggiunta di un ultimo livello Dense (fully connected) finale con un singolo neurone.
    # Questo livello restituisce il prezzo previsto delle azioni.
    model.add(Dense(1))

    # Configurazione del modello per il training:
    model.compile(optimizer='adam', loss='mean_squared_error')

```


L'addestramento del nostro modello è stata effettuata con dimensione del batch pari ad 1 per un totale di 2 epoche (dalle prove è emerso che la convergenza si raggiunge già solo con 2 epoche).

```python
    model.fit(x_train, y_train, batch_size=1, epochs=2)
```

```python
Inizio addestramento...
Epoch 1/2 - 1217/1217 ━━━━━━━━━━━━━━━━━━━━ 12s 9ms/step - loss: 0.0075    
Epoch 2/2 - 1217/1217 ━━━━━━━━━━━━━━━━━━━━ 11s 9ms/step - loss: 0.0012     
Addestramento completato!
```
La fase di addestramento ha determinato una loss di 0,0012 (valore molto interessante).

La predizione sul dataset di Test (ricavato sugli ultimi 50 campioni messi da parte nella costruzione del dataset di training), invece, ha visto una loss di 0,0732.

```python
    y_prev = model.predict(x_test)
    test_rmse = np.sqrt(np.mean(((y_prev - y_test) ** 2)))
```

```python
Inizio Predizione...
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step
Test Loss: 0.0732
Predizione completata!
```

Dopo aver validato il nostro modello, a questo punto possiamo trasformare le predizioni e riportarle sulla scala dei valori di prezzo originale. A questo punto possiamo misurare il vero valore di RSME pari a 8,9060 (dollari)

```python
    # Utilizzo dello scaler per riconvertire le previsioni (da scalate) ai loro valori originali
    y_prev = scaler.inverse_transform(y_prev)

    # Usiamo la metrica dell'errore quadratico medio (RMSE) per la valutazione reale del nostro modello.
    y_real = np_dataset[train_index:, :]
    test_rmse = np.sqrt(np.mean(((y_prev - y_real) ** 2)))
```

Il programma, per ultimo permette la generazione del grafico che visualizza i risultati delle previsioni sul titolo APPLE a confronto con i dati reali nel periodo preso in considerazione.

![Previsione prezzi APPLE con Keras](/img/Keras_Previsioni_AAPL.png)


La valutazione finale della previsione è tutto sommato buona: dall'analisi visiva l'andamento delle previsioni regge il confronto con i valori reali, soprattutto nella predizione di picchi e discese, mantenendo valida la predizione per quasi tutti i 50 giorni presi in esame.


## 3. Previsione prezzi con Pytorch
Il terzo programma realizzato permette di fare delle previsioni con un modello di rete neurale basato su moduli LSTM messi a disposizione dalla libreria PyTorch.

Anche in questo programma Python il titolo considerato è APPLE ed i prezzi e volumi scaricati con yfinance vanno dal 1 gennaio 2020 al 31 marzo 2025, fissando a 50 i giorni per la previsione finale.

Anche in questo caso, dopo aver rimodulato i dati con il MinMaxScaler, si creano i dataset di Training e di Test, rispettivamente per la fase di addestramento e di predizione finale.

Il modello di rete neurale, invece, è stato reso possibile con l'implementazione di una classe al cui interno sono stati definiti i seguenti layer:
 - Linear (da 1 a 32 neuroni)
 - LSTM (doppio layer da 32 unità)
 - Linear finale (da 32 neuroni di input ad uno solo di output)

```python
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

```
L'addestramento del modello è stato effettuato per 5 epoche, tramite un dataloader con batch_size pari ad uno (dalle diverse prove fatte emerge che non è sufficiente andare oltre le 5 epcohe). L'ottimizzatore scelto è stato Adam; la metrica per la valutazione della loss è la MSE.

```python
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

```
La fase di training ha determinato una loss finale di 0,0249

```python
    Inizio addestramento...
    Epoca 1/5: 100%|████████████████████████| 1217/1217 [00:12<00:00, 100.90batch/s]
    Epoca 1/5 - Loss: 0.0447
    Epoca 2/5: 100%|████████████████████████| 1217/1217 [00:11<00:00, 105.14batch/s]
    Epoca 2/5 - Loss: 0.0239
    Epoca 3/5: 100%|████████████████████████| 1217/1217 [00:11<00:00, 105.21batch/s]
    Epoca 3/5 - Loss: 0.0244
    Epoca 4/5: 100%|████████████████████████| 1217/1217 [00:11<00:00, 104.77batch/s]
    Epoca 4/5 - Loss: 0.0249
    Epoca 5/5: 100%|████████████████████████| 1217/1217 [00:11<00:00, 105.68batch/s]
    Epoca 5/5 - Loss: 0.0249
    Addestramento completato!
```

Abbiamo, quindi, impostato il modello per la predizione...

```python
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred.cpu(), y_test.cpu()).detach().numpy())            
        print("Test Loss: %.4f" % test_rmse)
```
... che ha prodotto una loss sui dati di test pari a 0,0238

```python
    Inizio Predizione...
    Test Loss: 0.0238
    Predizione completata!
```

Per la verifica finale, come per il modello Keras, abbiamo riportato le predizioni sulla scala dei prezzi naturali.


```python
    y_prev = scaler.inverse_transform(y_pred_numpy[0].reshape(-1, 1))

    y_real = np_dataset[train_index:, :]
    test_rmse = np.sqrt(np.mean(((y_prev - y_real) ** 2)))
    print("REAL RMSE: %.4f" % float(test_rmse))
```

Lo scarto RSME reale è pari a 15,9605 (dollari).
```python
REAL RMSE: 15.9605
```

Il programma, alla fine, realizza un grafico mettendo insieme tutti i prezzi del titolo APPLE, sia quelli di training, sia quelli di test (reali) che le previsioni finali.

![Previsione prezzi APPLE con PyTorch](/img/PyTorch_Previsioni_AAPL.png)

Dall'analisi visiva, la valutazione finale della previsione è tutto sommato discreta: le previsioni sono praticamente "sovrapponibili" nel primo periodo. Sono previsti picchi e discese, anche se con una certa distanza in termini di temporali. Il modello ha margini di miglioramento nel medio-lungo periodo.
