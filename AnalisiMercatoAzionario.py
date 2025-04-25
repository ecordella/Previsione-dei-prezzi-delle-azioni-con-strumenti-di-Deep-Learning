"""
Titolo: Analisi del Mercato Azionario
Autore: Enrico Cordella
Data: 22/04/2025

## Descrizione
Questo programma analizza i dati di mercato azionario per 4 diverse aziende quotate sul mercato azionario NASDAQ (è aperto anche a titoli quotati su Borsa Italiana).
Utilizza la libreria yfinance per scaricare i dati storici delle azioni e calcolare i rendimenti e i rischi associati.
Il programma permette di visualizzare i grafici dei prezzi e dei volumi delle azioni, calcolare e visualizzare la distribuzione dei rendimenti giornalieri
e identificare il miglior titolo in base all'Indice di Sharpe.

## Requisiti
 - Python 3.x
 - Librerie: pandas, numpy, matplotlib, yfinance, scipy

## Funzionalità
 1. Scarica i dati storici delle azioni da Yahoo Finance.
 2. Visualizza i grafici dei prezzi e dei volumi delle azioni.
 3. Calcola e visualizza la distribuzione dei rendimenti giornalieri.
 4. Identifica il miglior titolo in base all'Indice di Sharpe. 
 6. Visualizza la matrice rischio-rendimento.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from scipy.stats import norm

# Per leggere i dati finanziari
import yfinance as yf

# Per gestire la generazione delle date
from datetime import datetime



def scarica_dati_azioni(lst_ticker_azioni, inizio, fine):
    """
    Funzione per scaricare i dati delle azioni da Yahoo Finance
    :param lst_ticker_azioni: lista di ticker delle azioni da scaricare
    :param inizio: data di inizio per il download dei dati
    :param fine: data di fine per il download dei dati
    :return: lst_df_azioni: lista di dataframe con i dati delle azioni
    """

    lst_df_azioni = []

    # Per ogni singolo ticker si a yfinance per recuperare i dati dei prezzi delle azioni
    for tk_azione in lst_ticker_azioni:
        df = yf.download(tk_azione, inizio, fine)
        lst_df_azioni.append(df)

    return lst_df_azioni


def plot_prezzi_volumi(lst_df_azioni, salva=False):
    """
    Funzione per il disegno dei grafici dell'andamento dei prezzi e dei volumi delle azioni
    :param df_list: lista di dataframe con i dati delle azioni
    :param salva: se True salva il grafico come file immagine
    :return: None
    """

    # Creazione della figura con la griglia 2x2
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig)

    # Loop sulle aziende per il plot del relativo grafico
    for i in range(len(lst_df_azioni)):

        df_azienda = lst_df_azioni[i]
        nome_azienda = yf.Ticker(df_azienda.columns[0][1]).info['longName']

        df_azienda = df_azienda.reset_index()  # Reset index for plotting

        # Calcolo degli indici di riga e colonna
        row = i // 2
        col = i % 2

        # Creazione della GridSpec per il relativo subplot nella figura principale
        # In ogni singola cella della grigli vi sono due subplot
        inner_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, col], height_ratios=[3, 1], hspace=0.18)

        # Recupero delle informazioni sull'andamento del titolo
        date_prz = df_azienda['Date']
        prz_chiusura = df_azienda['Close']
        volumi = df_azienda['Volume']

        # Il primo grafico è quello dei prezzi
        ax_prezzo = fig.add_subplot(inner_gs[0])
        ax_prezzo.plot(date_prz, prz_chiusura, color='blue', linewidth=1.5, label='Chiusura')
        ax_prezzo.xaxis.label.set_color('grey')
        ax_prezzo.set_axisbelow(True)
        ax_prezzo.tick_params(axis='both')
        ax_prezzo.set_ylabel('Prezzo (in USD)', fontsize=13)
        ax_prezzo.yaxis.tick_right()
        ax_prezzo.yaxis.set_label_position("right")
        ax_prezzo.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)

        # Nel plot dell'andamento del prezzo di aggiunge anche quello delle medie mobili
        medie_mobili = {
            'SMA 50': {'Range': 50, 'Color': 'orange'},
            'SMA 100': {'Range': 100, 'Color': 'green'},
            'SMA 200': {'Range': 200, 'Color': 'red'}
        }
        for mm, mm_info in medie_mobili.items():
            ax_prezzo.plot(
                date_prz, prz_chiusura.rolling(mm_info['Range']).mean(),
                color=mm_info['Color'], label=mm, linewidth=2, ls='--'
        )

        # Disegno del bordo del plot_prezzo
        ax_prezzo.spines['right'].set_color('grey')
        ax_prezzo.spines['right'].set_linewidth(0.5)
        ax_prezzo.spines['bottom'].set_color('grey')
        ax_prezzo.spines['bottom'].set_linewidth(0.5)

        # Il secondo grafico è quello dei volumi
        ax_volumi = fig.add_subplot(inner_gs[1])
        ax_volumi.bar(date_prz, volumi.values.reshape(len(volumi)), width=15, color='#670f74')
        ax_volumi.set_xlabel('Data', fontsize=11)
        ax_volumi.xaxis.label.set_color('grey')
        ax_volumi.set_axisbelow(True)
        ax_volumi.yaxis.tick_right()
        ax_volumi.tick_params(axis='both')
        ax_volumi.yaxis.set_label_position("right")
        ax_volumi.set_ylabel('Volumi (in milioni)', fontsize=11)

        # Disegno del bordo del plot_volumi
        ax_volumi.spines['right'].set_color('grey')
        ax_volumi.spines['right'].set_linewidth(0.5)
        ax_volumi.spines['bottom'].set_color('grey')
        ax_volumi.spines['bottom'].set_linewidth(0.5)

        ax_prezzo.set_title(nome_azienda, size=16)
        ax_prezzo.legend(loc='upper left', bbox_to_anchor=(-0.005, 0.95), fontsize=13)

    if salva:
        # Se non esiste, crea la cartella "img"
        os.makedirs("img", exist_ok=True)

        # Percorso per salvare l'immagine
        nome_file = os.path.join("img", "PrezziVolumi.png")
        fig.savefig(nome_file, dpi=300, bbox_inches='tight')
        print("Grafico salvato come 'prezzi_volumi.png'")
    else:
        plt.show()


def plot_frequenze_rendimenti(lst_df_azioni, salva=False):
    """
    Funzione per il disegno dei grafici delle campane dei rendimemnti delle diverse azioni
    :param lst_df_azioni: lista di dataframe con i dati delle azioni
    :param salva: se True salva il grafico come file immagine    
    :return: None
    """
    rendimenti = pd.DataFrame()

    # Creazione della figura con la griglia 2x2
    fig = plt.figure(figsize=(15, 10))
    
    # Loop sulle aziende per il plot del relativo grafico
    for i in range(len(lst_df_azioni)):

        plt.subplot(2, 2, i+1)

        df_azienda = lst_df_azioni[i]
        ticker = df_azienda.columns[0][1]
        nome_azienda = yf.Ticker(ticker).info['longName']
       
        rendimenti[ticker] = df_azienda['Close'].pct_change() * 100
        df_azienda['Rendimento'] = df_azienda['Close'].pct_change() * 100
        
        np_azienda = df_azienda['Rendimento'].dropna().values
        mean = np.mean(np_azienda, axis=0)
        variance = np.var(np_azienda)
        sigma = np.sqrt(variance)
        minimo = min(np_azienda)
        massimo = max(np_azienda)
        plt.xlim(minimo, massimo)
        plt.hist(np_azienda, bins=100, label='Rendimento', density=True)
        x = np.linspace(minimo, massimo, 100)
        y_mean = 0.35
        plt.plot(x, norm.pdf(x, mean, sigma), color='green', label='Distribuzione', linewidth=1, ls='solid')
        plt.vlines(x=mean, ymin=0, ymax=y_mean, linewidth=1, linestyles='solid', colors='red', label=f'Media ({round(mean,2):+3.2f}%)')
        plt.vlines(x=mean+sigma, ymin=0, ymax=y_mean, linewidth=1, linestyles='solid', colors='orange')
        plt.vlines(x=mean-sigma, ymin=0, ymax=y_mean, linewidth=1, linestyles='solid', colors='orange', label=f'Dev. std (±{round(sigma,2):3.2f})')
        plt.title(nome_azienda, fontsize=16)
        plt.xlabel('Rendimenti giornalieri', fontsize=13)
        plt.ylabel('Frequenza', fontsize=13)
        plt.legend(loc='upper left', bbox_to_anchor=(-0.005, 0.95), fontsize=11)

    # Aggiungi spazio verticale tra i subplot
    plt.subplots_adjust(hspace=0.35)

    if salva:
        # Se non esiste, crea la cartella "img"
        os.makedirs("img", exist_ok=True)

        # Percorso per salvare l'immagine
        nome_file = os.path.join("img", "FrequenzeRendimenti.png")
        
        fig.savefig(nome_file, dpi=300, bbox_inches='tight')
        print("Grafico salvato come 'frequenze_rendimenti.png'")
    else:
        plt.show()


def ottieni_rendimenti(lst_df_azioni):
    """
    Funzione per il recupero dei rendimemnti delle diverse azioni
    :param lst_df_azioni: lista di dataframe con i dati delle azioni
    :return: df_rendimenti
    """
    df_rendimenti = pd.DataFrame()

    # Loop sulle aziende per il plot del relativo grafico
    for i in range(len(lst_df_azioni)):

        df_azienda = lst_df_azioni[i]
        ticker = df_azienda.columns[0][1]
       
        df_rendimenti[ticker] = df_azienda['Close'].pct_change() * 100

    return df_rendimenti


def plot_matrice_rischio_rendimento(df_rendimenti, tk_miglior_titolo, salva=False):
    """
    Funzione per il disegno della matrice rischio-rendimento
    :param df_rendimenti: dataframe dei reindimenti delle azioni
    :param salva: se True salva il grafico come file immagine    
    :return: df_rendimenti
    """

    area = np.pi * 30

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(df_rendimenti.std(), df_rendimenti.mean(), s=area)
    plt.xlim(left=df_rendimenti.std().min()*0.9, right=df_rendimenti.std().max()*1.1)
    plt.ylim(bottom=df_rendimenti.mean().min()*0.9, top=df_rendimenti.mean().max()*1.1) 
    plt.xlabel('Rischio (dev. standard)', fontsize=13)
    plt.ylabel('Rendimento (media)', fontsize=13)

    # Gestione degli assi del grafico
    ax = plt.gca()
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    ax.set_yticklabels(yticks, fontsize=11)  
    ax.set_xticklabels(xticks, fontsize=11)  
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))  

    plt.title(f'Matrice Rischio Rendimento', fontsize=14)

    for tk, sigma, mu in zip(df_rendimenti.columns, df_rendimenti.std(), df_rendimenti.mean()):
        plt.annotate(tk, fontsize=11, xy=(sigma, mu), xytext=(40, 40), textcoords='offset points', ha='right', va='bottom',
                    arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
        if (tk_miglior_titolo == tk):
            ax.scatter(sigma, mu, s=area*2, color='red', marker='*', label=tk_miglior_titolo)

    if salva:
        # Se non esiste, crea la cartella "img"
        os.makedirs("img", exist_ok=True)

        # Percorso per salvare l'immagine
        nome_file = os.path.join("img", "MatriceRischioRendimento.png")

        fig.savefig(nome_file, dpi=300, bbox_inches='tight')
        print("Grafico salvato come 'matrice_rr.png'")
    else:
        plt.show()


def calcola_indice_sharpe(df_rendimenti, tasso_privo_rischio=0.0):
    """
    Calcola l'Indice di Sharpe per ogni azione nel DataFrame dei rendimenti.
    
    :param df_rendimenti: DataFrame con i rendimenti delle azioni (colonne: ticker, righe: rendimenti giornalieri)
    :param tasso_privo_rischio: Tasso di rendimento privo di rischio (default: 0.0)
    :return: dizionario con ticker e relativo Indice di Sharpe
    """
    indici_sharpe = {}

    for ticker in df_rendimenti.columns:
        rendimento_medio = df_rendimenti[ticker].mean()  # Rendimento medio
        rischio = df_rendimenti[ticker].std()  # Deviazione standard (rischio)
        
        # Calcolo dell'Indice di Sharpe
        if rischio != 0:
            sharpe = (rendimento_medio - tasso_privo_rischio) / rischio
        else:
            sharpe = 0  # Evita divisioni per zero
        
        indici_sharpe[ticker] = sharpe

    return indici_sharpe


def ottieni_miglior_titolo(df_rendimenti, stampa=False):
    """
    Funzione per trovare il miglior titolo in base all'Indice di Sharpe"
    :param df_rendimenti: DataFrame dei rendimenti delle azioni
    :param stampa: se True stampa i risultati
    :return: ticker e indice del miglior titolo
    """

    # L'Indice di Sharpe è un ottimo strumento per sintetizzare il rischio e il rendimento di un'azione in un unico valore. 
    # Permette di confrontare diverse azioni e identificare quelle con il miglior rapporto rischio/rendimento. 

    # Calcola l'Indice di Sharpe per ogni azione   
    tasso_privo_rischio = 0.02  # Ad esempio, 2% annuo
    indici_sharpe = calcola_indice_sharpe(df_rendimenti, tasso_privo_rischio)

    # Stampa i risultati
    if(stampa):
        print("Indice di Sharpe per ogni azione:")
        for ticker, sharpe in indici_sharpe.items():
            print(f"{ticker}: {sharpe:.3f}")

    # Trova l'azione con il miglior Indice di Sharpe
    miglior_titolo = max(indici_sharpe, key=indici_sharpe.get)
    indice = indici_sharpe[miglior_titolo]

    if(stampa):
        print(f"L'azione con il miglior Indice di Sharpe è: {miglior_titolo} (Sharpe: {indice:.3f})")

    return miglior_titolo, indice


if __name__ == "__main__":

    # Definizione dei ticker delle azioni e dei nomi delle aziende
    lst_ticker_azioni = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    #lst_ticker_azioni = ['ENEL.MI', 'BMPS.MI', 'A2A.MI', 'TEN.MI']

    # Definizione del periodo di analisi
    inizio = datetime(2020, 1, 1)
    fine = datetime(2025, 3, 31)

    # Con la variabile MODALITA si lavora in tre diversi modi:
    # - 'GRAFICA' per il plot dei i dati
    # - 'ANALISI' per fare solo analisi sui dati
    MODALITA = 'GRAFICA' 
    verbose = True

    # Scarica i dati delle azioni
    lst_df_azioni = scarica_dati_azioni(lst_ticker_azioni, inizio, fine)

    # Plot dei prezzi e volumi delle azioni
    if (MODALITA == 'GRAFICA'):
        plot_prezzi_volumi(lst_df_azioni, salva=verbose)

    # Plot dei grafici frequenze dei rendimenti delle azioni
    if (MODALITA == 'GRAFICA'):
        plot_frequenze_rendimenti(lst_df_azioni, salva=verbose)

    # Recupero della matrice dei rendimenti
    df_rendimenti = ottieni_rendimenti(lst_df_azioni)

    # Scelta del miglior titolo in base all'Indice di Sharpe
    tk_miglior_titolo, indice = ottieni_miglior_titolo(df_rendimenti, stampa=verbose)

    # Plot del grafico di rischio/rendimento
    if (MODALITA == 'GRAFICA'):
        df_rendimenti = plot_matrice_rischio_rendimento(df_rendimenti, tk_miglior_titolo, salva=verbose)
