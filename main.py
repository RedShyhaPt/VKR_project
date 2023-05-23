from tkinter import *
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from tkinter.ttk import Combobox
import pandas as pd
import requests
import json
from IPython.display import clear_output

def comp():
    response = requests.get('http://localhost:5000/api/companies')
    s = json.loads(response.content)
    for ticker in s['tickers']:
        tickerCompany.append(ticker['ticker'])

def plot():
    clear_output(wait=True)
    name = combo.get()
    response = requests.get('http://localhost:5000/api/company/' + name)
    graf = json.loads(response.content)
    date = []
    value = []
    for info in graf['quotes']:
        date.append(info['date_'])
        value.append(info['open_'])
    rezult.configure(text=graf['suggest'])
    fig = Figure(figsize=(30, 10),  dpi=60)
    fig.clear()
    fig = Figure(figsize=(30, 10), dpi=60)
    df = pd.DataFrame({'Date': date, 'Val': value})
    plot1 = fig.add_subplot(111)
    plot1.plot(df['Date'], df['Val'])
    plot1.set_title("Анализ данных")
    plot1.set_xlabel('Дата')
    plot1.set_ylabel('Показатели')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.flush_events()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()

window = Tk()
window.title("Предсказание волатильности ценных бумаг компании на бирже")
window.geometry("1000x500")
lbl = Label(master=window, text="Выберите компанию:", font=("Arial Bold", 12))
lbl.pack(anchor='nw', padx=10, pady=10)
combo = Combobox(master=window, height=25)
tickerCompany = []
comp()
combo['values'] = (tickerCompany)
combo.current(0)
combo.pack(anchor='nw', fill=X, padx=[10, 10])
plot_button = Button(master=window,
                     command=plot,
                     height=2,
                     width=30,
                     text="Показать результаты")
plot_button.pack(anchor='nw', padx=10, pady=10)
lbl = Label(master=window, text="Вывод: ", font=("Arial Bold", 12))
lbl.pack(anchor='nw')
rezult = Label(master=window, text="", font=("Arial Bold", 12))
rezult.pack(anchor='nw')
window.mainloop()
