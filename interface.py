import tkinter as tk
import numpy as np
import pandas as pd
import holoviews as hv
hv.extension('bokeh' , 'matplotlib')
from tkinter import ttk


df = pd.read_csv('output.csv', encoding ="ISO-8859-1")
print(df.shape)

scatter = hv.Scatter(df,'compound', 'neg')

'''
class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

       #]\ <create the rest of your GUI here>

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Mozilla Feedback Analyzer 9000')
    root.geometry('500x500')
    #MainApplication(root).pack(side="top", fill="both", expand=True)

    #rows in our app
    rows = 0
    while rows < 50:
         root.rowconfigure(rows, weight=1)
         root.columnconfigure(rows, weight=1)
         rows += 1
    #Create notebook object and attach it to root
    nb = ttk.Notebook(root)
    nb.grid(row=1, column=0, columnspan=50, rowspan=49, sticky='NESW')

    trendingIssues = ttk.Frame(nb)
    nb.add(trendingIssues, text="Trending Issues")

    commonIssues = ttk.Frame(nb)
    nb.add(commonIssues, text="Common Issues")


    searchTab = ttk.Frame(nb)
    nb.add(searchTab, text="Search Issues")
    root.mainloop()
'''