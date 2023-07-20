import pandas as pd
import csv

input_file = open("Synthetic_entries.csv","r+")
reader_file = csv. reader(input_file)
value = len(list(reader_file))
value = value - 1

with open('C:/Users/HarshGupta/Desktop/size.txt', 'w') as f:
    print(value, file=f)
