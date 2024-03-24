import pandas as pd

df = pd.read_csv('stats.csv', sep=',', header=0)


print("Dati del punto minimo: ")

print(df['function'].min())
print(df['function'].idxmin())

print(df.iloc[df['function'].idxmin()])


print("Dati del punto massimo: ")

print(df['function'].max())
print(df['function'].idxmax())

print(df.iloc[df['function'].idxmax()])