import pandas as pd

df = pd.read_csv('stats.csv', sep=',', header=0)


print("Dati del punto minimo: ")

print(df['total_accuracy'].min())
print(df['total_accuracy'].idxmin())

print(df.iloc[df['total_accuracy'].idxmin()])


print("Dati del punto massimo: ")

print(df['total_accuracy'].max())
print(df['total_accuracy'].idxmax())

print(df.iloc[df['total_accuracy'].idxmax()])