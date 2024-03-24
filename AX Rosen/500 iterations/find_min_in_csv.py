import pandas as pd

df = pd.read_csv('rosenbrock.csv', sep=',', header=0)

print(df['function'].min())
print(df['function'].idxmin())


print(df.iloc[df['function'].idxmin()])
