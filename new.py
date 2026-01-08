from pathlib import Path
import pandas as pd

csv_path = Path(r"C:\Users\dhara\Desktop\python\data sets\AB_NYC_2019.csv")

df = pd.read_csv(csv_path)

print(df.head())
print(df.info())
print(df.describe(include='all'))
pd.set_option('display.float_format', '{:.0f}'.format)

print(df.isna().sum())
print(df.duplicated().sum())
print("done")
print("THIS IS THE END OF FILE")
# show full output (optional)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

