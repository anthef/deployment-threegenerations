import pandas as pd
df = pd.read_excel("data/Dataset Online Retail.xlsx")
df.to_csv("data/retail_data.csv", index=False)