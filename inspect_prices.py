import pickle

with open('embeddings_cache/df_cache.pkl', 'rb') as f:
    cached = pickle.load(f)
df = cached['df']

has_price = df[df['price'] != ''].index.tolist()
has_orig = df[df['original price'] != ''].index.tolist()
has_both = set(has_price) & set(has_orig)
has_neither = set(range(len(df))) - set(has_price) - set(has_orig)
print(f'Has price only: {len(set(has_price) - set(has_orig))}')
print(f'Has original price only: {len(set(has_orig) - set(has_price))}')
print(f'Has both: {len(has_both)}')
print(f'Has neither: {len(has_neither)}')
print()

print('=== PRICE COLUMN BY PRODUCT TYPE ===')
for pt in df['product type'].unique():
    subset = df[df['product type'] == pt]
    p1 = len(subset[subset['price'] != ''])
    p2 = len(subset[subset['original price'] != ''])
    print(f'{pt}: price={p1}/{len(subset)}, original price={p2}/{len(subset)}')

print()
print('=== SAMPLE: cricket bat product ===')
sports = df[df['product type'] == 'sports']
bat = sports[sports['product name'].str.contains('cricket bat', na=False)]
if len(bat) > 0:
    row = bat.iloc[0]
    for col in df.columns:
        val = row[col]
        if val != '':
            print(f'  {col}: {val}')

print()
print('=== ALSO CHECK RAW EXCEL ===')
import pandas as pd
all_sheets = pd.read_excel('Products Data.xlsx', sheet_name=None)
for name, sheet in all_sheets.items():
    cols = [c for c in sheet.columns if 'price' in str(c).lower()]
    print(f'Sheet "{name}": price columns = {cols}, rows = {len(sheet)}')
