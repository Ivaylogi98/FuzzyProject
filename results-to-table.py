import os
import pandas as pd

files = os.listdir('./output')
# print(files)

results = []
for fname in files:
    if 'fuzzy' in fname:
        with open('./output/' + fname, 'r') as f:
            val = f.readline().split('(')[0]
            k = fname.split('_')[2].split('=')[1]
            m = fname.split('_')[3].split('=')[1]
            rand = 10
            if 'rand' not in fname:
                m = m.split('.txt')[0]
            else:
                rand = fname.split('_')[4].split('=')[1].split('.')[0]
            results.append([int(k), float(m), int(rand), float(val)])

df_list = pd.DataFrame(results, columns=['k','m','rand','val'])
unique_k = set(sorted(df_list.loc[:, 'k']))
unique_m = set(sorted(df_list.loc[:, 'm']))
print(unique_k, unique_m)

table = pd.DataFrame(index=unique_k, columns=unique_m)

for row in results:
    table.at[row[0], row[1]] = row[3]

print(table)


table.to_excel('./output/table.xlsx')