import pandas as pd

df_A = pd.read_csv('dataset_2016.csv')
df_B = pd.read_csv('dataset_2017.csv')
df_1 = pd.read_csv('test_2016.csv')
df_2 = pd.read_csv('test_2018.csv')
df_3 = pd.read_csv('val_2016.csv')
df_4 = pd.read_csv('val_2018.csv')

df_A = df_A[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
df_B = df_B[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
df_1 = df_1[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
df_2 = df_2[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
df_3 = df_3[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]
df_4 = df_4[['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']]

df_C = pd.concat([df_A, df_B, df_1, df_2, df_3, df_4], ignore_index=True)

df_C.to_csv('main_dataset.csv', index=False)

print("File CSV đã được lưu thành công vào 'main_dataset.csv'.")
