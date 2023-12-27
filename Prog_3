import pandas as pd
df=pd.read_csv('data.csv')

print("First few rows: ")
print(df.head())

print("\nSummary stats")
print(df.describe())

filtered_data=df[df['age']>30]
print("\nFiltered data(Age>30)")
print(filtered_data)

sorted_data=df.sort_values(by='salary',ascending=False)
print("\nSorted data(by salary):")
print(sorted_data)

df["Bonus"]=df['salary']*0.1
print("\nData with new column (Bonus): ")
print(df)

df.to_excel('output.xlsx',index=False)
print("\nData written to output.xlsx")



(* First few rows: 
        Last      First                   Position  Salary  Hired
0        Lew      Allen         City Administrator  295000   2004
1    Sessoms      Allen                  President  295000   2008
2  HENDERSON  KAYATANYA  Superintendent Of Schools  275000   2007
3     Lanier      Cathy                      Chief  230743   1990
4      Arons    Bernard      Medical Officer Psych  206000   2008

Summary stats
              Salary         Hired
count   33424.000000  33424.000000
mean    56365.113631   2000.120542
std     33908.277908      9.817771
min         7.000000   1954.000000
25%     38060.250000   1993.000000
50%     58759.000000   2003.000000
75%     78743.000000   2008.000000
max    295000.000000   2011.000000 *)
