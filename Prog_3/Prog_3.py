import pandas as pd
df=pd.read_csv('dataa.csv')

print("First few rows: ")
print(df.head())

print("\nSummary stats")
print(df.describe())

filtered_data=df[df['Age']>30]
print("\nFiltered data(Age>30)")
print(filtered_data)

sorted_data=df.sort_values(by='Salary',ascending=False)
print("\nSorted data(by salary):")
print(sorted_data)

df["Bonus"]=df['Salary']*0.1
print("\nData with new column (Bonus): ")
print(df)

df.to_excel('output.xlsx',index=False)
print("\nData written to output.xlsx")


# First few rows: 
#    YearsExperience   Age  Salary
# 0              1.1  21.0   39343
# 1              1.3  21.5   46205
# 2              1.5  21.7   37731
# 3              2.0  22.0   43525
# 4              2.2  22.2   39891

# Summary stats
#        YearsExperience        Age         Salary
# count        30.000000  30.000000      30.000000
# mean          5.313333  27.216667   76003.000000
# std           2.837888   5.161267   27414.429785
# min           1.100000  21.000000   37731.000000
# 25%           3.200000  23.300000   56720.750000
# 50%           4.700000  25.000000   65237.000000
# 75%           7.700000  30.750000  100544.750000
# max          10.500000  38.000000  122391.000000

# Filtered data(Age>30)
#     YearsExperience   Age  Salary
# 22              7.9  31.0  101302
# 23              8.2  32.0  113812
# 24              8.7  33.0  109431
# 25              9.0  34.0  105582
# 26              9.5  35.0  116969
# 27              9.6  36.0  112635
# 28             10.3  37.0  122391
# 29             10.5  38.0  121872

# Sorted data(by salary):
#     YearsExperience   Age  Salary
# 28             10.3  37.0  122391
# 29             10.5  38.0  121872
# 26              9.5  35.0  116969
# 23              8.2  32.0  113812
# 27              9.6  36.0  112635
# 24              8.7  33.0  109431
# 25              9.0  34.0  105582
# 22              7.9  31.0  101302
# 21              7.1  30.0   98273
# 19              6.0  29.0   93940
# 20              6.8  30.0   91738
# 17              5.3  27.0   83088
# 18              5.9  28.0   81363
# 15              4.9  25.0   67938
# 16              5.1  26.0   66029
# 8               3.2  23.3   64445
# 10              3.9  23.9   63218
# 14              4.5  25.0   61111
# 6               3.0  23.0   60150
# 9               3.7  23.6   57189
# 13              4.1  24.0   57081
# 12              4.0  24.0   56957
# 5               2.9  23.0   56642
# 11              4.0  24.0   55794
# 7               3.2  23.3   54445
# 1               1.3  21.5   46205
# 3               2.0  22.0   43525
# 4               2.2  22.2   39891
# 0               1.1  21.0   39343
# 2               1.5  21.7   37731

# Data with new column (Bonus): 
#     YearsExperience   Age  Salary    Bonus
# 0               1.1  21.0   39343   3934.3
# 1               1.3  21.5   46205   4620.5
# 2               1.5  21.7   37731   3773.1
# 3               2.0  22.0   43525   4352.5
# 4               2.2  22.2   39891   3989.1
# 5               2.9  23.0   56642   5664.2
# 6               3.0  23.0   60150   6015.0
# 7               3.2  23.3   54445   5444.5
# 8               3.2  23.3   64445   6444.5
# 9               3.7  23.6   57189   5718.9
# 10              3.9  23.9   63218   6321.8
# 11              4.0  24.0   55794   5579.4
# 12              4.0  24.0   56957   5695.7
# 13              4.1  24.0   57081   5708.1
# 14              4.5  25.0   61111   6111.1
# 15              4.9  25.0   67938   6793.8
# 16              5.1  26.0   66029   6602.9
# 17              5.3  27.0   83088   8308.8
# 18              5.9  28.0   81363   8136.3
# 19              6.0  29.0   93940   9394.0
# 20              6.8  30.0   91738   9173.8
# 21              7.1  30.0   98273   9827.3
# 22              7.9  31.0  101302  10130.2
# 23              8.2  32.0  113812  11381.2
# 24              8.7  33.0  109431  10943.1
# 25              9.0  34.0  105582  10558.2
# 26              9.5  35.0  116969  11696.9
# 27              9.6  36.0  112635  11263.5
# 28             10.3  37.0  122391  12239.1
# 29             10.5  38.0  121872  12187.2

# Data written to output.xlsx
