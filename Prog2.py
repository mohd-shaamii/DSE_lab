import numpy as np

#create arrays
a=np.array([1,2,3,4,5])
b=np.array([6,7,8,9,10])

#basic operations
print("Array a: ",a)
print("Array b: ",b)
print("Sum of arrays a&b : ",np.add(a,b))
print("Difference of arrays a&b : ",np.subtract(a,b))
print("Product of arrays a&b : ",np.multiply(a,b))
print("Division of arrays a&b : ",np.divide(a,b))
print("Square root of array a: ",np.sqrt(a))
print("Exponential of array a: ",np.exp(a))

#aggregation operations
print("Minimum value of array a: ",np.min(a))
print("Maximum value of array b: ",np.max(b))
print("Mean of array a: ",np.mean(a))
print("Sandard Deviation of array b: ",np.std(b))
print("Sum of all the elements in array a: ",np.sum(a))

#reshaping arrays
c=np.array([[1,2],[3,4],[5,6]])
print("Array c:")
print(c)
print("Reshaped array c(2 rows, 3 columns):")
print(np.reshape(c,(2,3)))

#transposing arrays
d=np.array([[1,2,3],[4,5,6]])
print("Array d:")
print(d)
print("Transposed array d:")
print(np.transpose(d))
