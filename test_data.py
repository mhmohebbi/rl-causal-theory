from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

print(X.head())
  
# metadata 
# print(adult.metadata) 
  
# variable information 
# print(adult.variables) 
