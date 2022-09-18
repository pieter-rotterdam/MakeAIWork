from tkinter import W

#Sla de input ‘plat’​
#Initialiseer je matrixen​
#Maak een cost function​
#Maak een of meerdere activation function​
#Gebruik numpy​
#Maak je code zo generiek mogelijk, er komt een andere dataset om op uit te proberen​
#Extra uitdaging? Backpropagation + Stochastic Gradient Descent​
#z=x*W
#z= input_nodes * link_weights
"""
from cmath import exp
import math

input_nodes =  ((1,1,1),(1,0,1),(1,1,1)), #0 je wilt uit een set van 2 elementen, alleen eerste gebruiken.
                    ((0,1,0),(1,0,1),(0,1,0)), #0
                    ((0,1,0),(1,1,1),(0,1,0)), #X
                    ((1,0,1),(0,1,0),(1,0,1)), #X je geeft de trainingsset, en de testset zodat je kunt zien hoe goed hij leert
)

link_weights = ((1,1,1),(1,1,1),(1,1,1))

output_nodes =
"""

# create tuples with college id
# and name and store in a list
data=[(1,'sravan'),(2,'ojaswi'),
      (3,'bobby'),(4,'rohith'),
      (5,'gnanesh')]
  
# map with lambda expression to get first element
first_data = map(lambda x: x[0], data)
  
# display data
for i in first_data:
    print(i)
