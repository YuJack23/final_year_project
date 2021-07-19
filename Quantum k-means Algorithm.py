"""
Quantum k-Means Algorithm (Algorithm 4.5)
"""
##List of Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil, log, sqrt
from qiskit import Aer, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
qasm_sim = Aer.get_backend('qasm_simulator') # To use local qasm simulator
 
## Dataset
dataset = pd.read_csv('Mall_Customers.csv')

## Data Preprocessing
def min_max_rescaling(dataset):
    new_dataset = (dataset-np.min(dataset,axis=0))/(np.max(dataset,axis=0)-np.min(dataset,axis=0))
    return new_dataset

def undo_rescaling(data,dataset):
    new_data = data*(np.max(dataset,axis=0)-np.min(dataset,axis=0))+np.min(dataset,axis=0)
    return new_data

dataset = np.array(dataset.iloc[:,3:]).astype(float) # select last 2 columns 
X = min_max_rescaling(dataset) #rescale dataset

## Functions
# Canoncial mapping in Algorithm 4.3
def concatenate(data,centres):
    
    centres = np.array([centres])
    lenX,N = data.shape
    n = ceil(log(N,2))+1

    result = np.zeros((lenX,2**n))
    result[:,:N] = data[:,:]
    result[:,(2**(n-1)):(2**(n-1)+N)] = centres
   
    return result

# Distance Calculation using Destructive Interference (Algorithm 4.3)
def Q_distance(data,centres,iteration=20000): 

    N = data.shape[1]
    n = ceil(log(N,2))+1
    mat = concatenate(data,centres) 
    
    # Step 1: Normalisation & # Step 2: Initialisation
    norm = np.linalg.norm(mat,axis=1)
    initial_states = mat/norm[:,None]

    # Step 3: Apply H
    dist = np.zeros([initial_states.shape[0],1])
    
    for i,state in enumerate(initial_states):

        qr = QuantumRegister(n)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr,cr)               # Create a quantum circuit with n qubits and a classical register
        qc.initialize(state,[qr[::-1]])          
        qc.h(qr[0])                              
        qc.measure(qr[0],cr[0])                  # Perform measurement on first qubit 
      
        # Step 4: Distance Calculation & Step 5: Repetition
        qobj = assemble(qc, shots=iteration)
        counts = qasm_sim.run(qobj).result().get_counts()
        if len(counts) == 1:
            dist[i] = 0
        else:
            dist[i] = norm[i]*sqrt(2*(counts["1"]/iteration))
        
        # print(i)
        
    return dist 

# Initialisation Method of k-means++ Algorithm (Algorithm 4.4) 
def initialise_centres(data,K):
    
    # Step 1: Select First Centre 
    i = np.random.choice(data.shape[0])
    centres = np.array([data[i, :]])
    
    # Step 5: Repetition
    for k in range(0,K):
        # Step 2: Distance Calculation
        if k == 0:
            distance = Q_distance(data, centres[k])
            prob = distance/np.sum(distance)
            
        else:
            distance_1 = Q_distance(data, centres[k])
            distance = np.concatenate((distance,distance_1),axis=1)
            
            min_distance = np.min(distance, axis = 1)
            
            # Step 3: Find Probability Distribution 
            prob = min_distance/np.sum(min_distance)
            
        # Step 4: Select New Centre
        if k != (K-1):
            centres_new = data[np.random.choice(data.shape[0], p = prob.flatten())]  
            centres = np.append(centres, np.array([centres_new]), axis=0)

    return distance, centres 

# Find nearest cluster centre
def nearest_cluster(distances):
    
    assigned = np.zeros(len(distances))
    
    for n,i in enumerate(distances):
        # print(n)
        assigned[n] = np.argmin(i)
        
    return assigned
    
# Compute new cluster centres
def find_new_centres(data,assigned,k,n):
     
    new_centres = np.zeros([k,n])
    
    for i in range(k):
        new_centres[i,:] = np.average(data[assigned==i],axis=0)
        
    return new_centres

## Quantum k-means Algorithm (Algorithm 4.5)
eta = 0.001
min_clusters = 2
max_clusters = 10
totalSSE = []
total_iterations = []

### Step 1: Set Number of Clusters & Allocation of Initial Centres
all_distances, all_centres = initialise_centres(X,max_clusters)

for k in range(min_clusters,max_clusters+1):

    distances = all_distances[:,0:k]
    centres = all_centres[0:k]

    new_centres = np.zeros([centres.shape[0],centres.shape[1]])+10000
    count = 0
    
    while np.linalg.norm(centres - new_centres) > eta:
    
        count += 1     
        print("------"," Iteration: ",count,"------")
        
        ### Step 2: Distance Calculation & Choosing Nearest Centre
        if count != 1:
            centres = new_centres
            for i in range(k):
                if i == 0:
                    distances = Q_distance(X, centres[i])
                else:
                    distances_1 = Q_distance(X, centres[i])
                    distances = np.concatenate((distances,distances_1),axis=1)
        
        assigned = nearest_cluster(distances)
        
        ### Step 3: Recompute Centres
        new_centres = find_new_centres(X,assigned,k,centres.shape[1])
    
        # Plot Scatterplot
        plt.scatter(X[:,0],X[:,1], c=assigned, cmap='plasma')
        plt.xlabel('Rescaled Annual Income')
        plt.ylabel("Rescaled Spending Score")
        plt.scatter(new_centres[:,0],new_centres[:,1],s=200,color='black')
        plt.show()
    
        print(np.linalg.norm(centres - new_centres))
        
        # Step 4: Convergence
        if (np.linalg.norm(centres - new_centres) > eta):
            print("Not Converged")
        else:
            print("Converged")
        
    # Calculate SSE after convergence
    SSE = np.zeros(k)
    for i in range(k):    
        SSE[i] = sum((X[assigned==i][:,0]-centres[i][0])**2+(X[assigned==i][:,1]-centres[i][1])**2)
    totalSSE.append(sum(SSE))
    total_iterations.append(count)

# Plot SSE_k verses k
plt.plot(list(range(min_clusters,max_clusters+1)),totalSSE,marker='o')
plt.title('SSE for Different Values of k',fontsize=15)
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()

# Plot scatterplot with k=5 & unscaled feature vectors (set min_clusters = max_clusters = 5 for the loop above)
plt.scatter(dataset[:,0],dataset[:,1],c=assigned, cmap='plasma')
plt.scatter(undo_rescaling(centres,dataset)[:,0],undo_rescaling(centres,dataset)[:,1],s=200,color='black')
plt.xlabel('Annual Income (in US k$)')
plt.ylabel("Spending Score")