import qiskit as q
import numpy as np
import random as ra
#import pandas as pd

# this is an attempt of implementing a quantum reservoir on quantum circuits
# for details refer to https://www.researchgate.net/publication/345708326_Quantum_reservoir_computing_a_reservoir_approach_toward_quantum_machine_learning_on_near-term_quantum_devices
# pima-indians-diabetes.data.csv is a commmonly used database to predict diabetes risk, this file must be in the same directory as this code 

#from sklearn.linear_model import LinearRegression - not used currently

import matplotlib.pyplot as plt
# Tell matplotlib that we are in an Ipython notebook
%matplotlib inline



def normData(dataset):               # normalize data to yield figure between -1 and 1
    return 2*((dataset - np.min(dataset))/(np.max(dataset) - np.min(dataset)))-1

def ranAn():
    return ra.random()*2*np.pi # random angle between 0-2 pi degrees 
                            # is equivalent to a full rotation on the Bloch sphere


def gradientDes(y, w, b, out, eta):

    # linear regression function: y_pred=out*w+b, y is actual result
    # calculate derivative of squared error/ cost function
    
    y_pred=np.sum(out*w)+b    #  forward pass
    
    dC=2*(y_pred-y)
    dW=dC*out                 # calculate derivative of weight vector
    
    db=dC                     # calculate derivative of bias
    
    n=len(out)
    
    cost=1/2*np.sum(np.square(y_pred-y))/n

    # calculate changes

    w=w-eta*dW/n
    b=b-eta*db/n

    return cost,w,b    


def get_probability_distribution(result):
    
        probs={}
        
        states=np.zeros(stateN)
        
        for keys in result: 
            probs[keys]=result[keys]/NUM_SHOTS
            states[int(keys.replace(" ",""),2)]=probs[keys]   # strip empty spaces off binary bitstring and convert to decimal
        
        return states #probs.keys(),probs.values()

    

    
#main training parameters   
       
epochs=10
samples=300       # number of training samples
NUM_SHOTS=10000
Qnum=8            # number of Qbits used for training
Qtotal=12          # total number of Qbits

c=Qnum          # number of classical bits for measurements less or equal than total Qbits
stateN=2**c       # number of quantum states that will be measured



# read in Pima Indians csv file - must be in the same directory as this program

dataset = np.loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
xt = dataset[:,0:8]
yt = dataset[:,8]

for i in range(8):               # normalise each of the 8 columns of data
    xt[:,i]=normData(xt[:,i])
    
    
#print(xts,yts) checking normalised data

# intializing fixed random values for the reservoir

ranVal=np.empty(shape=(Qtotal,12,2))

for i in range(Qtotal):            # intializing reservoir with random values which stay fixed over all iterations
    for n in range(12):            # there are 12 elements; 4 times rx,rz,rx elements
        for r in range(2):         # this is repeated
            ranVal[i][n][r]=ranAn()

def ResCircuit(tr):    
    
    qc=q.QuantumCircuit(Qtotal,c)
    
    def initializeCircuit():
        
        for i in range(Qnum):               # these are the Qbits which are initialized with a parameterized value
                 
            qc.ry(np.tanh(xt[tr,i]),i) 
        
    def ConnectCircuit(a,b,i,r):
        
        
        qc.cz(a,b)
        
        qc.rx(ranVal[i][0][r],a)
        qc.rz(ranVal[i][1][r],a)
        qc.rx(ranVal[i][2][r],a)
    
        qc.rx(ranVal[i][3][r],b)
        qc.rz(ranVal[i][4][r],b)
        qc.rx(ranVal[i][5][r],b)
    
        qc.cz(a,b)
        
        qc.rx(ranVal[i][6][r],a)
        qc.rz(ranVal[i][7][r],a)
        qc.rx(ranVal[i][8][r],a)
    
        qc.rx(ranVal[i][9][r],b)
        qc.rz(ranVal[i][10][r],b)
        qc.rx(ranVal[i][11][r],b)
    
        
    initializeCircuit()


    for r in range(2):
        for i in range(Qtotal-1):
            a=0
            b=i+1
            ConnectCircuit(a,b,i,r)

    #qc.measure_all()
    #q0=qc.measure(0,0)
    #q1=qc.measure(1,1)
    #q2=qc.measure(2,2)
    #q3=qc.measure(3,3)
    
    qi=np.array(c)
    for i in range(c):
        qc.measure(i,i)


    #display(qc.draw(output="mpl"))


    # Drawing the histogram

    backend = q.Aer.get_backend('qasm_simulator')
    job = q.execute(qc, backend, shots=NUM_SHOTS)
    result = job.result().get_counts(qc)

    
    return get_probability_distribution(result)

    #fractionsOfCounts(counts)
    
    #graph = q.visualization.plot_histogram(counts)
    
    #display(graph)


counter=0  
x=[]
y=[]
w=np.random.rand(stateN)
b=0.5
    
for e in range(epochs): 
       
    for tr in range(samples):  # looping over all training samples
        
        probs=ResCircuit(tr)
        #print(probs)
        cost,w,b=gradientDes(yt[tr],w,b,probs, 0.001)
        
        counter+=1
        print(counter," ",cost)
        
        x.append(counter)
        y.append(cost)
        
#plt.xlabel("Iterations")              # plotting the progress of convergence 
#plt.ylabel("Cost function")        
#plt.plot(x,y)        
#plt.show()

correctPosRes=0                       # calculating sensitivity and specificity of predictions
correctNegRes=0
numberPos=0
numberNeg=0

for n in range(samples+1,767):        # database ends at len(xt)
    probs=ResCircuit(n)
    if yt[n]==0:
        numberNeg+=1
        if np.sum(probs*w)+b<.5:
            correctNegRes+=1
    else:
        numberPos+=1
        if np.sum(probs*w)+b>.5:
            correctPosRes+=1

sen=correctPosRes/numberPos       
spec=correctNegRes/numberNeg
   
print("sensitivity= {0},and specificity={1}".format(sen,spec))


        
