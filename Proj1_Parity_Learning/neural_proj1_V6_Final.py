import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl


def NeuronLayer(n_neuron, n_input_to_neuron):  #Function to generate arbitrary nodes in the hidden and output layer
        return 2 * np.random.random((n_input_to_neuron, n_neuron)) - 1

def sigmoid(s): #Sigmoidal activation function
    return 1 / (1 + np.exp(-s))
def D_Sigma(s):
    return s * (1-s)

def trial(s): #Function to give the output of arbitrary input sample for trial and debugging purposes
    return sigmoid(np.dot(sigmoid(np.dot(s[0], layer1)), layer2.T))

def forward(layer1,layer2,s):  #Forward feed
    return np.array([sigmoid(np.dot(sigmoid(np.dot(s,layer1)),layer2.T))])

def backprop(x,y,alpha,eta):  #Backprop feed
    iter = True
    epoch = []
    er_ar = []
    arr = [[] for _ in range(len(eta))]  #Stores all the errors for different learning rate value

    for j in range(len(eta)): #Loop over all the learning rate data
        np.random.seed(1)
        layer1 = NeuronLayer(4, 5)
        layer2 = NeuronLayer(4, 1)
        count = 1
        iter = True
        er_ar = []

        while iter == True: #Loop to check for the termination error
            er = []
            count += 1
            # dm=np.tile(0, (np.shape(layer1)[0],np.shape(layer1)[1]))
            dm1 = np.zeros((np.shape(layer1)[0], np.shape(layer1)[1]))
            dm2 = np.zeros((np.shape(layer2)[0], np.shape(layer2)[1]))
            for i in range(len(x)):  #Loop to go over all the samples and do the backprop algorithm
                out_1 = np.array([sigmoid(np.dot(x[i], layer1))])
                out_2 = sigmoid(np.dot(out_1, layer2.T))

                dk = out_2 * (1 - out_2) * (np.array([y[i]]) - out_2)
                dj = out_1 * (1 - out_1) * layer2 * dk
                dw2 = eta[j] * dk * out_1
                dw1 = eta[j] * np.array([x[i]]).T.dot(dj)

                dw2u = dw2 + np.multiply(alpha, dm2)     #Alpha term for momentum for layer2
                dw1u = dw1 + np.multiply(alpha, dm1)     #Alpha term for momentum for layer1(hidden layer)

                layer2 += dw2u
                layer1 += dw1u
                dm1 = dw1u
                dm2 = dw2u

            out = forward(layer1, layer2, x)
            er = np.abs(y - out)
            er_ar.append(np.max(er))

            if np.max(er) < 0.05:
                iter = False
            print(count, np.max(er))

        arr[j].append(er_ar)
        print("Eta=" + str(eta[j]) + "is converged")
        epoch.append(count)
        print(epoch[j], np.max(er))

    return epoch,arr

if __name__ == "__main__":
    x=np.array([[1,1,1,0,1],[0,1,1,0,1]])
    y=np.array([[1],[0]])

    x = np.array(([0, 0, 0, 0, 1], [1, 0, 0, 0, -1], [0, 1, 0, 0, 1], [0, 0, 1, 0, -1],
                  [0, 0, 0, 1, 1], [1, 1, 0, 0, -1], [1, 0, 1, 0, 1], [1, 0, 0, 1, -1],
                  [0, 1, 1, 0, 1], [0, 1, 0, 1, -1], [0, 0, 1, 1, 1], [1, 1, 1, 0, -1],
                  [1, 1, 0, 1, 1], [1, 0, 1, 1, -1], [0, 1, 1, 1, 1], [1, 1, 1, 1, -1]), dtype=int)

    y = np.array(([0], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0]), dtype=int)


    eta=np.arange(0.05,0.55,0.05)
    #eta = np.array([0.5])

[epoch_0, arr_0]= backprop(x,y,0,eta)
[epoch_5, arr_5]= backprop(x,y,0.5,eta)
[epoch_9, arr_9]= backprop(x,y,0.9,eta)

###################################
plt.figure()
plt.plot(eta,np.array(epoch_0), color='blue', label=r'$\alpha $'+"="+str(0))
plt.scatter(eta,np.array(epoch_0), color='blue')

plt.plot(eta,np.array(epoch_5), color='green', label=r'$\alpha $'+"="+str(0.5))
plt.scatter(eta,np.array(epoch_5), color='green')

plt.plot(eta,np.array(epoch_9), color='red', label=r'$\alpha $'+"="+str(0.9))
plt.scatter(eta,np.array(epoch_9), color='red')


plt.xlabel(r'$\eta $')
plt.ylabel("Number of Epochs")
plt.title("Comparing Epoch No. vs  " + r'$\eta $')
plt.legend()
#plt.gca().set_xlim(right=max(eta))

#plt.show()
plt.savefig('eta_epoch.pdf')


###################################
plt.figure()
plt.scatter(eta,np.array(epoch_0,dtype=np.float)/np.array(epoch_9,dtype=np.float), color='blue', label=r'$\alpha $'+"="+str(0))


plt.xlabel(r'$\eta $')
plt.ylabel(r'$\frac{\text{Epochs}_{\alpha=0}}{\text{Epochs}_{\alpha=0.9}} $')
plt.title("Convergence Speed (rate) vs  " + r'$\eta $')
#plt.legend()
#plt.gca().set_xlim(right=max(eta))

#plt.show()
plt.savefig('rate.pdf')


#################################


plt.figure()

plt.scatter(np.linspace(1,len(np.array(arr_0[0][0])),len(np.array(arr_0[0][0]))),np.array(arr_0[0][0]), color='blue', label=r'$\alpha $'+"="+str(0))
plt.scatter(np.linspace(1,len(np.array(arr_5[0][0])),len(np.array(arr_5[0][0]))),np.array(arr_5[0][0]), color='green', label=r'$\alpha $'+"="+str(0.5))
plt.scatter(np.linspace(1,len(np.array(arr_9[0][0])),len(np.array(arr_9[0][0]))),np.array(arr_9[0][0]), color='red', label=r'$\alpha $'+"="+str(0.9))


plt.xlabel("Epoch Number")
plt.ylabel("Error")
plt.title("Comparing Error vs Epoch No for"+r'$\eta $'+"="+str(0.05))
plt.legend()
#plt.gca().set_xlim(right=max(eta))

#plt.show()
plt.savefig('error_05.pdf')




######
plt.figure()

plt.scatter(np.linspace(1,len(np.array(arr_0[4][0])),len(np.array(arr_0[4][0]))),np.array(arr_0[4][0]), color='blue', label=r'$\alpha $'+"="+str(0))
plt.scatter(np.linspace(1,len(np.array(arr_5[4][0])),len(np.array(arr_5[4][0]))),np.array(arr_5[4][0]), color='green', label=r'$\alpha $'+"="+str(0.5))
plt.scatter(np.linspace(1,len(np.array(arr_9[4][0])),len(np.array(arr_9[4][0]))),np.array(arr_9[4][0]), color='red', label=r'$\alpha $'+"="+str(0.9))


plt.xlabel("Epoch Number")
plt.ylabel("Error")
plt.title("Comparing Error vs Epoch No for"+r'$\eta $'+"="+str(0.25))
plt.legend()
#plt.gca().set_xlim(right=max(eta))

#plt.show()
plt.savefig('error_25.pdf')



plt.figure()

plt.scatter(np.linspace(1,len(np.array(arr_0[9][0])),len(np.array(arr_0[9][0]))),np.array(arr_0[9][0]), color='blue', label=r'$\alpha $'+"="+str(0))
plt.scatter(np.linspace(1,len(np.array(arr_5[9][0])),len(np.array(arr_5[9][0]))),np.array(arr_5[9][0]), color='green', label=r'$\alpha $'+"="+str(0.5))
plt.scatter(np.linspace(1,len(np.array(arr_9[9][0])),len(np.array(arr_9[9][0]))),np.array(arr_9[9][0]), color='red', label=r'$\alpha $'+"="+str(0.9))


plt.xlabel("Epoch Number")
plt.ylabel("Error")
plt.title("Comparing Error vs Epoch No for"+r'$\eta $'+"="+str(0.5))
plt.legend()
#plt.gca().set_xlim(right=max(eta))

#plt.show()
plt.savefig('error_5.pdf')

print('done')