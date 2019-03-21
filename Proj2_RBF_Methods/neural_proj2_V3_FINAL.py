from __future__ import division
from itertools import cycle
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import random
import math

def sample_gen(x):
    # x:number
    #Generates a list of random inputs
    out=[]
    for i in range(x):
        out.append(np.random.random())
    out.sort()
    return out

def gauss(x,width):
    # x:input data (1 value), width:tuple(cluster_center,variance)
    # output: single value of gaussian
    #data: all input x sample
    #width(tuple): (center,var)
    center,var2=width
    fi=np.exp(-1/(2*var2) * (x-center)**2)

    return fi

def h(x):
    return 0.5+0.4*np.sin(2*np.pi*x)

def noise(x):
    return x+np.random.uniform(-0.1,0.1)

def kmean(data,n_neuron):
    # data: The list of all input, n_neuron: number of clusters in hidden layer
    # output: [(center,[all the members of cluster])]
    center=random.sample(set(data),n_neuron)
    iter=True
    count=1
    while(iter):
        ident = True
        cluster = [(x, 0) for x in data]
        center.sort()
        center_new=[]
        clus_div=[]
        for i in range(len(cluster)):
            x,j=cluster[i]
            dist=[(np.abs(x-c),t) for t,c in enumerate(center)]
            m=min(dist)[1]
            cluster[i]=(x,m)
        clus_div=[ (c , []) for c in center]
        for x , j in cluster:
            clus_div[j][1].append(x)

        for s in range(len(clus_div)):
            if (len(clus_div[s][1])==0 or clus_div[s][0]==clus_div[s][1]):
                center[s] = random.sample(set(data), 1)[0]
                ident=False #skip center update and reiterate the kmean

        if (ident==True): # Go ahead and update the centers
            center_new=[sum(clus_div[i][1])/len(clus_div[i][1])  for i in range(len(clus_div))]
            center_new.sort()

        if center==center_new:
            iter=False

        center = center_new


        # len_cent=[len(s) for l,s in clus_div]
        # if [s for s in len_cent if s == 0]:
        count+=1

    return clus_div


def width_dif(k_divided):
    # Different variance width method
    # input: [(center,[all the members of cluster])]
    # output: tuple (center,var^2)
    for i in range(len(k_divided)):
        if len(k_divided[i][1]) == 0:
            raise Exception("Error: Cluster Center Is Lonely Island!")
    result=[]

    for center,clust in k_divided:
        sig2=sum([(center-c)**2 for c in clust]) / len(clust)
        result.append((center,sig2))

    return result

def width_same(k_divided):
    # Same variance width method
    # input: [(center,[all the members of cluster])]
    # output: tuple (center,var^2)
    for i in range(len(k_divided)):
        if len(k_divided[i][1]) == 0:
            raise Exception("Error: Cluster Center Is Lonely Island!")
    result=[]
    center=[c for c,clus in k_divided]
    sig2= (max(center)-min(center))**2 /(2*len(center))
    for center,clust in k_divided:
        result.append((center,sig2))
    return result




def forward(x,width,weights):
    # Go to the output layer
    out=0
    #weight without bias
    weights_pure=weights[:len(weights)-1]

    for tup,w in zip(width,weights_pure):
        out+=gauss(x,tup) * w

    #bias
    out+=1*weights[len(weights)-1]

    desired=noise(h(x))
    return out,desired


def update(weights,width,eta,x,d,y):
    # LMS algorithm update
    weights_new=[]
    # weight without bias
    weights_pure=weights[:len(weights)-1]
    diff=d-y
    for tup, w in zip(width, weights_pure):
        weights_new.append( w + eta * diff * gauss(x,tup) )

    weights_new.append(weights[len(weights)-1]+eta*diff)

    return weights_new

def draw(data,desired,out,et,hid,er):
    #Plotting function
    plt.figure()
    plt.rc('text', usetex=True)
    plt.scatter(data, desired, color='blue', label="Desired")
    plt.plot(data, list(map(h, data)), color='green', label="Noise=0")
    plt.plot(data, out, color='red', label="RBF")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Evaluation With Same Gaussian Widths:"+'$\eta$'+"="+str(et)+",Clusters="+str(hid)+",Error="+str('%.3f' % er) )
    plt.gca().set_xlim(0, 1)
    plt.legend()
    plt.savefig('eta_'+str(et)+'_cluster_'+str(hid)+'.pdf')
    plt.clf()
    plt.cla()
    plt.close()
    plt.close('all')


if __name__ == "__main__":
    plt.rcParams.update({'figure.max_open_warning': 0})
    # np.random.seed(1)
    random.seed()
    neuron=[2,4,7,11,16]
    #neuron=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    eta=[0.01,0.02,0.03]
    epochs = 100
    smearing=1

    # eta=[0.02]
    # neuron = [10]


    data=sample_gen(75)


    summary=[]
    for et in eta:
        for hid in neuron:
            er=[]
            count=0
            for t in range(smearing):
                k_divided = kmean(data, hid + 1)
                #width = width_same(k_divided)
                width = width_dif(k_divided)

                while ( min([c for (b, c) in width]) < 1e-8):
                    print("Cluster Has a Lonely Member! Trying Again")
                    k_divided = kmean(data, hid + 1)
                    width = width_dif(k_divided)
                weights = [random.uniform(-1, 1) for i in range(hid + 1)]
                for i in range(epochs):
                    desired=[]
                    out=[]
                    for x in data:
                        y,d=forward(x, width, weights)
                        weights=update(weights,width,et,x,d,y)
                        desired.append(d)
                        out.append(y)
                er_trial=sum([abs(o-d) for o,d in zip(out,desired)]) / len(desired)
                er.append(er_trial)
                count+=1
                print(count)
            # summary=(eta,neuron in hidden,error)
            print('$\eta$'+"="+str(et)+",Clusters="+str(hid)+"is Done" )
            er=sum([c for c in er])/len(er)

            summary.append((et, hid, er))
            draw(data, desired, out, et, hid, er)

    cycol = cycle('bgrcmk')
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for et in eta:
        hid_conc=[d for (c, d, e) in summary if c==et]
        er_conc = [e for (c, d, e) in summary if c == et]
        plt.plot(hid_conc,er_conc, color=next(cycol),linewidth=2, label='$\eta$'+"="+str(et))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Error")
    plt.title("Errors For No. of Cluster vs Choice of $\eta$"+  " :No Smearing"+"Different Var Width")
    # plt.gca().set_xlim(0, 1)
    plt.legend()
    plt.savefig('Comparison.pdf')
    plt.clf()
    plt.cla()


print('done')