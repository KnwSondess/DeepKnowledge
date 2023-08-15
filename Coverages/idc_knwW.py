"""
# Code based on DeepImportance code release
# @ https://github.com/DeepImportance/deepimportance_code_release
"""

import numpy as np
from sklearn import cluster

from utils_ini import save_quantization, load_quantization, save_totalR, load_totalR
from utils_ini  import get_layer_outs_new, create_dir, get_non_con_neurons
from utils_ini import get_conv
from sklearn.metrics import silhouette_score

experiment_folder = 'experiments'
model_folder      = 'Networks'
def getneuron_layer(subset, neuron):
    for k in subset:
        if k[1] ==neuron:
            layer = k[0]


    return layer
class ImportanceDrivenCoverage:
    def __init__(self,model, model_name, num_relevant_neurons, selected_class, subject_layer,
                 train_inputs, train_labels):
        self.covered_combinations = ()

        self.model = model
        self.model_name = model_name
        self.num_relevant_neurons = num_relevant_neurons
        self.selected_class = selected_class
        self.subject_layer = subject_layer
        self.train_inputs = train_inputs
        self.train_labels = train_labels

        #create experiment directory if not exists
        create_dir(experiment_folder)


    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations




def quantize(out_vectors, conv, relevant_neurons, n_clusters=3):
    #if conv: n_clusters+=1
    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        out_i = []
        for l in out_vectors:
            if conv: #conv layer
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])

        #If it is a convolutional layer no need for 0 output check
        if not conv: out_i = filter(lambda elem: elem != 0, out_i)
        values = []

        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations

            kmeans = cluster.KMeans(n_clusters=n_clusters)
            kmeans.fit(np.array(out_i).reshape(-1, 1))
            values = kmeans.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)



        quantized_.append(values)


    quantized_ = [quantized_[rn] for rn in relevant_neurons]

    return quantized_

def quantizeSilhouette(out_vectors, conv, relevant_neurons):
    quantized_ = []


    for i in relevant_neurons:


        out_i = []
        for l in out_vectors:

            if conv: #conv layer
                out_i.append(np.mean(l[i[1]]))
            else:
                out_i.append(l[i[1]])

        #If it is a convolutional layer no need for 0 output check
        if not conv:
            out_i = [item for item in out_i if item != 0] # out_i = filter(lambda elem: elem != 0, out_i)

        values = []
        if not len(out_i) < 10:
            clusterSize = range(2, 5)#[2, 3, 4]
            clustersDict = {}
            for clusterNum in clusterSize:
                kmeans          = cluster.KMeans(n_clusters=clusterNum)
                clusterLabels   = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                silhouetteAvg   = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                clustersDict [silhouetteAvg] = kmeans

            maxSilhouetteScore = max(clustersDict.keys())
            bestKMean          = clustersDict[maxSilhouetteScore]

            values = bestKMean.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)


        if len(values) == 0:
            values.append(0)

        quantized_.append(values)

    return quantized_





def quantizeSilhouetteOld(out_vectors, conv, relevant_neurons):

    quantized_ = []

    for i in range(out_vectors.shape[-1]):
        if i not in relevant_neurons: continue

        out_i = []
        for l in out_vectors:
            if conv:
                out_i.append(np.mean(l[...,i]))
            else:
                out_i.append(l[i])


        if not conv:
            out_i = [item for item in out_i if item != 0]

        values = []

        if not len(out_i) < 10: #10 is threshold of number positives in all test input activations

            clusterSize = range(2, 6)
            clustersDict = {}
            for clusterNum in clusterSize:
                kmeans          = cluster.KMeans(n_clusters=clusterNum)
                clusterLabels   = kmeans.fit_predict(np.array(out_i).reshape(-1, 1))
                silhouetteAvg   = silhouette_score(np.array(out_i).reshape(-1, 1), clusterLabels)
                clustersDict [silhouetteAvg] = kmeans

            maxSilhouetteScore = max(clustersDict.keys())
            bestKMean          = clustersDict[maxSilhouetteScore]

            values = bestKMean.cluster_centers_.squeeze()
        values = list(values)
        values = limit_precision(values)

        #if not conv: values.append(0) #If it is convolutional layer we dont add  directly since thake average of whole filter.
        if len(values) == 0: values.append(0)

        quantized_.append(values)


    return quantized_


def limit_precision(values, prec=2):
    limited_values = []
    for v in values:
        limited_values.append(round(v,prec))

    return limited_values


def determine_quantized_cover(lout, quantized):
    covered_comb = []
    for idx, l in enumerate(lout):

            closest_q = min(quantized[idx], key=lambda x:abs(x-l))
            covered_comb.append(closest_q)

    return covered_comb


def measure_idc(model, model_name, test_inputs,relevant_neurons, subsetTop,sel_class,
                                   test_layer_outs, train_layer_outs,trainable_layers, skip_layers,
                                   covered_combinations=()):

    relevant= {}
    layers=[]
    neurons_list=[]
    for n in relevant_neurons:
        neurons_list.append(n[1])
        if n[0] not in layers:
            layers.append(n[0])


    for x in relevant_neurons:
        if x[0] in relevant:
            relevant[x[0]].append(x[1])
        else:
            relevant[x[0]] = [x[1]]
    total_max_comb=0

    for layer,neurons in relevant.items():
        subject_layer = layer


        if subject_layer not in skip_layers:

            is_conv = get_conv(subject_layer, model, train_layer_outs)

            qtizedlayer=quantizeSilhouette(train_layer_outs[subject_layer], is_conv,
                                  neurons)

            for test_idx in range(len(test_inputs)):
                if is_conv :
                    lout = []
                    for r in neurons:
                        lout.append(np.mean(test_layer_outs[subject_layer][test_idx][r[1]]))

                else:
                        lout = []

                        neuronsindices=get_non_con_neurons(neurons)

                        for i in neuronsindices:
                            lout.append(test_layer_outs[subject_layer][test_idx][i])


                comb_to_add = determine_quantized_cover(lout, qtizedlayer)

                if comb_to_add not in covered_combinations:
                        covered_combinations += (comb_to_add,)

            max_comb = 1
            for q in qtizedlayer:
                    max_comb *= len(q)

            total_max_comb += max_comb
        else:continue



    covered_num = len(covered_combinations)
    coverage = float(covered_num) / total_max_comb

    return coverage*100, covered_combinations, total_max_comb




