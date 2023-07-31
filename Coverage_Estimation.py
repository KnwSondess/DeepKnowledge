import argparse
import tensorflow as tf
from tensorflow import keras

from datetime import datetime
from tensorflow.keras.models import model_from_json, load_model, save_model
# from Coverages.knw import *
from Coverages.TrKnw import *
from tensorflow.keras import applications
from tensorflow.python.client import device_lib

__version__ = 0.4




def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """



    text = 'Knowledge Coverage for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.", choices=['lenet1','svhn', 'model_cifar10'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN"])
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                            to measure coverage", choices=['knw', 'idc'])
    parser.add_argument("-P", "--percentage", help="the percentage of TrKnw neurons to be deployed", type=float)
    parser.add_argument("-K", "--nbr_Trknw", help="the number of TrKnw neurons to be deployed", type=float)
    parser.add_argument("-HD", "--HD_thre", help="a threshold value used\
                            to identify the type of TrKnw neurons.", type=float)
    parser.add_argument("-Tr", "--TrKnw", help="Type of selected TrKnw neurons based on HD values range.", choices=['top', 'least'])
    parser.add_argument("-Sp", "--split", help="percentage of test data to be tested", type=float)
    parser.add_argument("-ADV", "--adv", help="name of adversarial attack", choices=['mim', 'bim', 'fgsm', ''])

    parser.add_argument("-C", "--class", help="the selected class", type=int)

    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)


    parser.add_argument("-LOG", "--logfile", help="path to log file")

    args = parser.parse_args()

    return vars(args)
def generate_coverage(approach,modelpath,dataset,TypeTrknw, percent,selected_class,threshold, attack,split):
    model_path = 'Networks/'+modelpath

    img_rows, img_cols, img_channel = 32, 32, 3

    model_name = model_path.split('/')[-1]
    if model_name == 'vgg16':
        model = applications.VGG16(weights='imagenet', include_top=False,
                                   input_shape=(img_rows, img_cols, img_channel))
        print("Model VGG 16 is loaded")
    else:

        try:
            json_file = open(model_path + '.json', 'r')
            file_content = json_file.read()
            json_file.close()

            model = model_from_json(file_content)
            model.load_weights(model_path + '.h5')

            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        except:
            print("exeception")
            model = load_model(model_path + '.hdf5')


    trainable_layers = get_trainable_layers(model)
    dense_layers=get_dense_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = [0]  # SKIP LAYERS FOR NC, KMNC, NBC
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print("Skipping layers:", skip_layers)
    ####################

    if approach == 'knw':
        model_folder = 'Networks'
        method = 'idc'
        if not os.path.exists("./data"):
            os.makedirs("./data")
        if not os.path.exists("./experiments"):
            os.makedirs("./experiments")
      

        knw = KnowledgeCoverage(model, dataset, model_name, subject_layer, trainable_layers,dense_layers, method, percent, threshold,attack, skip_layers, nbr_Trknw,selected_class=1)



        
        Knw_coverage, covered_TrKnw, combinations, max_comb, testsize, zero_size,Trkneurons= knw.run(split,TypeTrknw,use_adv=False)



        print("The model Transfer Knowledge Neurons number: ", covered_TrKnw)
        
        print("The percentage of the used neurons out of all Transfer Knowledge Neurons : ",percent)

        print("The test set coverage: %.2f%% for dataset  %s " % (Knw_coverage, dataset))

        print("Covered combinations: ", len(combinations))
        print("Total combinations:", max_comb)
        line=[model_name, dataset, testsize, zero_size, Knw_coverage, covered_TrKnw, TypeTrknw, (1-split),len(combinations),
                     max_comb,Trkneurons, attack]





    else:
        print("other method")

    logfile.close()
    return line


if __name__ == "__main__":
        args = parse_arguments()
        model = args['model'] if args['model'] else 'LeNet1'
        dataset = args['dataset'] if args['dataset'] else 'mnist'
        approach = args['approach'] if args['approach'] else 'knw'
        percent = args['percentage'] if args['percentage'] else 0.5
        nbr_Trknw= args['nbr_Trknw'] if args['nbr_Trknw'] else 10
        threshold = args['HD_thre'] if args['HD_thre'] else 0.05
        TypeTrknw = args['TrKnw'] if args['TrKnw'] else 'preferred'
        split = args['split'] if args['split'] else 0
        attack = args['adv'] if args['adv'] else 'mim'
        selected_class = args['class'] if not args['class'] == None else -1
        logfile_name = args['logfile'] if args['logfile'] else 'resultknw.log'
        logfile = open(logfile_name, 'a')
    
        startTime = time.time()

        results=generate_coverage(approach,model,dataset,TypeTrknw, percent,selected_class,threshold, attack,split)

        logfile.close()
        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed Time = %s" % elapsedTime)

