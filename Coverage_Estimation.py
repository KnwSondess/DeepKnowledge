import argparse
from Dataprocessing import *
from tensorflow.keras.models import model_from_json, load_model, save_model
from Coverages.TrKnw import *
from tensorflow.keras import applications
from tensorflow.python.client import device_lib
import tensorflow
import os

os.environ['TF_GPU_ALLOCATOR']="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
__version__ = 2.0


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    text = 'Knowledge-based Coverage for DNN models'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments
    parser.add_argument("-U", "--adv_use",  default=False, type=bool, help="use adversarial attacks")
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.", choices=['LeNet1','LeNet4','svhn', 'LeNet5', 'model_cifar10','DKox', 'ImagNet'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN",'GTSDB'])
    parser.add_argument("-K", "--nbr_Trknw", help="the number of TrKnw neurons to be deployed", type=float)
    parser.add_argument("-HD", "--HD_thre", help="a threshold value used\
                            to identify the type of TrKnw neurons.", type=float)
    parser.add_argument("-Tr", "--TrKnw", help="Type of selected TrKnw neurons based on HD values range.", choices=['top', 'least', 'preferred'])
    parser.add_argument("-Sp", "--split", help="percentage of test data to be tested", type=float)
    parser.add_argument("-ADV", "--adv", help="name of adversarial attack", choices=['mim', 'bim', 'fgsm', 'pgd'])
    parser.add_argument("-app", "--approach", help="the used approach", choices=['knw'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)

    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)


    parser.add_argument("-LOG", "--logfile", help="path to log file")

    args = parser.parse_args()

    return vars(args)






def estimate_coverage(approach,modelname,dataset,TypeTrknw, percent,selected_class,threshold, attack,split):
    model_path = 'Networks/'+modelname

    model_name = model_path.split('/')[-1]
    print(model_name)

    if model_name == 'DKox':
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.load_model("./Networks/leaf_disease_coloured.h5")

        print("Model for grape leaves disease detection is loaded")
    elif model_name == 'vgg16':
        img_rows, img_cols, img_channel = 32, 32, 3
        model = applications.VGG16(weights='imagenet', include_top=False,
                                   input_shape=(img_rows, img_cols, img_channel))
        print("Model VGG 16 is loaded")

    elif model_name == 'ImagNet':
        model =tf.keras.applications.MobileNetV2(include_top=True,
                                          weights='imagenet')
        model.trainable = False
        model.compile(optimizer='adam',
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])

        print("Model imagenet is loaded")
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
    isExist = os.path.exists(experiment_folder)
    if not isExist:
        os.makedirs(experiment_folder)
    dataset_folder = 'dataset'
    isExist = os.path.exists(dataset_folder)
    if not isExist:
        os.makedirs(dataset_folder)

    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = []  # SKIP LAYERS FOR NC, KMNC, NBC
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print("Skipping layers:", skip_layers)
    ####################

    if approach == 'knw':
        method = 'idc'

        knw = KnowledgeCoverage(model, dataset, model_name, subject_layer, trainable_layers,dense_layers, method, percent, threshold,attack, skip_layers, nbr_Trknw,selected_class=1)


        Knw_coverage, covered_TrKnw, combinations, max_comb, testsize, zero_size,Trkneurons= knw.run(split,TypeTrknw,use_adv)

        print("The model Transfer Knowledge Neurons number: ", covered_TrKnw)

        if use_adv:
            print("Deployed Adversarials attacks", attack)
        if split>0:
            print("Test set is splited and only %.2f%% is used" %(1-split))

        print("The test set coverage: %.2f%% for dataset  %s and the model %s " % (Knw_coverage, dataset, model_name))

        print("Covered combinations: ", len(combinations))
        print("Total combinations:", max_comb)






    else:
        print("other method")

    logfile.close()
    return 0


if __name__ == "__main__":

                args = parse_arguments()
                model = args['model'] if args['model'] else 'LeNet5'
                dataset = args['dataset'] if args['dataset'] else 'GTSDB'
                approach = args['approach'] if args['approach'] else 'knw'
                use_adv = args['adv_use'] if args['adv_use'] else False
                nbr_Trknw= args['nbr_Trknw'] if args['nbr_Trknw'] else 20
                threshold = args['HD_thre'] if args['HD_thre'] else 0.05
                TypeTrknw = args['TrKnw'] if args['TrKnw'] else 'preferred'
                split = args['split'] if args['split'] else 0
                attack = args['adv'] if args['adv'] else 'pgd'
                selected_class = args['class'] if not args['class'] == None else -1
                logfile_name = args['logfile'] if args['logfile'] else 'resultknw.log'
                logfile = open(logfile_name, 'a')

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # with tf.device('/gpu:2')
        #     gpus = tf.config.list_physical_devices('GPU')
        #     if gpus:
        #         # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        #         try:
        #             tf.config.set_logical_device_configuration(
        #                 gpus[0],
        #                 [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        #             logical_gpus = tf.config.list_logical_devices('GPU')
        #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #         except RuntimeError as e:
        #             # Virtual devices must be set before GPUs have been initialized
        #             print(e)

                startTime = time.time()
                percent= 0.7
                attack_lst=['bim', 'mim','fgsm', 'pgd']
                if dataset=='coco':
                        model='Vgg19_coco_animal_32_32_kera2_9'
                if dataset == "imagnet":
                        model = 'ImagNet'
                if dataset == "leaves":
                        model = 'grape'
                if dataset == "mnist":
                        model = 'LeNet1'
                if dataset == "GTSRB":
                        model = 'LeNet5'
                elif dataset == 'svhn':
                        model = 'svhn'# resnet18 trained with svhn
                elif dataset == 'cifar':
                        model = 'LeNet5_cifar10' #LeNet5 trained with cifar




                estimate_coverage(approach,model,dataset,TypeTrknw, percent,selected_class,threshold, attack,split)

                logfile.close()
                endTime = time.time()
                elapsedTime = endTime - startTime
                print("Elapsed Time = %s" % elapsedTime)







