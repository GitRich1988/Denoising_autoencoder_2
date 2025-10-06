# src/myDataSetMGR/myModelMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()
from src.myCircleParameters.myCircleParameters import myCircleParameters

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1" # this is an attempt to ensure a CNN trains to identical weights each time I run the program
import re
import json
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, models
import time
import pandas as pd
import h5py
import ast

#==============================================================================
class myModelMGR:

    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def __init__( self
                , a_DataSetMGR
                , a_FileNameCNNDefinition
                , a_FileNameHyperParameterSet):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.__init__()", "=", 1, 0)

        self.m_DataSetMGR = a_DataSetMGR
        self.m_ListOfHyperParameterSets = []
        self.m_ListOfCNNDefinitions = []
        self.m_DirHyperParameterSets = ""
        self.m_DirCNNDefinitions = ""
        self.m_FileNameCNNDefinition = a_FileNameCNNDefinition
        self.m_FileNameHyperParameterSet = a_FileNameHyperParameterSet
        self.m_RandomSeedNumpy = 0
        self.m_RandomSeedRandom =0
        self.m_RandomSeedTensorflow = 0
        self.m_NominalRadius = 0

        self.Initialise()

        l_GeneralFunctions.PrintMethodEND("myModelMGR.__init__()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Key input locations:
    # C:/Users/marti/Documents/Machine_learning/Denoising_autoencoder/Training_info/CNN_definitions/
    # C:/Users/marti/Documents/Machine_learning/Denoising_autoencoder/Training_info/Hyper_parameters/
    #
    # Key output locations:
    # C:/Users/marti/Documents/Machine_learning/Denoising_autoencoder/Training_info/Model_records/...
    # C:/Users/marti/Documents/Machine_learning/Work_stuff/Log_files/<Data_set_name>/Point_data/...
    def Initialise(self):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.Initialise()", "=", 0, 0)

        self.m_DirHyperParameterSets = "Training_info/Hyper_parameters/"
        self.m_DirCNNDefinitions = "Training_info/CNN_definitions/"
        self.m_RandomSeedNumpy = 42
        self.m_RandomSeedRandom = 42
        self.m_RandomSeedTensorflow = 42
        self.m_NominalRadius = 1

        self.SetListOfHyperParameterSets()
        self.SetListOfCNNDefinitions()

        l_GeneralFunctions.PrintMethodEND("myModelMGR.Initialise()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SetListOfHyperParameterSets(self):

        l_FullPathHyperParametersFile = self.m_DirHyperParameterSets + self.m_FileNameHyperParameterSet

        with open(l_FullPathHyperParametersFile, 'r') as f:
            l_JSONData = json.load(f)
            self.m_ListOfHyperParameterSets.append(l_JSONData)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SetListOfCNNDefinitions(self):

        l_FullPathCNNDefinitionsFile = self.m_DirCNNDefinitions + self.m_FileNameCNNDefinition

        with open(l_FullPathCNNDefinitionsFile, 'r') as f:
            l_JSONData = json.load(f)
            self.m_ListOfCNNDefinitions.append(l_JSONData)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def ListAvailableGPUs(self):
        # List available GPUs
        l_GPUs = tf.config.list_physical_devices('GPU')
        if l_GPUs:
            print(f"TensorFlow is using the following GPU(s): {l_GPUs}")
        else:
            print("No GPU detected")
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def Run(self):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.Run()", "=", 1, 0)

        self.ListAvailableGPUs()
        print("len(self.m_ListOfCNNDefinitions):", len(self.m_ListOfCNNDefinitions))
        print("len(self.m_ListOfHyperParameterSets):", len(self.m_ListOfHyperParameterSets))

        for l_HyperParameterSetIndex in range(0, len(self.m_ListOfHyperParameterSets)):
            print("\n===============================================================\nl_HyperParameterSetIndex: ", l_HyperParameterSetIndex)
            l_CurrentHyperParameterSet = self.m_ListOfHyperParameterSets[l_HyperParameterSetIndex]
            print(json.dumps(l_CurrentHyperParameterSet, indent=2))

            for l_CNNIndex in range(0, len(self.m_ListOfCNNDefinitions)):
                print("\n---------------------------------------------------------------\nl_CNNIndex: ", l_CNNIndex)
                l_CurrentCNNDefinition = self.m_ListOfCNNDefinitions[l_CNNIndex]
                print(json.dumps(l_CurrentCNNDefinition, indent=2))

                l_AutoEncoder = None
                l_ModelBuildingTime = 0
                l_ModelTrainingTime = 0
                l_ModelAndTimes = [l_AutoEncoder, l_ModelBuildingTime, l_ModelTrainingTime]
                l_ModelAndTimes = [l_AutoEncoder, l_ModelBuildingTime, l_ModelTrainingTime]
                self.DefineBuildAndTrainOneCNN( l_CurrentCNNDefinition
                                              , l_CurrentHyperParameterSet
                                              , l_ModelAndTimes)
                l_AutoEncoder = l_ModelAndTimes[0]
                l_ModelBuildingTime = l_ModelAndTimes[1]
                l_ModelTrainingTime = l_ModelAndTimes[2]

                # Test the model on a single example
                #"""
                l_ExampleIndex = 0
                self.TestOneCNNOnSingleExample_GPUVersion( l_AutoEncoder
                                                         , l_CurrentCNNDefinition
                                                         , l_ModelBuildingTime
                                                         , l_ModelTrainingTime
                                                         , l_ExampleIndex
                                                         , l_CNNIndex
                                                         , l_CurrentHyperParameterSet)
                #"""

                # Test the model on batches of examples
                """
                self.TestOneCNNOnBatches_GPUVersion( l_AutoEncoder
                                                   , l_CurrentCNNDefinition
                                                   , l_ModelBuildingTime
                                                   , l_ModelTrainingTime
                                                   , l_CNNIndex
                                                   , l_CurrentHyperParameterSet)
                """

        l_GeneralFunctions.PrintMethodEND("myModelMGR.Run()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def DefineBuildAndTrainOneCNN( self
                                 , a_CNNDefinition
                                 , a_HyperParameterSet
                                 , a_ModelAndTimes):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.DefineBuildAndTrainOneCNN()", "=", 1, 0)

        # Reset random seeds
        self.ResetRandomSeeds()

        # Build the model
        l_AutoEncoder = None
        l_ModelBuildingTime = 0
        [l_AutoEncoder, l_ModelBuildingTime] = self.DefineAndBuildOneCNN( a_CNNDefinition
                                                                        , a_HyperParameterSet)

        # Train the model
        l_ModelTrainingTime = 0;
        l_History = None
        l_FullPathCNNWeights = None
        l_TrainOneCNNList = [ l_AutoEncoder
                            , l_ModelTrainingTime
                            , l_History
                            , l_FullPathCNNWeights ]
        l_TrainOneCNNList = self.TrainOneCNN( l_TrainOneCNNList
                                            , a_HyperParameterSet
                                            , l_ModelBuildingTime)
        l_AutoEncoder = l_TrainOneCNNList[0]
        l_ModelTrainingTime = l_TrainOneCNNList[1]

        a_ModelAndTimes[0] = l_AutoEncoder
        a_ModelAndTimes[1] = l_ModelBuildingTime
        a_ModelAndTimes[2] = l_ModelTrainingTime

        # Save CNN weights to file
        ##self.DisplayCNNWeightsFromFile(l_FullPathCNNWeightsEarlierRun)
        ##self.SaveCNNWeights(l_AutoEncoder, a_CNNParameters)
        self.SaveFirstFewCNNWeights( l_AutoEncoder
                                   , a_CNNDefinition)

        l_GeneralFunctions.PrintMethodEND("myModelMGR.DefineBuildAndTrainOneCNN()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def ResetRandomSeeds(self):
        tf.keras.backend.clear_session()
        np.random.seed(self.m_RandomSeedNumpy)
        random.seed(self.m_RandomSeedRandom)
        tf.random.set_seed(self.m_RandomSeedTensorflow)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def DefineAndBuildOneCNN( self
                            , a_CNNDefinition
                            , a_HyperParameterSet):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.DefineAndBuildOneCNN()", "=", 1, 0)

        l_NumFiltersListStr = a_CNNDefinition[0]["NumFiltersList"]
        l_NumFiltersList = json.loads(l_NumFiltersListStr)
        #print("l_NumFiltersList:", l_NumFiltersList)
        for i in range(0, len(l_NumFiltersList)):
            print("l_NumFiltersList[", i, "]:", l_NumFiltersList[i])

        l_KernelSizesListStr = a_CNNDefinition[0]["KernelSizesList"]
        l_KernelSizesList = json.loads(l_KernelSizesListStr)
        #print("l_KernelSizesList:", l_KernelSizesList)
        for i in range(0, len(l_KernelSizesList)):
            print("l_KernelSizesList[", i, "]:", l_KernelSizesList[i])


        if(len(l_NumFiltersList) != len(l_KernelSizesList)):
            print("!!! ERROR: The number of filters and kernel sizes must match\n")
            return None
        else:
            print("--> The number of filters and kernel sizes match, so can proceed\n")

            # New way of building the model
            #l_Model = self.ConstructCNNArchitecture( l_NumFiltersList
            #                                       , l_KernelSizesList)
            #l_Model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=a_CurrentHyperParameterSet["LearningRate"])
            #l_Model.compile( optimizer='adam')
            #
            #return l_Model

            # Surviving from original program:
            l_NumRowsPerExampleDataFrame = self.m_DataSetMGR.m_NumRowsPerExampleDataFrame
            l_NumFeaturesPerPoint = self.m_DataSetMGR.m_NumFeaturesPerPoint
            l_InputLayer = tf.keras.layers.Input( shape=(l_NumRowsPerExampleDataFrame, l_NumFeaturesPerPoint)
                                                , dtype=tf.float32)

            l_Decoded = self.BuildCNNEncoderDecoderLayers( l_NumRowsPerExampleDataFrame
                                                         , l_NumFeaturesPerPoint
                                                         , l_InputLayer
                                                         , l_NumFiltersList
                                                         , l_KernelSizesList
                                                         , a_HyperParameterSet[0]["ActivationFunction"])

            # myCNNArchitecture class is lost
            #l_CNNArchitecture = myCNNArchitecture( l_InputScan
            #                                     , l_Decoded)
            l_CNNArchitecture = [l_InputLayer, l_Decoded]

            #self.m_CNNArchitectureList.append(l_CNNArchitecture)

            # Build the model
            l_AutoEncoder = None
            l_ModelBuildingTime = 0
            [l_AutoEncoder, l_ModelBuildingTime] = self.BuildOneCNN( l_CNNArchitecture
                                                                   , a_HyperParameterSet)

        l_GeneralFunctions.PrintMethodEND("myModelMGR.DefineAndBuildOneCNN()", "=", 0, 0)
        return [l_AutoEncoder, l_ModelBuildingTime]
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def ConstructCNNArchitecture( self
                                , a_NumFiltersList
                                , a_KernelSizesList):
        l_GeneralFunctions.PrintMethodSTART("myModelMGR.ConstructCNNArchitecture()", "=", 1, 0)

        l_NumRowsPerExampleDataFrame = self.m_DataSetMGR.m_NumRowsPerExampleDataFrame
        print("l_NumRowsPerExampleDataFrame:", l_NumRowsPerExampleDataFrame)
        l_NumFeaturesPerPoint = self.m_DataSetMGR.m_NumFeaturesPerPoint
        print("l_NumFeaturesPerPoint:", l_NumFeaturesPerPoint)

        l_InputLayer = tf.keras.layers.Input( shape=(l_NumRowsPerExampleDataFrame, l_NumFeaturesPerPoint)
                                            , dtype=tf.float32)


        # Encoder (downsampling)
        l_TempLayer = l_InputLayer
        for l_CurrentNumFilters, l_CurrentKernelSize in zip(a_NumFiltersList, a_KernelSizesList):
            l_TempLayer = layers.Conv1D( filters=l_CurrentNumFilters
                                       , kernel_size=l_CurrentKernelSize
                                       , activation='relu'  # I MAY NOT NECESSARILY HAVE TO SET THIS HERE AND THUS CAN RE-ORDER THE LOOPS IN Run()
                                       , padding='same')(l_TempLayer)

            l_TempLayer = layers.MaxPooling1D(pool_size=2, padding='same')(l_TempLayer)


        # Decoder (reversing the encoder)
        l_FinalDecoderLayerKernelSize = -1
        for l_CurrentNumFilters, l_CurrentKernelSize in reversed(list(zip(a_NumFiltersList, a_KernelSizesList))):
            l_TempLayer = tf.keras.layers.UpSampling1D(size=2)(l_TempLayer)
            l_TempLayer = tf.keras.layers.Conv1D (filters=l_CurrentNumFilters
                                               , kernel_size=l_CurrentKernelSize
                                               , activation='relu'
                                               , padding='same')(l_TempLayer)
            l_FinalDecoderLayerKernelSize = l_CurrentKernelSize

        # Final output layer to match input shape
        l_OutputLayer = tf.keras.layers.Conv1D( filters=l_NumFeaturesPerPoint
                                              , kernel_size=1
                                              #, kernel_size=l_FinalDecoderLayerKernelSize # This does not have to have the final decoder layer size but I needed to pick something
                                              , activation='linear'
                                              , padding='same')(l_TempLayer)

        # Model
        l_Model = tf.keras.models.Model( inputs=l_InputLayer
                                       , outputs=l_OutputLayer)
        l_Model.summary()

        return l_Model

        l_GeneralFunctions.PrintMethodEND("myModelMGR.ConstructCNNArchitecture()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Surviving from original program:
    def BuildCNNEncoderDecoderLayers( self
                                    , a_NumRowsPerExampleDataFrame
                                    , a_NumFeaturesPerPoint
                                    , a_InputLayer
                                    , a_NumFiltersList
                                    , a_KernelSizeList
                                    , a_ActivationFunction):
        #l_GeneralFunctions.PrintMethodSTART("BuildCNNEncoderDecoderLayers()", "=", 1, 0)

        # It is assumed that each layer will have its own size specified in the Lists a_NumFiltersList and a_KernelSizeList
        l_NumLayers = len(a_NumFiltersList)

        l_CurrentLayer = a_InputLayer

        # Encoding
        for l_LayerIndex in range(0, l_NumLayers):
            l_CurrentNumFilters = a_NumFiltersList[l_LayerIndex]
            l_TempEncoded = layers.Conv1D( l_CurrentNumFilters
                                         , kernel_size=a_KernelSizeList[l_LayerIndex]
                                         #, dtype=tf.float64
                                         , dtype=tf.float32
                                         , activation=a_ActivationFunction # "relu"
                                         , padding="same")(l_CurrentLayer)
            l_CurrentLayer = l_TempEncoded

        #l_Encoded = layers.Flatten(dtype=tf.float64)(l_CurrentLayer)
        l_Encoded = layers.Flatten(dtype=tf.float32)(l_CurrentLayer)

        # Latent space
        l_Latent = layers.Dense( a_NumFiltersList[len(a_NumFiltersList) - 1]
                               , activation = a_ActivationFunction
                               #, dtype=tf.float64
                               , dtype=tf.float32
                               )(l_Encoded)

        # Decoding (expand back to the sequence structure)
        l_Decoded = layers.Dense( a_NumRowsPerExampleDataFrame * a_NumFiltersList[len(a_NumFiltersList) - 1]
                                , activation = a_ActivationFunction
                                #, dtype=tf.float64
                                , dtype=tf.float32
                                )(l_Latent)

        l_Decoded = layers.Reshape( (a_NumRowsPerExampleDataFrame, a_NumFiltersList[len(a_NumFiltersList) - 1])
                                  #, dtype=tf.float64
                                  , dtype=tf.float32
                                  )(l_Decoded)

        # Convolutional Decoder to bring back to original feature space
        l_NumFiltersListReversed = a_NumFiltersList[::-1]
        l_KernelSizeListReversed = a_KernelSizeList[::-1]

        l_CurrentLayer = l_Decoded

        for l_LayerIndex in range(0, l_NumLayers):
            l_CurrentNumFilters = l_NumFiltersListReversed[l_LayerIndex]
            l_TempDecoded = layers.Conv1D( l_CurrentNumFilters
                                         , kernel_size=l_KernelSizeListReversed[l_LayerIndex]
                                         #, dtype=tf.float64
                                         , dtype=tf.float32
                                         , activation=a_ActivationFunction # "relu"
                                         , padding="same")(l_CurrentLayer)
            l_CurrentLayer = l_TempDecoded

        l_Decoded = layers.Conv1D( a_NumFeaturesPerPoint
                                 , kernel_size=l_KernelSizeListReversed[len(l_KernelSizeListReversed) - 1]
                                 , activation="linear" # why linear here?
                                 #, dtype=tf.float64
                                 , dtype=tf.float32
                                 , padding="same")(l_CurrentLayer)


        #l_GeneralFunctions.PrintMethodEND("BuildCNNEncoderDecoderLayers()", "=", 0, 0)
        return l_Decoded
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # surviving from original program:
    def BuildOneCNN( self
                   , a_CNNArchitecture
                   #, a_CNNParameters
                   , a_HyperParameterSet # a_HyperParameterSet is equivalent to the old a_CNNParameters arg
                   ):

        l_GeneralFunctions.PrintMethodSTART("BuildOneCNN()", "=", 1, 0)

        #l_InputLayer = a_CNNArchitecture.m_InputLayer
        #l_DecodedLayer = a_CNNArchitecture.m_DecodedLayer
        l_InputLayer = a_CNNArchitecture[0]
        l_DecodedLayer = a_CNNArchitecture[1]

        # Generate tensorflow.keras model
        #self.ResetAllRandomSeeds(self.m_RandomSeed)

        # 'models' is imported from tensorflow.keras
        l_AutoEncoder = models.Model( l_InputLayer
                                    , l_DecodedLayer)

        # Compile tensorflow.keras model
        l_StartTimeBuilding = time.time()
        #l_AutoEncoder.compile( optimizer = a_CNNParameters.m_Optimizer
        #                     , loss = a_CNNParameters.m_LossFunction)
        l_AutoEncoder.compile( optimizer = a_HyperParameterSet[0]["Optimizer"]
                             , loss = a_HyperParameterSet[0]["LossFunction"])
        l_EndTimeBuilding = time.time()

        #print("l_AutoEncoder.output.dtype:",l_AutoEncoder.output.dtype)  # should be float64 or float32
        #for layer in l_AutoEncoder.layers:
        #    print(f"{layer.name}: {layer.dtype}, output: {layer.output.dtype}")
        #print("Model dtype policy:", l_AutoEncoder.dtype_policy)

        l_ModelBuildingTime = l_EndTimeBuilding - l_StartTimeBuilding

        l_GeneralFunctions.PrintMethodEND("BuildOneCNN()", "=", 0, 0)
        return [l_AutoEncoder, l_ModelBuildingTime]
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def TrainOneCNN( self
                   , a_TrainOneCNNList
                   , a_HyperParameterSet
                   , a_ModelBuildingTime):
        l_GeneralFunctions.PrintMethodSTART("TrainOneCNN()", "=", 1, 0)

        # Ensure the raw and nominal file paths match (i.e., one forAutoEncoder.fit( each example)
        # For simplicity, assuming file paths are sorted correctly
        assert len(self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTraining) == len(self.m_DataSetMGR.m_ListOfPointDataFullPathsRCNomTraining), "Mismatch between raw and RCNom data files"

        l_StartTimeTraining = time.time()

        # Build dataset of filenames
        #l_DataSetRaw = tf.data.Dataset.from_tensor_slices(self.m_FilePathsRawDataTraining)
        #l_DataSetNominal = tf.data.Dataset.from_tensor_slices(self.m_FilePathsRCNomDataTraining)
        l_DataSetRaw = tf.data.Dataset.from_tensor_slices(self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTraining)
        l_DataSetNominal = tf.data.Dataset.from_tensor_slices(self.m_DataSetMGR.m_ListOfPointDataFullPathsRCNomTraining)


        # Zip them together
        filename_dataset = tf.data.Dataset.zip(( l_DataSetRaw
                                               , l_DataSetNominal))

        # Map the filenames into data
        l_NumFilePairsLoaded = 0
        train_dataset = filename_dataset.map( lambda raw_path
                                            , nominal_path: self.LoadRawAndRCNomFilePair(raw_path, nominal_path, l_NumFilePairsLoaded))

        #for raw_example in train_dataset.take(1):
        #    tf.print("Shape of one raw example:", tf.shape(raw_example))

        #for raw, nom in train_dataset.take(1):
        #    tf.print("raw dtype:", raw.dtype)
        #    tf.print("nom dtype:", nom.dtype)
        #    tf.print("nom sample row:", nom[0])

        # Set the batches
        #train_dataset = train_dataset.batch(self.m_HyperParameters.m_BatchSize)
        train_dataset = train_dataset.batch(int(a_HyperParameterSet[0]["TrainingBatchSize"]))


        # Now fit the model
        #self.ResetAllRandomSeeds(self.m_RandomSeed)

        l_AutoEncoder = a_TrainOneCNNList[0]
        weights_before = l_AutoEncoder.layers[1].get_weights()[0].copy()
        l_History = l_AutoEncoder.fit( train_dataset
                                     #, epochs=self.m_HyperParameters.m_NumEpochs
                                     , epochs=int(a_HyperParameterSet[0]["NumEpochs"])
                                     , verbose=0
                                     , shuffle=False)
        weights_after = l_AutoEncoder.layers[1].get_weights()[0]
        print("Weight difference:", np.abs(weights_before - weights_after).sum())
        print("Final training loss:", l_History.history['loss'][-1])

        l_EndTimeTraining = time.time()
        l_ModelTrainingTime = l_EndTimeTraining - l_StartTimeTraining

        #l_FullPathCNNWeights = self.SaveCNNWeights(l_AutoEncoder, a_CNNParameters)
        l_FullPathCNNWeights = None
        a_TrainOneCNNList = [ l_AutoEncoder
                            , l_ModelTrainingTime
                            , l_History
                            , l_FullPathCNNWeights]

        self.m_LoadedTrainingDataRawAndRCNom = train_dataset # unsure if necessary?

        l_GeneralFunctions.PrintMethodEND("TrainOneCNN()", "=", 0, 0)
        return a_TrainOneCNNList
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def LoadRawAndRCNomFilePair(self, raw_path, nominal_path, a_NumFilePairsLoaded = -1):
        #tf.print("===== START OF LoadRawAndRCNomFilePair() =====")
        #tf.print("Raw path:", raw_path)
        #tf.print("RCNom path:", nominal_path)

        time.sleep(0.01)  # 10 ms delay per item to reduce CPU load to keep temperatures down

        raw_data = self.load_and_parse(raw_path)
        nominal_data = self.load_and_parse(nominal_path)
        #tf.print("Parsed dtype raw_data:   ", raw_data.dtype)
        #tf.print("Parsed dtypenominal_data:", nominal_data.dtype)

        #tf.print("Raw loaded sample row 0:  ", raw_data[0, :], summarize=-1)
        #tf.print("RCNom loaded sample row 0:", nominal_data[0, :], summarize=-1)

        #tf.print("===== END OF LoadRawAndRCNomFilePair() =====")
        return raw_data, nominal_data
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def load_and_parse(self, path):
        # Read whole file
        #tf.print("In load_and_parse(): Loading file:", path)
        data = tf.io.read_file(path)
        data = tf.strings.strip(data)

        # Split into lines
        lines = tf.strings.split(data, '\n')
        lines = tf.strings.strip(lines)
        #tf.print("lines[0]: ", lines[0])

        # Split each line into parts (whitespace-separated)
        parts = tf.strings.split(lines)

        # Take only the first 6 columns (ignore any extra like column 7)
        parts = parts[:, :6]

        #tf.debugging.assert_equal(tf.shape(parts)[1], 6, message="Row does not have 6 elements after slicing.")

        # Flatten and convert to numbers
        flat_parts = tf.reshape(parts, [-1])
        #numbers = tf.strings.to_number(flat_parts, tf.float32)
        #numbers = tf.strings.to_number(flat_parts, out_type=tf.float64)
        numbers = tf.strings.to_number(flat_parts, out_type=tf.float32)

        # Reshape back to (n_lines, 6)
        n_lines = tf.shape(numbers)[0] // 6
        data = tf.reshape(numbers, (n_lines, 6))

        # Add extra dimension for CNN input
        #data = tf.expand_dims(data, axis=-1) # COMMENTED OUT 14/06/2025- UNSURE IF IT WILL BREAK THINGS?

        return data
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def TestOneCNNOnSingleExample_GPUVersion( self
                                            , a_AutoEncoder
                                            , a_CNNParameters
                                            , a_ModelBuildingTime
                                            , a_ModelTrainingTime
                                            , a_ExampleIndex
                                            , a_CNNIndex
                                            , a_CurrentHyperParameterSet):
        l_GeneralFunctions.PrintMethodSTART("TestOneCNNOnSingleExample_GPUVersion()", "=", 1, 0)

        # !!! CAN MOVE RESULTS-WRITING STUFF OUT INTO A myResultsMGR CLASS LATER?

        with tf.device('/GPU:0'):
            # Read in the raw example point data - this part could be done with GPU, could improve later
            #l_FilePathsRawDataSingleFile = []
            #l_FilePathsRawDataSingleFile.append(self.m_FilePathsRawData[a_ExampleIndex]);
            l_TestingBatchSize = 1

            #"""
            l_RawDataFullPath = self.m_DataSetMGR.m_ListOfPointDataFullPathsRawAll[a_ExampleIndex]
            l_TestingExampleRaw = l_GeneralFunctions.ReadFileIntoPandasDataframe(l_RawDataFullPath, a_Header=None)
            print("l_TestingExampleRaw.shape:\n", l_TestingExampleRaw.shape)

            l_TestingExampleRaw = l_TestingExampleRaw.iloc[:, :self.m_DataSetMGR.m_NumFeaturesPerPoint]
            l_TestingExampleRaw = l_TestingExampleRaw.to_numpy()
            l_TestingExampleRaw = l_TestingExampleRaw.reshape( 1
                                                             , self.m_DataSetMGR.m_NumRowsPerExampleDataFrame
                                                             , self.m_DataSetMGR.m_NumFeaturesPerPoint)
            #l_TestingExampleRaw = tf.convert_to_tensor(l_TestingExampleRaw, dtype=tf.float64)
            l_TestingExampleRaw = tf.convert_to_tensor(l_TestingExampleRaw, dtype=tf.float32)
            #"""

            l_DateTimeStampCurrentTest = l_GeneralFunctions.GetCurrentDateTimeStamp()

            # run the example's raw data through the CNN and get the denoised prediction
            l_DenoisingTimeStart = time.time()

            #l_DenoisedPrediction = a_AutoEncoder(l_TestingExampleRaw) # slightly less overhead than .predict() way
            #l_DenoisedPrediction = a_AutoEncoder.predict(l_TestingExampleRaw, training=False)  # 26/06/2025 - added training=False - Returns output as a NumPy array, not a tf.Tensor.
            #l_DenoisedPrediction = a_AutoEncoder.predict(l_TestingExampleRaw)  # 26/06/2025 - added training=False - Returns output as a NumPy array, not a tf.Tensor.

            l_DenoisedPrediction = None
            l_CountFailedPredictions = 0
            l_HasNANInFirstRow = True
            while l_HasNANInFirstRow == True:
                l_DenoisedPrediction = a_AutoEncoder.predict(l_TestingExampleRaw)
                l_HasNANInFirstRow = np.isnan(l_DenoisedPrediction[0, :]).any()
                if(l_HasNANInFirstRow == True):
                    l_CountFailedPredictions += 1
                if l_CountFailedPredictions > 10:
                    break
            if l_HasNANInFirstRow:
                return


            print("\n--> l_DenoisedPrediction.shape:", l_DenoisedPrediction.shape)
            l_DenoisingTimeEnd = time.time()
            l_DenoisingTimeTaken = l_DenoisingTimeEnd - l_DenoisingTimeStart
            print("--> Time taken make current denoised prediction:", l_DenoisingTimeTaken)


            self.WriteSingleTestingExampleResultsToFile( l_DenoisedPrediction
                                                       , a_ExampleIndex
                                                       , a_CNNIndex
                                                       , a_CNNParameters
                                                       , a_ModelBuildingTime
                                                       , a_ModelTrainingTime
                                                       , a_CurrentHyperParameterSet
                                                       , l_DenoisingTimeTaken
                                                       , l_DateTimeStampCurrentTest
                                                       , 0)

        l_GeneralFunctions.PrintMethodEND("TestOneCNNOnSingleExample_GPUVersion()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =



    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # - f.visititems() recursively visits every group and dataset in the file.
    # - It checks if the object is a Dataset, then:
    # - Prints the name (e.g., conv1d/conv1d/kernel:0)
    # - Optionally prints its shape and a few values.
    # Can Now:
    # - See if weights differ across saved models.
    # - Confirm determinism by checking whether weights from different training runs with the same seed are identical.
    # - Expand to save weights to .txt or .csv if needed.
    def DisplayCNNWeightsFromFile(self, a_FilePath):
        def walk_h5(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[:]
                print(f"  Dataset: {name}, shape: {data.shape}")
                # Print small datasets, skip large ones
                if data.size <= 10:
                    print(f"    Values: {data}")
                else:
                    print(f"    First few values: {data.flatten()[:5]}")

        with h5py.File(a_FilePath, "r") as f:
            print(f"=== Weights in file: {a_FilePath} ===")
            f.visititems(walk_h5)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SaveFirstFewCNNWeights(self, a_AutoEncoder, a_CNNParameters):
        l_GeneralFunctions.PrintMethodSTART("SaveFirstFewCNNWeights()", "=", 1, 0)

        l_DirModelRecords = l_ProjectInfo.GetDirModelRecords()
        l_DirCNNWeights = l_DirModelRecords + "CNN_weights/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DirCNNWeights)
        l_DirDateTimeStamp = l_DirCNNWeights + self.m_DataSetMGR.m_DateTimeStampOverall + "/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DirDateTimeStamp)

        print("\na_CNNParameters:", a_CNNParameters)
        l_NumFiltersFileEntry = a_CNNParameters[0]['NumFiltersList']
        l_NumFiltersList = ast.literal_eval(l_NumFiltersFileEntry)
        l_NumFiltersStr = ""
        for i in range(0, len(l_NumFiltersList)): # START FROM HERE TOMORROW
            l_NumFiltersStr += str(l_NumFiltersList[i])
            if(i < len(l_NumFiltersList) - 1):
                l_NumFiltersStr += "-"
        print("l_NumFiltersStr:", l_NumFiltersStr)

        l_KernelSizesFileEntry = a_CNNParameters[0]['KernelSizesList']
        l_KernelSizesList = ast.literal_eval(l_KernelSizesFileEntry)
        l_KernelSizesStr = ""
        for i in range(0, len(l_KernelSizesList)):
            l_KernelSizesStr += str(l_KernelSizesList[i])
            if(i < len(l_KernelSizesList) - 1):
                l_KernelSizesStr += "-"
        print("l_KernelSizesStr:", l_KernelSizesStr)

        #l_FileName = "Weights__" + l_NumFiltersStr + "__" + l_KernelSizesStr + "__" + l_GeneralFunctions.GetCurrentDateTimeStamp() + ".h5"
        l_FileName = "First_few_weights" + "__" + l_GeneralFunctions.GetCurrentDateTimeStamp() + ".h5"
        l_FullPath = l_DirDateTimeStamp + l_FileName;

        # Save the entire weights file - this might be large; only interested in recording a few values
        a_AutoEncoder.save_weights(l_FullPath)
        l_FirstFewWeights = self.GetFirstFewCNNWeightsFromFile(l_FullPath)

        # Delete the file of entire weights now that a few have been read from it
        if os.path.exists(l_FullPath):
            os.remove(l_FullPath)

        # Write the few stored weights to file
        l_FileName = l_GeneralFunctions.GetFileNameFromFullPath(l_FullPath)
        l_FileNameNoExtension = l_GeneralFunctions.GetFileNameWithoutExtension(l_FileName)
        l_FullPath2 = l_GeneralFunctions.GetPathBeforeFileName(l_FullPath) + "/" + l_FileNameNoExtension + ".txt"
        with open(l_FullPath2, "w") as f:
            for i in range(len(l_FirstFewWeights)):
                f.write(str(l_FirstFewWeights[i]))

        if os.path.exists(l_FullPath2):
            print("--> Saved a few weights to:\n", l_FullPath2)
        else:
            print("--> FAILED to save a few weights to:\n", l_FullPath2)

        l_GeneralFunctions.PrintMethodEND("SaveFirstFewCNNWeights()", "=", 0, 0)
        return l_FullPath
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GetFirstFewCNNWeightsFromFile(self, a_FilePath):
        l_Return0 = []

        def walk_h5(name, obj):
            l_Return = []
            if isinstance(obj, h5py.Dataset):
                data = obj[:]
                print(f"Layer: {name}, shape: {data.shape}")
                l_Return.append(f"Layer: {name}, shape: {data.shape}\n")

                # Print small datasets, skip large ones
                if data.size <= 10:
                    print(f"Values: {data}")
                    OutputStr = np.array2string(data)
                    l_Return.append(OutputStr)
                else:
                    print(f"First few values: {data.flatten()[:5]}")
                    OutputStr = np.array2string(data)
                    l_Return.append(OutputStr)

                return l_Return

        with h5py.File(a_FilePath, "r") as f:
            #print(f"=== Weights in file: {a_FilePath} ===")
            l_Return0 = f.visititems(walk_h5)

        return l_Return0
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteResultsForCNNParameters( self
                                    , a_CurrentCNNParameters
                                    , a_CircleParametersRaw
                                    , a_CircleParametersRCNominal
                                    , a_CircleParametersDenoised
                                    , a_TestingExampleIndex
                                    , a_TotalExamplesIndex
                                    , a_DateTimeStampOverall
                                    , a_DateTimeStampCurrentCNN
                                    , a_ModelBuildingTime
                                    , a_ModelTrainingTime
                                    , a_ModelTestingTime
                                    , a_DirCNNIndex
                                    , a_CurrentHyperParameterSet):

        l_ProjectInfo = myProjectInfo()
        l_DirModelRecords = l_ProjectInfo.GetDirModelRecords()
        l_DirDenoisingEncoder = l_DirModelRecords + "Denoising_autoencoder/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DirDenoisingEncoder)

        if(l_GeneralFunctions.CheckDirExists(l_DirDenoisingEncoder)):

            l_Data = [
                a_DateTimeStampOverall,
                a_DateTimeStampCurrentCNN,
                a_CurrentHyperParameterSet[0]['NumEpochs'],
                a_CurrentHyperParameterSet[0]['TrainingBatchSize'],
                a_CurrentHyperParameterSet[0]['ActivationFunction'],
                #self.GetTableFriendlyLossFunction(a_CurrentHyperParameterSet[0]['LossFunction']),
                a_CurrentHyperParameterSet[0]['LossFunction'],
                a_CurrentHyperParameterSet[0]['Optimizer'],
                round(a_CircleParametersRaw.m_Radius, 4),
                round(a_CircleParametersRCNominal.m_Radius, 4),
                round(a_CircleParametersDenoised.m_Radius, 4),
                round(a_CircleParametersRaw.m_Circularity, 4),
                round(a_CircleParametersDenoised.m_Circularity, 4),

                round(a_CircleParametersRaw.m_RMSDevFromNomRadius, 4),
                round(a_CircleParametersDenoised.m_RMSDevFromNomRadius, 4),

                #round(a_CircleParametersRaw.m_MeanSquareDeviation, 4),
                #round(a_CircleParametersDenoised.m_MeanSquareDeviation, 4),
                round(a_CircleParametersRaw.m_RMSDevFromXYRadialMean, 4),
                round(a_CircleParametersDenoised.m_RMSDevFromXYRadialMean, 4),

                #round(a_CircleParametersRaw.m_MeanXYRadialDistance, 8),
                round(a_CircleParametersRaw.m_XYRadialMean, 8),

                #round(a_CircleParametersRaw.m_XYRMSDevFromOwnMean, 8),
                round(a_CircleParametersRaw.m_RMSDevFromXYRadialMean, 8),

                #round(a_CircleParametersRaw.m_XYRMSDevFromRCNomPts, 8),
                round(a_CircleParametersRaw.m_RMSDevFromRCNomPts, 8),

                #round(a_CircleParametersDenoised.m_MeanXYRadialDistance, 8),
                round(a_CircleParametersDenoised.m_XYRadialMean, 8),

                #round(a_CircleParametersDenoised.m_XYRMSDevFromOwnMean, 8),
                #round(a_CircleParametersDenoised.m_XYRMSDevFromRCNomPts, 8),
                round(a_CircleParametersDenoised.m_RMSDevFromXYRadialMean, 8),
                round(a_CircleParametersDenoised.m_RMSDevFromRCNomPts, 8),

                round(a_CircleParametersRaw.m_CentreX, 4),
                round(a_CircleParametersRaw.m_CentreY, 4),
                round(a_CircleParametersRaw.m_CentreZ, 4),
                round(a_CircleParametersDenoised.m_CentreX, 4),
                round(a_CircleParametersDenoised.m_CentreY, 4),
                round(a_CircleParametersDenoised.m_CentreZ, 4),
                round(a_ModelBuildingTime, 6),
                round(a_ModelTrainingTime, 6),
                round(a_ModelTestingTime, 6),
                a_CurrentCNNParameters[0]['NumFiltersList'],
                a_CurrentCNNParameters[0]['KernelSizesList']
            ]

            l_ColumnWidths = [21, 21, 5, 5, 10, 20,
                              10, 10, 10, 10, 10, 10,
                              10, 10, 10, 10, 10, 10,
                              10, 10, 10, 10, 10, 10,
                              10, 10, 10, 10, 10, 20,
                              20]

            l_Columns=[ 'DateTimeOverall', 'DateTimeCurrent', 'Num_Epochs', 'Batch_Size'
                      , 'Activation', 'Loss', 'Optimizer', 'Radius_Raw'
                      , 'Radius_RCNom', 'Radius_Denoised', 'Circularity_Raw', 'Circularity_Denoised'
                      , 'MnSqDev_Raw', 'MnSqDev_Denoised', 'MeanXYRadial_Raw', 'XYRMSDevFromOwnMean_Raw'
                      , 'XYRMSDevFromRCNomPts_Raw', 'MeanXYRadial_Denoised', 'XYRMSDevFromOwnMean_Denoised', 'XYRMSDevFromRCNomPts_Denoised'
                      , 'CentreX_Raw', 'CentreY_Raw', 'CentreZ_Raw'
                      , 'CentreX_Denoised', 'CentreY_Denoised', 'Centrez_Denoised'
                      , 'Build_Time', 'Train_Time', 'Test_Time'
                      , 'Num_Filters', 'Kernel_Sizes' ]

            l_NewLineDataFrame = pd.DataFrame( [l_Data]
                                             , l_Columns)

            self.UpdateAllRecordsFile( l_DirDenoisingEncoder
                                     , l_NewLineDataFrame
                                     , l_ColumnWidths)

            self.WriteCurrentRecordsFile( l_DirDenoisingEncoder
                                        , l_NewLineDataFrame
                                        , l_ColumnWidths
                                        , a_DateTimeStampOverall)

            self.WriteIndividualQuantityTablesToFile( a_CurrentCNNParameters
                                                    , a_CircleParametersRaw
                                                    , a_CircleParametersRCNominal
                                                    , a_CircleParametersDenoised
                                                    , a_TestingExampleIndex
                                                    , a_TotalExamplesIndex
                                                    , a_DateTimeStampOverall
                                                    , a_DateTimeStampCurrentCNN
                                                    , a_ModelBuildingTime
                                                    , a_ModelTrainingTime
                                                    , a_ModelTestingTime
                                                    , l_ColumnWidths
                                                    , l_Columns
                                                    , a_CurrentHyperParameterSet)

            l_JSON = {}
            self.SetJSONOutputSingleExample( l_JSON
                                           , a_DateTimeStampOverall
                                           , a_DateTimeStampCurrentCNN
                                           , a_CurrentHyperParameterSet
                                           , a_CurrentCNNParameters
                                           , a_CircleParametersRaw
                                           , a_CircleParametersRCNominal
                                           , a_CircleParametersDenoised
                                           , a_ModelBuildingTime
                                           , a_ModelTrainingTime
                                           , a_ModelTestingTime)

            self.WriteJSONDataForDenoisedExample( l_JSON
                                                , a_DirCNNIndex)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def UpdateAllRecordsFile( self
                            , a_DirDenoisingEncoder
                            , a_NewLineDataFrame
                            , a_ColumnWidths):

        l_FileNameAllRecords = "All_records.txt"
        l_FullPathAllRecords = a_DirDenoisingEncoder + l_FileNameAllRecords
        l_FileExistsAllRecords = l_GeneralFunctions.CheckFileExists(l_FullPathAllRecords)

        if l_FileExistsAllRecords:
            l_GeneralFunctions.WritePandasDataFrameToFile( a_NewLineDataFrame
                                                         , l_FullPathAllRecords
                                                         , a_ColumnWidths
                                                         , 'a')
        else:
            l_GeneralFunctions.WritePandasDataFrameToFile( a_NewLineDataFrame
                                                         , l_FullPathAllRecords
                                                         , a_ColumnWidths
                                                         , 'w')
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteCurrentRecordsFile( self
                               , a_DirDenoisingEncoder
                               , a_NewLineDataFrame
                               , a_ColumnWidths
                               , a_DateTimeStampOverall):

        l_FileNameCurrentRecords = "Testing_" + a_DateTimeStampOverall + ".txt"
        l_FullPathCurrentRecords = a_DirDenoisingEncoder + l_FileNameCurrentRecords
        l_FileExistsCurrentRecords = l_GeneralFunctions.CheckFileExists(l_FullPathCurrentRecords)

        if l_FileExistsCurrentRecords:
            l_GeneralFunctions.WritePandasDataFrameToFile( a_NewLineDataFrame
                                                         , l_FullPathCurrentRecords
                                                         , a_ColumnWidths
                                                         , 'a')
        else:
            l_GeneralFunctions.WritePandasDataFrameToFile( a_NewLineDataFrame
                                                         , l_FullPathCurrentRecords
                                                         , a_ColumnWidths
                                                         , 'w')
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteIndividualQuantityTablesToFile( self
                                           , a_CurrentCNNParameters
                                           , a_CircleParametersRaw
                                           , a_CircleParametersRCNominal
                                           , a_CircleParametersDenoised
                                           , a_TestingExampleIndex
                                           , a_TotalExamplesIndex
                                           , a_DateTimeStampOverall
                                           , a_DateTimeStampCurrentCNN
                                           , a_ModelBuildingTime
                                           , a_ModelTrainingTime
                                           , a_ModelTestingTime
                                           , a_ColumnWidths
                                           , a_Columns
                                           , a_CurrentHyperParameterSet):

        l_QuantityNames=[ 'Radius_Raw', 'Radius_RCNom', 'Radius_Denoised'
                        , 'Circularity_Raw', 'Circularity_Denoised'
                        , 'MnSqDev_Raw', 'MnSqDev_Denoised', 'MeanXYRadial_Raw'
                        , 'XYRMSDevFromOwnMean_Raw', 'XYRMSDevFromRCNomPts_Raw', 'MeanXYRadial_Denoised'
                        , 'XYRMSDevFromOwnMean_Denoised', 'XYRMSDevFromRCNomPts_Denoised'
                        , 'CentreX_Raw', 'CentreY_Raw', 'CentreZ_Raw'
                        , 'CentreX_Denoised', 'CentreY_Denoised', 'Centrez_Denoised'
                        , 'Build_Time', 'Train_Time', 'Test_Time' ]


        # NOT QUITE RIGHT - THIS LOOP IS MAKING 10 LINES BE WRITTEN FOR EACH INTENDED 1 LINE - STILL TRUE 28/09/2025?
        for i in range(0, len(l_QuantityNames)):
            l_CurrentQuantityName = l_QuantityNames[i]
            l_CurrentQuantityColumnIndex = a_Columns.index(l_CurrentQuantityName)
            l_CurrentQuantityColumnWidth = a_ColumnWidths[l_CurrentQuantityColumnIndex]
            l_CurrentQuantityValue = None

            if(l_CurrentQuantityName == 'Radius_Raw'):
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_Radius, 4)
            if(l_CurrentQuantityName == 'Radius_RCNom'):
                l_CurrentQuantityValue = round(a_CircleParametersRCNominal.m_Radius, 4)
            if(l_CurrentQuantityName == 'Radius_Denoised'):
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_Radius, 4)
            if(l_CurrentQuantityName == 'Circularity_Raw'):
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_Circularity, 4)
            if(l_CurrentQuantityName == 'Circularity_Denoised'):
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_Circularity, 4)
            if(l_CurrentQuantityName == 'MnSqDev_Raw'):
                #l_CurrentQuantityValue = round(a_CircleParametersRaw.m_MeanSquareDeviation, 4)
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_RMSDevFromNomRadius, 4)
            if(l_CurrentQuantityName == 'MnSqDev_Denoised'):
                #l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_MeanSquareDeviation, 4)
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_RMSDevFromNomRadius, 4)
            if(l_CurrentQuantityName == 'MeanXYRadial_Raw'):
                #l_CurrentQuantityValue = round(a_CircleParametersRaw.m_MeanXYRadialDistance, 8)
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_XYRadialMean, 8)
            if(l_CurrentQuantityName == 'XYRMSDevFromOwnMean_Raw'):
                #l_CurrentQuantityValue = round(a_CircleParametersRaw.m_XYRMSDevFromOwnMean, 8)
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_RMSDevFromXYRadialMean, 8)
            if(l_CurrentQuantityName == 'XYRMSDevFromRCNomPts_Raw'):
                #l_CurrentQuantityValue = round(a_CircleParametersRaw.m_XYRMSDevFromRCNomPts, 8)
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_RMSDevFromRCNomPts, 8)
            if(l_CurrentQuantityName == 'MeanXYRadial_Denoised'):
                #l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_MeanXYRadialDistance, 8)
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_XYRadialMean, 8)
            if(l_CurrentQuantityName == 'XYRMSDevFromOwnMean_Denoised'):
                #l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_XYRMSDevFromOwnMean, 8)
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_RMSDevFromXYRadialMean, 8)
            if(l_CurrentQuantityName == 'XYRMSDevFromRCNomPts_Denoised'):
                #l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_XYRMSDevFromRCNomPts, 8)
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_RMSDevFromRCNomPts, 8)
            if(l_CurrentQuantityName == 'CentreX_Raw'):
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_CentreX, 4)
            if(l_CurrentQuantityName == 'CentreY_Raw'):
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_CentreY, 4)
            if(l_CurrentQuantityName == 'CentreZ_Raw'):
                l_CurrentQuantityValue = round(a_CircleParametersRaw.m_CentreZ, 4)
            if(l_CurrentQuantityName == 'CentreX_Denoised'):
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_CentreX, 4)
            if(l_CurrentQuantityName == 'CentreY_Denoised'):
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_CentreY, 4)
            if(l_CurrentQuantityName == 'Centrez_Denoised'):
                l_CurrentQuantityValue = round(a_CircleParametersDenoised.m_CentreZ, 4)
            if(l_CurrentQuantityName == 'Build_Time'):
                l_CurrentQuantityValue = round(a_ModelBuildingTime, 6)
            if(l_CurrentQuantityName == 'Train_Time'):
                l_CurrentQuantityValue = round(a_ModelTrainingTime, 6)
            if(l_CurrentQuantityName == 'Test_Time'):
                l_CurrentQuantityValue = round(a_ModelTestingTime, 6)

            l_OutputRow = [ a_DateTimeStampOverall
                          , a_DateTimeStampCurrentCNN
                          , a_CurrentHyperParameterSet[0]['NumEpochs']
                          , a_CurrentHyperParameterSet[0]['TrainingBatchSize']
                          , a_CurrentHyperParameterSet[0]['ActivationFunction']
                          #, self.GetTableFriendlyLossFunction(a_CurrentHyperParameterSet[0]['LossFunction'])
                          , a_CurrentHyperParameterSet[0]['LossFunction']
                          , a_CurrentHyperParameterSet[0]['Optimizer']
                          , l_CurrentQuantityValue
                          , a_CurrentCNNParameters[0]['NumFiltersList']
                          , a_CurrentCNNParameters[0]['KernelSizesList']
                          ]

            l_ColumnTitles = [ 'DateTimeOverall'
                             , 'DateTimeCurrent'
                             , 'Num_Epochs'
                             , 'Batch_Size'
                             , 'Activation'
                             , 'Loss'
                             , 'Optimizer'
                             , (l_CurrentQuantityName)
                             , 'Num_Filters'
                             , 'Kernel_Sizes' ]

            l_ColumnWidths = []
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('DateTimeOverall') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('DateTimeCurrent') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Num_Epochs') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Batch_Size') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Activation') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Loss') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Optimizer') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index(l_CurrentQuantityName) ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Num_Filters') ])
            l_ColumnWidths.append(a_ColumnWidths[ a_Columns.index('Kernel_Sizes') ])

            #l_NewLineDataFrame = pd.DataFrame( [l_OutputRow]
            #                                 , l_ColumnTitles)
            l_NewLineDataFrame = pd.DataFrame( [l_OutputRow] )
            l_NewLineDataFrame.columns = l_ColumnTitles
            #print("l_NewLineDataFrame.shape  ", l_NewLineDataFrame.shape)

            l_ProjectInfo = myProjectInfo()
            l_DirModelRecords = l_ProjectInfo.GetDirModelRecords()
            l_DirDenoisingEncoder = l_DirModelRecords + "Denoising_autoencoder/"
            l_GeneralFunctions.MakeDirIfNonExistent(l_DirDenoisingEncoder)

            l_DirDateTimeStampOverall = l_DirDenoisingEncoder + a_DateTimeStampOverall + "/"
            l_GeneralFunctions.MakeDirIfNonExistent(l_DirDateTimeStampOverall)

            l_DirSingleQuantityTables = l_DirDateTimeStampOverall + "Single_quantity_tables/"
            l_GeneralFunctions.MakeDirIfNonExistent(l_DirSingleQuantityTables)

            l_CurrentFileName = l_CurrentQuantityName + ".txt"
            l_CurrentFullPath = l_DirSingleQuantityTables + l_CurrentFileName;

            l_FileAlreadyExists = l_GeneralFunctions.CheckFileExists(l_CurrentFullPath)
            if l_FileAlreadyExists:
                l_GeneralFunctions.WritePandasDataFrameToFile( l_NewLineDataFrame
                                                             , l_CurrentFullPath
                                                             , l_ColumnWidths
                                                             , 'a')
            else:
                l_GeneralFunctions.WritePandasDataFrameToFile( l_NewLineDataFrame
                                                             , l_CurrentFullPath
                                                             , l_ColumnWidths
                                                             , 'w')
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteJSONDataForDenoisedExample(self, a_JSON, a_DirCNNIndex):
        l_FileName = "Results.json"
        l_FullPathOutput = a_DirCNNIndex + l_FileName

        with open(l_FullPathOutput, "w") as file:
            json.dump(a_JSON, file, indent=2)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SetAndWriteFittedParameters( self
                                   , a_TestingExampleRaw
                                   , a_TestingExampleRCNom
                                   , a_DenoisedPrediction
                                   , a_CNNParameters
                                   , a_ModelBuildingTime
                                   , a_ModelTrainingTime
                                   , a_ExampleIndex
                                   , a_CNNIndex
                                   , a_DenoisingTimeTaken
                                   , a_DirCNNIndex
                                   , a_DateTimeStampCurrentTest
                                   , a_DateTimeStampOverall
                                   , a_CurrentHyperParameterSet):

        # !!! SHOULD PERHAPS FIT THE POINTS FIRST THEN USE THE FITTED RADIUS AS THE 'MEAN XY RADIUS'

        # RCNom
        l_TestingExampleRCNom = a_TestingExampleRCNom[0] # Need to reshape it back into 2D (num_rows x 6) from (1, num_rows, 6)
        l_CircleParametersRCNom = myCircleParameters()
        l_CircleParametersRCNom.Initialise(l_TestingExampleRCNom, self.m_NominalRadius)
        l_CircleParametersRCNom.SetCircleFittedParameters()
        l_CircleParametersRCNom.SetAllRMSInfo(l_TestingExampleRCNom)
        l_CircleParametersRCNom.SetCircularity()
        #l_CircleParametersRCNom.PrintAllValues("l_CircleParametersRCNom")

        # Raw
        l_TestingExampleRaw = a_TestingExampleRaw[0] # Need to reshape it back into 2D (num_rows x 6) from (1, num_rows, 6)
        l_CircleParametersRaw = myCircleParameters()
        l_CircleParametersRaw.Initialise(l_TestingExampleRaw, self.m_NominalRadius)
        l_CircleParametersRaw.SetCircleFittedParameters()
        l_CircleParametersRaw.SetAllRMSInfo(l_TestingExampleRCNom)
        l_CircleParametersRaw.SetCircularity()
        #l_CircleParametersRaw.PrintAllValues("l_CircleParametersRaw")

        # Denoised
        #l_DenoisedPrediction = a_DenoisedPrediction[0]
        l_DenoisedPrediction = a_DenoisedPrediction
        l_CircleParametersDenoised = myCircleParameters()
        l_CircleParametersDenoised.Initialise(l_DenoisedPrediction, self.m_NominalRadius)
        l_CircleParametersDenoised.SetCircleFittedParameters()
        l_CircleParametersDenoised.SetAllRMSInfo(l_TestingExampleRCNom)
        l_CircleParametersDenoised.SetCircularity()
        #l_CircleParametersDenoised.PrintAllValues("l_CircleParametersDenoised")

        self.WriteResultsForCNNParameters( a_CNNParameters
                                         , l_CircleParametersRaw
                                         , l_CircleParametersRCNom
                                         , l_CircleParametersDenoised
                                         , 1 #<-- l_TotalTestingExamplesHandled
                                         , a_ExampleIndex
                                         , a_DateTimeStampOverall
                                         , a_DateTimeStampCurrentTest
                                         , a_ModelBuildingTime
                                         , a_ModelTrainingTime
                                         , a_DenoisingTimeTaken
                                         , a_DirCNNIndex
                                         , a_CurrentHyperParameterSet)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SetJSONOutputSingleExample( self
                                  , a_JSON
                                  , a_DateTimeStampOverall
                                  , a_DateTimeStampCurrentCNN
                                  , a_CurrentHyperParameterSet
                                  , a_CurrentCNNParameters
                                  , a_CircleParametersRaw
                                  , a_CircleParametersRCNominal
                                  , a_CircleParametersDenoised
                                  , a_ModelBuildingTime
                                  , a_ModelTrainingTime
                                  , a_ModelTestingTime):

        l_GeneralFunctions.PrintMethodSTART("SetJSONOutputSingleExample()", "=", 1, 0)

        a_JSON['DateTimeOverall']               = a_DateTimeStampOverall
        a_JSON['DateTimeCurrent']               = a_DateTimeStampCurrentCNN
        a_JSON['Num_Epochs']                    = a_CurrentHyperParameterSet[0]['NumEpochs']
        a_JSON['Batch_Size']                    = a_CurrentHyperParameterSet[0]['TrainingBatchSize']
        a_JSON['Activation']                    = a_CurrentHyperParameterSet[0]['ActivationFunction']
        #a_JSON['Loss']                         = self.GetTableFriendlyLossFunction(self.m_HyperParameters.m_LossFunction)
        a_JSON['Loss']                          = a_CurrentHyperParameterSet[0]['LossFunction']
        a_JSON['Optimizer']                     = a_CurrentHyperParameterSet[0]['Optimizer']
        a_JSON['Num_Filters']                   = a_CurrentCNNParameters[0]['NumFiltersList']
        a_JSON['Kernel_Sizes']                  = a_CurrentCNNParameters[0]['KernelSizesList']
        a_JSON['Build_Time']                    = round(a_ModelBuildingTime, 6)
        a_JSON['Train_Time']                    = round(a_ModelTrainingTime, 6)
        a_JSON['Test_Time']                     = round(a_ModelTestingTime, 6)

        # Fitted circle parameters
        a_JSON['Radius_Raw']                    = round(a_CircleParametersRaw.m_Radius, 6)
        a_JSON['Radius_RCNom']                  = round(a_CircleParametersRCNominal.m_Radius, 6)
        a_JSON['Radius_Denoised']               = round(a_CircleParametersDenoised.m_Radius, 6)

        a_JSON['CentreX_Raw']                   = round(a_CircleParametersRaw.m_CentreX, 4)
        a_JSON['CentreY_Raw']                   = round(a_CircleParametersRaw.m_CentreY, 4)
        a_JSON['CentreZ_Raw']                   = round(a_CircleParametersRaw.m_CentreZ, 4)

        a_JSON['CentreX_Denoised']              = round(a_CircleParametersDenoised.m_CentreX, 4)
        a_JSON['CentreY_Denoised']              = round(a_CircleParametersDenoised.m_CentreY, 4)
        a_JSON['Centrez_Denoised']              = round(a_CircleParametersDenoised.m_CentreZ, 4)

        # Circularity [UNSURE IF CALCULATED CORRECTLY]
        a_JSON['Circularity_Raw']               = round(a_CircleParametersRaw.m_Circularity, 4)
        a_JSON['Circularity_Denoised']          = round(a_CircleParametersDenoised.m_Circularity, 4)

        # Mean XY radial distance of points from nominal centre
        a_JSON['MeanXYRadialDist_Raw']          = round(a_CircleParametersRaw.m_XYRadialMean, 8)
        a_JSON['MeanXYRadialDist_RCNom']        = round(a_CircleParametersRCNominal.m_XYRadialMean, 8)
        a_JSON['MeanXYRadialDist_Denoised']     = round(a_CircleParametersDenoised.m_XYRadialMean, 8)

        # RMS deviation from nominal radius
        a_JSON['RMSDevFromNomRadius_Raw']       = round(a_CircleParametersRaw.m_RMSDevFromNomRadius, 6)
        a_JSON['RMSDevFromNomRadius_RCNom']     = round(a_CircleParametersRCNominal.m_RMSDevFromNomRadius , 6)
        a_JSON['RMSDevFromNomRadius_Denoised']  = round(a_CircleParametersDenoised.m_RMSDevFromNomRadius, 6)

        # RMS deviation from fitted radius
        a_JSON['RMSDevFromFittedRadius_Raw']    = round(a_CircleParametersRaw.m_RMSDevFromFittedRadius, 6)
        a_JSON['RMSDevFromFittedRadius_RCNom']  = round(a_CircleParametersRCNominal.m_RMSDevFromFittedRadius, 6)
        a_JSON['RMSDevFromFittedRadius_Denoised'] = round(a_CircleParametersDenoised.m_RMSDevFromFittedRadius, 6)

        # RMS deviation of points from mean XY radius
        a_JSON['RMSDevFromXYRadialMean_Raw']    = round(a_CircleParametersRaw.m_RMSDevFromXYRadialMean, 8)
        a_JSON['RMSDevFromXYRadialMean_RCNom']  = round(a_CircleParametersRCNominal.m_RMSDevFromXYRadialMean, 8)
        a_JSON['RMSDevFromXYRadialMean_Denoised'] = round(a_CircleParametersDenoised.m_RMSDevFromXYRadialMean, 8)

        # RMS deviation of points from corresponding RCNom points
        a_JSON['RMSDevFromRCNomPts_Raw']        = round(a_CircleParametersRaw.m_RMSDevFromRCNomPts, 8)
        a_JSON['RMSDevFromRCNomPts_RCNom']      = round(a_CircleParametersRCNominal.m_RMSDevFromRCNomPts, 8)
        a_JSON['RMSDevFromRCNomPts_Denoised']   = round(a_CircleParametersDenoised.m_RMSDevFromRCNomPts, 8)

        print(json.dumps(a_JSON, indent=2))

        l_GeneralFunctions.PrintMethodEND("SetJSONOutputSingleExample()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def TestOneCNNOnBatches_GPUVersion( self
                                      , a_AutoEncoder
                                      , a_CurrentCNNDefinition
                                      , a_ModelBuildingTime
                                      , a_ModelTrainingTime
                                      , a_CNNIndex
                                      , a_CurrentHyperParameterSet):

        l_GeneralFunctions.PrintMethodSTART("TestOneCNNOnBatches_GPUVersion()", "=", 3, 0)

        l_TestingBatchSize = int(a_CurrentHyperParameterSet[0]["TestingBatchSize"])

        print("l_TestingBatchSize:", l_TestingBatchSize)
        #l_TestDataset = tf.data.Dataset.from_tensor_slices(self.m_FilePathsRawDataTesting)
        l_TestDataset = tf.data.Dataset.from_tensor_slices(self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTesting)
        print("len(self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTesting):", len(self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTesting))
        print("m_DataSetMGR.m_ListOfPointDataFullPathsRawTesting[0]:\n", self.m_DataSetMGR.m_ListOfPointDataFullPathsRawTesting[0])

        l_TestDataset = l_TestDataset.map(self.load_and_parse).batch(l_TestingBatchSize)
        # Do I really need to use the same batch size for testing as I have used for training?
        # # Probably not, but it is easier to keep it the same for now.

        with tf.device('/GPU:0'):
            l_NumTestingExamplesHandled = 0
            l_BatchIndex = 0
            for l_CurrentBatch in l_TestDataset:
                l_BatchIndex += 1

                #if(l_BatchIndex > 1):
                # TESTING ONLY
                # break

                print("--> l_BatchIndex =", l_BatchIndex, " - l_NumTestingExamplesHandled =", l_NumTestingExamplesHandled)
                #print("--> l_CurrentBatch.shape:", l_CurrentBatch.shape)

                # Train on current training batch...
                #print("--> Passing current batch of testing set examples to trained model now to make predictions on...")
                l_DateTimeStampCurrentBatchTest = l_GeneralFunctions.GetCurrentDateTimeStamp()
                l_BatchDenoisingTimeStart = time.time()
                l_DenoisedPredictions = a_AutoEncoder(l_CurrentBatch)
                l_BatchDenoisingTimeEnd = time.time()
                l_BatchDenoisingTimeTaken = l_BatchDenoisingTimeEnd - l_BatchDenoisingTimeStart
                l_MeanDenoisedPredictionTimeTakenPerExampleInBatch = l_BatchDenoisingTimeTaken / l_TestingBatchSize
                #print("--> Time taken to make denoised predictions on current testing batch:", l_BatchDenoisingTimeTaken)
                #print("--> Mean time taken to predict denoised result for one example in batch:", l_MeanDenoisedPredictionTimeTakenPerExampleInBatch)
                #print("--> Will now calculate fitted circle parameters for Raw, RCNom, Denoised circles and write to file...")
                #print("\nl_DenoisedPredictions[0].shape:", l_DenoisedPredictions[0].shape)
                #print("l_DenoisedPredictions[0, 0:3,:]:\n",l_DenoisedPredictions[0, 0:3,:])
                #print("\nl_DenoisedPredictions[1].shape:", l_DenoisedPredictions[1].shape)
                #print("l_DenoisedPredictions[1, 0:3,:]:\n",l_DenoisedPredictions[1, 0:3,:])

                # NEED TO IMPLEMENT WRITING RESULTS TO FILE HERE

                self.WriteTestingBatchResultsToFile( l_DenoisedPredictions
                                                   , l_NumTestingExamplesHandled
                                                   , l_TestingBatchSize
                                                   , a_AutoEncoder
                                                   , a_CurrentCNNDefinition
                                                   , a_ModelBuildingTime
                                                   , a_ModelTrainingTime
                                                   , a_CNNIndex
                                                   , a_CurrentHyperParameterSet
                                                   , l_MeanDenoisedPredictionTimeTakenPerExampleInBatch
                                                   , l_DateTimeStampCurrentBatchTest)

                l_NumTestingExamplesHandled += l_TestingBatchSize

        l_GeneralFunctions.PrintMethodEND("TestOneCNNOnBatches_GPUVersion()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteTestingBatchResultsToFile( self
                                      , a_DenoisedPredictions
                                      , a_NumTestingExamplesHandled
                                      , a_TestingBatchSize
                                      , a_AutoEncoder # maybe unnecessary
                                      , a_CurrentCNNDefinition
                                      , a_ModelBuildingTime
                                      , a_ModelTrainingTime
                                      , a_CNNIndex
                                      , a_CurrentHyperParameterSet
                                      , a_MeanDenoisingTimeTaken
                                      , a_DateTimeStampCurrentBatchTest):

        l_GeneralFunctions.PrintMethodSTART("WriteTestingBatchResultsToFile()", "=", 1, 0)

        print("a_DenoisedPredictions.shape:", a_DenoisedPredictions.shape)

        for l_ExampleInBatchIndex in range(0, a_TestingBatchSize):
            print("l_ExampleInBatchIndex:", l_ExampleInBatchIndex)
            print("Total testing examples written for:", a_NumTestingExamplesHandled + l_ExampleInBatchIndex)


            #l_CurrentDenoisedPrediction = a_DenoisedPredictions[l_ExampleInBatchIndex]
            l_TestingSetIndex = a_NumTestingExamplesHandled + l_ExampleInBatchIndex
            l_ExampleIndex = self.m_DataSetMGR.m_ListOfTestingExampleIndices[l_TestingSetIndex]

            self.WriteSingleTestingExampleResultsToFile( #l_CurrentDenoisedPrediction
                                                         a_DenoisedPredictions
                                                       , l_ExampleIndex
                                                       , a_CNNIndex
                                                       , a_CurrentCNNDefinition
                                                       , a_ModelBuildingTime
                                                       , a_ModelTrainingTime
                                                       , a_CurrentHyperParameterSet
                                                       , a_MeanDenoisingTimeTaken
                                                       , a_DateTimeStampCurrentBatchTest
                                                       , l_ExampleInBatchIndex)

        l_GeneralFunctions.PrintMethodEND("WriteTestingBatchResultsToFile()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteSingleTestingExampleResultsToFile( self
                                              #a_DenoisedPrediction
                                              , a_DenoisedPredictions
                                              , a_ExampleIndex
                                              , a_CNNIndex
                                              , a_CNNParameters
                                              , a_ModelBuildingTime
                                              , a_ModelTrainingTime
                                              , a_CurrentHyperParameterSet
                                              , a_MeanDenoisingTimeTaken
                                              , a_DateTimeStampCurrentBatchTest
                                              , a_ExampleInBatchIndex):
        l_GeneralFunctions.PrintMethodSTART("WriteSingleTestingExampleResultsToFile()", "=", 1, 0)

        # Write the denoised prediction to file and read in the raw and RCNom data for circle parameter calculations
        l_DenoisedPrediction = a_DenoisedPredictions[a_ExampleInBatchIndex]
        #print("l_DenoisedPrediction.shape:", l_DenoisedPrediction.shape)

        #l_FilePathRawData = self.m_FilePathsRawData[a_ExampleIndex]
        l_FilePathRawData = self.m_DataSetMGR.m_ListOfPointDataFullPathsRawAll[a_ExampleIndex]
        #l_FilePathRCNomData = self.m_FilePathsRCNomData[a_ExampleIndex]
        l_FilePathRCNomData = self.m_DataSetMGR.m_ListOfPointDataFullPathsRCNomAll[a_ExampleIndex]
        l_FileName = l_GeneralFunctions.GetFileNameFromFullPath(l_FilePathRawData)

        #l_ExampleFolder = self.m_DirDataSets + "Csy_" + str(a_ExampleIndex) + "/"
        l_ExampleFolder = self.m_DataSetMGR.m_DirPointData + "Csy_" + str(a_ExampleIndex) + "/"

        l_DenoisedExampleFolder = l_ExampleFolder + "Denoised_data/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DenoisedExampleFolder)
        l_DateTimeStampDenoisedFileFolder = l_DenoisedExampleFolder + self.m_DataSetMGR.m_DateTimeStampOverall + "/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DateTimeStampDenoisedFileFolder)
        l_DateTimeStampDenoisedFileFolder = l_DenoisedExampleFolder + self.m_DataSetMGR.m_DateTimeStampOverall + "/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DateTimeStampDenoisedFileFolder)
        l_DirCNNIndex = l_DateTimeStampDenoisedFileFolder + "CNN_index_" + str(a_CNNIndex) + "/"
        l_GeneralFunctions.MakeDirIfNonExistent(l_DirCNNIndex)
        l_DenoisedExampleFullPath  = l_DirCNNIndex + l_FileName

        #l_DenoisedPredictionPandasDF = pd.DataFrame(a_DenoisedPredictions[0]) # Need a Pandas dataframe, not a Numpy dataframe
        l_DenoisedPredictionPandasDF = pd.DataFrame(l_DenoisedPrediction) # Need a Pandas dataframe, not a Numpy dataframe
        l_ColumnWidths = [10, 10, 10, 10, 10, 10]
        l_DenoisedPredictionPandasDF = l_DenoisedPredictionPandasDF.round(6)

        #l_GeneralFunctions.WriteDataFrameToFile( l_DenoisedPredictionPandasDF
        l_GeneralFunctions.WritePandasDataFrameToFile( l_DenoisedPredictionPandasDF
                                                     , l_DenoisedExampleFullPath
                                                     , l_ColumnWidths
                                                     , 'w')


        # Read in Raw data
        l_RawDataFullPath = self.m_DataSetMGR.m_ListOfPointDataFullPathsRawAll[a_ExampleIndex]
        l_TestingExampleRaw = l_GeneralFunctions.ReadFileIntoPandasDataframe(l_RawDataFullPath, a_Header=None)
        l_TestingExampleRaw = l_TestingExampleRaw.iloc[:, :self.m_DataSetMGR.m_NumFeaturesPerPoint]
        l_TestingExampleRaw = l_TestingExampleRaw.to_numpy()
        l_TestingExampleRaw = l_TestingExampleRaw.reshape( 1
                                                         , self.m_DataSetMGR.m_NumRowsPerExampleDataFrame
                                                         , self.m_DataSetMGR.m_NumFeaturesPerPoint)

        # Read in RCNom data
        #l_TestingExampleRCNom = l_GeneralFunctions.ReadFileIntoDataframe(l_FilePathRCNomData)
        l_TestingExampleRCNom = l_GeneralFunctions.ReadFileIntoPandasDataframe(l_FilePathRCNomData)

        l_TestingExampleRCNom = l_TestingExampleRCNom.iloc[:, :self.m_DataSetMGR.m_NumFeaturesPerPoint]
        l_TestingExampleRCNom = l_TestingExampleRCNom.to_numpy()
        l_TestingExampleRCNom = l_TestingExampleRCNom.reshape( 1
                                                             #, self.m_NumRowsPerExampleDataFrame
                                                             , self.m_DataSetMGR.m_NumRowsPerExampleDataFrame
                                                             #, 6
                                                             , self.m_DataSetMGR.m_NumFeaturesPerPoint)


        #print("\nRaw data frame first 3 rows:\n", l_FilePathRawData)
        #print(l_TestingExampleRaw[0, 0:3, :])
        #print("\nRCNom data frame first 3 rows:\n", l_FilePathRCNomData)
        #print(l_TestingExampleRCNom[0, 0:3, :])
        #print("\nDenoised prediction first 3 rows:\n", l_DenoisedExampleFullPath)
        #print(l_DenoisedPrediction[0:3, :])

        if isinstance(l_TestingExampleRaw, tf.Tensor):
            l_TestingExampleRaw = l_TestingExampleRaw.numpy()  # Convert to NumPy
            #l_TestingExampleRaw = l_TestingExampleRaw.astype(np.float64)
            l_TestingExampleRaw = l_TestingExampleRaw.astype(np.float32)

        self.SetAndWriteFittedParameters( l_TestingExampleRaw
                                        , l_TestingExampleRCNom
                                        #, a_DenoisedPrediction
                                        , l_DenoisedPrediction
                                        , a_CNNParameters
                                        , a_ModelBuildingTime
                                        , a_ModelTrainingTime
                                        , a_ExampleIndex
                                        , a_CNNIndex
                                        , a_MeanDenoisingTimeTaken
                                        , l_DirCNNIndex
                                        , a_DateTimeStampCurrentBatchTest
                                        , self.m_DataSetMGR.m_DateTimeStampOverall
                                        , a_CurrentHyperParameterSet)

        l_GeneralFunctions.PrintMethodEND("WriteSingleTestingExampleResultsToFile()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================
