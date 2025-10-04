# src/myDataSetMGR/myDataSetMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()
import re
import random


#==============================================================================
class myDataSetMGR:

    #--------------------------------------------------------------------------
    def __init__(self, a_DataSetName):
        l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.__init__()", "=", 1, 0)

        self.m_DataSetName = a_DataSetName
        self.m_DirAllDataSets = l_ProjectInfo.GetDirLogFiles()
        self.m_DirDataSet = self.m_DirAllDataSets + self.m_DataSetName + "/"
        self.m_DirPointData = self.m_DirDataSet + "Point_data/"
        self.m_ListOfPointDataExampleDirs = []
        self.m_ListOfPointDataExampleDirsRaw = []
        self.m_ListOfPointDataExampleDirsRCNom = []
        self.m_ListOfPointDataExampleDirsDenoised = []
        self.m_ListOfPointDataExampleDirsRawTraining = []
        self.m_ListOfPointDataExampleDirsRawTesting = []
        self.m_ListOfPointDataExampleDirsRCNomTraining = []
        self.m_ListOfPointDataExampleDirsRCNomTesting = []
        self.m_ListOfPointDataFullPathsRawAll = []
        self.m_ListOfPointDataFullPathsRawTraining = []
        self.m_ListOfPointDataFullPathsRawTesting = []
        self.m_ListOfPointDataFullPathsRCNomAll = []
        self.m_ListOfPointDataFullPathsRCNomTraining = []
        self.m_ListOfPointDataFullPathsRCNomTesting = []
        self.m_ListOfPointDataFullPathsDenoised = []
        self.m_ListOfPointDataFullPathsDenoisedTraining = []
        self.m_ListOfPointDataFullPathsDenoisedTesting = []
        self.m_ListOfTotalExampleIndices = []
        self.m_ListOfTrainingExampleIndices = []
        self.m_ListOfTestingExampleIndices = []
        self.m_RandomSeedTrainingTestingSplit = 42
        self.m_TrainingSetPercentageOfTotal = 70
        self.m_DateTimeStampOverall = "NOT_SET"
        self.m_NumRowsPerExampleDataFrame = -1
        self.m_NumFeaturesPerPoint = -1

        self.Initialise()

        l_GeneralFunctions.PrintMethodEND("myDataSetMGR.__init__()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def Initialise(self):
        #l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.Initialise()", "=", 1, 0)

        self.m_DateTimeStampOverall = l_GeneralFunctions.GetCurrentDateTimeStamp()
        self.SetListOfPointDataExampleDirs()
        self.SetListOfRawPointDataDirs()
        self.SetListOfRCNomPointDataDirs()
        self.SetListOfTotalExampleIndices()
        self.SetListOfTrainingAndTestingExampleIndices()
        self.SetListsOfPointDataDirsTrainingAndTesting()
        self.SetListOfTrainingAndTestingFullPathsRaw()
        self.SetListOfTrainingAndTestingFullPathsRCNom()
        self.SetNumRowsPerExampleDataFrame()
        self.SetNumFeaturesPerPoint()

        #l_GeneralFunctions.PrintMethodEND("myDataSetMGR.Initialise()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfPointDataExampleDirs(self):
        #l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.SetListOfPointDataExampleDirs()", "=", 1, 0)

        l_DirsInDirPointData = l_GeneralFunctions.GetDirsInDir(self.m_DirPointData)
        self.m_ListOfPointDataExampleDirs = self.ExtractCsyNDirsFromList(l_DirsInDirPointData)

        #l_GeneralFunctions.PrintMethodEND("myDataSetMGR.SetListOfPointDataExampleDirs()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def ExtractCsyNDirsFromList(self, a_FolderList):
        pattern = re.compile(r'^Csy_\d+$')
        return [folder for folder in a_FolderList if pattern.match(folder)]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfRawPointDataDirs(self):
        #l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.SetListOfRawPointDataDirs()", "=", 1, 0)

        self.m_ListOfPointDataExampleDirs
        for i in range(0, len(self.m_ListOfPointDataExampleDirs)):
            l_Dir = self.m_ListOfPointDataExampleDirs[i]
            l_RawDir = self.m_DirPointData + l_Dir + "/Raw_data/"
            self.m_ListOfPointDataExampleDirsRaw.append(l_RawDir)
            #print(l_RawDir)

        #l_GeneralFunctions.PrintMethodEND("myDataSetMGR.SetListOfRawPointDataDirs()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfRCNomPointDataDirs(self):
        #l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.SetListOfRCNomPointDataDirs()", "=", 1, 0)

        self.m_ListOfPointDataExampleDirs
        for i in range(0, len(self.m_ListOfPointDataExampleDirs)):
            l_Dir = self.m_ListOfPointDataExampleDirs[i]
            l_RCNomDir = self.m_DirPointData + l_Dir + "/Nominal_points_reconstructed_from_raw_data/"
            self.m_ListOfPointDataExampleDirsRCNom.append(l_RCNomDir)
            #print(l_RCNomDir)

        #l_GeneralFunctions.PrintMethodEND("myDataSetMGR.SetListOfRCNomPointDataDirs()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfTotalExampleIndices(self):
        for i in range(0, len(self.m_ListOfPointDataExampleDirs)):
            l_CurrentPointDataExampleDirPath = self.m_ListOfPointDataExampleDirs[i]
            l_BottomDirNameOnly = l_GeneralFunctions.GetFinalDirFromDirPath(l_CurrentPointDataExampleDirPath)
            l_CurrentExampleIDNumber = l_GeneralFunctions.ExtractNumberFromCsyDirName(l_BottomDirNameOnly)
            self.m_ListOfTotalExampleIndices.append(l_CurrentExampleIDNumber);
        #print(self.m_ListOfTotalExampleIndices)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfTrainingAndTestingExampleIndices(self):
        [self.m_ListOfTrainingExampleIndices, self.m_ListOfTestingExampleIndices] = l_GeneralFunctions.SplitListByPercentage( self.m_ListOfTotalExampleIndices
                                                                                                                            , self.m_TrainingSetPercentageOfTotal
                                                                                                                            , self.m_RandomSeedTrainingTestingSplit)

        self.m_ListOfTrainingExampleIndices = sorted(self.m_ListOfTrainingExampleIndices)
        self.m_ListOfTestingExampleIndices = sorted(self.m_ListOfTestingExampleIndices)
        #print("self.m_ListOfTrainingExampleIndices:\n", self.m_ListOfTrainingExampleIndices)
        #print("\nself.m_ListOfTestingExampleIndices:\n", self.m_ListOfTestingExampleIndices)
        print("len(self.m_ListOfTrainingExampleIndices):           ", len(self.m_ListOfTrainingExampleIndices))
        print("len(self.m_ListOfTestingExampleIndices):            ", len(self.m_ListOfTestingExampleIndices))
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListsOfPointDataDirsTrainingAndTesting(self):
        self.m_ListOfPointDataExampleDirsRawTraining = [self.m_ListOfPointDataExampleDirsRaw[i] for i in self.m_ListOfTrainingExampleIndices]
        self.m_ListOfPointDataExampleDirsRawTesting = [self.m_ListOfPointDataExampleDirsRaw[i] for i in self.m_ListOfTestingExampleIndices]

        self.m_ListOfPointDataExampleDirsRCNomTraining = [self.m_ListOfPointDataExampleDirsRCNom[i] for i in self.m_ListOfTrainingExampleIndices]
        self.m_ListOfPointDataExampleDirsRCNomTesting = [self.m_ListOfPointDataExampleDirsRCNom[i] for i in self.m_ListOfTestingExampleIndices]

        print("len(self.m_ListOfPointDataExampleDirsRawTraining):  ", len(self.m_ListOfPointDataExampleDirsRawTraining))
        print("len(self.m_ListOfPointDataExampleDirsRawTesting):   ", len(self.m_ListOfPointDataExampleDirsRawTesting))
        print("len(self.m_ListOfPointDataExampleDirsRCNomTraining):", len(self.m_ListOfPointDataExampleDirsRCNomTraining))
        print("len(self.m_ListOfPointDataExampleDirsRCNomTesting): ", len(self.m_ListOfPointDataExampleDirsRCNomTesting))

        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataExampleDirsRawTraining)
        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataExampleDirsRawTesting)
        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataExampleDirsRCNomTraining)
        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataExampleDirsRCNomTesting)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def ExtractCsyNumber(self, path):
        match = re.search(r'Csy_(\d+)', path)
        return int(match.group(1)) if match else -1  # fallback if no match found
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SortDirPathsByExampleIDNumber(self, a_List):
        a_List.sort(key=self.ExtractCsyNumber)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfTrainingAndTestingFullPathsRaw(self):
        for i in range(0, len(self.m_ListOfPointDataExampleDirsRawTraining)):
            l_CurrentDir = self.m_ListOfPointDataExampleDirsRawTraining[i]
            l_CurrentDir = l_CurrentDir + "EqualisedLength/Normalised_XY/"
            l_FullPathsToFilesInDir = l_GeneralFunctions.GetFullPathsOfFilesInDir(l_CurrentDir)
            l_CurrentFullPath = l_FullPathsToFilesInDir[0]
            self.m_ListOfPointDataFullPathsRawTraining.append(l_CurrentFullPath)

        for i in range(0, len(self.m_ListOfPointDataExampleDirsRawTesting)):
            l_CurrentDir = self.m_ListOfPointDataExampleDirsRawTesting[i]
            l_CurrentDir = l_CurrentDir + "EqualisedLength/Normalised_XY/"
            l_FullPathsToFilesInDir = l_GeneralFunctions.GetFullPathsOfFilesInDir(l_CurrentDir)
            l_CurrentFullPath = l_FullPathsToFilesInDir[0]
            self.m_ListOfPointDataFullPathsRawTesting.append(l_CurrentFullPath)

        self.m_ListOfPointDataFullPathsRawAll = self.m_ListOfPointDataFullPathsRawTraining + self.m_ListOfPointDataFullPathsRawTesting
        print("len(self.m_ListOfPointDataFullPathsRawTraining):    ", len(self.m_ListOfPointDataFullPathsRawTraining))
        print("len(self.m_ListOfPointDataFullPathsRawTesting):     ", len(self.m_ListOfPointDataFullPathsRawTesting))
        print("len(self.m_ListOfPointDataFullPathsRawAll):         ", len(self.m_ListOfPointDataFullPathsRawAll))
        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataFullPathsRawAll)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetListOfTrainingAndTestingFullPathsRCNom(self):
        for i in range(0, len(self.m_ListOfPointDataExampleDirsRCNomTraining)):
            l_CurrentDir = self.m_ListOfPointDataExampleDirsRCNomTraining[i]
            l_CurrentDir = l_CurrentDir + "Normalised_XY/"
            l_FullPathsToFilesInDir = l_GeneralFunctions.GetFullPathsOfFilesInDir(l_CurrentDir)
            l_CurrentFullPath = l_FullPathsToFilesInDir[0]
            self.m_ListOfPointDataFullPathsRCNomTraining.append(l_CurrentFullPath)

        for i in range(0, len(self.m_ListOfPointDataExampleDirsRCNomTesting)):
            l_CurrentDir = self.m_ListOfPointDataExampleDirsRCNomTesting[i]
            l_CurrentDir = l_CurrentDir + "Normalised_XY/"
            l_FullPathsToFilesInDir = l_GeneralFunctions.GetFullPathsOfFilesInDir(l_CurrentDir)
            l_CurrentFullPath = l_FullPathsToFilesInDir[0]
            self.m_ListOfPointDataFullPathsRCNomTesting.append(l_CurrentFullPath)

        self.m_ListOfPointDataFullPathsRCNomAll= self.m_ListOfPointDataFullPathsRCNomTraining + self.m_ListOfPointDataFullPathsRCNomTesting
        print("len(self.m_ListOfPointDataFullPathsRCNomTraining):  ", len(self.m_ListOfPointDataFullPathsRCNomTraining))
        print("len(self.m_ListOfPointDataFullPathsRCNomTesting):   ", len(self.m_ListOfPointDataFullPathsRCNomTesting))
        print("len(self.m_ListOfPointDataFullPathsRCNomAll):       ", len(self.m_ListOfPointDataFullPathsRCNomAll))
        self.SortDirPathsByExampleIDNumber(self.m_ListOfPointDataFullPathsRCNomAll)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetNumRowsPerExampleDataFrame(self):
        l_FirstRawDataFile = self.m_ListOfPointDataFullPathsRawTraining[0]
        with open(l_FirstRawDataFile, 'r') as file:
            l_LinesInFile = [line.strip() for line in file if line.strip()] # get only non-empty lines
            self.m_NumRowsPerExampleDataFrame = len(l_LinesInFile)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetNumFeaturesPerPoint(self):
        self.m_NumFeaturesPerPoint = 6
    #--------------------------------------------------------------------------


#==============================================================================