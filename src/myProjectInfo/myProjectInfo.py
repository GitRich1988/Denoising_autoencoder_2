# src/myProjectInfo/myProjectInfo.py
from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions

#==============================================================================
class myProjectInfo:

    def __init__(self):
        None

    def GetProjectDir(self):
       return "C:/Users/marti/Documents/Machine_learning/Denoising_autoencoder_2/"

    def GetDirLogFiles(self):
        return "C:/Users/marti/Documents/Machine_learning/Work_stuff/Log_files/"

    def GetDirTrainingInfo(self):
        return self.GetProjectDir() + "Training_info/"

    def GetDirModelRecords(self):
        return self.GetDirTrainingInfo() + "Model_records/"

    def Setup(self):
        l_GeneralFunctions.MakeDirIfNonExistent(self.GetDirTrainingInfo())
        l_GeneralFunctions.MakeDirIfNonExistent(self.GetDirModelRecords())
#==============================================================================