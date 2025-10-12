print("[IMPORT] from tensorflow.keras import mixed_precision")
from tensorflow.keras import mixed_precision
print("[IMPORT] mixed_precision.set_global_policy('float32')")
mixed_precision.set_global_policy('float32')
print("[IMPORT] from tensorflow.keras import layers")
from tensorflow.keras import layers
print("[IMPORT] from tensorflow.keras.models import Sequential")
from tensorflow.keras.models import Sequential

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()
from src.myDataSetMGR.myDataSetMGR import myDataSetMGR
from src.myModelMGR.myModelMGR import myModelMGR
from src.myInputParametersMGR.myInputParametersMGR import myInputParametersMGR


#==============================================================================
def main():
    l_GeneralFunctions.PrintMethodSTART("main()", "=", 1, 0)

    l_ProjectInfo.Setup()

    # Only uncomment these lines if you want to generate new CNN definitions:
    # (Can edit the upper and lower limits for the ranges of NumFilters and 
    # KernelSizes for each layer in GenerateNewCNNDefinitionsFile())
    #l_InputParametersMGR = myInputParametersMGR()
    #l_InputParametersMGR.GenerateNewCNNDefinitionsFile()
    

    l_DataSetMGR = myDataSetMGR("1000_scans__SphereCsy__EqX_4__ScanParSpeed_25")
    #l_ModelMGR = myModelMGR(l_DataSetMGR, "CNN_definition_1.json", "Hyper_parameters_1.json")
    #l_ModelMGR = myModelMGR(l_DataSetMGR, "CNN_definition_2.json", "Hyper_parameters_1.json")
    l_ModelMGR = myModelMGR(l_DataSetMGR, "CNN_definition__2025-10-12--11-07-10.json", "Hyper_parameters_1.json")
    l_ModelMGR.Run()

    l_GeneralFunctions.PrintMethodEND("main()", "=", 0, 0)
#==============================================================================

#==============================================================================
if __name__ == "__main__":
    main()
#==============================================================================
