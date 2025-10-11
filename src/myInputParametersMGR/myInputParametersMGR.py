# src/myDataSetMGR/myInputParametersMGR.py

from sre_constants import RANGE
from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()

from itertools import product

#==============================================================================
class myLimitPair:
    
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def __init__(self, a_LowerLimit, a_UpperLimit):
        self.m_LowerLimit = a_LowerLimit
        self.m_UpperLimit = a_UpperLimit
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================


#==============================================================================
class myLimitPairsList:
    
    
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def __init__(self):
        self.m_List = []
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================


#==============================================================================
class myInputParametersMGR:


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def __init__( self):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.__init__()", "=", 1, 0)

        self.m_ListOfNumFiltersLists = []
        self.m_ListOfKernelSizesLists = []

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.__init__()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfNumFiltersLists(self, a_NumLayers, a_MaxNumFilters):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateListOfNumFiltersLists()", "=", 1, 0)

        self.m_ListOfNumFiltersLists = self.GenerateListOfLists(a_NumLayers, a_MaxNumFilters)

        for l_NumFiltersList in self.m_ListOfNumFiltersLists:
            print(l_NumFiltersList)
        print("len(self.m_ListOfNumFiltersLists):", len(self.m_ListOfNumFiltersLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfNumFiltersLists()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfKernelSizesLists(self, a_NumLayers, a_MaxKernelSizes):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateListOfKernelSizesLists()", "=", 1, 0)

        self.m_ListOfKernelSizesLists = self.GenerateListOfLists(a_NumLayers, a_MaxKernelSizes)

        for l_KernelSizesList in self.m_ListOfKernelSizesLists:
            print(l_KernelSizesList)
        print("len(self.m_ListOfKernelSizesLists):", len(self.m_ListOfKernelSizesLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfKernelSizesLists()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfLists(self, a_NumLayers, a_MaxNumFilters):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateListOfLists()", "=", 1, 0)

        l_LimitPairsList = myLimitPairsList()

        l_CurrentMaxNumFilters = a_MaxNumFilters

        for l_LayerIndex in range(0, a_NumLayers):
            l_CurrentMinNumFilters = l_CurrentMaxNumFilters / 2
            if l_CurrentMinNumFilters % 2 != 0:
                l_CurrentMinNumFilters += 1

            l_LimitPair = myLimitPair( int(l_CurrentMinNumFilters), int(l_CurrentMaxNumFilters))
            l_LimitPairsList.m_List.append(l_LimitPair)
            l_CurrentMaxNumFilters = l_CurrentMinNumFilters

        self.DisplayLimitPairsList(l_LimitPairsList)


        l_ListOfLists = []
        for l_PairIndex in range(0, len(l_LimitPairsList.m_List)):
            l_CurrentPair = l_LimitPairsList.m_List[l_PairIndex]

            l_CurrentList = []
            for l_CurrentNumFilters in range(l_CurrentPair.m_UpperLimit, l_CurrentPair.m_LowerLimit, -1):
                l_CurrentList.append(l_CurrentNumFilters)

            l_ListOfLists.append(l_CurrentList)
            print("[",l_PairIndex,"] ",l_ListOfLists[l_PairIndex])


        # Generate all combinations using itertools
        l_Combinations = list(product(*l_ListOfLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfLists()", "=", 0, 0)
        return l_Combinations
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def DisplayLimitPairsList(self, a_LimitPairsList):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.DisplayNumFiltersList()", "=", 1, 0)

        for l_PairIndex in range(0, len(a_LimitPairsList.m_List)):
            l_CurrentLimitPair = a_LimitPairsList.m_List[l_PairIndex]
            print("(",l_PairIndex,")  [",l_CurrentLimitPair.m_LowerLimit,",",l_CurrentLimitPair.m_UpperLimit,"]")

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.DisplayNumFiltersList()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================
