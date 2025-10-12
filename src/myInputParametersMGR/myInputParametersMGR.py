# src/myDataSetMGR/myInputParametersMGR.py

import re
from sre_constants import RANGE
from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()

from itertools import product
from itertools import combinations_with_replacement

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

        #for l_NumFiltersList in self.m_ListOfNumFiltersLists:
        #    print(l_NumFiltersList)
        print("len(self.m_ListOfNumFiltersLists):", len(self.m_ListOfNumFiltersLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfNumFiltersLists()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfKernelSizesLists(self, a_NumLayers, a_MinKernelSize, a_MaxKernelSize):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateListOfKernelSizesLists()", "=", 1, 0)

        #for l_CurrentKernelSize in range(a_MaxKernelSize, a_MinKernelSize - 1, -1):
        #    l_List = []
        #    for l_LayerIndex in range(a_NumLayers, 1, -1):
        #        l_List.append(l_CurrentKernelSize)
        #    self.m_ListOfKernelSizesLists.append(l_List)

        self.m_ListOfKernelSizesLists = self.GenerateDescendingCombinations(a_MinKernelSize, a_MaxKernelSize, a_NumLayers)

        #for l_KernelSizesList in self.m_ListOfKernelSizesLists:
        #    print(l_KernelSizesList)
        print("len(self.m_ListOfKernelSizesLists):", len(self.m_ListOfKernelSizesLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfKernelSizesLists()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GetLimitPairsList(self, a_NumLayers, a_MaxNumFilters):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GetLimitPairsList()", "=", 1, 0)

        l_LimitPairsList = myLimitPairsList()

        l_CurrentMaxNumFilters = a_MaxNumFilters

        for l_LayerIndex in range(0, a_NumLayers):
            l_CurrentMinNumFilters = l_CurrentMaxNumFilters / 2
            if l_CurrentMinNumFilters % 2 != 0:
                l_CurrentMinNumFilters += 1

            l_LimitPair = myLimitPair( int(l_CurrentMinNumFilters), int(l_CurrentMaxNumFilters))
            l_LimitPairsList.m_List.append(l_LimitPair)
            l_CurrentMaxNumFilters = l_CurrentMinNumFilters

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GetLimitPairsList()", "=", 0, 0)
        return l_LimitPairsList
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def SetFullRangesOfEntriesForLimitPairsList(self, a_LimitPairsList, a_ListOfLists):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.SetFullRangesOfEntriesForLimitPairsList()", "=", 1, 0)

        for l_PairIndex in range(0, len(a_LimitPairsList.m_List)):
            l_CurrentPair = a_LimitPairsList.m_List[l_PairIndex]

            l_CurrentList = []
            for l_CurrentNumFilters in range(l_CurrentPair.m_UpperLimit, l_CurrentPair.m_LowerLimit, -1):
                l_CurrentList.append(l_CurrentNumFilters)

            a_ListOfLists.append(l_CurrentList)
            print("[",l_PairIndex,"] ",a_ListOfLists[l_PairIndex])

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.SetFullRangesOfEntriesForLimitPairsList()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def DisplayLimitPairsList(self, a_LimitPairsList):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.DisplayNumFiltersList()", "=", 1, 0)

        for l_PairIndex in range(0, len(a_LimitPairsList.m_List)):
            l_CurrentLimitPair = a_LimitPairsList.m_List[l_PairIndex]
            print("(",l_PairIndex,")  [",l_CurrentLimitPair.m_LowerLimit,",",l_CurrentLimitPair.m_UpperLimit,"]")

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.DisplayNumFiltersList()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfLists(self, a_NumLayers, a_MaxNumFilters):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateListOfLists()", "=", 1, 0)

        l_LimitPairsList = self.GetLimitPairsList(a_NumLayers, a_MaxNumFilters)
        self.DisplayLimitPairsList(l_LimitPairsList)

        l_ListOfLists = []
        self.SetFullRangesOfEntriesForLimitPairsList(l_LimitPairsList, l_ListOfLists)

        # Generate all combinations using itertools
        l_Combinations = list(product(*l_ListOfLists))

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateListOfLists()", "=", 0, 0)
        return l_Combinations
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateDescendingCombinations(self, lower, upper, length):
        # Generate non-decreasing combinations in the range
        combos = combinations_with_replacement(range(lower, upper + 1), length)
        # Reverse each to make it non-increasing
        descending_combos = [tuple(reversed(combo)) for combo in combos if list(reversed(combo)) == sorted(combo, reverse=True)]

        return descending_combos
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateListOfNumFiltersLists_2(self, a_NumFiltersRanges):

        #for i in range(0, len(a_NumFiltersRanges)):
        #    l_CurrentList = []
        #    for l_CurrentEntry in range(a_NumFiltersRanges[i][0], a_NumFiltersRanges[i][1], -1):
        #        l_CurrentList.append(l_CurrentEntry)

        l_LimitPairsList = myLimitPairsList()
        for i in range(0, len(a_NumFiltersRanges)):
            l_LimitPair = myLimitPair(a_NumFiltersRanges[i][0], a_NumFiltersRanges[i][1])
            l_LimitPairsList.m_List.append(l_LimitPair)

        l_ListOfLists = []
        self.SetFullRangesOfEntriesForLimitPairsList(l_LimitPairsList, l_ListOfLists)

        # Generate all combinations using itertools
        l_Combinations = list(product(*l_ListOfLists))

        self.m_ListOfNumFiltersLists = l_Combinations

        #for l_NumFiltersList in self.m_ListOfNumFiltersLists:
        #    print(l_NumFiltersList)
        print("len(self.m_ListOfNumFiltersLists):", len(self.m_ListOfNumFiltersLists))
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def WriteCNNDefinitions(self):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.WriteCNNDefinitions()", "=", 1, 0)

        l_DateTimeStamp = l_GeneralFunctions.GetCurrentDateTimeStamp()
        
        l_CNNDefinitionsFilePath = l_ProjectInfo.GetDirTrainingInfo() + "CNN_definitions/" + "CNN_definition__" + l_DateTimeStamp + ".json"
        l_CNNDefinitionsFile = open(l_CNNDefinitionsFilePath, "w")

        l_CNNDefinitionsFile.write("[\n")
        l_NumFiltersListIndex = -1
        l_LenListOfNumFiltersLists = len(self.m_ListOfNumFiltersLists)
        l_LenListOfKernelSizesLists = len(self.m_ListOfKernelSizesLists)
        l_IsFinalCNN = False

        for l_NumFiltersList in self.m_ListOfNumFiltersLists:
            l_NumFiltersListIndex += 1

            l_KernelSizesListIndex = -1
            for l_KernelSizesList in self.m_ListOfKernelSizesLists:
                l_KernelSizesListIndex += 1

                if(l_NumFiltersListIndex == l_LenListOfNumFiltersLists - 1):
                    if(l_KernelSizesListIndex == l_LenListOfKernelSizesLists - 1):
                        l_IsFinalCNN = True

                l_LinesOneCNNDefinition = self.GetLinesOneCNNDefinition(l_NumFiltersList, l_KernelSizesList, l_IsFinalCNN)
                for i in range(0, len(l_LinesOneCNNDefinition)):
                    l_CNNDefinitionsFile.write(l_LinesOneCNNDefinition[i])

        l_CNNDefinitionsFile.write("]")

        l_CNNDefinitionsFile.close()
        print("Wrote file:", l_CNNDefinitionsFilePath)

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.WriteCNNDefinitions()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GetLinesOneCNNDefinition(self, a_NumFiltersList, a_KernelSizesList, a_IsFinalCNN):
        l_Lines = []

        l_NumFiltersListStr = l_GeneralFunctions.GetNumberListAsString(a_NumFiltersList)
        l_KernelSizesList = l_GeneralFunctions.GetNumberListAsString(a_KernelSizesList)

        l_Lines.append('{\n')
        l_Lines.append('    "NumFiltersList": "[' + l_NumFiltersListStr + ']",\n')
        l_Lines.append('    "KernelSizesList": "[' + l_KernelSizesList + ']"\n')

        if a_IsFinalCNN:
            l_Lines.append('}\n')
        else:
            l_Lines.append('},\n')

        return l_Lines
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateNewCNNDefinitionsFile(self):
        #self.GenerateListOfNumFiltersLists(4, 30)
        #self.GenerateListOfNumFiltersLists(3, 35)
        l_NumFiltersRanges = [[25,35], [15,25], [5,15]]
        self.GenerateListOfNumFiltersLists_2(l_NumFiltersRanges)
        
        #self.GenerateListOfKernelSizesLists(4, 3, 7)
        self.GenerateListOfKernelSizesLists(3, 3, 3)
        self.WriteCNNDefinitions()
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================
