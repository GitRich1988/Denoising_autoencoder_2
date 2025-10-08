# src/myDataSetMGR/myInputParametersMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()


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

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.__init__()", "=", 0, 0)
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =


    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def GenerateNumFiltersList(self, a_NumLayers, a_MaxNumFilters):
        l_GeneralFunctions.PrintMethodSTART("myInputParametersMGR.GenerateNumFilterList()", "=", 1, 0)

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

 
        l_ListOfNumFiltersLists = []
        l_TempNumFiltersList = []
        self.DisplayNumberList(l_ListOfLists, 0, l_ListOfNumFiltersLists, l_TempNumFiltersList) 

        for i in range(0, len(l_ListOfNumFiltersLists)):
            print("[",i,"] ",l_ListOfNumFiltersLists[i])

        l_GeneralFunctions.PrintMethodEND("myInputParametersMGR.GenerateNumFiltersList()", "=", 0, 0)
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
    def DisplayNumberList(self, a_ListOfLists, a_ListIndex, a_ListOfNumFiltersLists, a_TempNumFiltersList):
        l_List = a_ListOfLists[a_ListIndex]
        l_ListIndex = a_ListIndex

        l_Spaces = ""
        for i in range(0, l_ListIndex):
            l_Spaces += " "

        l_CurrentNumber = -1
        for l_Index in range(0, len(l_List)):
            l_CurrentNumber = l_List[l_Index]
            #print(l_Spaces, l_CurrentNumber)
            
            l_NextListIndex = l_ListIndex + 1
            if l_NextListIndex < len(a_ListOfLists):
                a_TempNumFiltersList.append(l_CurrentNumber)
                self.DisplayNumberList(a_ListOfLists, l_NextListIndex, a_ListOfNumFiltersLists, a_TempNumFiltersList)
            
            if l_NextListIndex == len(a_ListOfLists) - 1:
                a_ListOfNumFiltersLists.append(a_TempNumFiltersList)
                a_TempNumFiltersList = []  
                
         return l_CurrentNumber
    #= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
#==============================================================================
