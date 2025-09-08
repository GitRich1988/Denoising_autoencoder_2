# src/myDataSetMGR/myDataSetMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
from src.myProjectInfo.myProjectInfo import myProjectInfo
l_ProjectInfo = myProjectInfo()
import re
import random


#==============================================================================
class myCircleParameters:

    #--------------------------------------------------------------------------
    def __init__(self, a_PointData):
        l_GeneralFunctions.PrintMethodSTART("myCircleParameters.__init__()", "=", 0, 0)

        self.m_PointData = a_PointData
        self.m_CentreX = 0
        self.m_CentreY = 0
        self.m_CentreZ = 0
        self.m_I = 0
        self.m_J = 0
        self.m_K = 0
        self.m_Radius = 0

        self.Initialise()

        l_GeneralFunctions.PrintMethodEND("myCircleParameters.__init__()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def Initialise(self):
        l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.Initialise()", "=", 1, 0)


        l_GeneralFunctions.PrintMethodEND("myDataSetMGR.Initialise()", "=", 0, 0)
    #--------------------------------------------------------------------------

#==============================================================================