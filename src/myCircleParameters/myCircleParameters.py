# src/myDataSetMGR/myDataSetMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
import pandas as pd
import re
import random


#==============================================================================
class myCircleParameters:

    #--------------------------------------------------------------------------
    def __init__(self):
        l_GeneralFunctions.PrintMethodSTART("myCircleParameters.__init__()", "=", 0, 0)

        self.m_PointData = pd.DataFrame()
        self.m_CentreX = 0
        self.m_CentreY = 0
        self.m_CentreZ = 0
        self.m_I = 0
        self.m_J = 0
        self.m_K = 0
        self.m_Radius = 0
        self.m_XYRadialDistancesNomCentre = []

        l_GeneralFunctions.PrintMethodEND("myCircleParameters.__init__()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def Initialise(self, a_PointData):
        l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.Initialise()", "=", 1, 0)

        self.m_PointData = a_PointData

        l_GeneralFunctions.PrintMethodEND("myDataSetMGR.Initialise()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialDistances( self
                            , a_NomCentreX
                            , a_NomCentreY):
        l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.SetXYRadialDistances()", "=", 1, 0)

        for l_PointIndex in range(0, len(self.m_PointData)):
            l_CurrentX = self.m_PointData[l_PointIndex][0]
            l_CurrentY = self.m_PointData[l_PointIndex][1]

            l_DiffX = l_CurrentX - a_NomCentreX
            l_DiffY = l_CurrentY - a_NomCentreY
            l_DiffXSquared = l_DiffX^2
            l_DiffYSquared = l_DiffY^2
            l_RadialDistanceSquared = l_DiffXSquared + l_DiffYSquared
            l_RadialDistance = sqrt(l_RadialDistanceSquared)

            self.m_XYRadialDistancesNomCentre.append(l_RadialDistance)

        l_GeneralFunctions.PrintMethodEND("myDataSetMGR.SetXYRadialDistances()", "=", 0, 0)
    #--------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------
    def SetCircleParameters(self):
        l_GeneralFunctions.PrintMethodSTART("myDataSetMGR.SetCircleParameters()", "=", 1, 0)


        l_GeneralFunctions.PrintMethodEND("myDataSetMGR.SetCircleParameters()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetRootMeanSquaredDeviation( self
                                   , l_CircleParametersRaw.m_PointData
                                   , l_TestingExampleRCNom):
        None
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromOwnXYRadialMean(self):
        None
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromRCNom( self
                                             , l_CircleParametersRCNom.m_XYRadialDistances):
        None
    #--------------------------------------------------------------------------
#==============================================================================