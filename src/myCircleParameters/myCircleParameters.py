# src/myDataSetMGR/myDataSetMGR.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
import math
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
            l_RadialDistance = math.sqrt(l_RadialDistanceSquared)

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
                                   , a_CircleParametersRaw.m_PointData
                                   , a_Radius
                                   , a_NominalCentreX
                                   , a_NominalCentreY):
       
        l_SumRadialDifferenceSquared = 0
        for l_PointIndex in range(0, len(a_CircleParametersRaw.m_PointData)):
            l_X = a_CircleParametersRaw.m_PointData[l_PointIndex, 0]
            l_Y = a_CircleParametersRaw.m_PointData[l_PointIndex, 1]

            l_DiffX = l_X - a_NominalCentreX
            l_DiffY = l_Y - a_NominalCentreY 
            l_DiffXSquared = l_DiffX * l_DiffX
            l_DiffYSquared = l_DiffY * l_DiffY
            l_RSquared = l_DiffXSquared + l_DiffYSquared
            l_R = math.sqrt(l_RSquared)
            l_RadialDifference = l_R - a_Radius
            l_RadialDifferenceSquared = l_RadialDifference * l_RadialDifference
            l_SumRadialDifferenceSquared += l_RadialDifferenceSquared

        l_MeanRadialDifferenceSquared = l_SumRadialDifferenceSquared / len(a_CircleParametersRaw.m_PointData)
        l_RMS = math.sqrt(l_MeanRadialDifferenceSquared)

        return l_RMS
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialMean( self
                       , a_CircleParametersRaw.m_PointData
                       , a_NominalCentreX
                       , a_NominalCentreY
                       , a_XYRadialMean):
        l_SumR = 0
        for l_PointIndex in range(0, len(a_CircleParametersRaw.m_PointData)):
            l_X = a_CircleParametersRaw.m_PointData[l_PointIndex, 0]
            l_Y = a_CircleParametersRaw.m_PointData[l_PointIndex, 1]

            l_DiffX = l_X - a_NominalCentreX
            l_DiffY = l_Y - a_NominalCentreY 
            l_DiffXSquared = l_DiffX * l_DiffX
            l_DiffYSquared = l_DiffY * l_DiffY
            l_RSquared = l_DiffXSquared + l_DiffYSquared
            l_R = math.sqrt(l_RSquared)
            l_SumR += l_R

        a_XYRadialMean = l_SumR / len(a_CircleParametersRaw.m_PointData)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromOwnXYRadialMean( self
                                                       , a_CircleParametersRaw.m_PointData
                                                       , a_NominalCentreX
                                                       , a_NominalCentreY):

        l_XYRadialMean = 0
        self.SetXYRadialMean( a_CircleParametersRaw.m_PointData
                            , a_NominalCentreX
                            , a_NominalCentreY
                            , l_XYRadialMean)

        self.SetRootMeanSquaredDeviation( a_CircleParametersRaw.m_PointData
                                        , l_XYRadialMean
                                        , a_NominalCentreX
                                        , a_NominalCentreY)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromRCNom( self
                                             , l_CircleParametersRCNom.m_XYRadialDistances):
        None
    #--------------------------------------------------------------------------
#==============================================================================