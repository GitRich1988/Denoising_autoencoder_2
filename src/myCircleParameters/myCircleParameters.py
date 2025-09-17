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

        self.m_NominalRadius = 0
        self.m_XYRadialDistancesNomCentre = []
        self.m_XYRadialMean = 0
        self.m_RMSDevFromXYRadialMean = 0
        self.m_RMSDevFromTrueNom = 0
        self.m_RMSDevFromRCNom = 0

        l_GeneralFunctions.PrintMethodEND("myCircleParameters.__init__()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def Initialise( self
                  , a_PointData
                  , a_NominalRadius):
        l_GeneralFunctions.PrintMethodSTART("myCircleParameters.Initialise()", "=", 1, 0)

        self.m_PointData = a_PointData
        self.m_NominalRadius = a_NominalRadius

        l_GeneralFunctions.PrintMethodEND("myCircleParameters.Initialise()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetAllRMSInfo( self
                     , a_TestingExampleRCNom = None):
        self.SetXYRadialDistances()
        self.SetXYRadialMean()
        self.SetXYRootMeanSquareDeviationFromOwnXYRadialMean()
        self.SetXYRootMeanSquareDeviationFromTrueNom()

        if(a_TestingExampleRCNom != None):
            self.SetRMSDevPointToPoint( a_TestingExampleRCNom
                                      , self.m_RMSDevFromRCNom)

    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialDistances(self):
        l_GeneralFunctions.PrintMethodSTART("myCircleParameters.SetXYRadialDistances()", "=", 1, 0)

        for l_PointIndex in range(0, len(self.m_PointData)):
            l_CurrentX = self.m_PointData[l_PointIndex][0]
            l_CurrentY = self.m_PointData[l_PointIndex][1]

            l_DiffX = l_CurrentX - self.m_NomCentreX
            l_DiffY = l_CurrentY - self.m_NomCentreY
            l_DiffXSquared = l_DiffX^2
            l_DiffYSquared = l_DiffY^2
            l_RadialDistanceSquared = l_DiffXSquared + l_DiffYSquared
            l_RadialDistance = math.sqrt(l_RadialDistanceSquared)

            self.m_XYRadialDistancesNomCentre.append(l_RadialDistance)

        l_GeneralFunctions.PrintMethodEND("myCircleParameters.SetXYRadialDistances()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialMean(self):
        l_SumR = sum(self.m_XYRadialDistancesNomCentre)
        self.m_XYRadialMean = l_SumR / len(self.m_PointData)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromFixedRadius( self
                                                   , a_FixedRadius
                                                   , a_RMS):

        l_SumSquaredDeviations = 0
        for l_PointIndex in range(0, len(self.m_XYRadialDistancesNomCentre)):
            l_CurrentDeviation = self.m_XYRadialDistancesNomCentre[l_PointIndex] - a_FixedRadius
            l_CurrentDeviationSquared = l_CurrentDeviation * l_CurrentDeviation
            l_SumSquaredDeviations += l_CurrentDeviationSquared

        l_MeanSquaredDeviation = l_SumSquaredDeviations / len(self.m_XYRadialDistancesNomCentre)
        a_RMS = math.sqrt(l_MeanSquaredDeviation);
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromOwnXYRadialMean( self):
        self.SetXYRootMeanSquareDeviationFromFixedRadius( self.m_XYRadialMean
                                                        , self.m_RMSDevFromXYRadialMean)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromTrueNom(self):
        self.SetXYRootMeanSquareDeviationFromFixedRadius( self.m_NominalRadius
                                                        , self.m_RMSDevFromTrueNom)
    #--------------------------------------------------------------------------

    

    #--------------------------------------------------------------------------
    def SetRMSDevPointToPoint( self
                             , a_PointDataForComparison
                             , a_RMS):
       
        l_SumRSquared =  0
        for l_PointIndex in range(0, len(self.m_PointData)):
            l_X = self.m_PointData[l_PointIndex, 0]
            l_Y = self.m_PointData[l_PointIndex, 1]

            l_CompX = a_PointDataForComparison[l_PointIndex, 0]
            l_CompY = a_PointDataForComparison[l_PointIndex, 1]

            l_DiffX = l_X - l_CompX
            l_DiffY = l_Y - l_CompY
            l_DiffXSquared = l_DiffX * l_DiffX
            l_DiffYSquared = l_DiffY * l_DiffY
            l_RSquared = l_DiffXSquared + l_DiffYSquared
            l_SumRSquared += l_RSquared

        l_MeanRadialDifferenceSquared = l_SumRSquared / len(self.m_PointData)
        a_RMS = math.sqrt(l_MeanRadialDifferenceSquared)
    #--------------------------------------------------------------------------




    #--------------------------------------------------------------------------
    def SetCircleFittedParameters(self):
        l_GeneralFunctions.PrintMethodSTART("myCircleParameters.SetCircleParameters()", "=", 1, 0)


        l_GeneralFunctions.PrintMethodEND("myCircleParameters.SetCircleParameters()", "=", 0, 0)
    #--------------------------------------------------------------------------

#==============================================================================