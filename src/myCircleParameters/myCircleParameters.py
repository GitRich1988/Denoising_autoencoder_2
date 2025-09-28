# src/myDataSetMGR/myCircleParameters.py

from src.myGeneralFunctions.myGeneralFunctions import myGeneralFunctions as l_GeneralFunctions
import math
import pandas as pd
import re
import random
import numpy as np
from scipy.optimize import least_squares


#==============================================================================
class myCircleParameters:

    #--------------------------------------------------------------------------
    def __init__(self):
        #l_GeneralFunctions.PrintMethodSTART("myCircleParameters.__init__()", "=", 0, 0)

        self.m_PointData = pd.DataFrame()

        self.m_CentreX = 0
        self.m_CentreY = 0
        self.m_CentreZ = 0
        self.m_I = 0
        self.m_J = 0
        self.m_K = 0
        self.m_Radius = 0

        self.m_NominalCentreX = 0
        self.m_NominalCentreY = 0
        self.m_NominalCentreZ = 0
        self.m_NominalI = 0
        self.m_NominalJ = 0
        self.m_NominalK = 0
        self.m_NominalRadius = 0

        self.m_XYRadialDistancesNomCentre = []
        self.m_XYRadialMean = 0
        self.m_RMSDevFromXYRadialMean = 0
        self.m_RMSDevFromTrueNom = 0
        self.m_RMSDevFromRCNom = 0
        self.m_Circularity = 0

        #l_GeneralFunctions.PrintMethodEND("myCircleParameters.__init__()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def Initialise( self
                  , a_PointData
                  , a_NominalRadius):
        #l_GeneralFunctions.PrintMethodSTART("myCircleParameters.Initialise()", "=", 1, 0)

        self.m_PointData = a_PointData
        self.m_NominalRadius = a_NominalRadius

        #l_GeneralFunctions.PrintMethodEND("myCircleParameters.Initialise()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetAllRMSInfo( self
                     , a_TestingExampleRCNom = None):
        self.SetXYRadialDistances()
        self.SetXYRadialMean()
        self.SetXYRootMeanSquareDeviationFromOwnXYRadialMean()
        self.SetXYRootMeanSquareDeviationFromTrueNom()

        if a_TestingExampleRCNom is not None:
            l_RMSDevFromRCNom = []
            l_RMSDevFromRCNom.append(0)
            self.SetRMSDevPointToPoint( a_TestingExampleRCNom
                                      , l_RMSDevFromRCNom)
            self.m_RMSDevFromRCNom = l_RMSDevFromRCNom[0]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialDistances(self):
        #l_GeneralFunctions.PrintMethodSTART("myCircleParameters.SetXYRadialDistances()", "=", 1, 0)

        for l_PointIndex in range(0, len(self.m_PointData)):
            l_CurrentX = self.m_PointData[l_PointIndex][0]
            l_CurrentY = self.m_PointData[l_PointIndex][1]

            l_DiffX = l_CurrentX - self.m_NominalCentreX
            l_DiffY = l_CurrentY - self.m_NominalCentreY
            l_DiffXSquared = l_DiffX*l_DiffX
            l_DiffYSquared = l_DiffY*l_DiffY
            l_RadialDistanceSquared = l_DiffXSquared + l_DiffYSquared
            l_RadialDistance = math.sqrt(l_RadialDistanceSquared)

            self.m_XYRadialDistancesNomCentre.append(l_RadialDistance)

        #l_GeneralFunctions.PrintMethodEND("myCircleParameters.SetXYRadialDistances()", "=", 0, 0)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRadialMean(self):
        l_SumR = sum(self.m_XYRadialDistancesNomCentre)
        self.m_XYRadialMean = l_SumR / len(self.m_PointData)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromFixedRadius( self
                                                   , a_FixedRadius
                                                   , a_RMSList):

        l_SumSquaredDeviations = 0
        for l_PointIndex in range(0, len(self.m_XYRadialDistancesNomCentre)):
            l_CurrentDeviation = self.m_XYRadialDistancesNomCentre[l_PointIndex] - a_FixedRadius
            l_CurrentDeviationSquared = l_CurrentDeviation * l_CurrentDeviation
            l_SumSquaredDeviations += l_CurrentDeviationSquared

        l_MeanSquaredDeviation = l_SumSquaredDeviations / len(self.m_XYRadialDistancesNomCentre)
        a_RMSList[0] = math.sqrt(l_MeanSquaredDeviation);
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromOwnXYRadialMean( self):
        l_List = [0]
        self.SetXYRootMeanSquareDeviationFromFixedRadius( self.m_XYRadialMean
                                                        , l_List)
        self.m_RMSDevFromXYRadialMean = l_List[0]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetXYRootMeanSquareDeviationFromTrueNom(self):
        l_List = [0]
        self.SetXYRootMeanSquareDeviationFromFixedRadius( self.m_NominalRadius
                                                        , l_List)
        self.m_RMSDevFromTrueNom = l_List[0]
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
        a_RMS[0] = math.sqrt(l_MeanRadialDifferenceSquared)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SetCircleFittedParameters(self):
        #l_GeneralFunctions.PrintMethodSTART("myCircleParameters.SetCircleParameters()", "=", 1, 0)

        l_Results = self.FitCircle2D()
        self.m_Radius = l_Results[1]
        self.m_CentreX = l_Results[0][0]
        self.m_CentreY = l_Results[0][1]
        self.m_CentreZ = 0
        self.m_I = 0
        self.m_J = 0
        self.m_K = 1

        #l_GeneralFunctions.PrintMethodEND("myCircleParameters.SetCircleParameters()", "=", 0, 0)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def FitCircle2D(self):
        """
        Fit a circle to 2D points using least squares.
    
        Parameters:
            points (array-like): List or array of (x, y) tuples or shape (N, 2).
    
        Returns:
            center (tuple): (x, y) coordinates of the circle center.
            radius (float): Radius of the fitted circle.
        """
        l_Points = np.asarray(self.m_PointData[:,0:2])
    
        # Initial guess: center at mean of points, radius as mean distance to center
        x_m, y_m = np.mean(l_Points, axis=0)
        r_guess = np.mean(np.sqrt((l_Points[:,0] - x_m)**2 + (l_Points[:,1] - y_m)**2))
        initial_guess = [x_m, y_m, r_guess]

        def residuals(params):
            x0, y0, r = params
            return np.sqrt((l_Points[:,0] - x0)**2 + (l_Points[:,1] - y0)**2) - r

        l_Result = least_squares(residuals, initial_guess)

        x0, y0, r = l_Result.x
        return (x0, y0), r
    #--------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------
    def SetCircularity(self):
        l_MaxNegativeDeviation = 0
        l_MaxPositiveDeviation = 0

        for l_PointIndex in range(0, len(self.m_XYRadialDistancesNomCentre)):
            l_CurrentXYRadialDistance = self.m_XYRadialDistancesNomCentre[l_PointIndex]
            l_Diff = l_CurrentXYRadialDistance - self.m_Radius
            
            if(l_Diff > 0):
                if(l_Diff > l_MaxPositiveDeviation):
                    l_MaxPositiveDeviation = l_Diff

            elif(l_Diff < 0):
                if(l_Diff < l_MaxNegativeDeviation):
                    l_MaxNegativeDeviation = l_Diff

        self.m_Circularity = l_MaxPositiveDeviation
        
        l_MaxNegativeDeviation = -1 * l_MaxNegativeDeviation
        if(l_MaxNegativeDeviation > l_MaxPositiveDeviation):
            self.m_Circularity = l_MaxNegativeDeviation
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def PrintAllValues( self
                      , a_Name = "NOT_SET"):

        if(a_Name != "NOT_SET"):
            print("\n------------------------\nFitted circle parameters:", a_Name)

        print("m_PointData.shape:", self.m_PointData.shape)

        print("m_CentreX:", self.m_CentreX)
        print("m_CentreY:", self.m_CentreY)
        print("m_CentreZ:", self.m_CentreZ)
        print("m_I:", self.m_I)
        print("m_J:", self.m_J)
        print("m_K:", self.m_K)
        print("m_Radius:", self.m_Radius)

        print("m_NominalCentreX:", self.m_NominalCentreX)
        print("m_NominalCentreY:", self.m_NominalCentreY)
        print("m_NominalCentreZ:", self.m_NominalCentreZ)
        print("m_NominalI:", self.m_NominalI)
        print("m_NominalJ:", self.m_NominalJ)
        print("m_NominalK:", self.m_NominalK)
        print("m_NominalRadius:", self.m_NominalRadius)

        print("len(self.m_XYRadialDistancesNomCentre):", len(self.m_XYRadialDistancesNomCentre))
        print("m_XYRadialMean:", self.m_XYRadialMean)
        print("m_RMSDevFromXYRadialMean:", self.m_RMSDevFromXYRadialMean)
        print("m_RMSDevFromTrueNom:", self.m_RMSDevFromTrueNom)
        print("m_RMSDevFromRCNom:", self.m_RMSDevFromRCNom)
        print("m_Circularity:", self.m_Circularity)
    #--------------------------------------------------------------------------
#==============================================================================