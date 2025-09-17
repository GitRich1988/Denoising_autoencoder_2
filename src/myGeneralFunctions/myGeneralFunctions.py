# src/myGeneralFunctions/myGeneralFunctions.py

import inspect
import os
import re
import random
from datetime import datetime
import pandas as pd


#==============================================================================
class myGeneralFunctions:

    #--------------------------------------------------------------------------
    def PrintMethodSTART( a_MethodName
                        , a_Symbol
                        , a_NumLeadingNewLines
                        , a_NumTrailingNewLines):
        l_OutputString = ""

        l_NumSymbolRepetitions = 5

        for i in range(0, a_NumLeadingNewLines):
            l_OutputString += "\n"

        for i in range(0, l_NumSymbolRepetitions):
            l_OutputString += a_Symbol

        l_OutputString += " START OF "
        l_OutputString += a_MethodName
        l_OutputString += " "

        for i in range(0, l_NumSymbolRepetitions):
            l_OutputString += a_Symbol

        for i in range(0, a_NumTrailingNewLines):
            l_OutputString += "\n"

        print(l_OutputString)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def PrintMethodEND( a_MethodName
                      , a_Symbol
                      , a_NumLeadingNewLines
                      , a_NumTrailingNewLines):
        l_OutputString = ""

        l_NumSymbolRepetitions = 5

        for i in range(0, a_NumLeadingNewLines):
            l_OutputString += "\n"

        for i in range(0, l_NumSymbolRepetitions):
            l_OutputString += a_Symbol

        l_OutputString += " END OF "
        l_OutputString += a_MethodName
        l_OutputString += " "

        for i in range(0, l_NumSymbolRepetitions):
            l_OutputString += a_Symbol

        for i in range(0, a_NumTrailingNewLines):
            l_OutputString += "\n"

        print(l_OutputString)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def CheckDirExists(a_FolderPath):
        #return os.path.isdir(os.path.normpath(a_FolderPath))
        return os.path.isdir(a_FolderPath)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetDirsInDir(a_DirPath):
        if not os.path.isdir(a_DirPath):
                raise ValueError(f"The path '{a_DirPath}' is not a valid directory.")

        return [name for name in os.listdir(a_DirPath)
                if os.path.isdir(os.path.join(a_DirPath, name))]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetFinalDirFromDirPath(a_FolderPath):
        return os.path.basename(os.path.normpath(a_FolderPath))
    #--------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------
    def ExtractNumberFromCsyDirName(a_FolderName):
        match = re.fullmatch(r"Csy_(\d+)", a_FolderName)
        if match:
            return int(match.group(1))
        return None
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def SplitListByPercentage(a_InputList, a_Percentage, a_Seed):
        if not (0 < a_Percentage < 100):
            raise ValueError("Percentage must be between 0 and 100 (exclusive).")
    
        random.seed(a_Seed)
        shuffled = a_InputList.copy()
        random.shuffle(shuffled)
    
        split_index = int(len(shuffled) * a_Percentage/ 100)
        return shuffled[:split_index], shuffled[split_index:]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetCurrentDateTimeStamp():
        return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetFullPathsOfFilesInDir(l_CurrentDirPath):
        file_list = []
        for entry in os.listdir(l_CurrentDirPath):
            full_path = os.path.join(l_CurrentDirPath, entry)
            if os.path.isfile(full_path):
                file_list.append(full_path)
        return file_list
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def ReadFileIntoPandasDataframe(a_FullPath, a_Delimiter=' ', a_Header=None):
        try:
            df = pd.read_csv( a_FullPath
                            , delimiter=a_Delimiter
                            , header=a_Header)
            return df
        except Exception as e:
            print(f"Error reading {a_FullPath}: {e}")
            return None
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def WritePandasDataFrameToFile( a_DF
                                  , a_FullPath
                                  , a_ColumnWidths=None):
        try:
            with open(a_FullPath, 'w') as f:
                # If no widths provided, calculate reasonable widths automatically
                if a_ColumnWidths is None:
                    a_ColumnWidths = []
                    for col in a_DF.columns:
                        max_data_len = a_DF[col].astype(str).map(len).max()
                        col_len = len(str(col))
                        a_ColumnWidths.append(max(max_data_len, col_len) + 2)  # +2 for spacing

                # Write header
                header_items = [
                    str(col).ljust(width)[:width] for col, width in zip(a_DF.columns, a_ColumnWidths)
                ]
                f.write("".join(header_items) + "\n")

                # Write each row
                for _, row in a_DF.iterrows():
                    row_items = [
                        str(val).ljust(width)[:width] for val, width in zip(row, a_ColumnWidths)
                    ]
                    f.write("".join(row_items) + "\n")

            print(f"DataFrame successfully written to fixed-width file: {a_FullPath}")

        except Exception as e:
            print(f"Error writing DataFrame to fixed-width file {a_FullPath}: {e}")
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetFileNameFromFullPath(a_FullPath):
        return os.path.basename(a_FullPath)
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def MakeDirIfNonExistent(a_DirPath):
        if not os.path.exists(a_DirPath):
            try:
                os.makedirs(a_DirPath)
                print(f"Directory created: {a_DirPath}")
            except Exception as e:
                print(f"Failed to create directory {a_DirPath}: {e}")
        else:
            print(f"Directory already exists: {a_DirPath}")
    #--------------------------------------------------------------------------

    
    #--------------------------------------------------------------------------
    def GetFileNameWithoutExtension(a_FileName):
        return os.path.splitext(os.path.basename(a_FileName))[0]
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def WritePandasDataFrameToFile( a_DataFrame
                                  , a_FullPath
                                  , a_ColumnWidths
                                  , a_AppendOrOverwrite):
        mode = 'a' if a_AppendOrOverwrite.lower() == 'a' else 'w'

        with open(a_FullPath, mode) as f:
            # Write column headers with specified widths
            #header = ''.join([f"{col:<{width}}" for col, width in zip(a_DataFrame.columns, a_ColumnWidths)])
            #f.write(header + '\n')

            # Write data rows with formatting
            for _, row in a_DataFrame.iterrows():
                formatted_values = []
                for val, width in zip(row, a_ColumnWidths):
                    if isinstance(val, float):
                        formatted_val = f"{val:.6f}"
                    else:
                        formatted_val = str(val)
                    formatted_values.append(f"{formatted_val:<{width}}")
                line = ''.join(formatted_values)
                f.write(line + '\n')
    #--------------------------------------------------------------------------


    #--------------------------------------------------------------------------
    def GetPathBeforeFileName(a_FullPath):
        return os.path.dirname(a_FullPath)
    #--------------------------------------------------------------------------

#==============================================================================
