# download_omniweb_data.py
# 
# downloads datasets from the NASA Goddard OmniWeb data portal at
# https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi , reads in the downloaded 
# file and the saves a second file formatted for easy reading in Python.
# The second file will have "_fmt.csv" appended to file name
#
# Revision:  6/7/2024 - changed columns written to reformatted data file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os



#############################     User Input     ##############################

# file to save data
file_name = 'omni_data.txt'

# begin and end dates
date_begin = 19860101    # format:  YYYYMMDD
date_end   = 20250110    # format:  YYYYMMDD

# columns to download (see allowed values in the dictionary below)
data_sets = ["Bz","Bx", "By","Dst","V","T"]




############   Dictonary converting dataset name to column number   ###########
#
# These lables will be written in the header in the formatted file

cd = {
    "YEAR":  0,
    "DOY":   1,
    "HR":    2,
    "BART":  3,
    "B":     8,
    "BMAG":  9,
    "Bx":    12,
    "By":    15,
    "Bz":    16,
    "T":     22,
    "N":     23,
    "V":     24,
    "P":     28,
    "Kp":    38,
    "SSN":   39,
    "Dst":   40,
    "AE":    41,
    "Ap":    49,
    "f10_7": 50
    }



#####################   UNIX command to download data   ######################

str = F"activity=retrieve&res=hour&spacecraft=omni2&start_date={date_begin}&end_date={date_end}"

for ttl in data_sets:
    str = str + f"&vars={cd[ttl]}"
    print(cd[ttl])
    
str = 'curl -d "' + str + '" https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi > ' + file_name

os.system(str)



############################   Load Data File   ###############################

# Number of data columns in file
n_data_cols = len(data_sets);

# read in a data file downloaded from https://omniweb.gsfc.nasa.gov
# skip the extra header rows leaving only a single row for header info
# df stands for "dataframe"
# Drop the last 15 rows that contain html commands  
nskip = 6 + n_data_cols;
df = pd.read_csv(file_name,skiprows=nskip,skipfooter=15,sep='\s+',engine='python')

print('saving raw data file '+file_name)
print('\n')

# reassign column labels and display header to confirm 
data_sets.insert(0,'YEAR')
data_sets.insert(1,'DOY')
data_sets.insert(2,'HR')
df.columns = data_sets

# convert YEAR, DOY and HR to datetime format
df['Date'] = pd.to_datetime(np.float32(df['YEAR'])*100000 + np.float32(df['DOY'])*100 + np.float32(df['HR']), format='%Y%j%H')

# delete YEAR, DOY and HR columns
df = df.drop('YEAR', axis=1)
df = df.drop('DOY', axis=1)
df = df.drop('HR', axis=1)

# shift column 'Date' to first position 
first_column = df.pop('Date') 
df.insert(0, 'Date', first_column) 
df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d %H'))


#####################   Write Nicely Formatted Data File   ####################

file_name_save = file_name.split('.')[0] +  '_fmt.csv'
print('saving nicely formatted file to '+file_name_save)
print('\n')
print(df.head())

# save nicely formatted file
df.to_csv(file_name_save, sep='\t', index=False)



############   Complete list of data columns available at OmniWeb   ###########
#
#  0 YEAR
#  1 DOY
#  2 Hour
#  3 Bartels rotation number
#  4 ID for IMF spacecraft
#  5 ID for SW Plasma spacecraft
#  6 # of points in IMF averages 
#  7 # of points in Plasma averag.
#  8 Scalar B, nT
#  9 Vector B Magnitude,nT
# 10 Lat. Angle of B (GSE)
# 11 Long. Angle of B (GSE)
# 12 BX, nT (GSE, GSM)
# 13 BY, nT (GSE)
# 14 BZ, nT (GSE)
# 15 BY, nT (GSM)
# 16 BZ, nT (GSM)
# 17 RMS_magnitude, nT
# 18 RMS_field_vector, nT
# 19 RMS_BX_GSE, nT
# 20 RMS_BY_GSE, nT
# 21 RMS_BZ_GSE, nT
# 22 SW Plasma Temperature, K
# 23 SW Proton Density, N/cm^3
# 24 SW Plasma Speed, km/s
# 25 SW Plasma flow long. angle
# 26 SW Plasma flow lat. angle
# 27 Alpha/Prot. ratio
# 28 Flow pressure
# 29 sigma-T,K
# 30 sigma-n, N/cm^3)
# 31 sigma-V, km/s
# 32 sigma-phi V, degrees
# 33 sigma-theta V, degrees
# 34 sigma-ratio
# 35 E elecrtric field 
# 36 Plasma Beta
# 37 Alfen mach number
# 38 Kp index
# 39 R (Sunspot No.)
# 40 Dst-index, nT
# 41 AE-index, nT
# 42 Proton flux (>1 Mev)
# 43 Proton flux (>2 Mev)
# 44 Proton flux (>4 Mev)
# 45 Proton flux (>10 Mev)
# 46 Proton flux (>30 Mev)
# 47 Proton flux (>60 Mev)
# 48 Flux FLAG
# 49 ap_index, nT
# 50 f10.7_index
# 51 pc-index
# 52 AL-index, nT
# 53 AU-index, nT
# 54 Magnetosonic Much num.
# 55 Lyman_alpha
# 56 Quasy-Invariant
