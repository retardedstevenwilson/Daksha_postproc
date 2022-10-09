# Program to read the .tdms file, made by ayush using sandeep's code to read one packet and plot it up

#**************** Read multiple pakts in tdms file (Should not use for single pkt data)**************************
#pkt_size=39087
# 10pkts in 1sec
# total samples=390870 in 1 sec
#********************************************

#from operator import index
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt

# Define global constants below
data_size=43 # Change nos. of bits in single event

# change index[x], index[y] for single packet reading in line 161

in_file_name="/home/ayushnema/Documents/work_DP2/tdms_conversions/Sandeep_files/TIFR_43bit_czt_pingpong_fcc_pingpong_23-9-2022_test7.tdms"

tdms_file = TdmsFile.read(in_file_name)
with TdmsFile.open(in_file_name) as tdms_file:
        group = tdms_file["Untitled"]
        all_groups = tdms_file.groups()
        print (all_groups)

        channel=group["Untitled"]
        all_group_channels = group.channels()
        print(all_group_channels)
        print("Single channel:",channel)

        all_channel_data = channel[:]
        print(all_channel_data)
        print("******************* Input TDMS file details **********************")
        print("Input file name:",in_file_name)
        print("***************** Total samples details **************************")
        #pkt_size=len(all_channel_data)
        #print(pkt_size)
        print("Total samples in acquired data:",len(all_channel_data)) # print total elemnts in array
        # Iterate through the array
full_pkt_data=[]
header=[]
event_data=[]
#val_data=[]

for i in range(0,(len(all_channel_data)),1):
    full_pkt_data.append(all_channel_data[i])
#print(full_pkt_data[293739:293783]) # 293783 means, print (293783-1)
print("Length of full_pkt_data:",len(full_pkt_data))

# find out header index using numpy    
arr1=np.array(full_pkt_data)
header_initial_index=np.where((arr1 == 1) & (np.roll(arr1,-1) == 1) & (np.roll(arr1,-2) == 1) & (np.roll(arr1,-3) == 1) & 
               (np.roll(arr1,-4) == 1) & (np.roll(arr1,-5) == 0) & (np.roll(arr1,-6) == 0) & 
               (np.roll(arr1,-7) == 1)  & (np.roll(arr1,-8) == 1)  & (np.roll(arr1,-9) == 0)
                & (np.roll(arr1,-10) == 1) & (np.roll(arr1,-11) == 0) & (np.roll(arr1,-12) == 0)
                & (np.roll(arr1,-13) == 1) & (np.roll(arr1,-14) == 0) & (np.roll(arr1,-15) == 0)
                & (np.roll(arr1,-16) == 0) & (np.roll(arr1,-17) == 0) & (np.roll(arr1,-18) == 1)) [0]
print("index:",header_initial_index)
print("Length of index:",len(header_initial_index))

pkt_size= header_initial_index[1]
print(pkt_size)

head_ctr=0
data_ctr=0
val_ctr=0
det_id_ctr=0
pix_energy_ctr=0
timetag_ctr=0
header_ctr=0
pix_id_ctr=0

timetag=[]
det_id=[]
pix_id=[]
pix_energy=[]
det_id_arr=[]
det_id_dec=[]
pix_id_arr=[]
pix_id_dec=[]
pix_energy_arr=[]
pix_energy_dec=[]
timetag_arr=[]
timetag_dec=[]
header_arr=[]
header_dec=[]
header_dec_part2=[]
header_dec_part3=[]

#timetag variables
bit_count=0
bin_weight=19
sum7=0

#det_id variable
bit_count1=0
bin_weight1=4
sum8=0

#pix_id variable
bit_count2=0
bin_weight2=7
sum9=0

#pix_energy variable
bit_count3=0
bin_weight3=9
sum10=0

#header variable
bit_count4=0
bin_weight4=15
sum11=0

timetag_err=[]
det_id_err=[]
pix_id_err=[]
pix_energy_err=[]
header_dec=[]


pkt0=[]

# Logic to not read partial packet data
print("******************** Packet data Report *****************")

total_full_pkt=int(len(all_channel_data)/pkt_size)
#print("total packets in acquired data:",full_pkt_size)


value=total_full_pkt*pkt_size

#print("value:",(total_full_pkt*pkt_size))
#print("value1",(len(all_channel_data)-(value*data_size)))

value1=len(all_channel_data)-(value)
print("samples in partial pkt data:",value1)
#print("total samples in partial packet:",(len(all_channel_data)-((len(all_channel_data))/data_size)*data_size))
full_pkt_size=len(all_channel_data)-value1
print("samples in full_pkt_data:",full_pkt_size)


header_data=[]
event_data=[]
event_indices=[]
header_indices=[]
index=0
header_counter=0

for index in header_initial_index:
    for i in range(0,42):
        header_indices.append(index+i)

#print(header_indices_index)
#print(full_pkt_data)
for index in range(len(full_pkt_data)):
    if index in header_indices:
        header_data.append(full_pkt_data[index])
        header_counter+=1
    else:
        event_data.append(full_pkt_data[index])
        event_indices.append(index)

print(len(event_data))
print(len(header_indices))
print(len(full_pkt_data))
print(len(event_indices))


#header into decimal
    #header_data
    #header_indices
    #event_data
    #event_indices
header_data_decimal_1=[]
header_data_decimal_2=[]
header_data_decimal_3=[]
sum3=0
sum5=0
sum6=0
