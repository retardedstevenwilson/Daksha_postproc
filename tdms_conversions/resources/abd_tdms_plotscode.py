"""
Created on Wed Jun 29 10:57:28 2022
@author: abhi
Each packet size is of 41969 bits
Plot of 5 quantities which is Header, Timetag, DetID, Pixel and Energy.
Header - 43-bit
Timetag - 20-bit 
detID - 5-bit
pixelID - 8-bit
Energy - 10-bit
"""

import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt

event_size = 43
packet_events = 975
tdms_file = TdmsFile("D:/Codes_MEB/LV_tdms/test1.tdms")
df = tdms_file['Untitled'].as_dataframe()
df_size = len(df.index)
packet = np.reshape(df.values,(df_size,))
head_count = int(len(packet) /((packet_events+1)*event_size))

# trim the extra zero that is extracted at the end of the packet, not to be done in good packets. After every 24 packets there is one additional zero

for i in range(1,head_count+1):
    packet = np.delete(packet,((packet_events+1)*event_size)*i)

# packet_trim = []
# val_ctr = 0
# for val in packet:
#     if (val_ctr<41969):
#         packet_trim.append(val)
#         val_ctr = val_ctr + 1
#     elif (val_ctr==41969):
#         val_ctr = 0
# packet = np.asarray(packet_trim)    
# zero padding to make 2d array of data possible, if we do not run previous cell then packet is not framed correctly
packet_size = len(packet)
packet_adjust = event_size - (packet_size % event_size)
packet = np.pad(packet,(0,packet_adjust)).astype(np.int64)[:-event_size]
#packet = np.pad(packet,(0,packet_adjust)).astype(np.int64)
# Checking and removal of sync word in this cell, this is currently not being used further down the code
sync_word = 'F9A42AB2'
header_word = np.pad(np.asarray(list(bin(int(sync_word,16))[2:])).astype(np.int64), (0, event_size-4*len(sync_word))) #makes proper header and does zero padding to make the sync word as 43 bit

# removal of headers and getting all the important packet information
events = np.reshape(packet,(int((packet_size+packet_adjust-event_size)/event_size),event_size)).astype(np.int64)
#unq, cnt = np.unique(events, axis=0, return_counts=True) #cnt variable should show 7 headers, but the headers are misaligned so we cannot precise match them

# collate all the headers to get a plot of that values
header = []
for i in range(0,head_count):
    header = np.append(header,events[i*packet_events+i,:],axis=0)
header = np.reshape(header,(head_count,event_size)).astype(np.int64)
header = (header[:,::-1]*(2**np.arange(header.shape[1]))).sum(1)

for i in range(0,head_count): #to delete all the headers, this code is commented as it is not general purpose for any sized packet
      events = np.delete(events,i*packet_events,axis=0)

# array slicing to get all the data segregated

timetag = events[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
detID = events[:,[20,21,22,23,24]]
pixID = events[:,[25,26,27,28,29,30,31,32]]
energy = events[:,[33,34,35,36,37,38,39,40,41,42]]

# The logic below is not the fastest logic but it works!
timetag = (timetag[:,::-1]*(2**np.arange(timetag.shape[1]))).sum(1)
detID = (detID[:,::-1]*(2**np.arange(detID.shape[1]))).sum(1)
pixID = (pixID[:,::-1]*(2**np.arange(pixID.shape[1]))).sum(1)
energy = (energy[:,::-1]*(2**np.arange(energy.shape[1]))).sum(1)

# Packet data information
print("Input file:",tdms_file)
print("Identified packets:",head_count)
print("Good packets:",head_count-2)
print("Used packets:",head_count)
#print("Saturated packets:",headers)
timetag 
print("Timetags analysed:",len(timetag))
print("Detector IDs analysed:",len(detID))
print("Pixel IDs analysed:",len(pixID))
print("Energy PHA (0-1024) values analysed:",len(energy))
# Plotting time

plt.plot(header)
plt.title('header')
# plt.savefig("D:/Plots/header_mock.png", dpi=800)
plt.show()

plt.plot(timetag)
plt.title('timetag')
# plt.savefig("D:/Plots/timetag_mock.png", dpi=800)
plt.show()

plt.plot(detID)
plt.title('detID')
# plt.savefig("D:/Plots/detid_mock.png", dpi=800)
plt.show()

plt.plot(pixID)
plt.title('pixel ID')
# plt.savefig("D:/Plots/pixid_mock.png", dpi=800)
plt.show()

plt.plot(energy)
plt.title('energy PHA')
# plt.savefig("D:/Plots/energy_mock.png", dpi=800)
plt.show()