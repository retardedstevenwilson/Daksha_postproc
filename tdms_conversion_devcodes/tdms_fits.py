import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from scipy.optimize import curve_fit
from nptdms import TdmsFile
from functools import reduce


in_file_name=r'/home/ayushnema/Documents/work_DP2/Daksha_postproc/detector_datadump/tdms_dump/TIFR_37bit_czt_pingpong_fcc_pingpong_6-10-2022_test7.tdms'

event_size=43              # Change nos. of bits in single event
header_bits=43  

header_pktid_bits=8
header_totevt_bits=10
header_other_bits=10

timestamp_bits=20
pixid_bits=8
detid_bits=5
pha_bits=10

timestamp_bits_base=2**np.arange(timestamp_bits)[::-1]
pixid_bits_base=2**np.arange(pixid_bits)[::-1]
detid_bits_base=2**np.arange(detid_bits)[::-1]
pha_bits_base=2**np.arange(pha_bits)[::-1]


tdms_file = TdmsFile.read(in_file_name)
with TdmsFile.open(in_file_name) as tdms_file:
        group = tdms_file["Untitled"]
        channel=group["Untitled"]
        all_channel_data = channel[:]

len_all_channel_data=len(all_channel_data)
event_counter_base_for_header=2**np.arange(header_totevt_bits)[::-1]
header_indices=[]
events_list=[]
total_events=0
total_headers=0
events_chunk=[]
events_data=[]   #chunktaker

#for bitid in range(0,len_all_channel_data):
bitid=0

while bitid <= len_all_channel_data:
    header=all_channel_data[bitid:bitid+header_bits]
    total_headers+=1
    
    # Read the header and calculate number of events
    header_pktid=header[0:header_pktid_bits]
    header_totevt=header[header_pktid_bits:header_pktid_bits + header_totevt_bits]  
    header_other=header[header_pktid_bits + header_totevt_bits:header_pktid_bits + header_totevt_bits+header_other_bits]

    bitid+=header_bits

    packet_no_of_events=np.dot(header_totevt,event_counter_base_for_header)
    packet_no_of_events=128

    total_events+=packet_no_of_events
    #for i in all_packets_data[bitid:(bitid+(packet_no_of_events*event_size))]:
    #    events_data.append(i)
    events_data.extend(all_channel_data[bitid:(bitid+(packet_no_of_events*event_size))])

    #1event_energies = np.sum(event_data[evt_start_bit:evt_start_bit+event_size] * 2 ** np.arange(event_size-1, -1, -1), axis=1)  #VB line
    
    
    bitid+=(packet_no_of_events*event_size)
    

partial_data_start=int(len(events_data)/event_size)*event_size
print(partial_data_start)
print(len(events_data))
events_data=events_data[0:partial_data_start]#int(events_data/event_size)*event_size)]
print(len(events_data))
events_data=np.array(events_data)
total_events=int(partial_data_start/event_size)

mask=np.ones((128,))
mask[80:]=0
mask=mask.astype(bool)
mask=np.tile(mask,total_events)
events_data=events_data[mask]

total_events=len(events_data)



events_data=events_data.reshape(total_events,event_size)
print(events_data)
events_list = np.recarray((total_events, ), dtype=[('FPGA Timestamp', np.uint32),  
                                                  ('Det ID',np.uint8),
                                                  ('Pixel ID', np.uint8),
                                                 ('PHA', np.uint16)])

timestamp_event=events_data[:,0:timestamp_bits]
events_list["FPGA Timestamp"]=np.dot(timestamp_event,timestamp_bits_base)

detid_event=events_data[:,timestamp_bits:timestamp_bits+detid_bits]
events_list["Det ID"]= np.dot(detid_event,detid_bits_base)


pixid_event=events_data[:,timestamp_bits+detid_bits:timestamp_bits+detid_bits+pixid_bits]         
events_list["Pixel ID"]=np.dot(pixid_event,pixid_bits_base)

pha_event=events_data[:,timestamp_bits+detid_bits+pixid_bits:timestamp_bits+detid_bits+pixid_bits+pha_bits]
events_list["PHA"]=np.dot(pha_event,pha_bits_base)



print(events_list)
np.savetxt('test7_fitsscript_verification.txt',events_list,fmt='%i',delimiter=",")
        
   ################################################################################################################################################# 
#print("all channel data is",all_packets_data)
#print("event data is",events_data[0:43])

#print("no of events are",total_events)
#print(len(events_data)+(total_headers*43))
##int(total_headers+total_events)
#print(total_packets)
#print(total_packets*129*43)
#print((events_data))

