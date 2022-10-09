
# Program to read the .tdms file but by using booolean array and its function

#**************** Read multiple pakts in tdms file (Should not use for single pkt data)**************************
#pkt_size=39087
# 10pkts in 1sec
# total samples=390870 in 1 sec
#********************************************

import bitarray as bit
#from operator import index
from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import pandas as pd


def tdms_to_bool_read(in_file_name)
        # Define global constants below
        event_size=43 # Change nos. of bits in single event
        packet_size=129
        # change index[x], index[y] for single packet reading in line 161

        #in_file_name="/home/ayushnema/Documents/work_DP2/tdms_conversions/Sandeep_files/TIFR_43bit_czt_pingpong_fcc_pingpong_23-9-2022_test7.tdms"

        tdms_file = TdmsFile.read(in_file_name)
        with TdmsFile.open(in_file_name) as tdms_file:
                group = tdms_file["Untitled"]
                all_groups = tdms_file.groups()
                print (all_groups)

                channel=group["Untitled"]
                all_group_channels = group.channels()
                all_channel_data = channel[:]

        print(len(all_channel_data))

        total_full_packets=int(len(all_channel_data)/packet_size)
        print(total_full_packets)

        total_events=total_full_packets

        df=np.zeros((total_events,4))
        df = pd.DataFrame(df, columns =["timestamps","detid","pixid","pha"])
        print(df)




        header=all_channel_data[0:43]
        print(len(header))
        #event_count=0
        header_count=0


        for element_index in range(0,len(all_channel_data)-(len(all_channel_data)%43),43):    
                event_elements=all_channel_data[element_index:element_index+43]
                # print(event_elements)
                #print(event_elements)
                # print(header)
                #print(event_elements)
                event_no=int(element_index/43)

                if (event_elements.all!=header).all():

                        timestamp_event=event_elements[0:20]
                        timestamp_event_str= ''.join(str(x) for x in timestamp_event)
                        timestamp_event_dec=int(timestamp_event_str, 2)
                        #print(timestamp_event_dec)
                        df.at[event_no, "timestamps"]= timestamp_event_dec

                        detID_event=event_elements[20:25]
                        detID_event_str="".join(str(x) for x in detID_event)
                        detID_event_dec=int(detID_event_str, 2)
                        #print(detID_event)
                        df.at[event_no, "detid"] = detID_event_dec
                        
                        pixid_event=event_elements[25:33]
                        pixid_event_str="".join(str(x) for x in pixid_event)
                        pixid_event_dec=int(pixid_event_str, 2)
                        df.at[event_no, "pixid"] =pixid_event_dec
                        
                        pha_event=event_elements[33:43]
                        pha_event_str="".join(str(x) for x in pha_event)
                        pha_event_dec=int(pha_event_str, 2)
                        df.at[event_no, "pha"]=pha_event_dec

        

        return df

#print(all_channel_data)
