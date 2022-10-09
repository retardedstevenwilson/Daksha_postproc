# Program to read the .tdms file

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
pkt_size=5547 # change nos. of samples in single pkt
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

#********************************
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

"""
for i in range(0,(full_pkt_size),1):
    full_pkt_data.append(all_channel_data[i])
#print(full_pkt_data[293739:293783]) # 293783 means, print (293783-1)
#print("Lenghth of full_pkt_data:",len(full_pkt_data))
#print(full_pkt_data)
full_pkt_size=int(len(all_channel_data)/pkt_size)
print("total packets in acquired data:",full_pkt_size)
print("total samples in partial packet:",(len(all_channel_data)%pkt_size))
"""

for i in range(0,(len(all_channel_data)),1):
    full_pkt_data.append(all_channel_data[i])
#print(full_pkt_data[293739:293783]) # 293783 means, print (293783-1)
print("Length of full_pkt_data:",len(full_pkt_data))

# find out header index using numpy    
arr1=np.array(full_pkt_data)
index=np.where((arr1 == 1) & (np.roll(arr1,-1) == 1) & (np.roll(arr1,-2) == 1) & (np.roll(arr1,-3) == 1) & 
               (np.roll(arr1,-4) == 1) & (np.roll(arr1,-5) == 0) & (np.roll(arr1,-6) == 0) & 
               (np.roll(arr1,-7) == 1)  & (np.roll(arr1,-8) == 1)  & (np.roll(arr1,-9) == 0)
                & (np.roll(arr1,-10) == 1) & (np.roll(arr1,-11) == 0) & (np.roll(arr1,-12) == 0)
                 & (np.roll(arr1,-13) == 1) & (np.roll(arr1,-14) == 0) & (np.roll(arr1,-15) == 0)
                 & (np.roll(arr1,-16) == 0) & (np.roll(arr1,-17) == 0) & (np.roll(arr1,-18) == 1)) [0]
print("index:",index)
print("Length of index:",len(index))

"""
m=0
n=1
for ind in index:
    a=index[m] #start index
    b=index[n] # stop index
    for data1 in range(a, b,1):
        pkt0.append(full_pkt_data[data1])
    if(n<len(index)-1):
        m=m+1
        n=n+1
"""        

for x in range(index[67],(index[68]),1): # change index[2], index[3] for single packet reading
    pkt0.append(full_pkt_data[x])
print("Length pkt0:",len(pkt0))


print("***********************************************************")

# segregate header and event data array from tdms file        
for element in pkt0:
        #print (element)
        if(head_ctr<=42):
                header.append(element) # Segregate to header data array
                head_ctr=head_ctr+1
                #print(head_ctr)
        elif (head_ctr>42 and head_ctr<=(index[1]-1)):
                event_data.append(element) # Segregate to event data array
                head_ctr=head_ctr+1
                if(head_ctr==pkt_size): # End of packet data bit '0' (Which is not required, but sent from cuurent program in FPGA Simulator). 
                        head_ctr=0   # This end bit will not be written in event data array
print("header:",header)

print("Length of header_data:",(len(header)))
print("Length of event_data:",(len(event_data)))
print("all_channel_data:",all_channel_data[43:86])
print("full_pkt_data:",full_pkt_data[0:43])
print("event_data:",event_data[0:43])
print("********** Length of various arrays (timetag, det_id, pix_is & pix_energy) *************")
#header into decimal
for header_val in header:
    if(header_ctr<=42):
        header_arr.append(header_val)
        header_ctr=header_ctr+1
        #print(timetag_val)
            #header_ctr=0

        if(header_ctr==43):
            sum3=((32768*header_arr[0])+(16384*header_arr[1])+(8192*header_arr[2])+(4096*header_arr[3])+(2048*header_arr[4])+(1024*header_arr[5])+(512*header_arr[6])+(256*header_arr[7])+(128*header_arr[8])+(64*header_arr[9])+(32*header_arr[10])+(16*header_arr[11])+(8*header_arr[12])+(4*header_arr[13])+(2*header_arr[14])+(1*header_arr[15]))
            header_dec.append(sum3)
            sum5=((32768*header_arr[16])+(16384*header_arr[17])+(8192*header_arr[18])+(4096*header_arr[19])+(2048*header_arr[20])+(1024*header_arr[21])+(512*header_arr[22])+(256*header_arr[23])+(128*header_arr[24])+(64*header_arr[25])+(32*header_arr[26])+(16*header_arr[27])+(8*header_arr[28])+(4*header_arr[29])+(2*header_arr[30])+(1*header_arr[31]))
            header_dec_part2.append(sum5)
            sum6=((1024*header_arr[32])+(512*header_arr[33])+(256*header_arr[34])+(128*header_arr[35])+(64*header_arr[36])+(32*header_arr[37])+(16*header_arr[38])+(8*header_arr[39])+(4*header_arr[40])+(2*header_arr[41])+(1*header_arr[42]))
            header_dec_part3.append(sum6)
            header_ctr=0
print("sums are " )
print(sum3,sum5,sum6)        
"""
# event_data into without '0' (End of packet data)
for val in event_data:
    if(val_ctr<=41925):
        val_data.append(val)
        val_ctr=val_ctr+1
    elif(val_ctr==41926):
        val_ctr=0
"""
"""
# convert header value binary to decimal
for header_arr_val in header_arr:
    if(bit_count4<=15):
        sum11=sum11+(header_arr_val*(pow(2,bin_weight4)))
        bin_weight4-=1
        bit_count4+=1
        #print("ele:",element)
        #print(sum)
    if (bit_count4>15 and bit_count4==16):
        #print("ele1:",element)
        header_dec.append(sum11)
        bit_count4=0
        bin_weight4=15
        sum11=0 
"""

# Segregate event_data array into time tag, det. ID, pixel ID and pixel energy array
for data in event_data:
        if(data_ctr<=19):
                timetag.append(data)
                data_ctr=data_ctr+1
        elif(data_ctr>19 and data_ctr<=24):
                det_id.append(data)
                data_ctr=data_ctr+1
        elif(data_ctr>24 and data_ctr<=32):
                pix_id.append(data)
                data_ctr=data_ctr+1
        elif(data_ctr>32 and data_ctr<=42):
                pix_energy.append(data)
                data_ctr=data_ctr+1
                if(data_ctr==43):
                        data_ctr=0
print("timetag:",timetag[345060:345080])
print("Length: of time tag:",len(timetag))

"""
#det_id into decimal
for det_id_val in det_id:
    if(det_id_ctr<=4):
        det_id_arr.append(det_id_val)
        det_id_ctr=det_id_ctr+1
        if(det_id_ctr==5):
            sum=((16*det_id_arr[0])+(8*det_id_arr[1])+(4*det_id_arr[2])+(2*det_id_arr[3])+(1*det_id_arr[4]))
            det_id_dec.append(sum)
            det_id_ctr=0
"""
#print(det_id)
for det_id_val in det_id:
    if(bit_count1<=4):
        sum8=sum8+(det_id_val*(pow(2,bin_weight1)))
        bin_weight1-=1
        bit_count1+=1
        #print("ele:",element)
        #print(sum)
    if (bit_count1>4 and bit_count1==5):
        #print("ele1:",element)
        det_id_dec.append(sum8)
        bit_count1=0
        bin_weight1=4
        sum8=0
        
#print("timetag_dec:",timetag_dec)
for k in det_id_dec:
    if (k>10 or k<10):
        det_id_err.append(k)
#print("det_id error:",det_id_err)
print("Length of det_id error:",len(det_id_err))


"""         
#pix_id into decimal
for pix_id_val in pix_id:
    if(pix_id_ctr<=7):
        pix_id_arr.append(pix_id_val)
        pix_id_ctr=pix_id_ctr+1
        if(pix_id_ctr==8):
            sum4=((128*pix_id_arr[0])+(64*pix_id_arr[1])+(32*pix_id_arr[2])+(16*pix_id_arr[3])+(8*pix_id_arr[4])+(4*pix_id_arr[5])+(2*pix_id_arr[6])+(1*pix_id_arr[7]))
            pix_id_dec.append(sum4)
            if(sum4!=129):
                print(sum4)
            pix_id_ctr=0  
"""

for pix_id_val in pix_id:
    if(bit_count2<=7):
        sum9=sum9+(pix_id_val*(pow(2,bin_weight2)))
        bin_weight2-=1
        bit_count2+=1
        #print("ele:",element)
        #print(sum)
    if (bit_count2>7 and bit_count2==8):
        #print("ele1:",element)
        pix_id_dec.append(sum9)
        bit_count2=0
        bin_weight2=7
        sum9=0            

#print("timetag_dec:",timetag_dec)
for m in pix_id_dec:
    if (m>129 or m<129):
        pix_id_err.append(m)
#print("Length of pix_id error:",len(pix_id_err))

print("*************************************************")
"""         
#pixel energy into decimal
for pix_energy_val in pix_energy:
    if(pix_energy_ctr<=9):
        pix_energy_arr.append(pix_energy_val)
        pix_energy_ctr=pix_energy_ctr+1
        if(pix_energy_ctr==10):
            sum1=((512*pix_energy_arr[0])+(256*pix_energy_arr[1])+(128*pix_energy_arr[2])+(64*pix_energy_arr[3])+(32*pix_energy_arr[4])+(16*pix_energy_arr[5])+(8*pix_energy_arr[6])+(4*pix_energy_arr[7])+(2*pix_energy_arr[8])+(1*pix_energy_arr[9]))
            pix_energy_dec.append(sum1)
            pix_energy_ctr=0
"""

for pix_energy_val in pix_energy:
    if(bit_count3<=9):
        sum10=sum10+(pix_energy_val*(pow(2,bin_weight3)))
        bin_weight3-=1
        bit_count3+=1
        #print("ele:",element)
        #print(sum)
    if (bit_count3>9 and bit_count3==10):
        #print("ele1:",element)
        pix_energy_dec.append(sum10)
        bit_count3=0
        bin_weight3=9
        sum10=0            

#print("timetag_dec:",timetag_dec)
for n in pix_energy_dec:
    if (n>1 or n<1):
        pix_energy_err.append(n)
#print("det_id error:",det_id_err)
print("Length of pix_energy error:",len(pix_energy_err))


for timetag_val in timetag:
    if(bit_count<=19):
        sum7=sum7+(timetag_val*(pow(2,bin_weight)))
        bin_weight-=1
        bit_count+=1
        #print("ele:",element)
        #print(sum)
    if (bit_count>19 and bit_count==20):
        #print("ele1:",element)
        timetag_dec.append(sum7)
        bit_count=0
        bin_weight=19
        sum7=0
        
#print("timetag_dec:",timetag_dec)
for j in timetag_dec:
    if (j>153 or j<153):
        timetag_err.append(j)
#print("time tag error:",timetag_err)
print("Length of timetag error:",len(timetag_err))

"""
for p in range(0,(len(timetag_dec)-1),1):
               if((timetag_dec[p])!=153):
                   print("timetag error val:",p)
"""
print("*************************************************")


"""
#time tag into decimal
for timetag_val in timetag:
    if(timetag_ctr<=39):
        timetag_arr.append(timetag_val)
        timetag_ctr=timetag_ctr+1
        #print(timetag_val)
        if(timetag_ctr==40):
            sum2=((524288*timetag_arr[20])+(262144*timetag_arr[21])+(131072*timetag_arr[22])+(65536*timetag_arr[23])+(32768*timetag_arr[24])+(16384*timetag_arr[25])+(8192*timetag_arr[26])+(4096*timetag_arr[27])+(2048*timetag_arr[28])+(1024*timetag_arr[29])+(512*timetag_arr[30])+(256*timetag_arr[31])+(128*timetag_arr[32])+(64*timetag_arr[33])+(32*timetag_arr[34])+(16*timetag_arr[35])+(8*timetag_arr[36])+(4*timetag_arr[37])+(2*timetag_arr[38])+(1*timetag_arr[39]))
            timetag_dec.append(sum2)
            timetag_ctr=0
            
            
#time tag into decimal
for timetag_val in timetag:
    if(timetag_ctr<=19):
        timetag_arr.append(timetag_val)
        timetag_ctr=timetag_ctr+1
        #print(timetag_val)
        if(timetag_ctr==20):
            sum2=((524288*timetag_arr[0])+(262144*timetag_arr[1])+(131072*timetag_arr[2])+(65536*timetag_arr[3])+(32768*timetag_arr[4])+(16384*timetag_arr[5])+(8192*timetag_arr[6])+(4096*timetag_arr[7])+(2048*timetag_arr[8])+(1024*timetag_arr[9])+(512*timetag_arr[10])+(256*timetag_arr[11])+(128*timetag_arr[12])+(64*timetag_arr[13])+(32*timetag_arr[14])+(16*timetag_arr[15])+(8*timetag_arr[16])+(4*timetag_arr[17])+(2*timetag_arr[18])+(1*timetag_arr[19]))
            timetag_dec.append(sum2)
            timetag_ctr=0   
"""
                   
                        
# Convert the binary values from arrays (timetag, det_id etc.) to decimal values and write to array
#pix_id_arr=np.array(pix_id)
#pix_id_dec=np.packbits(pix_id_arr, axis=0, bitorder="big")

# Plotting graph
# fig, ax = plt.subplots(1,2)
# ax.plot(pix_id_dec)
# ax.plot(timetag_dec)

plt.plot(header_dec)
plt.title('Header part1')
plt.xlabel('Sample number (Nos.)')
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/header_dec_part1.png", dpi=800)
plt.show()


plt.plot(header_dec_part2)
plt.title('Header part2')
plt.xlabel('Sample number (Nos.)')
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/header_dec_part2.png", dpi=800)
plt.show()

plt.plot(header_dec_part3)
plt.title('Header part3')
plt.xlabel('Sample number (Nos.)')
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/header_dec_part3.png", dpi=800)
plt.show()


# Python help: type #%% in between of lines to execute part of code

#plt.plot(timetag_dec[:-7000],linewidth = 0.1,alpha=0.5)

"""
# Below lines to select some part of array
timetag_dec1 = timetag_dec[850:]
plt.plot(timetag_dec1[:-15954],linewidth = 0.1,alpha=0.5)
plt.title('Time tag')
import numpy as np
plt.xticks(np.arange(0, len(timetag_dec1[:-16154]), step=40))
plt.xlabel('Sample number (Nos.)')
plt.ylabel('Value (Decimal no.)')
plt.margins(0)
plt.yscale('symlog')
#plt.savefig("D:/Plots/timetag_dec.png", dpi=800)
plt.savefig("D:/Plots/timetag_dec.pdf")
plt.show()
"""
plt.tight_layout()
plt.plot(index)
plt.title('Header index value')
plt.xlabel('Sample number (Nos.)')
plt.yticks(fontsize=7, rotation = 0)
plt.ylabel('Index (Decimal no.)')
#plt.savefig("D:/Plots/index.png", dpi=800)
plt.show()

fig = plt.figure()
plt.tight_layout()
plt.plot(timetag_dec)
plt.title('Time tag')
plt.xlabel('Sample number (Nos.)')
plt.yticks(fontsize=7, rotation = 0)
plt.ylabel('Value (Decimal no.)')

#fig.savefig("D:/Plots/timetag_dec.png", dpi=800)
plt.show()

plt.plot(det_id_dec)
plt.title('Detector ID')
plt.xlabel('Sample number (Nos.)')
plt.yticks(fontsize=7, rotation = 0)
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/det_id_dec.png", dpi=800)
plt.show()

plt.plot(pix_id_dec)
plt.title('Pixel ID')
plt.xlabel('Sample number (Nos.)')
plt.yticks(fontsize=7, rotation = 0)
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/pix_id_dec.png", dpi=800)
plt.show()

plt.plot(pix_energy_dec)
plt.title('Pixel Energy')
plt.xlabel('Sample number (Nos.)')
plt.yticks(fontsize=7, rotation = 0)
plt.ylabel('Value (Decimal no.)')
#plt.savefig("D:/Plots/pix_energy_dec.png", dpi=800)
plt.show()

# Display for testing purpose
#print("Length of header array:",len(header))
#print("Length of event data array:",len(event_data))

print("Length of timetag:",len(timetag))
print("Length of det_id:",len(det_id))
print("Length of pix_id:",len(pix_id))
print("Length of pix_energy:",len(pix_energy))

# Calculate length in fraction
length_header=int((len(header_arr)/43))
length_timetag=(len(timetag)/20)
# print(length_timetag)
length_det_id=(len(det_id)/5)
# print(length_det_id)
length_pix_id=(len(pix_id)/8)
# print(length_pix_id)
length_pix_energy=(len(pix_energy)/10)
# print(length_pix_energy)

print("************************ Total event data & Sync bit details *************************")
#print("Total event data in all packet:",(len(val_data)/data_size))
print("Sync bit part1 (First 16-bits):",hex(header_dec[0]))
print("Sync bit part2 (Second 16-bits):",hex(header_dec_part2[0]))
print("Sync bit part3 (Last 11-bits):",hex(header_dec_part3[0]))
print("************************ Length of header and event data parameters ******************")
print("Length of header:",len(header_dec))
print("Length of timetag:",length_timetag)
print("Length of det_id:",length_det_id)
print("Length of pix_id:",length_pix_id)
print("Length of pix_energy:",length_pix_energy)
print("**************************************************************************************")
print("pix_eneergy_dec:",pix_energy_dec)