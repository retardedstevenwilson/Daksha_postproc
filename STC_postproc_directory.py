import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from scipy.optimize import curve_fit
from nptdms import TdmsFile
from functools import reduce



def readfile_csv (file_name): 
    '''reads the csv file and converts the hexagonal data to decimal. Returns the total counts with the table'''
    df=pd.read_csv(file_name)
    df['Pixel ID'] = df['Pixel ID'].apply(int, base=16)
    df['PHA'] = df['PHA'].apply(int, base=16)
    df['FPGA TimeStamp']=df['FPGA TimeStamp'].apply(int,base=16)
    total_sample_count=df['Iteration'].iloc[-1]
    return df, total_sample_count
    

def tdms_to_bool_read(in_file_name):
        # Define global constants below
        event_size=38 # Change nos. of bits in single event
        packet_size=129
        timestamp_bits=20
        pixid_bits=8
        detid_bits=0
        pha_bits=10


        # change index[x], index[y] for single packet reading in line 161
        #in_file_name="/home/ayushnema/Documents/work_DP2/tdms_conversions/Sandeep_files/TIFR_43bit_czt_pingpong_fcc_pingpong_23-9-2022_test7.tdms"

        tdms_file = TdmsFile.read(in_file_name)
        with TdmsFile.open(in_file_name) as tdms_file:
                group = tdms_file["Untitled"]
                all_groups = tdms_file.groups()
                #print (all_groups)

                channel=group["Untitled"]
                all_group_channels = group.channels()
                all_channel_data = channel[:]

        #print(len(all_channel_data))

        total_full_packets=int(len(all_channel_data)/(packet_size*event_size))
        #print(total_full_packets)

        total_events=total_full_packets*event_size

        #df=np.zeros((total_events,4))
        #df = pd.DataFrame(df, columns =["FPGA Timestamp","Detid","Pixel ID","PHA"])            for det id included
        #print(df)
        #df=np.zeros((total_events,3))
        #df = pd.DataFrame(df, columns =["FPGA Timestamps","Pixel ID","PHA"])               
        
        df = np.recarray((total_events, ), dtype=[('FPGA Timestamps', np.float32),      #Sujay's recarray idea 1. 
                                                      ('Pixel ID', np.uint8),
                                                      ('PHA', np.uint16)])
                                                     

        total_sample_count=len(all_channel_data)

        header=all_channel_data[0:event_size]
        print(len(header))
        #event_count=0
        #header_count=0
        
        print("no. of packets are ",total_full_packets)

        for element_index in range(0,len(all_channel_data)-(len(all_channel_data)%event_size),event_size):    
                event_elements=all_channel_data[element_index:element_index+43]
                # print(event_elements)
                #print(event_elements)
                # print(header)
                #print(event_elements)
                event_no=int(element_index/event_size)

                if (event_elements.all!=header).all():

                        timestamp_event=event_elements[0:timestamp_bits]
                        timestamp_event_str= ''.join(str(x) for x in timestamp_event)
                        timestamp_event_dec=int(timestamp_event_str, 2)
                        #print(timestamp_event_dec)
                        #df.at[event_no, "FPGA Timestamps"]= timestamp_event_dec                      
                        df[event_no]["Pixel ID"] =timestamp_event_dec                              #Sujay
                        
                        """ detID_event=event_elements[20:25]
                        detID_event_str="".join(str(x) for x in detID_event)
                        detID_event_dec=int(detID_event_str, 2)
                        #print(detID_event)
                        df.at[event_no, "Det ID"] = detID_event_dec
                         """
                        pixid_event=event_elements[timestamp_bits:timestamp_bits+pixid_bits]         
                        pixid_event_str="".join(str(x) for x in pixid_event)
                        pixid_event_dec=int(pixid_event_str, 2)
                        #df.at[event_no, "Pixel ID"] =pixid_event_dec
                        df[event_no]["Pixel ID"] =pixid_event_dec                                   #sujay
                        
                        pha_event=event_elements[timestamp_bits+pixid_bits:timestamp_bits+pixid_bits+pha_bits]
                        pha_event_str="".join(str(x) for x in pha_event)
                        pha_event_dec=int(pha_event_str, 2)
                        #df.at[event_no, "PHA"]=pha_event_dec
                        df[event_no]["PHA"]=pha_event_dec                                           #sujay

        print(df)

        return df,total_sample_count


def plot_intensity_graph (table,folder_address):
    '''generates the DPH'''
    pixel_count=[]
    for i in range(256):
        pixel_count.append(np.count_nonzero(table['Pixel ID']==i)) 
    pixel_count=np.asarray(pixel_count)
    pixel_count=pixel_count.reshape(16,16)
    pixel_count=pixel_count.transpose()
    #for i in range(8):
    #    pixel_count[i,:],pixel_count[16-i-1,:]=pixel_count[16-i-1,:],pixel_count[i,:]
    plt.figure()
    plt.pcolormesh(pixel_count)
    plt.grid()
    plt.colorbar()
    #plt.show()
    #saving the plot
    DPH_plotname=folder_address + '/DPH.png'
    plt.savefig(DPH_plotname)
    plt.close()


def energy_spectrum(table,folder_address): 
    '''plots the PHA spectrum with respect to counts'''
    plt.figure()
#    plt.hist(table['PHA'],bins=1024,range=(0,2000))
    plt.hist(table['PHA'],bins=4096)

    plt.xlabel('PHA')
    plt.ylabel('counts')
    #plt.grid()
    
    #saving the plot
    E_spec_plotname=folder_address + '/PHA spectrum.png'
    plt.savefig(E_spec_plotname)
    plt.close()


def median_model(table):
    pixel_count=[]
    for i in range(256):
        pixel_count.append(np.count_nonzero(table['Pixel ID']==i))    
    median=np.median(pixel_count)
    deviations=np.abs(pixel_count-median)
    MAD=np.median(deviations)
    total=np.sum(pixel_count)
    return median, MAD,total

def PHA_resolution_Am(table):
    '''for a rough idea, the PHA range has been roughly taken and a gaussian is assumed between the range
    Not for lab use. This function will be reset by curve fitting for proper results'''
    PHA=pd.Series(table['PHA'].head())
    PHA=table['PHA'].to_numpy()
    mask=(PHA>700) & (PHA<1200)
    PHA_Am=PHA[mask]
    Am_mean=np.mean(PHA_Am)
    Am_std=np.std(PHA_Am)
    return Am_mean , Am_std
    
def PHAs_of_Pixel(table,ID):
    '''generates the PHA spectrum of given pixel ID. To be used in PHA_subplots'''
    pixid=[]    
    PHAs=[]
    pixid=np.where(table['Pixel ID']==ID)
    PHAs.append(table['PHA'].loc[pixid])
    return PHAs



def PHA_subplots(table,PHA_limit,count_limit,folder_address):
    '''generates a 16*16 array of graphs showing the PHA specturm of every pixel'''
    fig,ax=plt.subplots(16,16)
    fig.set_size_inches(30, 30)
    for x in range(16):
        for y in range(16):    
            ax[x][y].set_xlim(0, PHA_limit)
            ax[x][y].set_ylim(0,count_limit)
            ax[x, y].text(PHA_limit/2,count_limit/2, str((x, y)),fontsize=6, ha='center')    
        
    for ID in range(256):
        xaxis=(ID%16)
        yaxis=((ID//16))
        PHAs=PHAs_of_Pixel(table, ID)
        ax[xaxis][yaxis].hist(PHAs,range=[0,2000],bins=60,histtype='stepfilled',alpha=1,color='red')
        
    PHA_subplot_name=folder_address + '/pixel PHA spectrum.png'
    plt.savefig(PHA_subplot_name)
    plt.close()




def Am_res_simple(table,folder_address):

    pha=table['PHA']
    pha=pha.to_numpy()
    type(pha)

    Am_spec=[]
    PHA_axis=[]
    binsize=10
    for i in range(0,int(1200/binsize)):
        PHA_axis.append(400+i*binsize)
        count=np.count_nonzero((pha>(400+(i*binsize))) & (pha<(400+((i+1)*binsize))))
        Am_spec.append(count)

    from scipy.optimize import curve_fit

    x = np.asarray(PHA_axis)
    y = np.asarray(Am_spec)

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    amplitude_fit=popt[0]
    mean_fit=popt[1]
    stddev_fit=popt[2]
    
    E_resolution=2.355*stddev_fit/mean_fit
    

    
    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
    plt.legend()
    plt.title('Fig. - Am peak')
    plt.xlabel('PHA')
    plt.ylabel('counts')
    Am_spec_plotname=folder_address + '/Am_fit_gauss.png'
    plt.savefig(Am_spec_plotname)
    plt.close()
    
    return popt , E_resolution

'''

def am_res_skewed_gauss(table,folder_address):
    from lmfit.models import SkewedGaussianModel
    
    
    pha=table['PHA']
    pha=pha.to_numpy()
    Am_spec=[]
    PHA_axis=[]
    binsize=10
    for i in range(0,int(1200/binsize)):
        PHA_axis.append(600+i*binsize)
        count=np.count_nonzero((pha>(600+(i*binsize))) & (pha<(600+((i+1)*binsize))))
        Am_spec.append(count)

    x = np.asarray(PHA_axis)
    y = np.asarray(Am_spec)

    
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
    xvals = np.asarray(PHA_axis)
    yvals = np.asarray(Am_spec)
    model = SkewedGaussianModel()    
    # set initial parameter values
    gamma_gauss=(popt[1]-np.median(Am_spec))/popt[2]


    params = model.make_params(amplitude=4000, center=900, sigma=70, gamma=gamma_gauss)

    # adjust parameters  to best fit data.
    result = model.fit(yvals, params, x=xvals)

    
    
    print(result.fit_report())
    plt.plot(xvals, yvals,'b+:', label='data')
    plt.plot(xvals, result.best_fit,'r-', label='fit') 
    plt.title('Fig. - Am peak for skewed gauss')
    plt.xlabel('PHA')
    plt.ylabel('counts')
    Am_spec_plotname=folder_address + '/Am_fit_skewedgauss.png'
    plt.savefig(Am_spec_plotname)
    plt.close()
    
    

    skew_amp=result.params['amplitude'].value
    skew_std=result.params['sigma'].value
    skew_mean=result.params['center'].value

    E_res_skew=skew_std*2.355/skew_mean
    return skew_amp,skew_std,skew_mean,E_res_skew



def noise_fileread(infile):
    table_noise=readfile_csv(infile)
    pha_noise=table_noise['PHA']
    pha_noise=pha_noise.to_numpy()
    counts_noise=[]
    PHA_axis_noise=[]
    binsize=10
    for i in range(0,int(4000/binsize)):
        PHA_axis_noise.append(i*binsize)
        count=np.count_nonzero((pha_noise>((i*binsize))) & (pha_noise<(((i+1)*binsize))))
        counts_noise.append(count)
    import scipy.interpolate
    y_interp = scipy.interpolate.interp1d(PHA_axis_noise, counts_noise,kind='cubic')
    return y_interp

def noise_subtracted(infile,noise_infile):
    noise_interp=noise_fileread(noise_infile)
    table=readfile_csv(infile)
    pha=table['PHA']
    pha=pha.to_numpy()
    counts=[]
    PHA_axis=[]
    binsize=10
    for i in range(0,int(4000/binsize)):
        PHA_axis.append(i*binsize)
        count=np.count_nonzero((pha>((i*binsize))) & (pha<(((i+1)*binsize))))
        counts.append(count)
    
    
    pha_noise_subtracted=pha-noise_interp(pha)
    
'''



def iterative_3sig1sig_fit(table,folder_address):
    
    pha=(table['PHA'])
    
    pha=pha.to_numpy()
    Am_spec_y=[]
    PHA_axis_x=[]
    Am_min=600
    Am_max=1200
    binsize=10
    for i in range(0,int(Am_max/binsize)):
        PHA_axis_x.append(Am_min+i*binsize)
        count=np.count_nonzero((pha>(Am_min+(i*binsize))) & (pha<(Am_min+((i+1)*binsize))))
        Am_spec_y.append(count)

    from scipy.optimize import curve_fit

    x = np.asarray(PHA_axis_x)
    y = np.asarray(Am_spec_y)

    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    
    Am_mean_iter=popt[1]
    Am_stddev_iter=popt[2]
    Am_min_iter=popt[1]-popt[2]
    Am_max_iter=popt[1]+3*popt[2]
        
    for i in range(10):
        range_iter=np.where((x>Am_min_iter)&(x<Am_max_iter))[0]
        print(range_iter)
        popt_iter,_=curve_fit(Gauss, x[range_iter], y[range_iter], p0=[max(y), Am_mean_iter, Am_stddev_iter])
        print(popt)
        Am_min_iter=popt_iter[1]-popt_iter[2]
        print([Am_min_iter,Am_max_iter])
        
        Am_max_iter=popt_iter[1]+3*popt_iter[2]
    plt.plot(x, y, 'b+:', label='data')
    plt.plot(x, Gauss(x, *popt_iter), 'r-', label='fit')
    plt.legend()
    plotname=folder_address + '/Am_fit_gaussiterations3sigma1sigma.png'
    plt.savefig(plotname)
    plt.close()
    
    fwhm_fit_iter=2.355*popt[2]
    E_resolution_iter=fwhm_fit_iter/popt[1]
    return popt_iter,E_resolution_iter


    
def save_data(infile):
    '''runs all the defined functions and saves the plots and data in an identical directory created alongside the
    source folder.'''
    
    table,total_sample_count=tdms_to_bool_read(infile) #reading the file
    
    path=Path(infile).parent.absolute() 
    out_dir=output_path+'\\' + os.path.basename(path)+'\\'+ os.path.basename(infile)[:-4]  # making the directory 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
   
    #using the written functions to acquire the graphs and values which are needed to be saved
    median_table,MAD_table,MAD_total_count= median_model(table) 
    energy_spectrum(table,out_dir)
    plot_intensity_graph(table,out_dir)
    PHA_subplots(table,4096,100,out_dir)
    #popt,E_res=Am_res_simple(table,out_dir)
    #skew_amp,skew_std,skew_mean,E_res_skew=am_res_skewed_gauss(table,out_dir)
    
    #popt_iter,E_res_iter=iterative_3sig1sig_fit(infile,out_dir)
    
    #creating a text file for conclusions 
    #fh=open(out_dir+'\\remarks.txt','w+')
    #fh.write("Total counts of the sample are %d \n" %total_sample_count )
    #fh.write("The median of the PHA of the sample is %d \n" % median_table)
    #fh.write("the median standard deviation of sample is %d \n" %MAD_table)
    #fh.write("the MAD total of sample is %d \n" %MAD_total_count)
    #fh.write("the iterated gaussian values are %d \n" %popt_iter)
    #fh.write("the iterated resolution is %d \n" %E_res_iter)
    
    
    #fh.write("the mean of Am241 is %d \n" %popt[1])
    #  fh.write("the std of Am241 is %d \n" %popt[2])
    #fh.write("the skewed gaussian amplitude is %d \n" %skew_amp )
    #fh.write("the resolution of Am241 line is %d\n " %E_res)
    #fh.write("the skewed gaussian std is %d \n" %skew_std )
    #fh.write("the skewed gaussian mean is %d \n" %skew_mean )
    #fh.write("the skewed gaussian resoluton is %d \n" %E_res_skew )
    
    #fh.close()
    
    
    

 
    


'''running the code'''

input_path = r'/home/ayushnema/Documents/work_DP2/Daksha_postproc/detector_datadump/tdms_dump'
#output_path =r'/home/ayushnema/Documents/work_DP2/Daksha_postproc/detector_datadump/tdms_dump_result'
output_path =input_path+ r'tdms_dump_result'

processedfilecount=0            
for file in glob.glob(input_path +"/*active/*.tdms" ):
   print(file)
   save_data(file)
   processedfilecount+=1
   print("no.of processed files",processedfilecount)