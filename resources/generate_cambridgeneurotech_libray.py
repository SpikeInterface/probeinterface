'''
202-01-07 CambridgeNeurotech
Original script:
  * contact: Thal Holtzman
  * email: info@cambridgeneurotech.com

2021-03-01
The script have been modified by Smauel Garcia (samuel.garcia@cnrs.fr):
  * more pytonic
  * improve code readability
  * not more channel_device_index the order is the conatct index
  * simpler function for plotting.

2021-04-02
Samuel Garcia:
  * add "contact_id" one based in Probe.

Derive probes to be used with SpikeInterface base on Cambridgeneurotech databases
Probe library to match and add on
https://gin.g-node.org/spikeinterface/probeinterface_library/src/master/cambridgeneurotech

see repos https://github.com/SpikeInterface/probeinterface

In the 'Probe Maps 2020Final.xlsx'
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from probeinterface.plotting import plot_probe
from probeinterface import generate_multi_columns_probe, combine_probes, write_probeinterface

from pathlib import Path


# work_dir = r"C:\Users\Windows\Dropbox (Scripps Research)\2021-01-SpikeInterface_CambridgeNeurotech"
# work_dir = '.'
# work_dir = '/home/samuel/Documents/SpikeInterface/2021-03-01-probeinterface_CambridgeNeurotech/'
work_dir = '/home/samuel/Documents/SpikeInterface/2022-05-20-probeinterface_CambridgeNeurotech/'
work_dir = Path(work_dir).absolute()

export_folder = work_dir / 'export_2021_05_20'
probe_map_file = work_dir /  'Probe Maps 2020Final_patch2022.xlsx'
probe_info_table_file = work_dir  / 'ProbesDataBase.csv'


# graphing parameters
plt.rcParams['pdf.fonttype'] = 42 # to make sure it is recognize as true font in illustrator
plt.rcParams['svg.fonttype'] = 'none'  # to make sure it is recognize as true font in illustrator


def convert_probe_shape(listCoord):
    '''
    This is to convert reference point probe shape inputed in excel
    as string 'x y x y x y that outline the shape of one shanck
    and can be converted to an array to draw the porbe
    '''
    listCoord = [float(s) for s in listCoord.split(' ')]
    res = [[listCoord[i], listCoord[i + 1]] for i in range(len(listCoord) - 1)]
    res = res[::2]

    return res

def convert_contact_shape(listCoord):
    '''
    This is to convert reference shift in electrodes
    '''
    listCoord = [float(s) for s in listCoord.split(' ')]
    return listCoord

def get_channel_index(connector, probe_type):
    """
    Get the channel index given a connector and a probe_type.
    This will help to re-order the probe contact later on.
    """

    # first part of the function to opne the proper connector based on connector name
    
    # header [0,1] is used to create a mutliindex 
    df = pd.read_excel(probe_map_file, sheet_name=connector, header=[0,1])

    # second part to get the proper channel in the
    if probe_type == 'E-1' or probe_type == 'E-2':
        probe_type = 'E-1 & E-2'

    if probe_type == 'P-1' or probe_type == 'P-2':
        probe_type = 'P-1 & P-2'

    if probe_type == 'H3' or probe_type == 'L3':
        probe_type = 'H3 & L3'

    if probe_type == 'H5' or probe_type == 'H9':
        probe_type = 'H5 & H9'

    tmpList = []
    for i in df[probe_type].columns:
        if len(df[probe_type].columns) == 1:
            tmpList = np.flip(df[probe_type].values.astype(int).flatten())
        else:
            tmp = df[probe_type][i].values
            tmp = tmp[~np.isnan(tmp)].astype(int) # get rid of nan and convert to integer
            tmp = np.flip(tmp) # this flips the value to match index that goes from tip to headstage of the probe
            tmpList = np.append(tmpList, tmp)
            tmpList = tmpList.astype(int)

    return tmpList


def generate_CN_probe(probe_info, probeIdx):
    """
    Generate a mono shank CN probe
    """
    if probe_info['part'] == 'Fb' or probe_info['part'] == 'F':
        probe = generate_multi_columns_probe(
                num_columns=probe_info['electrode_cols_n'],
                num_contact_per_column=[int(x) for x in convert_probe_shape(probe_info['electrode_rows_n'])[probeIdx]],
                 xpitch=float(probe_info['electrodeSpacingWidth_um']),
                 ypitch=probe_info['electrodeSpacingHeight_um'],
                 y_shift_per_column=convert_probe_shape(probe_info['electrode_yShiftCol'])[probeIdx],
                contact_shapes=probe_info['ElectrodeShape'],
                contact_shape_params={'width': probe_info['electrodeWidth_um'], 'height': probe_info['electrodeHeight_um']}
            )
        probe.set_planar_contour(convert_probe_shape(probe_info['probeShape']))

    else:
        probe = generate_multi_columns_probe(
                num_columns=probe_info['electrode_cols_n'],
                num_contact_per_column=int(probe_info['electrode_rows_n']),
                xpitch=float(probe_info['electrodeSpacingWidth_um']),
                ypitch=probe_info['electrodeSpacingHeight_um'],
                 y_shift_per_column=convert_contact_shape(probe_info['electrode_yShiftCol']),
                contact_shapes=probe_info['ElectrodeShape'],
                contact_shape_params={'width': probe_info['electrodeWidth_um'], 'height': probe_info['electrodeHeight_um']}
            )
        probe.set_planar_contour(convert_probe_shape(probe_info['probeShape']))

    if type(probe_info['electrodesCustomPosition']) == str:
        probe._contact_positions = np.array(convert_probe_shape(probe_info['electrodesCustomPosition']))

    return probe

def generate_CN_multi_shank(probe_info):
    """
    Generate a multi shank probe
    """
    sub_probes = []
    for probeIdx in range(probe_info['shanks_n']):
        sub_probe = generate_CN_probe(probe_info, probeIdx)
        sub_probe.move([probe_info['shankSpacing_um']*probeIdx, 0])
        sub_probes.append(sub_probe)
        
    multi_shank_probe = combine_probes(sub_probes)
    return multi_shank_probe


def create_CN_figure(probe_name, probe):
    """
    Create custum figire for CN with custum colors + logo
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)    
    
    n = probe.get_contact_count()
    plot_probe(probe, ax=ax,
            contacts_colors = ['#5bc5f2'] * n,  # made change to default color
            probe_shape_kwargs = dict(facecolor='#6f6f6e', edgecolor='k', lw=0.5, alpha=0.3), # made change to default color
            with_channel_index=True)

    ax.set_xlabel(u'Width (\u03bcm)') #modif to legend
    ax.set_ylabel(u'Height (\u03bcm)') #modif to legend
    ax.spines['right'].set_visible(False) #remove external axis
    ax.spines['top'].set_visible(False) #remove external axis

    ax.set_title('\n' +'CambridgeNeuroTech' +'\n'+  probe.annotations.get('name'), fontsize = 24)
    
    fig.tight_layout() #modif tight layout
    
    im = plt.imread(work_dir / 'CN_logo-01.jpg')
    newax = fig.add_axes([0.8,0.85,0.2,0.1], anchor='NW', zorder=0)
    newax.imshow(im)
    newax.axis('off')
    
    return fig
    

def export_one_probe(probe_name, probe):
    """
    Save one probe in "export_folder" + figure.
    """
    probe_folder = export_folder / probe_name
    probe_folder.mkdir(exist_ok=True, parents=True)
    probe_file = probe_folder / (probe_name + '.json')
    figure_file = probe_folder / (probe_name + '.png')
    
    write_probeinterface(probe_file, probe)
    
    fig = create_CN_figure(probe_name, probe)
    fig.savefig(figure_file)
    
    # plt.show()
    # avoid memory error
    plt.close(fig)
    

def generate_all_probes():
    """
    Main function.
    Generate all probes.
    """
    probe_info_table = pd.read_csv(probe_info_table_file)
    #~ print(probe_info_list)
    
    for i, probe_info in probe_info_table.iterrows():
        print(i, probe_info['part'])
        
        
        if probe_info['shanks_n'] == 1:
            # one shank
            probe_unordered = generate_CN_probe(probe_info, 0)
        else: 
            # multi shank
            probe_unordered = generate_CN_multi_shank(probe_info)
        
        # loop over connector case that re order the probe contact index
        for connector in list(probe_info[probe_info.index.str.contains('ASSY')].dropna().index):
            probe_name = connector+'-'+probe_info['part']
            print('  ', probe_name)
            
            channelIndex = get_channel_index(connector = connector, probe_type = probe_info['part'])
            order = np.argsort(channelIndex)
            probe = probe_unordered.get_slice(order)

            probe.annotate(name=probe_name,
                            manufacturer='cambridgeneurotech',
                            first_index=1) 
            
            # one based in cambridge neurotech
            contact_ids = np.arange(order.size) + 1
            contact_ids =contact_ids.astype(str)
            probe.set_contact_ids(contact_ids)

            export_one_probe(probe_name, probe)


if __name__ == '__main__':
    generate_all_probes()
