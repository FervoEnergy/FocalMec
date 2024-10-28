from matplotlib import pyplot as plt

from IPython import display
import pyproj
import obspy
from obspy.geodetics import gps2dist_azimuth
from joblib import Parallel,delayed
from tqdm.auto import tqdm
import pandas as pd
from scipy.signal import find_peaks, find_peaks_cwt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from obspy import Trace, UTCDateTime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import glob
import os

import warnings
warnings.filterwarnings("ignore")



# model_weight = "/home/ahmadfervo/cape/polarity_prediction/convnetlstm_cls01.pth"
# csvjoined = "/home/ahmadfervo/cape/catalog/b060_catalog_ges_halliburton2.csv"
waveform_path = "/home/ahmadfervo/cape/20240601-10_earthquake_mseed"
stationxml = "/home/ahmadfervo/cape/stationxml"
eqnet_csv = "/home/ahmadfervo/cape/results_eqnet06/picks.csv"
# nori_picks = "/home/ahmadfervo/cape/nori_picks"
# csv06 = "/home/ahmadfervo/cape/catalog/GES_FervoCAPECirculation_allevents.csv"


# # Define the NAD83 UTM zone 12 coordinate system
# nad83_utm12 = pyproj.CRS("EPSG:26912")

# # Define the WGS84 coordinate system
# wgs84 = pyproj.CRS("EPSG:4326")

# events = pd.read_csv(csvjoined)
# events["origin_time"] = pd.to_datetime(events["origin_time"])
# # events = events.rename(columns={"latitude_deg":"latitude","longitude_deg":"longitude"})

# transformer = pyproj.Transformer.from_crs(nad83_utm12,wgs84,always_xy=True)

# events['longitude'], events['latitude'] = transformer.transform(events['ges_Easting(m)'].values,events['ges_Northing(m)'].values)
# events = events[(events["ges_time_diff_sec"]<1)]
# # events = events[events["magnitude"]>0.5]
# events = events.sort_values(by='magnitude',ascending=False)
# events["depth"] = events["ges_Depth(m)"]*(-0.001)
# for idx,row in events.iterrows():
#     events.loc[idx,["event_index"]] = f"eq{row['id']:05d}"
# print(len(events))


events = pd.read_csv("/home/ahmadfervo/cape/catalog/GES_202406_mag01events.csv")
# events = events[events["magnitude"]>=0.5]

inv = obspy.read_inventory(f"{stationxml}/*.xml")
stations = []

for net in inv:
    for sta in net:
        for cha in sta:
            station = {}
            sid = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code[:-1]}"
            station["station_id"] = sid
            station["latitude"] = cha.latitude
            station["longitude"] = cha.longitude
            station["depth_km"] = cha.depth/1000
            station["elevation"] = cha.elevation

            stations.append(station)

stations = pd.DataFrame(stations)
stations = stations.drop_duplicates(subset=["station_id"])


eqpicks=pd.read_csv(f"{eqnet_csv}")

picks = pd.concat([eqpicks],ignore_index=True)
_picks = picks.add_prefix("pick_")
_picks = _picks.rename(columns={"pick_station_id":"station_id","pick_event_index":"event_index"})
_stations = stations.add_prefix("station_")
_stations = _stations.rename(columns={"station_station_id":"station_id"})
_events = events.add_prefix("origin_")
_events = _events.rename(columns={"origin_event_index":"event_index","origin_origin_time":"origin_time"})
_picks = _picks.merge(_stations, on="station_id", suffixes=("_pick", "_station"))
_picks = _picks.merge(_events, on="event_index", suffixes=("_pick", "_origin"))

print(len(_picks))


def predict_polarity(pick,phase="P"):
    before_p = 16
    stream = obspy.read(f"{waveform_path}/{pick.event_index}/{pick.event_index}_{pick.station_id}*.mseed")
    inv = obspy.read_inventory(f"{stationxml}/{pick.station_id}.xml")
    stream = stream.remove_sensitivity(inv)
    stream.rotate(method="->ZNE", inventory=inv)
    dist, az, baz = gps2dist_azimuth(pick.origin_latitude,pick.origin_longitude,
                                        pick.station_latitude,pick.station_longitude)
    for tr in stream:
        tr.stats.azimuth = az
        tr.stats.back_azimuth = baz

    stream.rotate(method="NE->RT")
    stream.resample(200)
    smp = stream[0].stats.sampling_rate

    if phase=="snr":
        zcomp = stream.select(channel="*Z")[0]
        zcomp.filter("bandpass",freqmin=1,freqmax=95,zerophase=True)
        phase_time = obspy.UTCDateTime(pick.pick_phase_time)
        start_time = zcomp.stats.starttime
        phase_arrival = phase_time-start_time
        
        pindx = int(phase_arrival/zcomp.stats.delta)
        ppl=0.0
        ppl_score = 0.0
        zdata = zcomp.data[pindx-int(smp*0.1):pindx+int(smp*0.15)]
        # print(pindx,zdata.shape[0])
        amp = np.max(np.abs(zdata))
        noisedata = zcomp.data[pindx-int(smp*0.3):pindx-int(smp*0.1)]
        noise = np.max(np.abs(noisedata))
        amp = amp/noise
    
    if phase=="Pz":
        zcomp = stream.select(channel="*Z")[0]
        zcomp.filter("bandpass",freqmin=1,freqmax=55,zerophase=True)
        phase_time = obspy.UTCDateTime(pick.pick_phase_time)
        start_time = zcomp.stats.starttime
        phase_arrival = phase_time-start_time
        
        pindx = int(phase_arrival/zcomp.stats.delta)
        
        # prob , predicted = 0.0,0.0
        
        # pred = int(predicted[0])
        ppl = 0.0
        ppl_score = 0.0
        zdata = zcomp.data[pindx-int(smp*0.05):pindx+int(smp*0.1)]
        amp = np.mean(np.abs([zdata.min(),zdata.max()]))

    if phase=="Pr":
        zcomp = stream.select(channel="*R")[0]
        zcomp.filter("bandpass",freqmin=1,freqmax=55,zerophase=True)
        phase_time = obspy.UTCDateTime(pick.pick_phase_time)
        start_time = zcomp.stats.starttime
        phase_arrival = phase_time-start_time
        
        pindx = int(phase_arrival/zcomp.stats.delta)

        # prob , predicted = 0.0,0.0
        
        # pred = int(predicted[0])
        ppl = 0.0
        ppl_score = 0.0
        zdata = zcomp.data[pindx-int(smp*0.05):pindx+int(smp*0.1)]
        amp = np.mean(np.abs([zdata.min(),zdata.max()]))

    elif phase=="Sh":
        tcomp = stream.select(channel="*T")[0]
        tcomp.filter("bandpass",freqmin=1,freqmax=55,zerophase=True)
        phase_time = obspy.UTCDateTime(pick.pick_phase_time)
        start_time = tcomp.stats.starttime
        phase_arrival = phase_time-start_time
        
        pindx = int(phase_arrival/tcomp.stats.delta)

        # prob , predicted = 0.0,0.0
        
        # pred = int(predicted[0])
        ppl = 0.0
        ppl_score = 0.0
        tdata = tcomp.data[pindx-int(smp*0.05):pindx+int(smp*0.15)]
        amp = np.mean(np.abs([tdata.min(),tdata.max()]))

    elif phase=="Sv":
        rcomp = stream.select(channel="*R")[0]
        rcomp.filter("bandpass",freqmin=1,freqmax=55,zerophase=True)
        phase_time = obspy.UTCDateTime(pick.pick_phase_time)
        start_time = rcomp.stats.starttime
        phase_arrival = phase_time-start_time
        
        pindx = int(phase_arrival/rcomp.stats.delta)

        # prob , predicted = 0.0,0.0
        
        # pred = int(predicted[0])
        ppl = 0.0
        ppl_score = 0.0
        rdata = rcomp.data[pindx-int(smp*0.05):pindx+int(smp*0.15)]
        amp = np.mean(np.abs([rdata.min(),rdata.max()]))

    return ppl, ppl_score,amp



def fix_amplitude(event_id):
    event_picks = _picks[_picks["event_index"]==event_id]
    event_picks = event_picks.sort_values(by=["pick_phase_amplitude"],ascending=False)
    event_picks = event_picks.drop_duplicates(subset=["station_id","pick_phase_type"])

    event_df = pd.concat(Parallel(n_jobs=48)(delayed(fix_station_picks)(event_picks[event_picks["station_id"]==station]) for station in event_picks["station_id"].unique()),ignore_index=True)

    return event_df


def fix_station_picks(station_pick):
    ppick = station_pick[station_pick["pick_phase_type"]=="P"]
    spick = station_pick[station_pick["pick_phase_type"]=="S"]
    column_names = ['station_id', 'phase_index', 'phase_time', 'phase_score','psnr','phase_type',
            'dt_s', 'phase_polarity','phase_amplitude', 'event_index']


    df_list = []
    if len(ppick)>0:
        ppick = station_pick[station_pick["pick_phase_type"]=="P"].iloc[0]
        df_list.append(ppick)

        try:
            snrpick = ppick.copy()
            _,_,snr = predict_polarity(snrpick,phase="snr")

            pzpick = ppick.copy()
            ppl,_,amp = predict_polarity(pzpick,phase="Pz")
            pzpick["pick_phase_polarity"] = ppl*_
            pzpick["pick_phase_type"] = "Pz"
            pzpick["pick_phase_amplitude"] = amp
            df_list.append(pzpick)

            prpick = ppick.copy()
            ppl,_,amp = predict_polarity(pzpick,phase="Pr")
            prpick["pick_phase_polarity"] = ppl*_
            prpick["pick_phase_type"] = "Pr"
            prpick["pick_phase_amplitude"] = amp
            df_list.append(prpick)
        except Exception as e:
            # print(e)
            # raise e
            pass
    if len(spick)>0:
        spick = station_pick[station_pick["pick_phase_type"]=="S"].iloc[0]
        df_list.append(spick)

        try:
            shpick = spick.copy()
            ppl,_,amp = predict_polarity(shpick,phase="Sh")
            shpick["pick_phase_polarity"] = ppl*_
            shpick["pick_phase_type"] = "Sh"
            shpick["pick_phase_amplitude"] = amp
            df_list.append(shpick)

            svpick = spick.copy()
            ppl,_,amp = predict_polarity(svpick,phase="Sv")
            svpick["pick_phase_polarity"] = ppl*_
            svpick["pick_phase_type"] = "Sv"
            svpick["pick_phase_amplitude"] = amp
            df_list.append(svpick)
        except:
            pass

    df = pd.DataFrame(df_list)
    df["psnr"] = snr
    df.columns = df.columns.str.replace("pick_","")
    df = df.loc[:,df.columns.str.contains("|".join(column_names))]

    return df



evids = _picks["event_index"].unique()

if __name__=="__main__":
    data_frames = []
    for event_id in tqdm(_picks["event_index"].unique()):
        data_frames.append(fix_amplitude(event_id))


    new_picks = pd.concat(data_frames,ignore_index=True)
    new_picks.to_csv("/home/ahmadfervo/cape/picks_amp_06.csv")