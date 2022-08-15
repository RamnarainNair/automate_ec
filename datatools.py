import csv
import datetime
import pandas as pd
from utils import get_measurements, get_sensors, convert_measurement, get_nodes_by_serial, get_nodes_by_zoneId_farmId


def historical_measurements(node_serial, startTime, endTime, raw_vwc=False):
    """Returns a list of measurements from a node with serial node_serial taken between startTime and endTime"""
    measurements = get_measurements(node_serial,startTime,endTime)
    sensors = get_sensors(node_serial)
    computed_measurements = []
    for measurement in measurements:
        computed_measurements.append(convert_measurement(measurement, sensors, raw_vwc))
    return computed_measurements


def to_dataframe(measurements, with_raw_vwc = False):
    """Fits measurements to a Pandas Dataframe"""
    df = pd.DataFrame(columns=["Depth", "DT", "Date", "Time", "EC", "VWC"] + (["Raw VWC"] if with_raw_vwc else []) + ["Temperature"])
    for measurement in measurements:
        data = measurement["topToBottom"]
        date_time = datetime.datetime.utcfromtimestamp(int(measurement["timestamp"]) // 1000)
        for dm in data:
            row = [dm["depth"], date_time, date_time.date(), date_time.time(), dm["electricalConductivity"], dm["volumetricWaterContent"]] + ([dm["rawVolumetricWaterContent"]] if "rawVolumetricWaterContent" in dm.keys() else ["-"]) + [dm["temperature"]]
            df.loc[len(df)] = row
    return df