import os
import logging

normalizationMediumU1 = {'description':'air', 'epsilonR': 1, 'vwc': -0.2025}
normalizationMediumU2 = {'description':'propan-2-ol', 'epsilonR': 20.1, 'vwc': 23.7848}
ecPolynomialA = -3.0992195e-5
ecPolynomialB = 4.20087056e-8
ecPolynomialC = -2.40624321e-11
ecPolynomialD = 4.06407987e-2
ecFactor = 10.0 / 93.5

def get_connection_string():
    return os.getenv('MONGO_CONN')

def get_database(version='prod'):
    from pymongo import MongoClient

    CONNECTION_STRING = get_connection_string()
    client = MongoClient(CONNECTION_STRING)
    return client['agurotech-' + version], client

def get_measurements(nodeSerial, startTime, endTime):
    db, client = get_database()
    measurements = list(db['measurements'].find({
        "nodeSerial": nodeSerial,
        "timestamp":{
            "$gte": startTime,
            "$lte": endTime
        }
    }))
    client.close()
    return measurements

def get_sensors(nodeSerial):
    db, client = get_database()
    node = list(db['nodes'].find({
        "_id": nodeSerial,
    }))
    node = node[0]
    client.close()
    return node['sensors']

def get_nodes_by_serial(nodeSerials=None):
    db, client = get_database()
    query = {} if nodeSerials is None else {
        "_id": {"$in":nodeSerials}
    }
    nodes = list(db['nodes'].find(query))
    client.close()
    return nodes

def get_nodes_by_zoneId_farmId(zoneId, farmId):
    from bson.objectid import ObjectId
    db, client = get_database()
    # TODO: change to latestDeviceOwner after migrations are applied
    query = {
        "latestNodeOwner.farmId": farmId,
       "latestDeployment.zoneId": ObjectId(zoneId),
    }
    nodes = list(db['nodes'].find(query))
    client.close()
    return nodes

def resToVwc(res):
    if (res <= 0.5):
         return 0.0
    if (res >= 1.0):
         return 100.0
    return (res - 0.5) / 0.5 * 100

def isPCBCalibration(sensor):
    coefs = ['intersect', 'slopeA', 'slopeB', 'slopeC']
    for coef in coefs:
        if coef not in sensor.keys():
            return False    
    return True

def convertMeasurementToVWCWithFactors(sensor, U1,U2, raw_vwc=False):
    if isPCBCalibration(sensor):
        if 'offset' in sensor.keys():
            offset = sensor['offset']
        else:
            offset = 0.0
            logging.warning(f"Offset for sensor {sensor['_id']}, depth sensor['depth'] is not present, defaulted to 0.0")
        diff = U1 - U2
        intersect = sensor['intersect']
        slopeA = sensor['slopeA']
        slopeB = sensor['slopeB']
        slopeC = sensor['slopeC']
        vwc = -(100.0 / 79.0) + (100.0 / 79.0) * (intersect + slopeA * diff + slopeB * diff ** 2.0 + slopeC * diff ** 3.0)
        return vwc + offset, None if not raw_vwc else vwc
    else:
        raise Exception(f"No calibration factors for sensor {sensor['_id']}")

def getNormalizationU1(sensor_measurements):
    for measurement in sensor_measurements:
        if measurement['medium']['description'] == normalizationMediumU1['description']:
            return measurement['U1Average']
    raise Exception(f"Missing measurement for normalization U1 {normalizationMediumU1['description']}")

def getNormalizationU2(sensor_measurements):
    for measurement in sensor_measurements:
        if measurement['medium']['description'] == normalizationMediumU2['description']:
            return measurement['U2Average']
    raise Exception(f"Missing measurement for normalization U2 {normalizationMediumU2['description']}")

def convertMeasurementToVWC(sensor, U1,U2, raw_vwc=False):
    diff = (U1 - getNormalizationU1(sensor['measurements'])) - (U2 - getNormalizationU2(sensor['measurements']))
    if 'curves' in sensor.keys():
        curves = sensor['curves']
    else:
        raise Exception(f"No calib curves for sensor {sensor}")
    
    hasOneCurve = len(curves) == 1

    offset = sensor['offset'] if 'offset' in sensor.keys() else 0.0

    # try with first curve
    firstCurve = curves[0]
    slope = firstCurve['slope']
    intersect = firstCurve['intersect']
    vwc = diff * slope + intersect
    if hasOneCurve or vwc <= firstCurve['mediumHigh']['vwc']:
        return vwc + offset, None if not raw_vwc else vwc
    
    # try with middle curves
    for curve in curves[1:len(curves)-1]:
        slope = curve['slope']
        intersect = curve['intersect']
        vwc = diff * slope + intersect
        if vwc > curve['mediumLow']['vwc'] and vwc <= curve['mediumHigh']['vwc']:
            return vwc + offset, None if not raw_vwc else vwc

    # try with last curves
    lastCurve = curves[-1]
    slope = lastCurve['slope']
    intersect = lastCurve['intersect']
    vwc = diff * slope + intersect
    if vwc > lastCurve['mediumLow']['vwc']:
        return vwc + offset, None if not raw_vwc else vwc

    raise Exception(f"Missing calibration curve for (U1-U2) = {diff}, VWC on last curve = {vwc}")

def convertMeasurementToEc(U3):
    mV = U3 * 1000
    polynomial = ecPolynomialA * mV + ecPolynomialB * (mV ** 2) + ecPolynomialC * (mV ** 3) + ecPolynomialD
    return ecFactor * (10.0 ** (0.1 / polynomial))

def convert_measurement(measurement, sensors, raw_vwc=False):
    computed_measurement = {
        "deviceContextId": measurement["dci"],
        "timestamp":measurement['timestamp'],
        "topToBottom":[]}
    sensors = { sensor['depth']:sensor for sensor in sensors }
    single_measurements = measurement['topToBottom']
    for single_measurement in single_measurements:
        # Compute VWC
        vwc = None
        vwc_raw = None
        try:
            corresponding_sensor = sensors[single_measurement['depth']]
            if 'value' in single_measurement['highCap'].keys() and 'value' in single_measurement['highRes'].keys():
                try:
                    if isPCBCalibration(corresponding_sensor):
                        vwc, vwc_raw = convertMeasurementToVWCWithFactors(corresponding_sensor, single_measurement['highCap']['value'], single_measurement['highRes']['value'], raw_vwc)
                    else:
                        vwc, vwc_raw = convertMeasurementToVWC(corresponding_sensor, single_measurement['highCap']['value'], single_measurement['highRes']['value'], raw_vwc)
                except Exception as e:
                    logging.error(str(e))
        except KeyError:
            vwc = resToVwc(single_measurement['highRes']['value'])
            vwc_raw = vwc

        # Compute EC
        ec = convertMeasurementToEc(single_measurement['lowCap']['value'])
        cm = {
            "depth": single_measurement['depth'],
            "temperature": single_measurement['temperature']['value'],
            "volumetricWaterContent": vwc,
            "electricalConductivity": ec
        }

        if vwc_raw is not None:
            cm["rawVolumetricWaterContent"]= vwc_raw

        computed_measurement['topToBottom'].append(cm)

    return computed_measurement


def get_computed_measurements(node_serial, startTime, endTime):
    measurements = get_measurements(node_serial,startTime,endTime)
    sensors = get_sensors(node_serial)
    computed_measurements = []
    for measurement in measurements:
        computed_measurements.append(convert_measurement(measurement, sensors))
    return computed_measurements
