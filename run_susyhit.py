#!/usr/bin/env python3
import numpy as np 
import pandas as pd 
import os, sys 
import pyslha
import math

pwd = os.path.abspath(os.path.dirname(__file__))

def calc(point):
    sinTW = math.sqrt(0.22290)
    cosTW = math.sqrt(1.0 - 0.22290)
    beta =  math.atan(point['TB'])
    sinBT = math.sin(beta)
    cosBT = math.cos(beta)
    sinAP = math.sin(point['Alpha'])
    cosAP = math.cos(point['Alpha'])
    Gamma0 = 1.12e-11 

    def kappa_Gamma(ii):
        Ni1 = point["N{}1".format(ii)]
        Ni2 = point["N{}2".format(ii)]
        return (Ni1 * cosTW + Ni2 * sinTW) ** 2
    
    def kappa_H(ii):
        Ni3 = point["N{}3".format(ii)]
        Ni4 = point["N{}4".format(ii)]
        return (Ni3 * sinAP - Ni4 * cosAP) ** 2
    
    def kappa_ZT(ii):
        Ni1 = point["N{}1".format(ii)]
        Ni2 = point["N{}2".format(ii)]
        return (Ni1 * sinTW - Ni2 * cosTW) ** 2

    def kappa_ZL(ii):
        Ni3 = point["N{}3".format(ii)]
        Ni4 = point["N{}4".format(ii)]
        return (Ni3 * cosBT - Ni4 * sinBT) ** 2
    
    def kappa_WT(jj):
        Vi1 = point["V{}1".format(jj)]
        Ui1 = point["U{}1".format(jj)]
        return 0.5 * (Vi1 ** 2 + Ui1 ** 2)
    
    def kappa_WL(jj):
        Vi2 = point["V{}2".format(jj)]
        Ui2 = point["U{}2".format(jj)]
        return Vi2 ** 2 * sinBT ** 2 + Ui2 ** 2 * cosBT ** 2 

    res = {
        "WN1Gamma": kappa_Gamma(1) * (abs(point['mN1']) / 100. ) ** 5 * Gamma0, 
        "WN2Gamma": kappa_Gamma(2) * (abs(point['mN2']) / 100. ) ** 5 * Gamma0, 
        "WN3Gamma": kappa_Gamma(3) * (abs(point['mN3']) / 100. ) ** 5 * Gamma0, 
        "WN4Gamma": kappa_Gamma(4) * (abs(point['mN4']) / 100. ) ** 5 * Gamma0, 
        "WN1H":     0.5 * kappa_H(1) * (abs(point['mN1']) / 100. ) ** 5 * (1 - 125.0/abs(point['mN1']))**4 * Gamma0, 
        "WN2H":     0.5 * kappa_H(2) * (abs(point['mN2']) / 100. ) ** 5 * (1 - 125.0/abs(point['mN2']))**4 * Gamma0, 
        "WN3H":     0.5 * kappa_H(3) * (abs(point['mN3']) / 100. ) ** 5 * (1 - 125.0/abs(point['mN3']))**4 * Gamma0, 
        "WN4H":     0.5 * kappa_H(4) * (abs(point['mN4']) / 100. ) ** 5 * (1 - 125.0/abs(point['mN4']))**4 * Gamma0, 
        "WN1Z":     (kappa_ZT(1) + 0.5 * kappa_ZL(1)) * (abs(point['mN1']) / 100. ) ** 5 * (1 - 91.20/abs(point['mN1']))**4 * Gamma0, 
        "WN2Z":     (kappa_ZT(2) + 0.5 * kappa_ZL(2)) * (abs(point['mN2']) / 100. ) ** 5 * (1 - 91.20/abs(point['mN2']))**4 * Gamma0, 
        "WN3Z":     (kappa_ZT(3) + 0.5 * kappa_ZL(3)) * (abs(point['mN3']) / 100. ) ** 5 * (1 - 91.20/abs(point['mN3']))**4 * Gamma0, 
        "WN4Z":     (kappa_ZT(4) + 0.5 * kappa_ZL(4)) * (abs(point['mN4']) / 100. ) ** 5 * (1 - 91.20/abs(point['mN4']))**4 * Gamma0, 
        "WC1W":     (kappa_WT(1) + 0.5 * kappa_WL(1)) * (abs(point['mC1']) / 100. ) ** 5 * (1 - 80.37/abs(point['mN1']))**4 * Gamma0, 
        "WC2W":     (kappa_WT(2) + 0.5 * kappa_WL(2)) * (abs(point['mC2']) / 100. ) ** 5 * (1 - 80.37/abs(point['mN2']))**4 * Gamma0
    }
    return res 

def calculate(row):
    # print(row)
    point = {
        "M1":   row['M1'],
        "M2":   row['M2'],
        "Mu":   row['mu'],
        "TB":   row['TanBeta'],
        "ID":   int(row['index']),
        "LogL": row['Loglike']
    }
    shp = os.path.join(pwd, "External")
    tem = os.path.join(pwd, "suspect2_lha.in.template")
    ipf = os.path.join(pwd, "External/suspect2_lha.in")
    # print(ipf)
    opf = os.path.join(pwd, "External/susyhit_slha.out")
    op1 = os.path.join(pwd, "External/suspect2.out")
    op2 = os.path.join(pwd, "External/slhaspectrum.in")
    with open(tem, 'r') as f1:
        with open(ipf, "w") as f2:
            inp = f1.read()
            inp = inp.replace(">>>M1<<<", "{}".format(float(point['M1']))) 
            inp = inp.replace(">>>M2<<<", "{}".format(float(point['M2']))) 
            inp = inp.replace(">>>MU<<<", "{}".format(float(point['Mu']))) 
            inp = inp.replace(">>>TB<<<", "{}".format(float(point['TB']))) 
            f2.write(inp)

    # os.system("rm {}".format(opf))
    os.system("rm {}".format(op1))
    os.system("rm {}".format(op2))

    os.chdir(shp)
    os.system("./run > log")
    os.system("mv {} {}".format(opf, os.path.join(pwd, "spectr/{}.dat".format(point['ID']))))
    spe = pyslha.read(os.path.join(pwd, "spectr/{}.dat".format(point['ID'])))

    point.update({
        "mN1":  spe.blocks['MASS'][1000022],
        "mN2":  spe.blocks['MASS'][1000023],
        "mC1":  spe.blocks['MASS'][1000024],
        "mN3":  spe.blocks['MASS'][1000025],
        "mN4":  spe.blocks['MASS'][1000035],
        "mC2":  spe.blocks['MASS'][1000037],
        "N11":  spe.blocks['NMIX'][1,1],
        "N12":  spe.blocks['NMIX'][1,2],
        "N13":  spe.blocks['NMIX'][1,3],
        "N14":  spe.blocks['NMIX'][1,4],
        "N21":  spe.blocks['NMIX'][2,1],
        "N22":  spe.blocks['NMIX'][2,2],
        "N23":  spe.blocks['NMIX'][2,3],
        "N24":  spe.blocks['NMIX'][2,4],
        "N31":  spe.blocks['NMIX'][3,1],
        "N32":  spe.blocks['NMIX'][3,2],
        "N33":  spe.blocks['NMIX'][3,3],
        "N34":  spe.blocks['NMIX'][3,4],
        "N41":  spe.blocks['NMIX'][4,1],
        "N42":  spe.blocks['NMIX'][4,2],
        "N43":  spe.blocks['NMIX'][4,3],
        "N44":  spe.blocks['NMIX'][4,4],
        "U11":  spe.blocks['UMIX'][1,1],
        "U12":  spe.blocks['UMIX'][1,2],
        "U21":  spe.blocks['UMIX'][2,1],
        "U22":  spe.blocks['UMIX'][2,2],
        "V11":  spe.blocks['VMIX'][1,1],
        "V12":  spe.blocks['VMIX'][1,2],
        "V21":  spe.blocks['VMIX'][2,1],
        "V22":  spe.blocks['VMIX'][2,2],    
        "WN2":  spe.decays[1000023].totalwidth,
        "WC1":  spe.decays[1000024].totalwidth,
        "WN3":  spe.decays[1000025].totalwidth,
        "WN4":  spe.decays[1000035].totalwidth,
        "WC2":  spe.decays[1000037].totalwidth,
        "Alpha":    spe.blocks['ALPHA'][None]
        })
    point.update(calc(point))

    spe = pyslha.read(op2)
    point.update({
        "SUFT01":   spe.blocks["SU_FINETUNE"][1],
        "SUFT02":   spe.blocks["SU_FINETUNE"][2],
        "SUFT03":   spe.blocks["SU_FINETUNE"][3],
        "SUFT04":   spe.blocks["SU_FINETUNE"][4]
    })
    # print(pd.Series(point))
    return point
    




if __name__ == "__main__":
    pdata = pd.read_csv("param.csv")
    sams = []
    for index, row in pdata.iterrows():
        if index % 100 == 0:
            print("Calculating the mass spectrum of ID ->\t{} ".format(index))
        info = calculate(dict(row))
        # print(pd.Series(info))
        sams.append(info)
        # # break
        # if index == 100:
        #     break
        
    df = pd.DataFrame(sams)
    print(df.shape)
    os.chdir(pwd)
    df.to_csv("tot_data_1107.csv")
 