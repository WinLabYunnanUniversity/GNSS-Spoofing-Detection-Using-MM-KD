import json, os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def reSave_ds0():
    ds_num = 0
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\0\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(Observation.shape[0]):
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def readerJson():
    ds_num = 0
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\0\ds%s' % ds_num
    with open(dp + r'\0001.json', 'r') as f:
        obsC = json.load(f)
    obsC = np.array(obsC)
    print(obsC.shape)

def reSave_ds2():
    #  1:1101, 1102:4100
    ds_num = 2
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\1\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(1101,Observation.shape[0]):
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def reSave_ds3():
    # ds3 1:1197, 1198:4049
    ds_num = 3
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\1\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(1197,Observation.shape[0]): # Observation.shape[0]
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def reSave_ds4():
    # ds4 1:1139, 1140:4040
    ds_num = 4
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\1\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(1139,Observation.shape[0]): # Observation.shape[0]
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def reSave_ds5():
    # ds5 normal 1:454, abnormal: 1023:4049
    ds_num = 5
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\1\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(1022,Observation.shape[0]): # Observation.shape[0]
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def reSave_ds6():
    # ds6 normal 1:450, abnormal: 1044:4051
    ds_num = 6
    dp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs'
    sp = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\0\ds%s' % ds_num
    with open(dp + r'\obs_ds%s.json' % ds_num, 'r') as f:
        obsC = json.load(f)

    CN0 = np.array(obsC['CN0fromSNR'])
    MCN0 = np.array(obsC['meanCN0fromSNR'])
    VCN0 = np.array(obsC['varianceCNOfromSNR'])
    dop = np.array(obsC['doppler'])
    ranged = np.array(obsC['range'])
    rangeR = np.array(obsC['rangeResid'])

    minL = min([len(CN0), len(MCN0), len(VCN0), len(dop), len(ranged), len(rangeR)])
    shape = (minL, 32, 6)
    Observation = np.empty(shape)
    for i in range(minL):
        Observation[i, :, 0] = CN0[i, :]
        Observation[i, :, 1] = MCN0[i, :]
        Observation[i, :, 2] = VCN0[i, :]
        Observation[i, :, 3] = dop[i, :]
        Observation[i, :, 4] = ranged[i, :]
        Observation[i, :, 5] = rangeR[i, :]
    print(Observation.shape)

    for i in range(450): # Observation.shape[0]
        iObs = Observation[i]  # (32, 6)
        if i + 1 < 10:
            saveP = sp + r'\000%s.json' % (i + 1)
        elif i + 1 >= 10 and i + 1 < 100:
            saveP = sp + r'\00%s.json' % (i + 1)
        elif i + 1 >= 100 and i + 1 < 1000:
            saveP = sp + r'\0%s.json' % (i + 1)
        else:
            saveP = sp + r'\%s.json' % (i + 1)

        idx = saveP[-9:-5]

        # save_list = ['ds','idx','data']
        # save_data = ['ds2',idx,list(iObs)]
        # save_dict = dict(zip(save_list, save_data))
        # new_data = json.loads(str(save_dict).replace("'", "\""))

        with open(saveP, 'w', encoding='utf8') as f2:
            json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def reSave_ubx():
    sp = r'F:\GNSS_SD_MMKD\dataset\GNSS\obs\0'
    dataP = r'F:\GNSS_SD_MMKD\dataset\GNSS\obs_normal.json'
    abdataP = r'F:\GNSS_SD_MMKD\dataset\GNSS\obs_abnormal.json'

    with open(dataP, 'r') as f:
        obsC = json.load(f)
    CN01 = np.array(obsC['CN0fromSNR'])
    MCN01 = np.array(obsC['meanCN0fromSNR'])
    VCN01 = np.array(obsC['varianceCNOfromSNR'])
    dop1 = np.array(obsC['doppler'])
    ranged1 = np.array(obsC['range'])
    rangeR1 = np.array(obsC['rangeResid'])

    with open(dataP, 'r') as f2:
        abobsC = json.load(f2)
    CN02 = np.array(abobsC['CN0fromSNR'])
    MCN02 = np.array(abobsC['meanCN0fromSNR'])
    VCN02 = np.array(abobsC['varianceCNOfromSNR'])
    dop2 = np.array(abobsC['doppler'])
    ranged2 = np.array(abobsC['range'])
    rangeR2 = np.array(abobsC['rangeResid'])

    CN0 = np.vstack((CN01,CN02))
    MCN0 = np.vstack((MCN01,MCN02))
    VCN0 = np.vstack((VCN01,VCN02))
    dop = np.vstack((dop1,dop2))
    ranged = np.vstack((ranged1,ranged2))
    rangeR = np.vstack((rangeR1,rangeR2))

    x_min = np.nanmin()
    #
    #
    # minL = len(ts)
    # shape = (minL, 32, 6)
    # Observation = np.empty(shape)
    #
    # for d in days:
    #     for h in hours:
    #         idp = fileName + r'\%s\observation%s.json'%(d,h)
    #         with open(idp, 'r') as f:
    #             obsC = json.load(f)
    #
    #         CN0 = np.array(obsC['CN0fromSNR'])
    #         MCN0 = np.array(obsC['meanCN0fromSNR'])
    #         VCN0 = np.array(obsC['varianceCNOfromSNR'])
    #         dop = np.array(obsC['doppler'])
    #         ranged = np.array(obsC['range'])
    #         rangeR = np.array(obsC['rangeResid'])
    #
    #         i=0
    #         Observation[i, :, 0] = CN0[i, :]
    #         Observation[i, :, 1] = MCN0[i, :]
    #         Observation[i, :, 2] = VCN0[i, :]
    #         Observation[i, :, 3] = dop[i, :]
    #         Observation[i, :, 4] = ranged[i, :]
    #         Observation[i, :, 5] = rangeR[i, :]
    # print(Observation.shape)  # (258727, 32, 6)
    #
    # for i in range(Observation.shape[0]):
    #     iObs = Observation[i]  # (32, 6)
    #     if i + 1 < 10:
    #         saveP = sp + r'\000%s.json' % (i + 1)
    #     elif i + 1 >= 10 and i + 1 < 100:
    #         saveP = sp + r'\00%s.json' % (i + 1)
    #     elif i + 1 >= 100 and i + 1 < 1000:
    #         saveP = sp + r'\0%s.json' % (i + 1)
    #     else:
    #         saveP = sp + r'\%s.json' % (i + 1)
    #
    #     idx = saveP[-9:-5]
    #
    #     with open(saveP, 'w', encoding='utf8') as f2:
    #         json.dump(iObs.tolist(), f2, ensure_ascii=False, indent=2)

def getObsHeatmap():
    obsPath = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\obs\1\ds6'
    saveHeatPath = r'F:\workspace\MultiModal_GNSSAD\TEXBAT\dataset\heatmap'
    with open(r'F:\workspace\MultiModal_GNSSAD\TEXBAT\obs\obs_normalizer.json', 'r') as f0:
        normalizer = json.load(f0)
    allObsPath = []
    for root, dirs, files in os.walk(obsPath):
        for file in files:
            file_path = os.path.join(root, file)
            allObsPath.append(file_path)

            f_name = file_path
            with open(file_path, 'r') as f1:
                obs = json.load(f1)
            obs = np.array(obs)

            normalizer_min = np.array(normalizer['min'])
            normalizer_max = np.array(normalizer['max'])

            nor_obs = (obs - normalizer_min) / (normalizer_max - normalizer_min)
            nor_obs = np.nan_to_num(nor_obs)

            sp = os.path.join(saveHeatPath + os.path.split(f_name)[0][-6:], os.path.split(f_name)[1][-9:-5])+'.jpg'

            plt.subplots(figsize=(4, 4))
            sns.heatmap(nor_obs, vmin=0, vmax=1, cmap='PuRd', cbar=False)
            plt.xticks([]), plt.yticks([])
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
            plt.savefig(sp, bbox_inches='tight')
            plt.close()
            print(sp)

# reSave_ubx()
getObsHeatmap()