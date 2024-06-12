from sklearn.model_selection import KFold
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
import netCDF4 as nc
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
start = 1
import csv
gro = "D:\pyCharm\python\helloworld\data\changchun_gro.nc"
sky = "D:\pyCharm\python\helloworld\data\changchun_sky.nc"

gro_data = nc.Dataset(gro)
sky_data = nc.Dataset(sky)

# print(sky_data.variables.keys())

new_keys500 = ['500h_' + key for key in sky_data.variables.keys()]
new_keys850 = ['850h_' + key for key in sky_data.variables.keys()]

# csv_head = list(gro_data.variables.keys())
csv_head = ['longitude', 'latitude', 'u100', 'v100', 'u10', 'v10', 'd2m', 't2m','ilspf', 'lsp', 'lspf',
            'lsrr', 'lsf', 'lssfr', 'mcpr', 'mer', 'msl', 'mtpr','sf', 'sp']
# csv_head = csv_head + new_keys500[4:] + new_keys850[4:]
sky_500 = ['500h_cc','500h_r','500h_q','500h_t','500h_v','500h_u']
sky_850 = ['850h_cc','850h_r','850h_q','850h_t','850h_v','850h_u']
# csv_head = csv_head + new_keys500[4:] + new_keys850[4:]
csv_head = csv_head + sky_500 + sky_850
# elements_to_add = ["Altitude","Precipitation","Air_Temperature","Vapour_Pressure","O18","H2"]
# elements_to_add = ["Altitude","O18","H2"]
# elements_to_add = ["Precipitation","Air_Temperature","Altitude","Vapour_Pressure","O18"]
elements_to_add = ["Precipitation","Altitude","Vapour_Pressure","O18"]
csv_head.extend(elements_to_add)
# print(csv_head[4:])
# print(csv_head)

col = csv_head
print(col)

baotou = pd.read_csv(r"baotou.csv")[col].values
changchun = pd.read_csv(r"changchun.csv")[col].values
changsha = pd.read_csv(r"changsha.csv")[col].values
chengdu = pd.read_csv(r"chengdu.csv")[col].values
chongqing = pd.read_csv(r"chongqing.csv")[col].values
fuzhou = pd.read_csv(r"fuzhou.csv")[col].values
guangzhou = pd.read_csv(r"guangzhou.csv")[col].values
guilin = pd.read_csv(r"guilin.csv")[col].values
guiyang = pd.read_csv(r"guiyang.csv")[col].values
haerbing = pd.read_csv(r"haerbing.csv")[col].values
haikou = pd.read_csv(r"haikou.csv")[col].values
hetian = pd.read_csv(r"hetian.csv")[col].values
jinbian = pd.read_csv(r"jinbian.csv")[col].values
jinzhou = pd.read_csv(r"jinzhou.csv")[col].values
kunming = pd.read_csv(r"kunming.csv")[col].values
lanzhou = pd.read_csv(r"lanzhou.csv")[col].values
LHASA = pd.read_csv(r"LHASA.csv")[col].values
liuzhou = pd.read_csv(r"liuzhou.csv")[col].values
nanjing = pd.read_csv(r"nanjing.csv")[col].values
pingliang = pd.read_csv(r"pingliang.csv")[col].values
qiqihaer = pd.read_csv(r"qiqihaer.csv")[col].values
shijiazhaung = pd.read_csv(r"shijiazhaung.csv")[col].values
taiyuan = pd.read_csv(r"taiyuan.csv")[col].values
tianjing = pd.read_csv(r"tianjing.csv")[col].values
wuhan = pd.read_csv(r"wuhan.csv")[col].values
wulumuqi = pd.read_csv(r"wulumuqi.csv")[col].values
xian = pd.read_csv(r"xian.csv")[col].values
xianggang = pd.read_csv(r"xianggang.csv")[col].values
yantai = pd.read_csv(r"yantai.csv")[col].values
yinchuan = pd.read_csv(r"yinchuan.csv")[col].values
zhangye = pd.read_csv(r"zhangye.csv")[col].values
zhengzhou = pd.read_csv(r"zhengzhou.csv")[col].values
zunyi = pd.read_csv(r"zunyi.csv")[col].values

AMDERMA = pd.read_csv(r"AMDERMA.csv")[col].values
ANTALYA = pd.read_csv(r"ANTALYA.csv")[col].values
ARKHANGELSK = pd.read_csv(r"ARKHANGELSK.csv")[col].values
PENDELI = pd.read_csv(r"ATHENS-PENDELI.csv")[col].values
THISSION = pd.read_csv(r"ATHENS-THISSION.csv")[col].values
BAHRAIN = pd.read_csv(r"BAHRAIN.csv")[col].values
BALTI = pd.read_csv(r"BALTI.csv")[col].values
BANGKOK = pd.read_csv(r"BANGKOK.csv")[col].values
BRAVICEA = pd.read_csv(r"BRAVICEA.csv")[col].values
CAHUL = pd.read_csv(r"CAHUL.csv")[col].values
CESTAS = pd.read_csv(r"CESTAS-PIERROTON.csv")[col].values
CHISINAU = pd.read_csv(r"CHISINAU.csv")[col].values
DHAKA = pd.read_csv(r"DHAKA (SAVAR).csv")[col].values
DIYARBAKIR = pd.read_csv(r"DIYARBAKIR.csv")[col].values
EDIRNE = pd.read_csv(r"EDIRNE.csv")[col].values
FARO = pd.read_csv(r"FARO.csv")[col].values
GIBRALTAR = pd.read_csv(r"GIBRALTAR.csv")[col].values
GOR = pd.read_csv(r"GOR.csv")[col].values
HOHENPEISSENBERG = pd.read_csv(r"HOHENPEISSENBERG.csv")[col].values
KALININ = pd.read_csv(r"KALININ.csv")[col].values
KANDALAKSA = pd.read_csv(r"KANDALAKSA.csv")[col].values
KARACHI = pd.read_csv(r"KARACHI.csv")[col].values
KARLSRUHE = pd.read_csv(r"KARLSRUHE.csv")[col].values
KHARKIV = pd.read_csv(r"KHARKIV.csv")[col].values
KIROV = pd.read_csv(r"KIROV.csv")[col].values
KONSTANZ = pd.read_csv(r"KONSTANZ.csv")[col].values
KOZHIKODE = pd.read_csv(r"KOZHIKODE (CALICUT).csv")[col].values
KRAKOW = pd.read_csv(r"KRAKOW.csv")[col].values
KUOPIO = pd.read_csv(r"KUOPIO.csv")[col].values
KURSK = pd.read_csv(r"KURSK.csv")[col].values
LEOVA = pd.read_csv(r"LEOVA.csv")[col].values
LIPTOVSKY = pd.read_csv(r"LIPTOVSKY MIKULAS-ONDRASOVA.csv")[col].values
LJUBLJANA = pd.read_csv(r"LJUBLJANA.csv")[col].values
MONACO = pd.read_csv(r"MONACO.csv")[col].values
MOSCOW = pd.read_csv(r"MOSCOW.csv")[col].values
MURMANSK = pd.read_csv(r"MURMANSK.csv")[col].values
ALESUND = pd.read_csv(r"NY ALESUND.csv")[col].values
ODENSE = pd.read_csv(r"ODENSE.csv")[col].values
ODESSA = pd.read_csv(r"ODESSA.csv")[col].values
PECHORA = pd.read_csv(r"PECHORA.csv")[col].values
PERM = pd.read_csv(r"PERM.csv")[col].values
DELGADA = pd.read_csv(r"PONTA DELGADA (AZO.csv")[col].values
PORTALEGRE = pd.read_csv(r"PORTALEGRE.csv")[col].values
RAMNICU = pd.read_csv(r"RAMNICU VALCEA.csv")[col].values
REYKJAVIK = pd.read_csv(r"REYKJAVIK.csv")[col].values
RIZE = pd.read_csv(r"RIZE.csv")[col].values
RJAZAN = pd.read_csv(r"RJAZAN.csv")[col].values
ROSTOV = pd.read_csv(r"ROSTOV-NA-DONU.csv")[col].values
ROVANIEMI = pd.read_csv(r"ROVANIEMI.csv")[col].values
RYORI = pd.read_csv(r"RYORI.csv")[col].values
SALEKHARD = pd.read_csv(r"SALEKHARD.csv")[col].values
SARATOV = pd.read_csv(r"SARATOV.csv")[col].values
SINGAPORE = pd.read_csv(r"SINGAPORE.csv")[col].values
SINOP = pd.read_csv(r"SINOP.csv")[col].values
ST_PETERSBURG = pd.read_csv(r"ST_PETERSBURG.csv")[col].values
STUTTGART = pd.read_csv(r"STUTTGART.csv")[col].values
TAMBOV = pd.read_csv(r"TAMBOV.csv")[col].values
TARTU = pd.read_csv(r"TARTU.csv")[col].values
TOKYO = pd.read_csv(r"TOKYO.csv")[col].values
VIENNA = pd.read_csv(r"VIENNA.csv")[col].values
WALLINGFORD = pd.read_csv(r"WALLINGFORD.csv")[col].values

data = np.concatenate((baotou,changchun,changsha,chengdu,chongqing,fuzhou,guangzhou,guilin,guiyang,haerbing,haikou,hetian,jinbian,jinzhou,kunming,lanzhou,LHASA,liuzhou,nanjing,pingliang,qiqihaer,shijiazhaung,taiyuan,tianjing,wuhan,wulumuqi,xian,xianggang,yantai,yinchuan,zhangye,zhengzhou,zunyi,
AMDERMA,ANTALYA,ARKHANGELSK,PENDELI,THISSION,BAHRAIN,BALTI,BANGKOK,BRAVICEA,CAHUL,CESTAS,CHISINAU,DHAKA,DIYARBAKIR,EDIRNE,FARO,
GIBRALTAR,GOR,HOHENPEISSENBERG,KALININ,KANDALAKSA,KARACHI,KARLSRUHE,KHARKIV,KIROV,KONSTANZ,KOZHIKODE,KRAKOW,KUOPIO,KURSK,
LEOVA,LIPTOVSKY,LJUBLJANA,MONACO,MOSCOW,MURMANSK,ALESUND,ODENSE,ODESSA,PECHORA,PERM,DELGADA,PORTALEGRE,RAMNICU,REYKJAVIK,
RIZE,RJAZAN,ROSTOV,ROVANIEMI,RYORI,SALEKHARD,SARATOV,SINGAPORE,SINOP,ST_PETERSBURG,STUTTGART,TAMBOV,TARTU,TOKYO,WALLINGFORD,VIENNA
), axis=0)
# print(data.shape)

data = data[~np.isnan(data[:, -1])]
data = data[~np.isnan(data[:, -2])]
data = data[~np.isnan(data[:, -4])]
data = data[~np.isnan(data[:, -3])]

x = data[:, :-1]
y = data[:, -1]
print(x.shape)
# print(x.shape)
X_train_1, X_validation_1, y_train, y_validation_1 = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_1)
X_validation_scaled = scaler_X.transform(X_validation_1)





# 初始化每个模型的预测结果数组
preds_cat = np.zeros_like(y_train)
preds_xgb = np.zeros_like(y_train)
preds_cat = preds_cat.reshape(-1,1)
preds_xgb = preds_xgb.reshape(-1,1)

test_cat = np.zeros_like(y_validation_1)
test_xgb = np.zeros_like(y_validation_1)
test_cat = test_cat.reshape(-1,1)
test_xgb = test_xgb.reshape(-1,1)

# print(y_train.shape)
# print(y_validation_1.shape)
# 定义模型   cat [0.33, 10.0, 7.0]    xgb [10, 0.2, 200]
models = {
    'cat': RandomForestRegressor(
        ),
    'xgb': xgb.XGBRegressor(
        )
}
# ElasticNet
# 设置 KFold 交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 对每个模型进行交叉验证
for model_name, model in models.items():
    # 遍历每个折叠
    for train_index, val_index in kf.split(X_train_scaled):
        # 分割数据
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # 训练模型
        model.fit(X_train_fold, y_train_fold)
        # 预测验证集
        val_preds = model.predict(X_val_fold)
        test_preds = model.predict(X_validation_scaled)

        val_preds = val_preds.reshape(-1,1)
        test_preds = test_preds.reshape(-1, 1)
        # print(train_index)

        # 保存预测结果
        if model_name == 'cat':
            preds_cat[val_index] = val_preds
            # test_cat = test_preds + test_cat
            test_cat = np.add(test_preds, test_cat)
            print("cat")
        elif model_name == 'xgb':
            preds_xgb[val_index] = val_preds
            # test_xgb = test_preds + test_xgb
            test_xgb = np.add(test_preds, test_xgb)
            print("xgb")


# 将预测结果按列拼接起来
combined_preds = np.concatenate((preds_cat,preds_xgb),axis=1)

test_cat = test_cat/5
test_xgb = test_xgb/5


# combined_test = np.concatenate((test_ridge,test_rf,test_svr),axis=1)
combined_test = np.concatenate((test_cat,test_xgb),axis=1)


ensemble = LinearRegression()

ensemble.fit(combined_preds,y_train)
ensemble_pre= ensemble.predict(combined_test[0:300])
print('ensemble_mae：{}'.format(metrics.mean_absolute_error(y_validation_1[0:300], ensemble_pre)))
print('ensemble_mse：{}'.format(metrics.mean_squared_error(y_validation_1[0:300], ensemble_pre)))
print('ensemble_rmse：{}'.format(np.sqrt(metrics.mean_squared_error(y_validation_1[0:300], ensemble_pre))))
print('ensemble_R2：{}'.format(metrics.r2_score(y_validation_1[0:300], ensemble_pre)))

if start:
    with open('./../ensembles.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['true', 'real'])  # 写入列名
        writer.writerows(zip(y_validation_1[0:300], ensemble_pre))  # 写入数据

    print("Data has been saved to data.csv.")
    # 指定CSV文件的名称
    # csv_file = './sky_predictions.csv'
    #
    # # 尝试读取现有的CSV文件
    # try:
    #     # 如果文件存在，读取现有的CSV文件到DataFrame
    #     df_predictions = pd.read_csv(csv_file)
    # except FileNotFoundError:
    #     # 如果文件不存在，创建一个空的DataFrame
    #     # 这里创建一个与 ensemble_pre 长度相同、但没有任何列的DataFrame
    #     df_predictions = pd.DataFrame(index=range(len(ensemble_pre)))
    #
    # # 生成新列的名称，这里使用 DataFrame 中的列数来生成唯一的列名
    # new_column_name = f'pre_900h_u'
    #
    # # 将新的预测结果作为新列添加到DataFrame
    # df_predictions[new_column_name] = ensemble_pre
    #
    # # 将更新后的DataFrame写入CSV文件
    # df_predictions.to_csv(csv_file, index=False)