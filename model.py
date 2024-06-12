# from cuml.ensemble import RandomForestRegressor
# from cuml.linear_model import Ridge
# from cuml.linear_model import LinearRegression
# from cuml.linear_model import Lasso
import sys
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import copy
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
import netCDF4 as nc
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
from sklearn.model_selection import cross_val_score

X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
# X_train_scaled = scaler_X.fit_transform(X_train)
X_train_scaled = scaler_X.fit_transform(X_train)
X_validation_scaled = scaler_X.transform(X_validation)

def f(points_max,points_rate,points_n):
    result = []
    for i in range(len(points_max)):
        print("第" + str(i + 1) + "个粒子")
        max = points_max[i][0]
        rate = points_rate[i][0]
        n = points_n[i][0]
        rmse = master(max,rate,n)
        result.append(rmse)
    return result



def master(points_max,points_rate,points_n):
    # with config_context(target_offload = "gpu:0"):
    # xgb_model = xgb.XGBRegressor(
    #     max_depth=int(points_max),
    #     learning_rate=points_rate,
    #     n_estimators=int(points_n),
    #     objective='reg:squarederror',
    #     booster='gbtree',
    #     random_state=0)
    xgb_model = RandomForestRegressor(
        n_estimators=int(points_max), max_depth=int(points_rate), min_samples_leaf=int(points_n))


    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_validation_scaled)
    print('mse：{}'.format(metrics.mean_squared_error(y_validation, y_pred)))
    print('mae：{}'.format(metrics.mean_absolute_error(y_validation, y_pred)))
    print('R2：{}'.format(metrics.r2_score(y_validation, y_pred)))
    print(f"rmse：{np.sqrt(mean_squared_error(y_validation, y_pred))}")
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # errors = abs(y_validation_pred - y_validation)
    # mape = errors / y_validation
    # print(f'MAPE:{mape.mean():.2%}')
    return metrics.mean_squared_error(y_validation, y_pred)

#先实现普通的粒子群算法
def PSO_org(init,demensions,iters,tol,limit,vlimit,w,c1,c2,f):
    '''
    print(PSO_org(20,1,50,0.1,[0,50],[-10,10],0.8,0.5,0.5,f))

    粒子群搜索求解(求最小值)
    :param init:初始化多少只鸟
    :param demensions: 数据维度
    :param iters: 最大迭代轮数
    :param tol: 最小差值（没有使用）
    :param limit: 位置限制
    :param vlimit:速度限制
    :param w:惯性权重
    :param c1:自我学习因子
    :param c2:总体学习因子
    :param f:适应度函数
    :return:最小值点
    '''
    # 模拟退火的初始值
    T0 = 100  # 初始温度
    T = T0  # 迭代中温度会发生改变，第一次迭代时温度就是T0
    Lk = 2 # 每个温度下的迭代次数
    alfa = 0.8 # 温度衰减系数
    #生成N行d列的随机点
    points_max=np.random.uniform(limit[0],limit[1],[init,demensions])
    points_rate=np.random.uniform(limit[2],limit[3],[init,demensions])
    points_n=np.random.uniform(limit[4],limit[5],[init,demensions])
    #初始各个随机点的速度
    v_max=np.random.rand(init,demensions)
    v_rate=0.2 * np.random.rand(init,demensions)
    v_n=np.random.rand(init,demensions)
    #每个个体的最佳位置
    indiviual_best_max=copy.deepcopy(points_max)
    indiviual_best_rate=copy.deepcopy(points_rate)
    indiviual_best_n=copy.deepcopy(points_n)
    #整个种群的历史最佳位置
    tot_best_max=np.zeros(demensions)
    tot_best_rate=np.zeros(demensions)
    tot_best_n=np.zeros(demensions)
    #每个个体的历史最佳适应度
    f_indiviual_best=np.array([float('inf') for i in range(init)])
    #种群的历史最佳适应度
    f_tot_best=float('inf')

    #迭代更新
    for i in tqdm(range(iters)):
        print("第" + str(i + 1) + "代")
        for i in range(Lk):  # 每个温度下的迭代次数
            print("温度下第" + str(i + 1) + "次迭代次数")
            # 位置更新
            new_points_max = points_max + v_max
            new_points_rate = points_rate + v_rate
            new_points_n = points_n + v_n
            #边界位置处理
            new_points_max[new_points_max>limit[1]]=limit[1]
            new_points_max[new_points_max<limit[0]]=limit[0]
            new_points_rate[new_points_rate > limit[3]] = limit[3]
            new_points_rate[new_points_rate < limit[2]] = limit[2]
            new_points_n[new_points_n > limit[5]] = limit[5]
            new_points_n[new_points_n < limit[4]] = limit[4]
            # 计算个体适应度
            new_points_max = np.round(new_points_max)
            # new_points_rate = np.round(new_points_rate,2)
            new_points_rate = np.round(new_points_rate)
            new_points_n = np.round(new_points_n)
            fx=f(new_points_max,new_points_rate,new_points_n) #个体的适应度
            #更新个体历史适应度
            for i in tqdm(range(init)):
                if fx[i]<f_indiviual_best[i]:
                    f_indiviual_best[i]=fx[i]
                    indiviual_best_max[i]=new_points_max[i]
                    indiviual_best_rate[i]=new_points_rate[i]
                    indiviual_best_n[i]=new_points_n[i]
                else:
                    # 以一定概率接受更差的解
                    p = np.exp(-(fx[i] - f_indiviual_best[i]) / T)
                    r = np.random.uniform(0, 1)
                    # 接受
                    if r < p:
                        new_points_max[i] = new_points_max[i]
                        new_points_rate[i] = new_points_rate[i]
                        new_points_n[i] = new_points_n[i]
                    # 不接受
                    else:
                        new_points_max[i] = points_max[i]
                        new_points_rate[i] = points_rate[i]
                        new_points_n[i] = points_n[i]
                #更新种群历史适应度最小值
                if min(f_indiviual_best)<f_tot_best:
                    f_tot_best=min(f_indiviual_best)
                    index=np.argmin(f_indiviual_best)
                    tot_best_max=indiviual_best_max[index]
                    tot_best_rate=indiviual_best_rate[index]
                    tot_best_n=indiviual_best_n[index]

            points_max = new_points_max
            points_rate = new_points_rate
            points_n = new_points_n
            #更新速度
            v_max=v_max*w+c1*np.random.rand(1)*(indiviual_best_max-points_max)+np.random.rand(1)*c2*(tot_best_max-points_max)
            # v_rate=v_rate*w+c1 * 0.2*np.random.rand(1)*(indiviual_best_rate-points_rate)+np.random.rand(1)*c2 * 0.2*(tot_best_rate-points_rate)
            v_rate=v_rate*w+c1 * 0.2*np.random.rand(1)*(indiviual_best_rate-points_rate)+np.random.rand(1)*c2 * 0.2*(tot_best_rate-points_rate)
            v_n=v_n*w+c1 * 0.2*np.random.rand(1)*(indiviual_best_n-points_n)+np.random.rand(1) *0.2* c2*(tot_best_n-points_n)
            #边界速度处理，超速的和太慢的都要修改
            v_max[v_max>vlimit[1]]=vlimit[1]
            v_max[v_max<vlimit[0]]=vlimit[0]
            v_rate[v_rate > vlimit[3]] = vlimit[3]
            v_rate[v_rate < vlimit[2]] = vlimit[2]
            v_n[v_n > vlimit[5]] = vlimit[5]
            v_n[v_n < vlimit[4]] = vlimit[4]

            #记录当前的适应度最小值
        record.append(f_tot_best)
        T = alfa * T
    #     plt.clf()
    #     plt.plot(scale, f(scale))
    #     plt.scatter(points, f(points),c='r')
    #     plt.pause(0.1)
    # plt.ioff()
    # plt.show()
    #print(record)
        c1 = 2.5 - i*(2.5-1.25)/20
        c2 = 1.25 + i*(2.5-1.25)/20
        w*=(1.35/2) + (sigmoid((-4+8*(20-i)/20))*2 - 1)*(0.55/2)
    tot_best = [tot_best_max,tot_best_rate,tot_best_n]
    return tot_best

if __name__=='__main__':
    record = []
    # plt.figure(1)
    # plt.plot(scale,f(scale))
    # plt.show()
    init = 30
    iters = 50
    lb = np.array([3, 0.01, 100])
    ub = np.array([10.0, 0.2, 500])
    result =  PSO_org(init, 1, iters, 0.1, [10, 150,1, 10, 1, 10], [-1, 1, -1, 1, -1, 1], 0.8, 0.5, 0.5, f)
    # result =  PSO_org(init, 1, iters, 0.1, [3, 10,0.01, 0.2, 100, 500], [-1, 1, -0.01, 0.01, -1, 1], 0.8, 0.5, 0.5, f)
    iterations = np.arange(0, iters, 1)  # 以0.1为单位，生成0到6的数据
    max = int(result[0])
    rate = float(result[1])
    n = float(result[2])
    result = [max, rate, n]
    print(result)
    with open('model_para.txt', 'w') as f:
        f.write(str(result))