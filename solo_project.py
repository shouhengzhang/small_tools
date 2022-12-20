import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
import datetime


plt.rcParams['figure.dpi'] = 150
np.set_printoptions(precision=4, suppress=True)
pd.options.display.float_format = '{:.4f}'.format

import yfinance as yf
import pandas_datareader as pdr
import requests_cache
session = requests_cache.CachedSession()

# performance measures
def annual_return(r):
    years = (r.index[-1] - r.index[0]).days / 365.25
    return (1 + r).product() ** (1/years) - 1


def total_return(r):
    return (1 + r).prod() - 1


def annual_volatility(r):
    return np.sqrt(252) * r.std()


def sharpe_ratio(r):
    return np.sqrt(252) * r.mean() / r.std()


# drawdown is value today relative to previous max value
def drawdown(r):
    value = (1 + r).cumprod()
    return value / value.cummax() - 1


def max_drawdown(r):
    return drawdown(r).min()


def calmar_ratio(r):
    return annual_return(r) / np.abs(max_drawdown(r))


# def stability(r):
#     df = pd.DataFrame({'cumlogr': np.log(1 + r).cumsum(), 'time': np.arange(cumlogr.shape[0])})
#     from statsmodels.formula.api import ols
#     mod = ols(formula = 'cumlogr ~ 1 + time', data = df).fit()
#     return mod.rsquared


def sortino_ratio(r):
    negr = r.loc[r < 0]
    return np.sqrt(252) * r.mean() / negr.std()


brk = yf.download(tickers = 'BRK-A', session = session)

brk_monthly = brk.resample('M').last()
brk_monthly['ret'] = brk_monthly['Adj Close'].pct_change()
brk_monthly

ff_all = pdr.get_data_famafrench('F-F_Research_Data_Factors', start = '1980', session = session)

ff = ff_all[0] / 100
ff.index = ff.index.to_timestamp(freq = 'M')
ff

portfolios_all = pdr.get_data_famafrench('Portfolios_Formed_on_BE-ME', start = '1980', session = session)

portfolios = portfolios_all[0] / 100
portfolios.index = portfolios.index.to_timestamp(freq = 'M')
portfolios.drop(columns = portfolios.columns[np.arange(9)], inplace = True)

df = brk_monthly.join([ff, portfolios], how = 'inner')

df

BEME_list = portfolios.columns.to_list()
dfr = df[BEME_list]
dfrf = df['RF']

dfr_b07 = dfr[dfr.index <= datetime.datetime(2007, 12, 31)]
dfr_a07 = dfr[dfr.index >= datetime.datetime(2008, 1, 1)]
dfrf_b07 = dfrf[dfrf.index <= datetime.datetime(2007, 12, 31)]
dfrf_a07 = dfrf[dfrf.index >= datetime.datetime(2008, 1, 1)]
brk_monthly_b07 = brk_monthly[brk_monthly.index <= datetime.datetime(2007, 12, 31)]
brk_monthly_a07 = brk_monthly[brk_monthly.index >= datetime.datetime(2008, 1, 1)]


def get_portf_rtn(w, avg_rtns):
    return np.sum(avg_rtns * w)

def get_portf_vol(w, cov_mat):
    return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

def get_portf_sr(w, avg_rtns, cov_mat):
    return get_portf_rtn(w, avg_rtns) / get_portf_vol(w, cov_mat)


def add_rf_portfolio(dfr_input, dfrf_input, brk_input, mv_weight):
    replication_returns = []
    l_w_p = []
    l_w_rf = []
    for i in range(0, 2001):
        w_p = i * 0.001
        w_rf = 1 - w_p
        portfolio_return = dfr_input.dot(mv_weight)
        df_rf_portfolio = pd.DataFrame()
        df_rf_portfolio['portfolio'] = portfolio_return
        df_rf_portfolio['rf'] = dfrf_input
        rf_portfolio_weight = np.asarray([w_p, w_rf])
        rf_portfolio_return = df_rf_portfolio.dot(rf_portfolio_weight)
        this_replication_return = total_return(rf_portfolio_return)
        replication_returns.append(this_replication_return)

        l_w_p.append(w_p)
        l_w_rf.append(w_rf)
    df_replication = pd.DataFrame()
    df_replication['w_p'] = l_w_p
    df_replication['w_rf'] = l_w_rf
    df_replication['replication'] = replication_returns
    df_replication['BRK'] = total_return(brk_input['ret'])
    df_replication['diff'] = abs(df_replication['replication'] - df_replication['BRK'])
    return df_replication


def replication(dfr_input, dfrf_input, brk_input):

    avg_returns = 12 * dfr_input.mean()
    cov_mat = 12 * dfr_input.cov()
    initial_weight = np.ones(dfr_input.shape[1]) / dfr_input.shape[1]  # equal weights

    target_returns = np.linspace(avg_returns.min(), avg_returns.max(), 50)
    df = pd.DataFrame(columns=['return', 'volatility'], index=np.arange(target_returns.shape[0]))

    i = 0
    all_weights = np.zeros((len(target_returns), len(dfr.columns)))
    for target_return in target_returns:
        res = sco.minimize(
            fun=get_portf_vol,  # function we want to minimize the output of
            x0=initial_weight,  # first set of portfolio weights
            args=(cov_mat),  # additional arguments for "fun"
            bounds=tuple((0,1) for i in avg_returns),  # bounds for each portfolio weight
            constraints=(
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # we want this lambda function to evaluate to 0
                # {'type': 'eq', 'fun': lambda x: np.sum(x[0:10]) - x[-1] - 1},
                {'type': 'eq', 'fun': lambda x: get_portf_rtn(x, avg_returns) - target_return}
            # we want this lambda function to evaluate to 0
            )
        )
        assert res['success']

        df['return'].iloc[i] = get_portf_rtn(res['x'], avg_returns)
        df['volatility'].iloc[i] = res['fun']
        all_weights[i, :] = res['x']
        i += 1

    df['return'] = df['return'].astype(float)
    df['volatility'] = df['volatility'].astype(float)

    # min variance portfolio weight
    mv_loc = df['volatility'].idxmin()
    mv_weight = all_weights[mv_loc, :]
    df['volatility'].min()

    # test only
    mv_loc = 34
    mv_weight = all_weights[mv_loc, :]

    df_replication = add_rf_portfolio(dfr_input, dfrf_input, brk_input, mv_weight)
    while df_replication['replication'].max() < df_replication['BRK'].mean():
        mv_loc += 1
        print(mv_loc)
        mv_weight = all_weights[mv_loc, :]
        df_replication = add_rf_portfolio(dfr_input, dfrf_input, brk_input, mv_weight)

    best_w_p = df_replication['w_p'][df_replication['diff'].idxmin()]
    best_w_rf = 1 - best_w_p

    return best_w_p, best_w_rf, mv_weight


best_w_p_b07, best_w_rf_b07, mv_weight_b07 = replication(dfr_b07, dfrf_b07, brk_monthly_b07)
best_w_p_a07, best_w_rf_a07, mv_weight_a07 = replication(dfr_a07, dfrf_a07, brk_monthly_a07)

portfolio_b07_return = dfr_b07.dot(mv_weight_b07)
portfolio_a07_return = dfr_a07.dot(mv_weight_a07)

df_rf_portfolio_b07 = pd.DataFrame()
df_rf_portfolio_b07['portfolio'] = portfolio_b07_return
df_rf_portfolio_b07['rf'] = dfrf_b07
rf_portfolio_weight_b07 = np.asarray([best_w_p_b07, best_w_rf_b07])
rf_portfolio_b07_return = df_rf_portfolio_b07.dot(rf_portfolio_weight_b07)

df_rf_portfolio_a07 = pd.DataFrame()
df_rf_portfolio_a07['portfolio'] = portfolio_a07_return
df_rf_portfolio_a07['rf'] = dfrf_a07
rf_portfolio_weight_a07 = np.asarray([best_w_p_a07, best_w_rf_a07])
rf_portfolio_a07_return = df_rf_portfolio_a07.dot(rf_portfolio_weight_a07)

def plot_cumulative_return(r):
    (1 + r).cumprod().plot()
    plt.title('Cumulative Return on $1 Investment')
    plt.ylabel('Cumulative Return ($)')
    plt.axhline(1, color='k', linestyle='--')
    plt.show()
    return None

def plot_rolling_sharpe_ratio(r, n=126):
    (np.sqrt(252) * r.rolling(n).mean() / r.rolling(n).std()).plot()
    plt.title('Rolling Sharpe Ratio (' + str(n) + ' Trading Day)')
    plt.ylabel('Sharpe Ratio')
    plt.axhline(sharpe_ratio(r), color='k', linestyle='--')
    plt.legend(['Rolling', 'Mean'])
    plt.show()
    return None

def plot_underwater(r):
    (100 * drawdown(r)).plot()
    plt.title('Underwater Plot')
    plt.ylabel('Drawdown (%)')
    plt.show()
    return None

def tear_sheet(r, plots=True):
    dic = {
        'Annual Return': annual_return(r),
        # 'Total Return': total_return(r),
        # 'Annual Volatility': annual_volatility(r),
        'Sharpe Ratio': sharpe_ratio(r),
        'Calmar Ratio': calmar_ratio(r),
        # 'Stability': stability(r),
        'Max Drawdown': max_drawdown(r),
        # 'Omega Ratio': omega_ratio(r),
        'Sortino Ratio': sortino_ratio(r),
        # 'Skew': skew(r),
        # 'Kurtosis': kurtosis(r),
        # 'Tail Ratio': tail_ratio(r),
        # 'Daily Value at Risk': daily_value_at_risk(r)
    }
    df = pd.DataFrame(data=dic.values(), columns = ['Backtest'], index=dic.keys())
    # display(df)
    if plots:
        plot_cumulative_return(r)
        plot_underwater(r)
    return None

tear_sheet(portfolio_b07_return)
tear_sheet(brk_monthly_b07['ret'])
tear_sheet(portfolio_a07_return)
tear_sheet(brk_monthly_a07['ret'])


# BRK 5 year rolling window
annual_return_rolling5y = []
sharpe_ratio_rolling5y = []
calmar_ratio_rolling5y = []
mdd_rolling5y = []
sortino_ratio_rolling5y = []

for i in range(60, len(brk_monthly)):
    start = i - 60
    end = i
    window_brk_monthly = brk_monthly.iloc[start:end]['ret']
    annual_return_rolling5y.append(annual_return(window_brk_monthly))
    sharpe_ratio_rolling5y.append(sharpe_ratio(window_brk_monthly))
    calmar_ratio_rolling5y.append(calmar_ratio(window_brk_monthly))
    mdd_rolling5y.append((max_drawdown(window_brk_monthly)))
    sortino_ratio_rolling5y.append((sortino_ratio(window_brk_monthly)))

df_BRK_5y = pd.DataFrame()
df_BRK_5y.index = brk_monthly.index[60:]
df_BRK_5y['annual_return'] = annual_return_rolling5y
df_BRK_5y['sharpe_ratio'] = sharpe_ratio_rolling5y
df_BRK_5y['calmar_ratio'] = calmar_ratio_rolling5y
df_BRK_5y['max_drawdown'] = mdd_rolling5y
df_BRK_5y['sortino_ratio'] = sortino_ratio_rolling5y

# # test only
# #  arrays for stock weights in each portfolio
# all_weights = np.zeros((num_ports, len(dfr.columns)))
#
# # expected return of each portfolio
# ret_arr = np.zeros(num_ports)
#
# # volatility of each portfolio
# vol_arr = np.zeros(num_ports)
#
# # Sharpe ratio of each portfolio
# sharpe_arr = np.zeros(num_ports)
#
# for x in range(num_ports):
#     # generate random weights
#     weights = np.array(np.random.random(len(dfr.columns)))
#     weights = weights/np.sum(weights)
#     this_portfolios_return = np.sum((dfr.mean() * weights * len(dfr)))
#     # replace the zero arrays of portfolio weights with simulated weights
#     all_weights[x, :] = weights
#     # compute the return for the portfolio
#     ret_arr[x] = np.sum((dfr.mean() * weights * 12))
#     # compute the volatility for the portfolio
#     vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(dfr.cov()*12, weights)))
#     # calculate Sharpe ratio
#     sharpe_arr[x] = ret_arr[x] / vol_arr[x]
#
# max_sr = sharpe_arr.max()
# max_sr_loc = sharpe_arr.argmax()
# best_allocation = all_weights[max_sr_loc, :]
# max_sr_ret = ret_arr[max_sr_loc]
# max_sr_vol = vol_arr[max_sr_loc]


# def replication(dfr, dfrf, brk_monthly):
#     # arrays for stock weights in each portfolio
#     all_weights = np.zeros((num_ports, len(dfr.columns)))
#
#     # expected return of each portfolio
#     ret_arr = np.zeros(num_ports)
#
#     # volatility of each portfolio
#     vol_arr = np.zeros(num_ports)
#
#     # Sharpe ratio of each portfolio
#     sharpe_arr = np.zeros(num_ports)
#
#     # calculate brk total return first
#     brk_total_return = total_returns(brk_monthly['ret'])
#
#     # generate portfolios with loop
#     for x in range(num_ports):
#         # generate random weights
#         weights = np.array(np.random.random(len(dfr.columns)))
#         weights = weights/np.sum(weights)
#         this_portfolios_return = np.sum((dfr.mean() * weights * len(dfr)))
#         if this_portfolios_return < 0.5 * brk_total_return:
#             continue
#         # replace the zero arrays of portfolio weights with simulated weights
#         all_weights[x, :] = weights
#         # compute the return for the portfolio
#         ret_arr[x] = np.sum((dfr.mean() * weights * 12))
#         # compute the volatility for the portfolio
#         vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(dfr.cov()*12, weights)))
#         # calculate Sharpe ratio
#         sharpe_arr[x] = ret_arr[x] / vol_arr[x]
#
#     max_sr = sharpe_arr.max()
#     max_sr_loc = sharpe_arr.argmax()
#     best_allocation = all_weights[max_sr_loc, :]
#     max_sr_ret = ret_arr[max_sr_loc]
#     max_sr_vol = vol_arr[max_sr_loc]
#
#     # brk_total_return = total_returns(brk_monthly['ret'])
#     RF_total_return = total_returns(dfrf)
#     df_portfolio_return = pd.DataFrame()
#     l_portfolio_return = []
#     for i in range(0, len(dfr)):
#         l_portfolio_return.append((dfr.iloc[i] * best_allocation).sum())
#     df_portfolio_return['portfolio_return'] = l_portfolio_return
#     portfolio_total_return = total_returns(df_portfolio_return['portfolio_return'])
#
#     replication_returns = []
#     l_w_p = []
#     l_w_rf = []
#     for i in range(0, 2001):
#         w_p = i * 0.001
#         w_rf = 1 - w_p
#         replication_returns.append(w_p * portfolio_total_return + w_rf * RF_total_return)
#         l_w_p.append(w_p)
#         l_w_rf.append(w_rf)
#     df_replication = pd.DataFrame()
#     df_replication['w_p'] = l_w_p
#     df_replication['w_rf'] = l_w_rf
#     df_replication['replication'] = replication_returns
#     df_replication['BRK'] = brk_total_return
#     df_replication['diff'] = df_replication['replication'] - df_replication['BRK']
#     best_w_p = df_replication['w_p'][df_replication['diff'].idxmin()]
#     best_w_rf = 1 - best_w_p

    # # create a figure
    # plt.figure(figsize=(12, 8))
    # # scatter plot
    # plt.scatter(vol_arr, ret_arr, c=sharpe_arr)
    # plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50)
    # # color bar to show levels of Sharpe ratio
    # plt.colorbar(label='Sharpe Ratio')
    # plt.xlabel('Volatility')
    # plt.ylabel('Return')
    # plt.show()

#     return best_w_p, best_w_rf, best_allocation
#
#
# best_w_p, best_w_rf, best_allocation = replication(dfr_b07, dfrf_b07, brk_monthly_b07)

