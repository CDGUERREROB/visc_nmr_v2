import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.model_selection import FeatureImportances

import warnings
warnings.filterwarnings("ignore")

def dist_box_stats(dataframe, feature, scale,  bins):
    f, (ax_box, ax_hist) = plt.subplots(
        2,
        sharex=True,
        gridspec_kw= {"height_ratios": (.15, .85)},
        figsize=(6, 4))
    color  = np.random.rand(3,)

    sns.boxplot(dataframe[feature], ax=ax_box, color=color)
    sns.histplot(data=dataframe[feature], ax=ax_hist, color=color, stat='density', kde='True', bins=bins)
    plt.xscale(scale)

    summary = pd.DataFrame((dataframe[feature]).describe().round(2)).T
    plt.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc='center',
        rowLoc='left',
        loc='left',
        bbox=[0., -0.45, 1, .20],
        fontsize=25
        )
    plt.savefig(f"../reports/figures/KDE-Boxplot_{feature}.png", quality=100, bbox_inches='tight', dpi=100
    )
    plt.show();

def t2lm_visc(dataset):
    f, ax = plt.subplots(figsize=(6, 6))
    x= dataset['T2lm (ms)']
    y= dataset['Viscosity (cP)']
    # s= dataset['Temperature (°F)']
    # cm = plt.cm.get_cmap('plasma')

    ax = plt.scatter(
        x,
        y,
        # c=s,
        # cmap=cm,
        alpha=0.5
        )
    # cmap =plt.colorbar(ax)
    # cmap.set_label('Temperature (°F)')
    plt.xlabel('T2lm (ms)')
    plt.ylabel('Viscosity (cP)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Viscosity (cP) vs T2LM (ms)', pad=10)
    # plt.title('Color: Temperature (°F)', pad=5)

    plt.grid(True, alpha=0.5)

    plt.savefig(f"../reports/figures/ViscvsT2LM_Temp.png", quality=100, bbox_inches='tight', dpi=1000)

def pplot(df):
    pp = sns.pairplot(df, diag_kind='kde')

    log_columns = ['Viscosity (cP)', 'T2lm (ms)']

    for ax in pp.axes.flat:
        if ax.get_xlabel() in log_columns:
            ax.set(xscale="log")
            ax.set(yscale="log")

    plt.show();

def corr_plot(df, method, figsize, name):
    plt.figure(figsize=figsize)
    plt.title(f'Correlation Coeficient Matrix - {method}', pad=10)
    sns.heatmap(df.corr(method), annot = True, cmap='coolwarm')
    plt.savefig(f"../reports/figures/CorrCoefMatrix_{name}_{method}.png", quality=100, bbox_inches='tight', dpi=1000)
    plt.show();

def regression_plots(y, y_pred, name):

    Y = 10**y
    Y_pred = 10**y_pred

    fig, ax = plt.subplots(figsize=(5.25, 5))
    plt.scatter(
        Y,
        Y_pred,
        s=75,
        alpha=0.75,
        label=
        f"R{chr(0x00b2)}: {r2_score(y, y_pred).round(3)}",
        edgecolor='k'
    )
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=1.5, label="Identity")
    # ax.plot([Y.min(), Y.max()], [Y.min()+10, Y.max()+10], 'k:', lw=3, alpha=0.75)
    ax.set_title(f'Error plot - {name}', pad=15)
    ax.set_xlabel('y')
    ax.set_ylabel('y\u0302')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(frameon=True)
    plt.savefig(f'../reports/figures/Errorplot_{name}.png', dpi=500, bbox_inches='tight')
    plt.show()

    # fig, ax2 = plt.subplots(figsize=(6, 5))
    # ax2 = sns.residplot(
    #     Y,
    #     Y_pred,
    #     lowess=True,
    #     color="b",
    #     line_kws={
    #         'color': 'black',
    #         "lw": 1
    #             },
    #     scatter_kws={
    #         's': 75,
    #         'alpha': 0.75,
    #         'edgecolor':'b'
    #                 })
    # ax2.set_title(f'Residuals - {name}', pad=15)
    # ax2.set_xlabel("Predicción")
    # ax2.set_ylabel("y\u0302")
    # ax2.set_xscale('log')

    # plt.savefig(f'../reports/figures/Residuos_{name}.png', dpi=500, bbox_inches='tight')
    # plt.show()

def plot_regression(model, X_train, y_train, X_test, y_test):

    _, axes = plt.subplots(ncols=2, figsize=(12,5))
    
    residuals = ResidualsPlot(model, train_color = 'g',test_color = 'b', ax=axes[1], show=False)
    residuals.fit(X_train, y_train)
    residuals.score(X_test, y_test)
    residuals.finalize()
    
    error = PredictionError(model, bestfit=False, ax=axes[0], show=False )
    error.fit(X_train, y_train)  # Fit the training data to the visualizer
    error.score(X_test, y_test)  # Evaluate the model on the test data 
    error.finalize(outpath=f"Predition Error.png")
    
    plt.show();


from yellowbrick.model_selection import FeatureImportances

def feature_importance(model, X, y):
    plt.figure(figsize=(6,4))
    viz = FeatureImportances(model)
    viz.fit(X, y)
    plt.savefig(f'../reports/figures/feature_importance.png', dpi=500, bbox_inches='tight')

    viz.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

def regression_metrics(model_name, y_train, y_train_pred, y_test, y_test_pred):
    mae_test =  mean_absolute_error(10**y_test, 10**y_test_pred).round(2)
    rmse_test =  math.sqrt(mean_squared_error(10**y_test, 10**y_test_pred))
    r2_test =  r2_score(y_test, y_test_pred).round(2)

    mae_train =  mean_absolute_error(10**y_train, 10**y_train_pred).round(2)
    rmse_train =  math.sqrt(mean_squared_error(10**y_train, 10**y_train_pred))
    r2_train =  r2_score(y_train, y_train_pred).round(2)

    print(f"============== Regression Metrics : {model_name} ===============")
    print(f'MAE_train: {mae_train}\t\t MAE_test: {mae_test}')
    print(f'RMSE_train: {rmse_train}\t RMSE_test: {rmse_test}')
    print(f'R{chr(0x00b2)}_train: {r2_train}\t\t\t R{chr(0x00b2)}_test: {r2_test}')


def regression_metrics_models(model_name, y_test, y_test_pred):

    mae = mean_absolute_error((y_test), (y_test_pred)).round(2)
    rmse = (math.sqrt(mean_squared_error((y_test), (y_test_pred))))
    r2 = r2_score(np.log10(y_test), np.log10(y_test_pred)).round(3)

    print(f"============== Regression Metrics : {model_name} ===============")
    print(f'R{chr(0x00b2)}: {r2}\tMAE: {mae}\tRMSE: {rmse}\n')

def regression_metrics_models2(model_name, y_test, y_test_pred):

    mae = mean_absolute_error((y_test), (y_test_pred)).round(2)
    rmse = (math.sqrt(mean_squared_error((y_test), (y_test_pred))))
    # r2 = r2_score(np.log10(y_test), np.log10(y_test_pred)).round(3)

    print(f"============== Regression Metrics : {model_name} ===============")
    print(f'df:{abs(y_test-y_test_pred).round(1)} \tMAE: {mae}\tRMSE: {rmse}\n')


def triple_combo_plot(data,top_depth,bottom_depth):

    data = data.copy()
    # data = data.replace('-999.00000', np.nan)
    data = data.rename(columns=(
        {
            'MD': 'DEPT',
            'DEN': 'DEN',
            'NEUT': 'NEUT',
            'RT': 'RT',
            'T2LM': 'T2lm',
            'GR': 'GR',
            'VISC_NMR': 'VISCOSITY',
            'PRESS': 'PRESSURE',
            'TEMP':'TEMPERATURE'
        }
        )
        )

    tops=(data.DEPT.min(), data.DEPT.max())
    tops_depths=(data.DEPT.min(), data.DEPT.max())
    
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,6), sharey=True)
    fig.subplots_adjust(top=0.825,wspace=0.125)

    for axes in ax:
        axes.set_ylim (bottom_depth,top_depth)
        # axes.invert_yaxis(True)
        axes.yaxis.grid(True)
        axes.get_xaxis().set_visible(False) 
        # for (i,j) in zip(tops_depths,tops):
            # if ((i>=top_depth) and (i<=bottom_depth)):
        #         axes.axhline(y=i, linewidth=1.5, color='black')
        #         axes.text(0.05, i ,j, horizontalalignment='center',verticalalignment='center')

#1st track: GR, Temp
    ax01=ax[0].twiny()
    ax01.grid(True)
    ax01.set_xlim(data.GR.min(),data.GR.max())
    ax01.spines['top'].set_position(('outward',10))
    ax01.spines['top'].set_edgecolor('green')
    ax01.set_xlabel("GR [GAPI]")
    ax01.plot(data.GR, data.DEPT, label='GR [GR-API]', color='green', lw =0.5)
    ax01.set_xlabel('GR [GAPI]',color='green')   
    ax01.get_yticklabels()
    ax01.tick_params(axis='x', colors='green')

    ax02 = ax[0].twiny()
    ax02.grid(False)
    ax02.set_xlim(data.PRESSURE.min(),data.PRESSURE.max())
    ax02.spines['top'].set_position(('outward',50))
    ax02.spines['top'].set_edgecolor('darkblue')
    ax02.set_xlabel('Pressure [psia]')
    ax02.plot(data.PRESSURE, data.DEPT, label='Pressure [psia]', color='darkblue', lw =0.75)
    ax02.set_xlabel('Pressure [psia]', color='darkblue')    
    ax02.tick_params(axis='x', colors='darkblue')

#2nd track: Resistividades
    ax11=ax[1].twiny()
    ax11.set_xlim(10 ,10000)
    ax11.set_xscale('log')
    ax11.grid(True, which="both")
    ax11.spines['top'].set_position(('outward',10))
    ax11.spines['top'].set_edgecolor('purple')
    ax11.set_xlabel('Rt[m.ohm]', color='purple')
    ax11.plot(data.RT, data.DEPT, label='Rt[ohm.m]', color='purple', lw =0.75)
    ax11.tick_params(axis='x', colors='purple') 

    ax12=ax[1].twiny()
    ax12.grid(True, which="both", alpha=0.5)
    ax12.set_xlim(data.TEMPERATURE.min() ,data.TEMPERATURE.max())
    ax12.set_xlim(data.TEMPERATURE.min(),data.TEMPERATURE.max())
    ax12.spines['top'].set_position(('outward',50))
    ax12.spines['top'].set_edgecolor('orange')
    ax12.set_xlabel('Temperature [°F]')
    ax12.plot(data.TEMPERATURE, data.DEPT, label='Temperature [°F]', color='orange', lw =0.75)
    ax12.set_xlabel('Temperature [°F]', color='orange')    
    ax12.tick_params(axis='x', colors='orange') 

#3nd track: Densidad-Neutron

    ax21=ax[2].twiny()
    ax21.set_xlim(-15,45)
    ax21.invert_xaxis()
    ax21.grid(False)
    ax21.plot(data.NEUT*100, data.DEPT, label='NPHI [%]', color='blue', lw =0.75) 
    ax21.spines['top'].set_position(('outward',10))
    ax21.spines['top'].set_edgecolor('blue')
    ax21.set_xlabel('NPHI[%]', color='blue')    
    ax21.tick_params(axis='x', colors='blue')
    
    ax22=ax[2].twiny()
    ax22.set_xlim(1.95,2.95)
    ax22.grid(False)
    ax22.plot(data.DEN, data.DEPT ,label='RHOB [g/cc]', color='red', lw =0.75,) 
    ax22.spines['top'].set_position(('outward',50))
    ax22.spines['top'].set_edgecolor('red')
    ax22.set_xlabel('RHOB[g/cc]',color='red')
    ax22.tick_params(axis='x', colors='red')

#4nd track: T2LM-Viscosidad

    ax41=ax[3].twiny()
    ax41.grid(False)
    ax41.set_xlim(10, 1000)
    ax41.spines['top'].set_position(('outward',10))
    ax41.spines['top'].set_edgecolor('brown')
    ax41.set_xlabel('Viscosidad [cP]')
    ax41.set_xscale('log')
    ax41.scatter([120.167], [8230.5], label='Viscosidad [cP]', color='brown', lw =1.25, marker='s')
    ax41.set_xlabel('Viscosity [cP]', color='brown')    
    ax41.tick_params(axis='x', colors='brown')

    ax42=ax[3].twiny()
    ax42.grid(True, which="both")
    ax42.set_xlim(10, 1000)
    ax42.set_xscale('log')
    ax42.set_xlim(data.T2lm.min(),data.T2lm.max())
    ax42.spines['top'].set_position(('outward',50))
    ax42.spines['top'].set_edgecolor('k')
    ax42.set_xlabel('T2LM [ms]')
    ax42.plot(data.T2lm, data.DEPT, label='T2LM [ms]', color='k', lw =1)
    ax42.set_xlabel('T2LM [ms]', color='k')    
    ax42.tick_params(axis='x', colors='k') 

    plt.savefig('../reports/figures/well_log.png', dpi=500, bbox_inches='tight', transparent = True)

def visc_well():
    visc_test = pd.DataFrame(
        {
            'Pressure (psia)':[4003.7,3514.7,3014.7,2516.7,2016.7,1519.7,1220.7,1020.7,818.7,522.7,415.7,316.7,216.7,114.7,],
            'Viscosity (cP)':[154.485,143.103,131.165,120.167,110.262,101.525,95.932,92.713,90.053,84.352,83.327,81.814,80.285,78.467]
        }
        )
    x = visc_test['Pressure (psia)']
    y = visc_test['Viscosity (cP)']
    plt.figure(figsize=(6,4.5))
    plt.plot(x, y, '--bo', alpha=0.75)
    plt.plot(2516.7, 120.167, color='brown', marker='s')
    plt.title('Dinamic Viscosity @ 200 °F', pad=15)
    plt.xlabel('Pressure (psia)')
    plt.ylabel('Viscosity (cP)')
    for xitem,yitem in np.nditer([x,y]):
        etiqueta = "{:.1f}".format(yitem)
        plt.annotate(etiqueta, (xitem,yitem), textcoords="offset points",xytext=(0,10),ha="center",fontsize=8)
    plt.savefig('../reports/figures/visc_log.png', dpi=500, bbox_inches='tight', transparent = True)
    plt.show();

def compare_ml(metrics_ml, name):
    import numpy as np
    fig,ax = plt.subplots(figsize=(11,5))
    plt.title('Compare metrics Machine Learning Models', pad=15)

    labels = metrics_ml.index.tolist()

    x = np.arange(len(labels))

    ax2 = ax.twinx()
    ax3 = ax.twinx()

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(0.9, 1.01)
    ax2.set_ylim(6000, metrics_ml['RMSE'].max()+300)
    ax3.set_ylim(2000,metrics_ml['MAE'].max()+300)

    ax.set_xlabel('Machine Learning Models', size=12)
    ax.set_ylabel(f'R{chr(0x00b2)}',)
    ax2.set_ylabel('RMSE')
    ax3.set_ylabel('MAE')
    ax.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    width = 0.1
    p1 = ax.bar(x-(width*2.5), metrics_ml['R2'], width=width, color='#006837', align='center', label='R2')
    p4 = ax2.bar(x, metrics_ml['RMSE'], width=width, color='#042845', align='center', label='RMSE')
    p6 = ax3.bar(x+(width*2), metrics_ml['MAE'], width=width, color='gray', align='center', label='MAE')

    lns = [p1,p4,p6]
    ax.legend(handles=lns, loc='best')

    ax3.spines['right'].set_position(('outward', 60))  
    ax3.xaxis.set_ticks([])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for p in p1:
        height = p.get_height()
        ax.text(x=p.get_x()+0.05, y=height+0.001,s="{}".format(height),ha='center', fontsize=10)

    for p in p4:
        height = p.get_height()
        ax2.text(x=p.get_x()+0.05, y=height+50,s="{}".format(height),ha='center', fontsize=10)

    for p in p6:
        height = p.get_height()
        ax3.text(x=p.get_x()+0.075, y=height+25,s="{}".format(height),ha='center', fontsize=10)

    plt.savefig(f'../reports/figures/compare_ML{name}.png', dpi=500, bbox_inches='tight', transparent = True)
    plt.show();

def compare_prev(metrics_ml, name):
    import numpy as np
    fig,ax = plt.subplots(figsize=(12,5))
    plt.title('Compare Previous Models Viscosity - T2LM NMR', pad=15)

    labels = metrics_ml.index.tolist()

    x = np.arange(len(labels))

    ax2 = ax.twinx()
    ax3 = ax.twinx()

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0.0, 1.01)
    ax2.set_ylim(1000, 100000)
    ax3.set_ylim(0, metrics_ml['MAE'].max()+500)

    ax.set_xlabel('Authors Models', size=12)
    ax.set_ylabel(f'R{chr(0x00b2)}',)
    ax2.set_ylabel('RMSE')
    ax3.set_ylabel('MAE')
    ax.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    width = 0.1
    p1 = ax.bar(x-(width*2.75), metrics_ml['R2'], width=width, color='#006837', align='center', label='R2')
    p4 = ax2.bar(x, metrics_ml['RMSE'], width=width, color='#042845', align='center', label='RMSE')
    p6 = ax3.bar(x+(width*2.75), metrics_ml['MAE'], width=width, color='gray', align='center', label='MAE')

    lns = [p1,p4,p6]
    ax.legend(handles=lns, loc='best')
    ax2.set_yscale('log')
    ax3.spines['right'].set_position(('outward', 60))  
    ax3.xaxis.set_ticks([])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for p in p1:
        height = p.get_height()
        ax.text(x=p.get_x()+0.05, y=height+0.0125,s="{}".format(height),ha='center', fontsize=10)

    for p in p4:
        height = p.get_height()
        ax2.text(x=p.get_x()+0.05, y=height+1500,s="{}".format(height),ha='center', fontsize=10)

    for p in p6:
        height = p.get_height()
        ax3.text(x=p.get_x()+0.075, y=height+30,s="{}".format(height),ha='center', fontsize=10)

    plt.savefig(f'../reports/figures/compare_ML{name}.png', dpi=500, bbox_inches='tight', transparent = True)
    plt.show();