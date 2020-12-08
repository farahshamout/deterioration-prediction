import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from scipy.interpolate import interp1d
import random
#import tensorflow as tf
from math import sqrt
#import gpflow 
#import keras


pd.options.mode.chained_assignment = None
# ignore warnings 
import math

def visualize(X, Y, xmin, xmax, ymin, ymax, m, v, descrip, figs_path, hadmid,p, code ):
    print(v)
    print('plotting')    
    #m.plot(plot_limits= [xmin, xmax] )
    X_all= np.linspace(xmax, xmin, 50).reshape(-1,1)
    X_lines = np.linspace (xmax, xmin, 48).reshape(-1,1)
    pred = m.predict(X_all)
    
    
    pred_lines = m.predict(X_lines)[0]
    
    mean_yp = pred[0]
    q =m.predict_quantiles(X_all)
    lb_ci = q[0]
    ub_ci = q[1]
    #plt.rc('font', family='serif')
    #plt.rc('xtick', labelsize='large')
    #plt.rc('ytick', labelsize='large')
    #plt.rc('xlabel', labelsize='large')
    plt.figure(num=None, figsize=(4.5, 2), dpi=80)
    plt.plot(X_all, mean_yp, linewidth=2, color='k', label='Mean' )   
    plt.fill_between(X_all.flatten(), lb_ci.flatten(), ub_ci.flatten(), color='silver', label= '95% CI')
    plt.scatter(X, Y, marker='x', color='k', label='Training Data')
    X_lines = X_lines.flatten()
    for i in range(0, len(pred_lines)):
        plt.plot([X_lines[i], X_lines[i]], [0, pred_lines[i]], linestyle='--', color='k')
    
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xlabel('Hours from admission', fontsize=20, labelpad=7)
    h=plt.ylabel(v, fontsize=20, labelpad=20)
    h.set_rotation(0)
    plt.xticks(np.arange(xmin, xmax, step=6), fontsize = 17)
    steps={'HR': 30, 'SBP':30, 'TEMP':4, 'SPO2': 20, 'RR': 7}
    
    plt.yticks(np.arange(ymin, ymax, step= steps[v]), fontsize=17)

    #pb.title(str(hadmid) + ': ' + v + ' label='+str(label))
    #plt.legend()
    save_flag =1
    if save_flag == 1:
        print('saving plot')
        # plt.savefig(figs_path+'/GPs/'+ v+'/' +descrip+'/'+str(hadmid)+', point='+ str(p)+v+'.png')
        plt.savefig(figs_path+ v +'_'+str(hadmid)+', point='+ str(p)+'code='+ code+v+'.png', bbox_inches='tight')
    # plt.show()
    plt.close()

def visualize_gpflow(X, Y, xmin, xmax, ymin, ymax, m, v, descrip, figs_path, hadmid,p, code ):
    print('plotting')    
    #m.plot(plot_limits= [xmin, xmax] )
    X_all= np.linspace(xmax, xmin, 50).reshape(-1,1)
    #X_lines = np.linspace (xmax, xmin, 12).reshape(-1,1)
    mean, var = m.predict_y(X_all)
    
    plt.figure(num=None, figsize=(4.5, 2), dpi=80)
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(X_all, mean, 'C0', lw=2)
    plt.fill_between(X_all[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
    
    
    
    
    #plt.plot(X_all, mean_yp, linewidth=2, color='k', label='Mean' )   
    #plt.fill_between(X_all.flatten(), lb_ci.flatten(), ub_ci.flatten(), color='silver', label= '95% CI')
    #plt.scatter(X, Y, marker='x', color='k', label='Training Data')
    #X_lines = X_lines.flatten()
    #for i in range(0, len(pred_lines)):
    #    plt.plot([X_lines[i], X_lines[i]], [0, pred_lines[i]], linestyle='--', color='k')
    
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xlabel('Hours from admission', fontsize=20, labelpad=7)
    h=plt.ylabel(v, fontsize=20, labelpad=20)
    h.set_rotation(0)
    plt.xticks(np.arange(xmin, xmax, step=6), fontsize = 17)
    steps={'HR': 30, 'SBP':30, 'TEMP':4, 'SPO2': 20, 'RR': 7}
    
    plt.yticks(np.arange(ymin, ymax, step= steps[v]), fontsize=17)

    #pb.title(str(hadmid) + ': ' + v + ' label='+str(label))
    #plt.legend()
    save_flag =1
    if save_flag == 1:
        print('saving plot')
        # plt.savefig(figs_path+'/GPs/'+ v+'/' +descrip+'/'+str(hadmid)+', point='+ str(p)+v+'.png')
        plt.savefig(figs_path+ v +'_gpflow_'+str(hadmid)+', point='+ str(p)+'code='+ code+v+'.png', bbox_inches='tight')
    plt.show()
    plt.close()


# This optimizes the GPS using GPy for each vital sign of each point 
def GP(hadmid, obs, vital_signs, N, figs_path, s, priors, store_cols, code, point_n, num_test_points=0, p=0, m=0, mean_vitals=0):
    # Create a dataframe to return the predicted mean and variance of each vital sign    
    samples_df = pd.DataFrame(columns=store_cols)
    tree = lambda: defaultdict(tree)
    hyperparam = tree()
    
    
    # Extract patient profile with vital signs and add an additional column based on timeframe of data       
    label = obs.iloc[-1].label 
    eventtime = obs.iloc[-1].hrs_to_firstevent + obs.iloc[-1].hrs_from_adm
    descrip = ('normals', 'abnormals')[label>0]
    temp_dict_df={}
    # Get each vital sign's observations
    for v in vital_signs:
        # prior mean of GP 
        if len(mean_vitals)!=0:
            prior_mean = np.round(mean_vitals[v].values[0],4)
            
        # 1.  Get vital sign dataframe and perform data pre-processing 
        X = obs.hrs_from_adm.values   
        Y = obs[v]
        # drop null values but store time of observation
        xmin = X.min()
        xmax = X.max()
        nan_pos = np.argwhere(np.isnan(Y.values))

        if len(nan_pos) == len(Y):
            Y = Y.fillna(prior_mean)
        else:
            X = np.delete(X, nan_pos)
            Y= np.delete(Y.values, nan_pos)

        # transpose matrices
        X=np.atleast_2d(X).T
        Y=np.atleast_2d(Y).T

        # 2. Choose kernel type and initial values for hyperparameters 
        # Choose kernel
        kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 1) 
        # 3. Choose mean function OR leave it as zero
        mf = GPy.core.Mapping(1,1)
        mf.f = lambda x: prior_mean     
        mf.update_gradients = lambda a,b: None
        mf.gradients_X = lambda a,b: 0
        m = GPy.models.GPRegression(X=X,Y=Y,kernel=kernel,mean_function=mf)
        # 5. Set priors on the hyperparameters
        m.rbf.lengthscale.set_prior(priors[v]['lengthscale'], warning=False)
        m.Gaussian_noise.set_prior(priors[v]['noise'], warning=False)
        m.rbf.variance.set_prior(priors[v]['variance'], warning=False)

        # Plot initial GP model prior to optimization
        #m.plot()
        # 6. Optimize the kernel parameters and choose the best one out of 10
        np.random.seed(0)
        m.optimize_restarts(num_restarts =10, verbose=False, robust=True)
        #print(m)
        
        # Store final results for each patient 
        loglikelihood = m.objective_function()
        hyperparam[v]['obj_func']=loglikelihood
        hyperparam[v]['rbf_lengthscales']=m.kern.lengthscale[0]
        hyperparam[v]['rbf_variance']=m.kern.variance[0]
        hyperparam[v]['Gaussian_noise_variance']=m.Gaussian_noise.variance[0]

        ymin = priors[v]['plot_ylims'][0]
        ymax = priors[v]['plot_ylims'][1]
    
        if xmax >=24:
            xmin=xmax-24.00
        else:
            xmin=0
        # 8. plot and save figure 
        plot=1
        if plot ==1:
            visualize(X.flatten(), Y.flatten(), xmin, xmax, ymin, ymax,m, v, descrip, figs_path, hadmid,p )
        # 9. Subsample predicted values in the previous 24 hours of point at every hour
        start = xmax - 24.00
        if xmax < 2:
            Xp = np.array([xmax])
        elif xmax <24:
            Xp=[]
            st = xmax
            while st>=2:
                Xp = Xp + [st]
                st= st-2
            Xp = Xp+[st]
            Xp = np.asarray(Xp)
            Xp.sort()
        elif xmax>=24:
            Xp = np.linspace( start, xmax, 12, endpoint=True)  
       
        # 10. Compute MSE over training points
        y_pred = m.predict(X)
        rmse = mean_squared_error(Y.flatten(), y_pred[0])**(1/2)        
        mll = (0.5*np.log(2*np.pi*(y_pred[1]**2)) + (Y.flatten()-y_pred[0])**2/(2*(y_pred[1]**2))).sum()/len(Y.flatten())
        
        Xp.resize((len(Xp),1))
        
        # mean
        Yp =np.append(np.round(m.predict(Xp[0:-1])[0],3), Y[-1])
        len_sequence = len(Yp)
        Yp = np.pad(Yp, (0,s-len_sequence), 'constant', constant_values= np.nan)
        Yp.resize((1, s))
        # variance
        Vp = np.append(np.round(m.predict(Xp[0:-1])[1],3), 0)
        Vp = np.pad(Vp, (0,s-len_sequence), 'constant', constant_values= np.nan)
        Vp.resize((1, s))
        
        l = len(Yp[0])
        temp_dict_df['hadm_id'] =  [hadmid]*l
        temp_dict_df[v+'_mean'] =  Yp[0]
        temp_dict_df[v+'_var'] =  Vp[0]
        temp_dict_df['point'] = [point_n]*l
        temp_dict_df['len_sequence'] = [len_sequence]*l       
        temp_dict_df['hrs_to_firstevent'] = (eventtime - Xp.flatten(),[np.nan]*l)[math.isnan(eventtime)]
        temp_dict_df['code'] = [code]*l
        temp_dict_df['hrs_from_adm'] = np.pad(Xp.flatten(), (0, s-len_sequence), 'constant', constant_values = np.nan) 
        if len(temp_dict_df['hrs_to_firstevent']) <= len_sequence:
            temp_dict_df['hrs_to_firstevent']=np.pad(temp_dict_df['hrs_to_firstevent'], (0, s-len(temp_dict_df['hrs_to_firstevent'])), 'constant', constant_values=np.nan)
        
        
        temp_df = pd.DataFrame({'hrs_from_adm': Xp.flatten(), 
                                'avpu': [np.nan]*len(Xp.flatten()), 
                                'supplemental_oxygen':[np.nan]*len(Xp.flatten())})
        am = pd.concat([obs[['hrs_from_adm', 'avpu', 'supplemental_oxygen']],temp_df])
        am = am.sort_values(by=['hrs_from_adm'])
        am = am.fillna(method='ffill')
        am = am.drop_duplicates(subset=['hrs_from_adm'])
        df_temp = pd.DataFrame(temp_dict_df)
        final_df = pd.merge(df_temp, am, how='left',on='hrs_from_adm' )   
 
    samples_df = samples_df.append(final_df)
    return samples_df,  hyperparam, m




# This runs GP + linear interpolation + carry last value forward 
def modelling(hadmid, obs, vital_signs, N, figs_path, s, priors, store_cols, code, point_n, num_test_points=0, p=0, m=0, mean_vitals=0):
    # Create a dataframe to return the predicted mean and variance of each vital sign    
    samples_df = pd.DataFrame(columns=store_cols)
    samples_df_rmse = pd.DataFrame()
    tree = lambda: defaultdict(tree)
    hyperparam = tree()
    
    # Extract patient profile with vital signs and add an additional column based on timeframe of data       
    #label = obs.iloc[-1].y_true 
    #eventtime = obs.iloc[-1].hrs_to_firstevent + obs.iloc[-1].hrs_from_adm
    #descrip = ('normals', 'abnormals')[label>0]
    temp_dict_df={}
    temp_dict_df_rmse={}
    # Get each vital sign's observations
    for v in vital_signs:
            
        # 1.  Get vital sign dataframe and perform data pre-processing 
        X = obs.hrs_from_adm.values   
        Y = obs[v]
        # drop null values but store time of observation
        xmin = X.min()
        xmax = X.max()
        nan_pos = np.argwhere(np.isnan(Y.values))
        
        
        # prior mean of GP 
        if len(mean_vitals)!=0:
            if (v!= 'avpu') & (v != 'supplemental_oxygen'):
                #prior_mean = np.round(mean_vitals[v].values[0],4)
                prior_mean = np.round(mean_vitals[v],4)
            else:
                #prior_mean =  np.round(mean_vitals[v].values[0],0)
                prior_mean =  np.round(mean_vitals[v],0)
        else:
            prior_mean = np.round(Y.mean(), 0)

        if len(nan_pos) == len(Y):
            Y = Y.fillna(prior_mean)
            Y=Y.values
        else:
            X = np.delete(X, nan_pos)
            Y= np.delete(Y.values, nan_pos)

        # Subsmaple 48 points because it could be too dense
        if len(X)>48:
            rn_sample = random.sample(list(np.arange(len(X))), 48)
            X_model = [X[i] for i in rn_sample]
            Y_model = [Y[i] for i in rn_sample]
        else: 
            X_model = X
            Y_model = Y
        
        # transpose matrices
        X = np.atleast_2d(X).T
        Y = np.atleast_2d(Y).T
        X_model=np.atleast_2d(X_model).T
        Y_model=np.atleast_2d(Y_model).T
        
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            # 2. Choose kernel type and initial values for hyperparameters 
            # Choose kernel
            kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 1) 
            # 3. Choose mean function OR leave it as zero
            mf = GPy.core.Mapping(1,1)
            mf.f = lambda x: prior_mean     
            mf.update_gradients = lambda a,b: None
            mf.gradients_X = lambda a,b: 0
            m = GPy.models.GPRegression(X=X_model,Y=Y_model,kernel=kernel,mean_function=mf)
            # 5. Set priors on the hyperparameters
            m.rbf.lengthscale.set_prior(priors[v]['lengthscale'], warning=False)
            m.Gaussian_noise.set_prior(priors[v]['noise'], warning=False)
            m.rbf.variance.set_prior(priors[v]['variance'], warning=False)
    
            # Plot initial GP model prior to optimization
            #m.plot()
            # 6. Optimize the kernel parameters and choose the best one out of 10
            np.random.seed(0)
            m.optimize_restarts(num_restarts =20, verbose=False, robust=True)
            #print(m)
            
            # Store final results for each patient 
            loglikelihood = m.objective_function()
            hyperparam[v]['obj_func']=loglikelihood
            hyperparam[v]['rbf_lengthscales']=m.kern.lengthscale[0]
            hyperparam[v]['rbf_variance']=m.kern.variance[0]
            hyperparam[v]['Gaussian_noise_variance']=m.Gaussian_noise.variance[0]
    
            ymin = priors[v]['plot_ylims'][0]
            ymax = priors[v]['plot_ylims'][1]
    
        if xmax >=48:
            xmin=xmax-48
        else:
            xmin=0
        # 8. plot and save figure 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            plot=0
        else:
            plot=0
        if plot ==1:
            visualize(X.flatten(), Y.flatten(), xmin, xmax, ymin, ymax,m, v, descrip, figs_path, hadmid,p, code )
            
        
        # 9. Subsample predicted values in the previous 24 hours of point at every hour
        start = xmax - 48
        if xmax < 2:
            Xp = np.array([xmax])
        elif xmax <48:
            Xp=[]
            st = xmax
            while st>=2:
                Xp = Xp + [st]
                st= st-2
            Xp = Xp+[st]
            Xp = np.asarray(Xp)
            Xp.sort()
        elif xmax>=48:
            Xp = np.append(np.linspace( start, xmax, 48, endpoint=False)[1:], np.array([xmax]))
       
        Xp = np.linspace(0,48,48)
        Xp.resize((len(Xp),1))
        
        # original length of sequence 
        len_sequence = len(Xp) 
        
        # Option 1: predict GP mean and variance 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            gp_pred = m.predict(Xp[0:-1])
            Ygp =np.append(np.round(gp_pred[0],2), Y[-1]) # mean of GP posterior 
            Ygp_rmse = sqrt(mean_squared_error(Y, m.predict(X)[0]))   # mean posterior to calculate rmse using all of the training data
            
            Vp = np.append(np.round(gp_pred[1],2), 0) # variance of GP posterior 
            # pad and resize mean and variance values 
            Ygp = np.pad(Ygp, (0,s-len_sequence), 'constant', constant_values= np.nan) 
            Ygp.resize((1, s))
            Vp = np.pad(Vp, (0,s-len_sequence), 'constant', constant_values= np.nan)
            Vp.resize((1, s))
            temp_dict_df[v+'_mean'] =  Ygp[0]
            temp_dict_df[v+'_var'] =  Vp[0]
            
            temp_dict_df_rmse[v+'_gp_rmse'] = Ygp_rmse
        
        l = s
        temp_dict_df['hadm_id'] =  [hadmid]*l
        temp_dict_df['point'] = [point_n]*l
        temp_dict_df['len_sequence'] = [len_sequence]*l       
        #temp_dict_df['hrs_to_firstevent'] = (eventtime - Xp.flatten(),[np.nan]*l)[math.isnan(eventtime)]
        temp_dict_df['code'] = [code]*l
        temp_dict_df['hrs_from_adm'] = np.pad(Xp.flatten(), (0, s-len_sequence), 'constant', constant_values = np.nan) 
        #if len(temp_dict_df['hrs_to_firstevent']) <= len_sequence:
         #   temp_dict_df['hrs_to_firstevent']=np.pad(temp_dict_df['hrs_to_firstevent'], (0, s-len(temp_dict_df['hrs_to_firstevent'])), 'constant', constant_values=np.nan)
        
        temp_dict_df_rmse['code'] = [code]
        
        X=list(X.flatten())
        Y=list(Y.flatten())
        Xp =list( Xp.flatten())
        if len(X) == 1:
            X.insert(0, max(X)-1)            
            Y.insert(0, prior_mean)
        
        if min(X) > min(Xp):
            X.insert(0, min(Xp))            
            Y.insert(0, prior_mean)
        
        if max(Xp) > max(X):
            X = np.array(X)
            X = np.append(X, np.array([xmax]))
            Y = np.array(Y)
            Y = np.append(Y, np.array([prior_mean]))
        
        # Option 2: interpolation (carry last value forward)
        f1 = interp1d(X, Y, kind='previous', fill_value='extrapolate')
        Y_prev = f1(Xp)
        temp_dict_df[v+'_prev'] = np.pad(Y_prev, (0,s-len_sequence), 'constant', constant_values= np.nan)
        #temp_dict_df_rmse[v+'_prev'] =  sqrt(mean_squared_error(Y, f1(X)))  
        
        
        # Option 3:  interpolation (nearest)
        f2 = interp1d(X, Y, kind='nearest', fill_value='extrapolate')
        Y_nearest = f2(Xp)
        temp_dict_df[v+'_nearest'] = np.pad(Y_nearest, (0,s-len_sequence), 'constant', constant_values= np.nan)
        #temp_dict_df_rmse[v+'_nearest'] =  sqrt(mean_squared_error(Y, f2(X)))  
        
        # Option 4: interpolation (linear)
        f3 = interp1d(X, Y, kind='linear', fill_value='extrapolate')
        Y_cubic = np.round(f3(Xp),1)
        temp_dict_df[v+'_linear'] = np.pad(Y_cubic, (0,s-len_sequence), 'constant', constant_values= np.nan)
        #temp_dict_df_rmse[v+'_linear'] =  sqrt(mean_squared_error(Y, f3(X)))  

        
        if (v == 'avpu' ) | (v=='supplemental_oxygen'):
            temp_dict_df[v+'_linear'] = np.round( temp_dict_df[v+'_linear'], 0)
        
        # 10. Compute MSE over training points
        #y_pred = m.predict(X)
        #rmse = mean_squared_error(Y.flatten(), y_pred[0])**(1/2)        
        #mll = (0.5*np.log(2*np.pi*(y_pred[1]**2)) + (Y.flatten()-y_pred[0])**2/(2*(y_pred[1]**2))).sum()/len(Y.flatten())
    
    samples_df = samples_df.append(pd.DataFrame(temp_dict_df))
    samples_df_rmse = samples_df_rmse.append(pd.DataFrame(temp_dict_df_rmse))
    return samples_df, samples_df_rmse#,  hyperparam, m



# This runs GP + linear interpolation + carry last value forward 
def modelling_rmse(hadmid, obs, vital_signs, N, figs_path, s, priors, store_cols, code, point_n, num_test_points=0, p=0, m=0, mean_vitals=0):
    # Create a dataframe to return the predicted mean and variance of each vital sign    
    samples_df = pd.DataFrame(columns=store_cols)
    tree = lambda: defaultdict(tree)
    hyperparam = tree()
    
    # Extract patient profile with vital signs and add an additional column based on timeframe of data       
    temp_dict_df={}
    # Get each vital sign's observations
    for v in vital_signs:

        # prior mean of GP 
        if len(mean_vitals)!=0:
            if (v!= 'avpu') & (v != 'supplemental_oxygen'):
                prior_mean = np.round(mean_vitals[v].values[0],4)
            else:
                prior_mean =  np.round(mean_vitals[v].values[0],0)
            
        # 1.  Get vital sign dataframe and perform data pre-processing 
        X = obs.hrs_from_adm.values   
        Y = obs[v]
        xmin = X.min()
        xmax = X.max()
        # drop null values but store time of observation
        nan_pos = np.argwhere(np.isnan(Y.values))

        if len(nan_pos) == len(Y):
            Y = Y.fillna(prior_mean)
            Y=Y.values
        else:
            X = np.delete(X, nan_pos)
            Y= np.delete(Y.values, nan_pos)
            
        if len(Y)>2:
            idx_test = random.sample(list(np.arange(1,len(Y)-1,1)), int(np.round(0.2*len(Y),0)))
            X_train= np.delete(X, idx_test)
            Y_train = np.delete(Y, idx_test)
            Xp = X[idx_test]
            Y_test = Y[idx_test]
        

            # transpose matrices
            X=np.atleast_2d(X_train).T
            Y=np.atleast_2d(Y_train).T
            
            if (v!= 'avpu') & (v != 'supplemental_oxygen'):
                # 2. Choose kernel type and initial values for hyperparameters 
                # Choose kernel
                kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 1) 
                # 3. Choose mean function OR leave it as zero
                mf = GPy.core.Mapping(1,1)
                mf.f = lambda x: prior_mean     
                mf.update_gradients = lambda a,b: None
                mf.gradients_X = lambda a,b: 0
                m = GPy.models.GPRegression(X=X,Y=Y,kernel=kernel,mean_function=mf)
                # 5. Set priors on the hyperparameters
                m.rbf.lengthscale.set_prior(priors[v]['lengthscale'], warning=False)
                m.Gaussian_noise.set_prior(priors[v]['noise'], warning=False)
                m.rbf.variance.set_prior(priors[v]['variance'], warning=False)
        
                # Plot initial GP model prior to optimization
                #m.plot()
                # 6. Optimize the kernel parameters and choose the best one out of 10
                np.random.seed(0)
                m.optimize_restarts(num_restarts =10, verbose=False, robust=True)
                #print(m)
                
                loglikelihood = m.objective_function()
            
           
            Xp.resize((len(Xp),1))
            
            # Option 1: predict GP mean and variance 
            if (v!= 'avpu') & (v != 'supplemental_oxygen'):
                gp_pred = m.predict(Xp)
                Ygp =gp_pred[0] # mean of GP posterior 
                rmse = sqrt(mean_squared_error(Y_test, Ygp))
                temp_dict_df[v+'_gp'] =  rmse
            
            temp_dict_df['hadm_id'] =  [hadmid]
            temp_dict_df['point'] = [point_n]
            temp_dict_df['code'] = [code]
            
            X=list(X.flatten())
            Y=list(Y.flatten())
            Xp =list( Xp.flatten())
            if len(X) == 1:
                X.insert(0, max(X)-1)            
                Y.insert(0, prior_mean)
            
            if min(X) > min(Xp):
                X.insert(0, min(Xp))            
                Y.insert(0, prior_mean)
            
            if max(Xp) > max(X):
                X = np.array(X)
                X = np.append(X, np.array([xmax]))
                Y = np.array(Y)
                Y = np.append(Y, np.array([prior_mean]))
                
            X = np.array(X)
            Y = np.array(Y)
            
            # Option 2: interpolation (carry last value forward)
            f1 = interp1d(X, Y, kind='previous')
            Y_prev = f1(Xp)
            temp_dict_df[v+'_prev'] = sqrt(mean_squared_error(Y_test, Y_prev))

            # Option 3:  interpolation (nearest)
            f2 = interp1d(X, Y, kind='nearest')
            Y_nearest = f2(Xp)
            temp_dict_df[v+'_nearest'] = sqrt( mean_squared_error(Y_test, Y_nearest))  

            # Option 4: interpolation (linear (cubic))
            f3 = interp1d(X, Y, kind='linear')
            Y_cubic = f3(Xp)
            temp_dict_df[v+'_linear'] =  sqrt(mean_squared_error(Y_test, Y_cubic))  
        
        # 10. Compute MSE over training points
        #y_pred = m.predict(X)
        #     
        #mll = (0.5*np.log(2*np.pi*(y_pred[1]**2)) + (Y.flatten()-y_pred[0])**2/(2*(y_pred[1]**2))).sum()/len(Y.flatten())
    
    df_temp = pd.DataFrame(temp_dict_df)
    samples_df = samples_df.append(df_temp)
    return samples_df,  hyperparam, m



#from optimize_code import timing_decorator


# Plot per window using GP from previous window December 3rd (optimized)

def plot_GP_point_2(hadmid, hadmid_dataset, dataset, vital_signs, n, figs_path, vitals_priors, s, store_cols, mean_vitals):
    sampled_obs_train_df = pd.DataFrame( columns=store_cols)
    #print(hadmid)
    m=0 
    for p in range(0, len(hadmid_dataset)):
        print(str(p)+'/'+str(len(hadmid_dataset)))
        # obtain ID's data points within the last 24 hours
        end_time = hadmid_dataset.iloc[p].hrs_from_adm
        subset_df = dataset.loc[(dataset.hrs_from_adm<= end_time)&(dataset.hrs_from_adm> end_time-24)]

        code = hadmid_dataset.iloc[p].code
        point_n = hadmid_dataset.iloc[p].point
        
        # model using GP
        patient_df = modelling_gpflow(hadmid,  subset_df, vital_signs, n, figs_path, s, vitals_priors, store_cols, code, point_n, 0, p, m, mean_vitals)
        sampled_obs_train_df =sampled_obs_train_df.append(patient_df)
            
    return sampled_obs_train_df

def plot_GP(hadmid, hadmid_dataset, dataset, vital_signs, n, figs_path, vitals_priors, s, store_cols, mean_vitals=0):
    sampled_obs_train_df = pd.DataFrame( columns=store_cols)
    sampled_obs_train_df_rmse = pd.DataFrame(columns=store_cols)
    #print(hadmid)
    m=0 
    
    print(str(hadmid)+'_'+str(len(hadmid_dataset)))

    subset_df = dataset

    code = dataset.iloc[0].code
    point_n = dataset.iloc[0].point
        
    # model using GP
    #patient_df = modelling_gpflow_mimic(hadmid,  subset_df, vital_signs, n, figs_path, s, vitals_priors, store_cols, code, point_n, 0, 0, m, mean_vitals)
    patient_df, patient_df_rmse = modelling(hadmid,  subset_df, vital_signs, n, figs_path, s, vitals_priors, store_cols, code, point_n, 0, point_n, m, mean_vitals)
    sampled_obs_train_df =sampled_obs_train_df.append(patient_df)
    sampled_obs_train_df_rmse = sampled_obs_train_df_rmse.append(patient_df_rmse)
            
    return patient_df, patient_df_rmse


# Plot per window using GP from previous window December 3rd (optimized)
def summarize_window(hadmid, hadmid_dataset, dataset, vital_signs, n, figs_path, vitals_priors, s, store_cols, mean_vitals):
    sampled_obs_train_df = pd.DataFrame( columns=store_cols)
    print(hadmid)
    m=0 
    for p in range(0, len(hadmid_dataset)):
        
        # obtain ID's data points within the last 24 hours
        end_time = hadmid_dataset.iloc[p].hrs_from_adm
        subset_df = dataset.loc[(dataset.hrs_from_adm<= end_time)&(dataset.hrs_from_adm> end_time-24)]

        code = hadmid_dataset.iloc[p].code
        point_n = hadmid_dataset.iloc[p].point
        
        # model using GP
        patient_df, opt_hyperparams_patient,  m = modelling(hadmid,  subset_df, vital_signs, n, figs_path, s, vitals_priors, store_cols, code, point_n, 0, p, m, mean_vitals)
        sampled_obs_train_df =sampled_obs_train_df.append(patient_df)
            
    return sampled_obs_train_df




# Plot per window using GP from previous window December 3rd (optimized)
def plot_GP_point_2_rmse(hadmid, hadmid_dataset, dataset, vital_signs, n, figs_path, vitals_priors, s, store_cols, mean_vitals):
    sampled_obs_train_df = pd.DataFrame( columns=store_cols)
    print(hadmid)
    m=0 
    for p in range(0, len(hadmid_dataset)):
        #print(p)
        # obtain ID's data points within the last 24 hours
        end_time = hadmid_dataset.iloc[p].hrs_from_adm
        subset_df = dataset.loc[(dataset.hrs_from_adm<= end_time)&(dataset.hrs_from_adm> end_time-24)]

        code = hadmid_dataset.iloc[p].code
        point_n = hadmid_dataset.iloc[p].point
        
        # model using GP
        patient_df, opt_hyperparams_patient,  m = modelling_rmse(hadmid,  subset_df, vital_signs, n, figs_path, s, vitals_priors, store_cols, code, point_n, 0, p, m, mean_vitals)
        sampled_obs_train_df =sampled_obs_train_df.append(patient_df)
            
    return sampled_obs_train_df


def modelling_gpflow(hadmid, obs, vital_signs, N, figs_path, s, priors, store_cols, code, point_n, num_test_points=0, p=0, m=0, mean_vitals=0):
    # Create a dataframe to return the predicted mean and variance of each vital sign    
    samples_df = pd.DataFrame(columns=store_cols)
    tree = lambda: defaultdict(tree)
    hyperparam = tree()


    # Extract patient profile with vital signs and add an additional column based on timeframe of data       
    label = obs.iloc[-1].label 
    eventtime = obs.iloc[-1].hrs_to_firstevent + obs.iloc[-1].hrs_from_adm
    descrip = ('normals', 'abnormals')[label>0]
    temp_dict_df={}
    # Get each vital sign's observations
    for v in vital_signs:
        # print(v)
        # prior mean of GP 
        if len(mean_vitals)!=0:
            if (v!= 'avpu') & (v != 'supplemental_oxygen'):
                prior_mean = np.round(mean_vitals[v].values[0],4)
            else:
                prior_mean =  np.round(mean_vitals[v].values[0],0)
            
        # 1.  Get vital sign dataframe and perform data pre-processing 
        X = obs.hrs_from_adm.values   
        Y = obs[v]
        # drop null values but store time of observation
        xmin = X.min()
        xmax = X.max()
        nan_pos = np.argwhere(np.isnan(Y.values))

        if len(nan_pos) == len(Y):
            Y = Y.fillna(prior_mean)
            Y=Y.values
        else:
            X = np.delete(X, nan_pos)
            Y= np.delete(Y.values, nan_pos)

        # transpose matrices
        X=np.atleast_2d(X).T
        Y=np.atleast_2d(Y).T
        
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            # 2. Choose kernel type and initial values for hyperparameters 
            # Choose kernel
            #import tensorflow as tf
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.3))
            gpflow.reset_default_session(config=config)

            kernel = gpflow.kernels.RBF(input_dim=1, variance=1, lengthscales=1) #+ gpflow.kernels.White(3)
            #GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 1) 
            # 3. Choose mean function OR leave it as zero
            #gpflow.mean_functions.Constant(prior_mean)
            meanf = gpflow.mean_functions.Constant(prior_mean)
            m = gpflow.models.GPR(X, Y, kernel, meanf)
            
            
            #mf = GPy.core.Mapping(1,1)
            #mf.f = lambda x: prior_mean  
            #mf.update_gradients = lambda a,b: None
            #mf.gradients_X = lambda a,b: 0
            #m = GPy.models.GPRegression(X=X,Y=Y,kernel=kernel,mean_function=mf)
            
            
            # 5. Set priors on the hyperparameters
            m.clear()
            m.kern.lengthscales.prior = priors[v]['lengthscale']
            m.kern.variance.prior = priors[v]['variance']
            m.likelihood.variance.prior = priors[v]['noise']
            m.compile()
            
            #m.likelihood.variance.prior = priors[v]['noise']
            # m.rbf.lengthscale.set_prior(priors[v]['lengthscale'], warning=False)
            # m.Gaussian_noise.set_prior(priors[v]['noise'], warning=False)
            # m.rbf.variance.set_prior(priors[v]['variance'], warning=False)
    
            # Plot initial GP model prior to optimization
            #m.plot()
            # 6. Optimize the kernel parameters and choose the best one out of 10
            np.random.seed(0)
            
            gpflow.train.ScipyOptimizer().minimize(m, maxiter=10, anchor = False)
            #print(m)

            #m.optimize_restarts(num_restarts =10, verbose=False, robust=True)
            #print(m)
            
            # Store final results for each patient 
            #loglikelihood = m.objective_function()
            #hyperparam[v]['obj_func']=loglikelihood
            hyperparam[v]['rbf_lengthscales']=m.kern.lengthscales.value
            hyperparam[v]['rbf_variance']=m.kern.variance.value
            hyperparam[v]['Gaussian_noise_variance']=m.likelihood.variance.value
    
            ymin = priors[v]['plot_ylims'][0]
            ymax = priors[v]['plot_ylims'][1]
    
        if xmax >=24:
            xmin=xmax-24.00
        else:
            xmin=0
        # 8. plot and save figure 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            plot=0
        else:
            plot=0
        
        if plot ==1:
            visualize_gpflow(X.flatten(), Y.flatten(), xmin, xmax, ymin, ymax,m, v, descrip, figs_path, hadmid,p, code )
        
        # 9. Subsample predicted values in the previous 24 hours of point at every hour
        start = xmax - 24.00
        if xmax < 2:
            Xp = np.array([xmax])
        elif xmax <24:
            Xp=[]
            st = xmax
            while st>=2:
                Xp = Xp + [st]
                st= st-2
            Xp = Xp+[st]
            Xp = np.asarray(Xp)
            Xp.sort()
        elif xmax>=24:
            Xp = np.append(np.linspace( start, xmax, 12, endpoint=False)[1:], np.array([xmax]))
       
        Xp.resize((len(Xp),1))
        # original length of sequence 
        len_sequence = len(Xp) 
        
        # Option 1: predict GP mean and variance 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            gp_pred = m.predict_y(Xp[0:-1])
            Ygp =np.append(np.round(gp_pred[0],3), Y[-1]) # mean of GP posterior 
            Vp = np.append(np.round(gp_pred[1],3), 0) # variance of GP posterior 
            # pad and resize mean and variance values 
            Ygp = np.pad(Ygp, (s-len_sequence,0), 'constant', constant_values= np.nan) 
            Ygp.resize((1, s))
            Vp = np.pad(Vp, (s-len_sequence,0), 'constant', constant_values= np.nan)
            Vp.resize((1, s))
            temp_dict_df[v+'_mean'] =  Ygp[0]
            temp_dict_df[v+'_var'] =  Vp[0]
        tf.reset_default_graph()
        l = s
        temp_dict_df['hadm_id'] =  [hadmid]*l
        temp_dict_df['point'] = [point_n]*l
        temp_dict_df['len_sequence'] = [len_sequence]*l       
        temp_dict_df['hrs_to_firstevent'] = (eventtime - Xp.flatten(),[np.nan]*l)[math.isnan(eventtime)]
        temp_dict_df['code'] = [code]*l
        temp_dict_df['hrs_from_adm'] = np.pad(Xp.flatten(), (s-len_sequence,0), 'constant', constant_values = np.nan) 
        if len(temp_dict_df['hrs_to_firstevent']) <= len_sequence:
            temp_dict_df['hrs_to_firstevent']=np.pad(temp_dict_df['hrs_to_firstevent'], (s-len(temp_dict_df['hrs_to_firstevent']),0), 'constant', constant_values=np.nan)
        
        
        X=list(X.flatten())
        Y=list(Y.flatten())
        Xp =list( Xp.flatten())
        if len(X) == 1:
            X.insert(0, max(X)-1)            
            Y.insert(0, prior_mean)
        
        if min(X) > min(Xp):
            X.insert(0, min(Xp))            
            Y.insert(0, prior_mean)
        
        if max(Xp) > max(X):
            X = np.array(X)
            X = np.append(X, np.array([xmax]))
            Y = np.array(Y)
            Y = np.append(Y, np.array([prior_mean]))
        
        # Option 2: interpolation (carry last value forward)
        f1 = interp1d(X, Y, kind='previous')
        Y_prev = f1(Xp)
        temp_dict_df[v+'_prev'] = np.pad(Y_prev, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        # Option 3:  interpolation (nearest)
        f2 = interp1d(X, Y, kind='nearest')
        Y_nearest = f2(Xp)
        temp_dict_df[v+'_nearest'] = np.pad(Y_nearest, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        # Option 4: interpolation (linear (cubic))
        f3 = interp1d(X, Y, kind='linear')
        Y_cubic = f3(Xp)
        temp_dict_df[v+'_linear'] = np.pad(Y_cubic, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        if (v == 'avpu' ) | (v=='supplemental_oxygen'):
            temp_dict_df[v+'_linear'] = np.round( temp_dict_df[v+'_linear'], 0)
        
        # 10. Compute MSE over training points
        #y_pred = m.predict(X)
        #rmse = mean_squared_error(Y.flatten(), y_pred[0])**(1/2)        
        #mll = (0.5*np.log(2*np.pi*(y_pred[1]**2)) + (Y.flatten()-y_pred[0])**2/(2*(y_pred[1]**2))).sum()/len(Y.flatten())

    #keras.backend.clear_session()
    df_temp = pd.DataFrame(temp_dict_df)
    samples_df = samples_df.append(df_temp)
    return samples_df 
#hyperparam

def modelling_gpflow_mimic(hadmid, obs, vital_signs, N, figs_path, s, priors, store_cols, code, point_n, num_test_points=0, p=0, m=0, mean_vitals=0):
    # Create a dataframe to return the predicted mean and variance of each vital sign    
    samples_df = pd.DataFrame(columns=store_cols)
    tree = lambda: defaultdict(tree)
    hyperparam = tree()


    # Extract patient profile with vital signs and add an additional column based on timeframe of data       
    label = obs.iloc[-1].y_true 
    #eventtime = obs.iloc[-1].hrs_to_firstevent + obs.iloc[-1].hrs_from_adm
    descrip = ('normals', 'abnormals')[label>0]
    temp_dict_df={}
    # Get each vital sign's observations
    for v in vital_signs:
        # print(v)
        # prior mean of GP 
        #if len(mean_vitals)!=0:
         #   if (v!= 'avpu') & (v != 'supplemental_oxygen'):
          #      prior_mean = np.round(mean_vitals[v].values[0],4)
           # else:
            #    prior_mean =  np.round(mean_vitals[v].values[0],0)
            
        # 1.  Get vital sign dataframe and perform data pre-processing 
        X = obs.Hours.values   
        Y = obs[v]
        # drop null values but store time of observation
        xmin = X.min()
        xmax = X.max()
        nan_pos = np.argwhere(np.isnan(Y.values))

        if len(nan_pos) == len(Y):
            #Y = Y.fillna(prior_mean)
            Y=Y.values
        else:
            X = np.delete(X, nan_pos)
            Y= np.delete(Y.values, nan_pos)

        # transpose matrices
        X=np.atleast_2d(X).T
        Y=np.atleast_2d(Y).T
        
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            # 2. Choose kernel type and initial values for hyperparameters 
            # Choose kernel
            #import tensorflow as tf
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.3))
            gpflow.reset_default_session(config=config)

            kernel = gpflow.kernels.RBF(input_dim=1, variance=1, lengthscales=1) #+ gpflow.kernels.White(3)
            #GPy.kern.RBF(input_dim=1, variance=1, lengthscale = 1) 
            # 3. Choose mean function OR leave it as zero
            #gpflow.mean_functions.Constant(prior_mean)
            #meanf = gpflow.mean_functions.Constant(prior_mean)
            m = gpflow.models.GPR(X, Y, kernel)
            
            
            #mf = GPy.core.Mapping(1,1)
            #mf.f = lambda x: prior_mean  
            #mf.update_gradients = lambda a,b: None
            #mf.gradients_X = lambda a,b: 0
            #m = GPy.models.GPRegression(X=X,Y=Y,kernel=kernel,mean_function=mf)
            
            
            # 5. Set priors on the hyperparameters
            m.clear()
            m.kern.lengthscales.prior = priors[v]['lengthscale']
            m.kern.variance.prior = priors[v]['variance']
            m.likelihood.variance.prior = priors[v]['noise']
            m.compile()
            
            #m.likelihood.variance.prior = priors[v]['noise']
            # m.rbf.lengthscale.set_prior(priors[v]['lengthscale'], warning=False)
            # m.Gaussian_noise.set_prior(priors[v]['noise'], warning=False)
            # m.rbf.variance.set_prior(priors[v]['variance'], warning=False)
    
            # Plot initial GP model prior to optimization
            #m.plot()
            # 6. Optimize the kernel parameters and choose the best one out of 10
            np.random.seed(0)
            
            gpflow.train.ScipyOptimizer().minimize(m, maxiter=10, anchor = False)
            #print(m)

            #m.optimize_restarts(num_restarts =10, verbose=False, robust=True)
            #print(m)
            
            # Store final results for each patient 
            #loglikelihood = m.objective_function()
            #hyperparam[v]['obj_func']=loglikelihood
            hyperparam[v]['rbf_lengthscales']=m.kern.lengthscales.value
            hyperparam[v]['rbf_variance']=m.kern.variance.value
            hyperparam[v]['Gaussian_noise_variance']=m.likelihood.variance.value
    
            ymin = priors[v]['plot_ylims'][0]
            ymax = priors[v]['plot_ylims'][1]
    
        #if xmax >=24:
        #    xmin=xmax-24.00
        #else:
        xmin=0
        # 8. plot and save figure 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            plot=1
        else:
            plot=0
        
        if plot ==1:
            visualize_gpflow(X.flatten(), Y.flatten(), xmin, xmax, ymin, ymax,m, v, descrip, figs_path, hadmid,p, code )
        
        # 9. Subsample predicted values in the previous 24 hours of point at every hour
        start = xmax - 24.00
        if xmax < 2:
            Xp = np.array([xmax])
        elif xmax <24:
            Xp=[]
            st = xmax
            while st>=2:
                Xp = Xp + [st]
                st= st-2
            Xp = Xp+[st]
            Xp = np.asarray(Xp)
            Xp.sort()
        elif xmax>=24:
            Xp = np.append(np.linspace( start, xmax, 12, endpoint=False)[1:], np.array([xmax]))
       
        Xp.resize((len(Xp),1))
        # original length of sequence 
        len_sequence = len(Xp) 
        
        # Option 1: predict GP mean and variance 
        if (v!= 'avpu') & (v != 'supplemental_oxygen'):
            gp_pred = m.predict_y(Xp[0:-1])
            Ygp =np.append(np.round(gp_pred[0],3), Y[-1]) # mean of GP posterior 
            Vp = np.append(np.round(gp_pred[1],3), 0) # variance of GP posterior 
            # pad and resize mean and variance values 
            Ygp = np.pad(Ygp, (s-len_sequence,0), 'constant', constant_values= np.nan) 
            Ygp.resize((1, s))
            Vp = np.pad(Vp, (s-len_sequence,0), 'constant', constant_values= np.nan)
            Vp.resize((1, s))
            temp_dict_df[v+'_mean'] =  Ygp[0]
            temp_dict_df[v+'_var'] =  Vp[0]
        tf.reset_default_graph()
        l = s
        temp_dict_df['hadm_id'] =  [hadmid]*l
        temp_dict_df['point'] = [point_n]*l
        #temp_dict_df['len_sequence'] = [len_sequence]*l       
        #temp_dict_df['hrs_to_firstevent'] = (eventtime - Xp.flatten(),[np.nan]*l)[math.isnan(eventtime)]
        temp_dict_df['code'] = [code]*l
        temp_dict_df['hrs_from_adm'] = np.pad(Xp.flatten(), (s-len_sequence,0), 'constant', constant_values = np.nan) 
        #f len(temp_dict_df['hrs_to_firstevent']) <= len_sequence:
         #   temp_dict_df['hrs_to_firstevent']=np.pad(temp_dict_df['hrs_to_firstevent'], (s-len(temp_dict_df['hrs_to_firstevent']),0), 'constant', constant_values=np.nan)
        
        
        X=list(X.flatten())
        Y=list(Y.flatten())
        Xp =list( Xp.flatten())
        #if len(X) == 1:
         #   X.insert(0, max(X)-1)            
          #  Y.insert(0, prior_mean)
        
        #if min(X) > min(Xp):
        #    X.insert(0, min(Xp))            
        #    Y.insert(0, prior_mean)
        
        #if max(Xp) > max(X):
        #    X = np.array(X)
        #    X = np.append(X, np.array([xmax]))
         #   Y = np.array(Y)
          #  Y = np.append(Y, np.array([prior_mean]))
        
        # Option 2: interpolation (carry last value forward)
       # f1 = interp1d(X, Y, kind='previous')
        #Y_prev = f1(Xp)
        #temp_dict_df[v+'_prev'] = np.pad(Y_prev, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        # Option 3:  interpolation (nearest)
        #f2 = interp1d(X, Y, kind='nearest')
        #Y_nearest = f2(Xp)
        #temp_dict_df[v+'_nearest'] = np.pad(Y_nearest, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        # Option 4: interpolation (linear (cubic))
        #f3 = interp1d(X, Y, kind='linear')
        #Y_cubic = f3(Xp)
        #temp_dict_df[v+'_linear'] = np.pad(Y_cubic, (s-len_sequence,0), 'constant', constant_values= np.nan)
        
        #if (v == 'avpu' ) | (v=='supplemental_oxygen'):
         #   temp_dict_df[v+'_linear'] = np.round( temp_dict_df[v+'_linear'], 0)
        
        # 10. Compute MSE over training points
        #y_pred = m.predict(X)
        #rmse = mean_squared_error(Y.flatten(), y_pred[0])**(1/2)        
        #mll = (0.5*np.log(2*np.pi*(y_pred[1]**2)) + (Y.flatten()-y_pred[0])**2/(2*(y_pred[1]**2))).sum()/len(Y.flatten())

    #keras.backend.clear_session()
    df_temp = pd.DataFrame(temp_dict_df)
    samples_df = samples_df.append(df_temp)
    return samples_df 
