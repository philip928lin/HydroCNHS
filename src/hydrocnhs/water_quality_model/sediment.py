import numpy as np
import pandas as pd

class Sediment:
    def __init__(self, prec, Q_frac, model_dict, dam_agts_TSS_dict=None):
        """
        Parameters
        ----------
        prec : dict
            Precipitation data of all subbasins over time (t) [cm].
        Q_frac : dict
            Fraction of flow of all outlets over time (t).
        model_dict : dict
            Model dictionary.
        dam_agts_TSS_dict : dict, optional
            . A dictionary contains dam agents' monthly TSS data in pd.DataFrame
        """
        self.prec = prec
        model_dict = model_dict
        sediment_setting = model_dict["Sediment"]
        dam_agts = model_dict["SystemParsedData"]["DamAgents"]
        sim_seq = model_dict["SystemParsedData"]["SimSeq"]
        
        self.sediment_setting = sediment_setting
        self.dam_agts = dam_agts
        self.sim_seq = sim_seq
        # Annual cycle start month
        # If the annual cycle start month is None, we use 12 month moving window to calculate the sediment transportation ratio.
        # If it is given, e.g., =4, SX of April will be distrbited from April to March of the next year.
        # SX of May will be distrbited from May to March of the next year.
        # .....
        # SX of March will be distrbited only to March. The month that accumulated sed being flushed out.
        # This month is ususally identify by look into the observation data to find the month that has the highest sediment yield.
        # Then, use the next month as the annual cycle start month.
        self.annual_cycle_start_month = model_dict["WaterSystem"]["Sediment"].get("StartMonth")
        
        if dam_agts is not None:
            assert dam_agts_TSS_dict is not None, "Dam agents' monthly TSS data is required."
        else:
            self.dam_agts = []
        self.dam_agts_TSS_dict = dam_agts_TSS_dict
        
        self.hydro_start_date = pd.to_datetime(model_dict["WaterSystem"]["StartDate"], format="%Y/%m/%d")
        self.hydro_end_date = pd.to_datetime(model_dict["WaterSystem"]["EndDate"], format="%Y/%m/%d")
        self.pd_hydro_dates = pd.date_range(
            start=self.hydro_start_date, 
            end=self.hydro_end_date, 
            freq="D"
            )
        self.max_sed_end_date = self.hydro_end_date - pd.DateOffset(years=1)
            
        self.Q_frac_m = {
            ro: pd.DataFrame(sbs, index=self.pd_hydro_dates).resample("MS").sum() \
            for ro, sbs in Q_frac.items()
            }
            
        # Transportation capacity (we precalculate this to save time)
        self.Q_frac_m_sq = Sediment.cal_Q_frac_m_sq(self.Q_frac_m, self.sediment_setting)
        
        # Sediment yield record (monthly X subbasins)
        self.SX_record = pd.DataFrame(
            0.0,  # Fill value
            index=self.pd_hydro_dates, 
            columns=list(sediment_setting.keys()) + list(self.dam_agts)
            )
        # SX is stored in monthly resolution but caluclated in daily resolution
        self.SX_record = self.SX_record.resample("MS").sum() 
        
        self.simulated_start_date_tracker = None
        self.simulated_end_date_tracker = self.hydro_start_date
        self.TSS_M = self.SX_record.copy()
        self.TSS_Y = self.TSS_M.resample("YS").sum()
    
    def get_monthly_TSS(self):
        return self.TSS_M.loc[
            self.simulated_start_date_tracker:self.simulated_end_date_tracker, 
            self.sim_seq
            ]
    
    def get_yearly_TSS(self):
        return self.TSS_Y.loc[
            self.simulated_start_date_tracker:self.simulated_end_date_tracker, 
            self.sim_seq
            ]
    
    def set_sed_dates(self, sed_start_date, sed_end_date):
        """Set sediment simulation dates."""
        if sed_start_date is None:
            if self.simulated_end_date_tracker == self.hydro_start_date:
                self.sed_start_date = self.hydro_start_date
            else:
                sed_start_date=self.simulated_end_date_tracker + pd.DateOffset(months=1)
        else:
            self.sed_start_date = pd.to_datetime(sed_start_date, format="%Y/%m/%d")
            assert self.sed_start_date >= self.hydro_start_date, \
            "Sediment simulation start date must be after hydrological simulation start date."
        
        if self.simulated_start_date_tracker is None:
            self.simulated_start_date_tracker = self.sed_start_date
        
        if sed_end_date is None:
            self.sed_end_date = self.max_sed_end_date
        else:
            self.sed_end_date = pd.to_datetime(sed_end_date, format="%Y/%m/%d")
            assert self.sed_end_date <= self.max_sed_end_date, \
            "Sediment simulation end date must be 1 year before hydrological simulation end date." 
        
        self.pd_sed_dates = pd.date_range(
            start=self.sed_start_date, 
            end=self.sed_end_date, 
            freq="D"
            )
        
        self.pd_sed_dates_m = pd.date_range(
            start=f"{self.sed_start_date.year}/{self.sed_start_date.month}", 
            end=f"{self.sed_end_date.year}/{self.sed_end_date.month}", 
            freq="MS"
        )    
        
        self.from_index = self.pd_hydro_dates.get_loc(self.sed_start_date)
        self.to_index = self.pd_hydro_dates.get_loc(self.sed_end_date)     
    
    def run_TSS(self, sed_start_date=None, sed_end_date=None):
        """
        Run sediment transport simulation.
        
        Parameters
        ----------
        sed_start_date : str, optional
            Start date of sediment simulation. The default is None.
        sed_end_date : str, optional
            End date of sediment simulation. The default is None.
            
        Returns
        -------
        SX_routed_M : pd.DataFrame
            Monthly sediment yield of all outlets.
        SX_routed_Y : pd.DataFrame
            Yearly sediment yield of all outlets.
        """
        self.set_sed_dates(sed_start_date, sed_end_date)
        from_index = self.from_index
        to_index = self.to_index
        pd_sed_dates = self.pd_sed_dates
        pd_sed_dates_m = self.pd_sed_dates_m
        prec_sed = {sb: v[from_index:to_index+1] for sb, v in self.prec.items()}
        
        sediment_setting = self.sediment_setting
        
        # Calculate rainfall erosivity for each subbasin
        RE_dict = {}
        for sb, v in sediment_setting.items():
            pars = v["Pars"]
            inputs = v["Inputs"]
            RE_dict[sb] = Sediment.cal_RE(
                prec_sb=prec_sed[sb],
                cool_months=inputs["CoolMonths"], 
                ac=pars["Ac"], 
                aw=pars["Aw"], 
                sa=pars["Sa"],
                sb=pars["Sb"], 
                pd_sed_dates=pd_sed_dates
                )

        # Calculate sediment supply for each subbasin
        # Note that the order of the outlets in SX is the same as the order of the
        SX_df_dict = {}
        for sb, v in sediment_setting.items():
            inputs = v["Inputs"]
            pars = v["Pars"]
            RE = RE_dict[sb]
            LS = pars["LS"]
            CP = pars["CP"]
            # Expand CP to the length of LS if assuming all area segments have the same CP
            if isinstance(CP, (np.ndarray, np.generic, list)) is False:
                CP = [CP]*len(LS)
            K = pars["K"]
            Areas = inputs["Areas"]
            X = Sediment.cal_usle(RE, K, CP, LS, Areas)
            DR = pars["DR"]
            SX_sb = Sediment.cal_SX(
                DR=DR, X=X, pd_sed_dates=pd_sed_dates
                )
            SX_df_dict[sb] = SX_sb
        
        # Add DamAgts' month sediment yield 
        # User has to mannually assign monthly sediment yield for each dam agent. We currently 
        # do not have a way to calculate this.
        dam_agts_TSS_dict = self.dam_agts_TSS_dict # a dictionary of monthly TSS data for each dam agent
        dam_agts = self.dam_agts
        for agt in dam_agts:
            SX_df_dict[agt] = dam_agts_TSS_dict.loc[pd_sed_dates_m, agt]
        
        SX = pd.concat(SX_df_dict, axis=1) # t (month) x sb

        # add to the records
        self.SX_record.loc[SX.index, SX.columns] = SX
        
        # Retrive the SX records since the past 11 months SX will contribute to the current month
        # sediment yield.
        annual_cycle_start_month = self.annual_cycle_start_month
        Q_frac_m_sq = self.Q_frac_m_sq
        ind = self.SX_record.index.get_loc(SX.index[0])
        num_of_back_track_months = ind - max(ind-11, 0)
        SX_routed_M = self.SX_record.iloc[ind-num_of_back_track_months:ind+SX.shape[0], :].copy()
        for ro in self.sim_seq:
            l = SX_routed_M.shape[0]
            Q_frac_m_sq_ro = Q_frac_m_sq[ro] # a df of monthly Q_frac_sq
            ind = Q_frac_m_sq_ro.index.get_loc(SX_routed_M.index[0])
            for sb in Q_frac_m_sq_ro.columns:
                ratios = np.zeros((l, l+11))
                for i, v in enumerate(list(SX_routed_M.index)):
                    
                    Q = Q_frac_m_sq_ro[sb][ind:ind+12]
                    if annual_cycle_start_month is not None:
                        acm = annual_cycle_start_month
                        if acm != Q.index[0].month:
                            annual_cycle_mask_i = [i.month for i in Q.index].index(acm)
                            Q = Q.iloc[:annual_cycle_mask_i]
    
                    Q = Q.values/np.sum(Q)
                    ratios[i, i:i+len(Q)] = Q
                SX_routed_M_array = SX_routed_M[sb].values.reshape(-1,1)
                sb_sed_routed = np.multiply(SX_routed_M_array, ratios).sum(axis=0)
                # The last 11 months extend to the future, which will be descarded
                # in this calculation. They will be re-routed when moving into 
                # the next sediment simulation period.
                SX_routed_M[sb] = sb_sed_routed[:-11] 
                
            # Sum the subbasin sediment yield to get the routing outlet's sediment yield
            SX_routed_M[ro] = SX_routed_M[Q_frac_m_sq_ro.columns].sum(axis=1) 
        
        # Collect the routed dates within pd_sed_dates_m
        SX_routed_M = SX_routed_M.loc[pd_sed_dates_m, :]
        
        # Update the records
        self.TSS_M.loc[SX_routed_M.index, SX_routed_M.columns] = SX_routed_M
        SX_routed_Y = SX_routed_M.resample("YS").sum()
        self.TSS_Y.loc[SX_routed_Y.index, SX_routed_Y.columns] = SX_routed_Y

        # Update the simulated date tracker
        self.simulated_end_date_tracker = pd_sed_dates[-1]
        
        return SX_routed_M, SX_routed_Y
    
    def run_next_year_TSS(self):
        """Run sediment transport simulation for the next year."""
        if self.simulated_end_date_tracker == self.hydro_start_date:
            SX_routed_M, SX_routed_Y = self.run_TSS(
                sed_start_date=self.simulated_end_date_tracker,
                sed_end_date=self.simulated_end_date_tracker + pd.DateOffset(months=11)
            )
        else:
            SX_routed_M, SX_routed_Y = self.run_TSS(
                sed_start_date=self.simulated_end_date_tracker + pd.DateOffset(months=1),
                sed_end_date=self.simulated_end_date_tracker + pd.DateOffset(months=12)
            )
        return SX_routed_M, SX_routed_Y
    
    @staticmethod
    def cal_Q_frac_m_sq(Q_frac_m, sediment_setting):
        Q_frac_m_sq = {}
        for ro, df in Q_frac_m.items():
            for sb in df.columns:
                df[sb] = df[sb]**sediment_setting[sb]["Pars"]["Sq"]
            Q_frac_m_sq[ro] = df
        return Q_frac_m_sq
    
    @staticmethod
    def cal_usle(RE, K, CP, LS, Areas):
        """Calculate the erosion using USLE.

        Parameters
        ----------
        RE : 1darray
            Rainfall erosivity over time (t).
        K : 1darray
            Soil erodibility factor of each area segment (k).
        CP : 1darray
            Vegetation management factor = cover and management factor * support
            practice factor of each area segment (k).
        LS : 1darray
            Topographic factor of each area segment (k).
        Areas : 1darray
            [ha] Area of each area segment (k).

        Returns
        -------
        2darray
            [Mg] Daily sediment supply
        """
        RE = np.array(RE).reshape((-1,1))
        K = np.array(K).reshape((-1,1))
        CP = np.array(CP).reshape((-1,1))
        LS = np.array(LS).reshape((-1,1))
        Areas = np.array(Areas).reshape((-1,1))
        X = 0.132 * np.dot(K*CP*LS*Areas, RE.T)
        return X    # area segement x time, k x t
    
    @staticmethod
    def cal_RE(prec_sb, cool_months, ac, aw, sa, sb, pd_sed_dates):
        """
        Calculate rainfall erosibility with Sa, Sb calibrated parameters for a subbasin.
        
        Parameters
        ----------
        prec : 1darray
            Precipitation of a subbasin over time (t) [cm].
        cool_months : list
            List of cool months.
        ac : float
            Erosibility factor of cool months.
        aw : float
            Erosibility factor of warm months.
        sa : float
            Erosivity factor.
        sb : float
            Erodibility factor.
        pd_sed_dates : pd.Datetime
            Pandas datetime list.
        
        Returns
        -------
        1darray
            Daily rainfall erosibility over time (t).
        """
        cm = cool_months
        ac_ab = np.array([ac if d.month in cm else aw for d in pd_sed_dates])
        RE = sa * np.array(prec_sb)**sb * ac_ab
        return RE
    
    @staticmethod
    def cal_SX(DR, X, pd_sed_dates):
        """Calculate total sediment supply, SX, of subbasins.

        Parameters
        ----------
        DR : 1darray
            Delivery ratio of each subbasin.
        X : 2darray
            Sediment supply.
        pd_sed_dates : pd.Datetime
            Pandas datetime index.

        Returns
        -------
        1darray
            [Mg] Monthly SX
        """
        DR = np.array(DR)
        SX = DR * pd.DataFrame(X.T, index=pd_sed_dates).sum(axis=1).resample("MS").sum()
        #return SX.to_numpy(), SX.index # 1 x t (monthly)
        return SX
    
    @staticmethod
    def cal_LS(SL, PS):
        """Calculate topographic factor, LS, for all area segments in a subbasin.

        Parameters
        ----------
        SL : 1darray
            Slope length of area segements (k).
        PS : 1darray
            Percent slope of area segements (k).

        Returns
        -------
        1darray
            LS
        """
        SL = np.array(SL)
        PS = np.array(PS)
        if PS >= 5:
            b = 0.5
        elif PS < 5 and PS > 3:
            b = 0.4
        elif PS <= 3 and PS >= 1:
            b = 0.3
        elif PS <1:
            b = 0.2
        theta = np.arctan(PS/100)
        LS = (0.045*SL)**b * (65.41*np.sin(theta)**2 + 4.56*np.sin(theta) + 0.065)
        return LS   # k x 1
    