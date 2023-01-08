import pandas as pd
import numpy as np
from tools import create_df, create_dict, divide_by


class variableAnnuity:
    def __init__(self, file_name=None, inputs_max_row=None, output_col_names=None, list_pvs_names=None):
        if file_name is not None:
            self.input_df = pd.read_excel(file_name, sheet_name='Main', header=None)
            self.decrement_assumptions = pd.read_excel(file_name, sheet_name='DecrermentAssumptions', header=None)
            self.actuarial_assumptions = create_dict(self.input_df, keys_list=['Initial Premium',
                                                                               'First Withdrawal Age',
                                                                               'Annuity Commencement Date/Age',
                                                                               'Last Death Age',
                                                                               'Mortality',
                                                                               'Withdrawal Rate',
                                                                               'Fixed Allocation Funds '
                                                                               'Automatic Rebalancing Target'])
            self.product_properties = create_dict(self.input_df, keys_list=['Product',
                                                                            'Step-Up',
                                                                            'Step-Up Period (Contract Years)',
                                                                            'Rider Charge'])
            self.fees = create_dict(self.input_df, keys_list=['M&E', 'Fund Fees'])
            self.return_assumptions = create_dict(self.input_df, keys_list=['Risk Free Rate', 'Volatility'])
            self.max_withdrawal = create_df(self.input_df, columns_list=['Ages (>=)', 'Maximum Annual Withdrawal'],
                                            max_row=inputs_max_row)
            self.mortality_table = create_df(self.decrement_assumptions, columns_list=['Age', 'Mortality Rate'])
            self.lapse_table = create_df(self.decrement_assumptions, columns_list=['Dur', 'ING LifePay Plus Base'])
        # initialize the outputs to be filled by methods of this class
        self.fund_returns = None
        self.actuarial_cfs = pd.DataFrame(columns=output_col_names)
        self.actuarial_pv = pd.DataFrame(columns=list_pvs_names)

    def generate_fund_returns(self, generate_funds, years, from_excel):
        if from_excel:
            self.fund_returns = pd.read_excel('hardcoded_fundReturns.xlsx', index_col='Year')
        else:
            list_years = range(0, years + 1, 1)
            temp_fund_returns = pd.DataFrame(index=list_years, columns=generate_funds.keys())
            temp_fund_returns.index.name = 'Year'
            # for each fund generate either 'risk_free' or 'random_normal' returns and fill the appropriate column
            for i_col in temp_fund_returns.columns:
                i_val_generator = generate_funds[i_col]['Generator']
                if i_val_generator == 'risk_free':
                    temp_fund_returns.loc[1:, i_col] = self.return_assumptions['Risk Free Rate']
                elif i_val_generator == 'random_normal':
                    temp_fund_returns.loc[1:, i_col] = np.exp(np.log(1 + self.return_assumptions['Risk Free Rate']) -
                                                              0.5 * self.return_assumptions['Volatility'] ** 2 +
                                                              self.return_assumptions['Volatility'] *
                                                              np.random.normal(loc=0, scale=1, size=years)) - 1

            self.fund_returns = temp_fund_returns.fillna(0)
        return

    def incremental_actuarial_cfs(self, plan_beneficiary, fund_characteristics, years):
        # calculate all fo the rules based columns like Year, Anniversary, and Age
        list_years = list(range(0, years + 1, 1))
        list_age = list(range(plan_beneficiary['Start Age'], plan_beneficiary['Start Age'] + years + 1, 1))
        list_anniversary = pd.date_range(plan_beneficiary['Anniversary'], periods=years + 1, freq='Y')
        self.actuarial_cfs = self.actuarial_cfs.assign(Year=list_years, Anniversary=list_anniversary, Age=list_age)
        # get all the columns that are by Fund and create a new column for each fund defined in the fund returns
        by_fund_cols = self.actuarial_cfs.columns[self.actuarial_cfs.columns.str.contains('by Fund')]
        replace_by_fund_cols = [s.replace(' by Fund', '') for s in by_fund_cols if 'by Fund' in s]
        all_by_fund_cols = [' '.join([s1, s2]) for s1 in replace_by_fund_cols for s2 in self.fund_returns.columns]
        self.actuarial_cfs.drop(by_fund_cols, axis=1, inplace=True)
        self.actuarial_cfs[all_by_fund_cols] = 0
        self.actuarial_cfs['AV Pre-Fee'] = 0
        # Sort funds so that risk-free fund(s) rebalancing values so that we rebalance those first
        sorted_fund_char = {k: v for k, v in sorted(fund_characteristics.items(),
                                                    key=lambda x: list(x[1].values())[0], reverse=True)}
        # GENERATE DATA
        # Calculate vectors for Growth Phase, Last Death, Eligible Step Up, Discount Factor (DF),
        # Maximum Annual Withdrawal Rate
        self.actuarial_cfs['Growth Phase'] = np.where((self.actuarial_cfs['Age'] <=
                                                       self.actuarial_assumptions['First Withdrawal Age']) &
                                                      (self.actuarial_cfs['Age'] <=
                                                       self.actuarial_assumptions['Annuity Commencement Date/Age']) &
                                                      (self.actuarial_cfs['Age'] <
                                                       self.actuarial_assumptions['Last Death Age']), 1, 0)
        self.actuarial_cfs['Last Death'] = np.where((self.actuarial_cfs['Age'] >=
                                                     self.actuarial_assumptions['Last Death Age']), 1, 0)
        self.actuarial_cfs['Eligible Step-Up'] = np.where((self.actuarial_cfs['Year'] <=
                                                           self.product_properties['Step-Up Period (Contract Years)']) &
                                                          (self.actuarial_cfs['Growth Phase'] == 1), 1, 0)
        self.actuarial_cfs['DF'] = ((1 + self.return_assumptions['Risk Free Rate']) ** (-self.actuarial_cfs['Year']))
        self.max_withdrawal = self.max_withdrawal.apply(pd.to_numeric)
        self.max_withdrawal.rename(columns={'Maximum Annual Withdrawal': 'Max Withdrawal Rates by Age'}, inplace=True)
        self.actuarial_cfs['Age_float'] = self.actuarial_cfs['Age'].astype('float64')
        self.actuarial_cfs = pd.merge_asof(self.actuarial_cfs, self.max_withdrawal, left_on='Age_float',
                                           right_on='Ages (>=)', direction='backward')
        self.actuarial_cfs['Maximum Annual Withdrawal Rate'] = np.where(self.actuarial_cfs['Growth Phase'] == 1, 0,
                                                                        self.actuarial_cfs[
                                                                            'Max Withdrawal Rates by Age'])
        # iterate over all the rows to generate the values of each column
        for i_row in self.actuarial_cfs.iterrows():
            mask_age_mortality_table = (self.mortality_table['Age'] == i_row[1]['Age'])
            prb_mortality = (1 - self.mortality_table.loc[mask_age_mortality_table, 'Mortality Rate']).item()
            # fill the Pre-Fee columns
            for i_fund in self.fund_returns.columns:
                if i_row[0] == 0:
                    temp_col_val = (self.actuarial_assumptions['Initial Premium'] *
                                    fund_characteristics[i_fund]['Start Allocation'])
                else:
                    temp_col_val = ((self.fund_returns.loc[i_row[1]['Year'], i_fund] + 1) *
                                    lag_i_row['Post-Rebalance ' + i_fund])
                temp_col_name = 'Pre-Fee ' + i_fund
                i_row[1][temp_col_name] = temp_col_val
                i_row[1]['AV Pre-Fee'] = temp_col_val + i_row[1]['AV Pre-Fee']
            if i_row[0] == 0:
                # logic for if row is the first row
                i_row[1][['AV Pre-Withdrawal', 'M&E/Fund Fees', 'Withdrawal Amount', 'Rider Charge', 'Death Payments',
                          'Cumulative Withdrawal', 'Maximum Annual Withdrawal', 'Eligible Step-Up', 'Growth Phase',
                          'Withdrawal Phase', 'Automatic Periodic Benefit Status', 'Rebalance Indicator',
                          'Withdrawal Claims']] = 0
                i_row[1][['ROP Death Base', 'Death Benefit Base', 'Withdrawal Base']] = \
                    self.actuarial_assumptions['Initial Premium']
            else:
                i_row[1]['M&E/Fund Fees'] = lag_i_row['AV Post-Death Claims'] * (self.fees['M&E'] +
                                                                                 self.fees['Fund Fees'])
                i_row[1]['ROP Death Base'] = (prb_mortality * lag_i_row['ROP Death Base'])
                i_row[1]['Withdrawal Phase'] = np.where((((i_row[1]['Age'] >
                                                           self.actuarial_assumptions['First Withdrawal Age']) |
                                                          (i_row[1]['Age'] >
                                                           self.actuarial_assumptions['Annuity Commencement Date/Age']))
                                                         & (i_row[1]['Age'] <
                                                            self.actuarial_assumptions['Last Death Age']) &
                                                         (lag_i_row['AV Post-Death Claims'] > 0)), 1, 0).item()
                i_row[1]['Automatic Periodic Benefit Status'] = np.where((i_row[1]['Age'] >=
                                                                          self.actuarial_assumptions['Last Death Age']),
                                                                         0,
                                                                         np.where((lag_i_row['Withdrawal Phase'] == 1)
                                                                                  &
                                                                                  (lag_i_row['AV Post-Death Claims']
                                                                                   == 0), 1, lag_i_row[
                                                                                      'Automatic Periodic Benefit '
                                                                                      'Status'])).item()
                i_row[1]['Rebalance Indicator'] = (i_row[1]['Withdrawal Phase'] +
                                                   i_row[1]['Automatic Periodic Benefit Status'])
                i_row[1]['Death Payments'] = 0 if (i_row[1][['Rebalance Indicator', 'Growth Phase', 'Last Death']].sum()
                                                   == 0) else (lag_i_row[['ROP Death Base', 'Death Benefit Base']].max()
                                                               * self.mortality_table.loc[mask_age_mortality_table,
                        'Mortality Rate']).item()

            i_row[1]['AV Pre-Withdrawal'] = (plan_beneficiary['Annual Contribution'] + i_row[1]['AV Pre-Fee'] -
                                             i_row[1]['M&E/Fund Fees'])
            # todo make sure to review the logic so that withdrawals can only happen during the withdrawal phase, i.o.w.
            # todo dependencies from other phases are of no importance when we are not in the withdrawal phase
            if i_row[0] != 0:
                i_row[1]['Withdrawal Base'] = max(
                    i_row[1]['AV Post-Death Claims'] if i_row[1]['Growth Phase'] == 1 else 0,
                    lag_i_row['Withdrawal Base'] * prb_mortality + plan_beneficiary['Annual Contribution'],
                    lag_i_row['Withdrawal Base'] * prb_mortality * (1 + self.product_properties['Step-Up']) +
                    plan_beneficiary['Annual Contribution'] - i_row[1]['M&E/Fund Fees'] - i_row[1]['Rider Charge']
                    if i_row[1]['Eligible Step-Up'] == 1 else 0)
                i_row[1]['Maximum Annual Withdrawal'] = (i_row[1]['Maximum Annual Withdrawal Rate'] *
                                                         i_row[1]['Withdrawal Base'])
                i_row[1]['Withdrawal Amount'] = ((i_row[1]['Withdrawal Base'] *
                                                  self.actuarial_assumptions['Withdrawal Rate'])
                                                 if i_row[1]['Withdrawal Phase'] == 1 else (
                    i_row[1]['Maximum Annual Withdrawal'] if i_row[1]['Automatic Periodic Benefit Status'] == 1 else 0))

            i_row[1]['AV Post-Withdrawal'] = max(0, i_row[1]['AV Pre-Withdrawal'] - i_row[1]['Withdrawal Amount'])

            if i_row[0] != 0:
                i_row[1]['Rider Charge'] = (self.product_properties['Rider Charge'] * i_row[1]['AV Post-Withdrawal'])
                # todo dependent on withdrawal amount of prior period
                i_row[1]['Death Benefit Base'] = max(0, (lag_i_row['Death Benefit Base'] * prb_mortality) +
                                                     plan_beneficiary['Annual Contribution'] - i_row[1][
                                                         'M&E/Fund Fees'] - lag_i_row['Withdrawal Amount'] -
                                                     i_row[1]['Rider Charge'])

            i_row[1]['AV Post-Charges'] = i_row[1]['AV Post-Withdrawal'] - i_row[1]['Rider Charge']
            i_row[1]['AV Post-Death Claims'] = max(i_row[1]['AV Post-Charges'] - i_row[1]['Death Payments'], 0)
            i_row[1]['NAR Death Claims'] = max(-i_row[1]['AV Post-Death Claims'], 0)

            if i_row[0] != 0:
                i_row[1]['Withdrawal Base'] = max(
                    i_row[1]['AV Post-Death Claims'] if i_row[1]['Growth Phase'] == 1 else 0,
                    lag_i_row['Withdrawal Base'] * prb_mortality + plan_beneficiary['Annual Contribution'],
                    lag_i_row['Withdrawal Base'] * prb_mortality * (1 + self.product_properties['Step-Up']) +
                    plan_beneficiary['Annual Contribution'] - i_row[1]['M&E/Fund Fees'] - i_row[1]['Rider Charge']
                    if i_row[1]['Eligible Step-Up'] == 1 else 0)
                i_row[1]['Maximum Annual Withdrawal'] = (i_row[1]['Maximum Annual Withdrawal Rate'] *
                                                         i_row[1]['Withdrawal Base'])
                i_row[1]['Withdrawal Amount'] = ((i_row[1]['Withdrawal Base'] *
                                                  self.actuarial_assumptions['Withdrawal Rate'])
                                                 if i_row[1]['Withdrawal Phase'] == 1 else (
                    i_row[1]['Maximum Annual Withdrawal'] if i_row[1]['Automatic Periodic Benefit Status'] == 1 else 0))
                i_row[1]['Cumulative Withdrawal'] = i_row[1]['Withdrawal Amount'] + lag_i_row['Cumulative Withdrawal']
                i_row[1]['Withdrawal Claims'] = max(i_row[1]['Withdrawal Amount'] - lag_i_row['AV Post-Death Claims'],
                                                    0)
            i_row[1]['AV Post-Charges'] = i_row[1]['AV Post-Withdrawal'] - i_row[1]['Rider Charge']
            i_row[1]['AV Post-Death Claims'] = max(i_row[1]['AV Post-Charges'] - i_row[1]['Death Payments'], 0)
            i_row[1]['NAR Death Claims'] = max(- i_row[1]['AV Post-Charges'] + i_row[1]['Death Payments'], 0)

            # Calculate by Fund the: Pre-Withdrawal, Post-Withdrawal, Post-Charges, Post-Death Claims,
            # and Post-Rebalancing values, need to get the next period fund balances
            temp_val_risk_free = 0
            for i_fund in sorted_fund_char.keys():
                temp_val_pre_withdrawal = i_row[1]['Pre-Fee ' + i_fund] * divide_by(i_row[1]['AV Pre-Withdrawal'],
                                                                                    i_row[1]['AV Pre-Fee'])
                temp_val_post_withdrawal = temp_val_pre_withdrawal * divide_by(i_row[1]['AV Post-Withdrawal'],
                                                                               i_row[1]['AV Pre-Withdrawal'])
                temp_val_post_charges = temp_val_post_withdrawal * divide_by(i_row[1]['AV Post-Charges'],
                                                                             i_row[1]['AV Post-Withdrawal'])
                temp_val_post_death = temp_val_post_charges * divide_by(i_row[1]['AV Post-Death Claims'],
                                                                        i_row[1]['AV Post-Charges'])
                # According to the policy (see section on Fixed Allocation Funds Automatic Rebalancing),
                # any Death Payments should come off of the risk-free fund allocations. Since I added the capability of
                # having multiple risk-free investment funds (as well as risk-on investment funds) the logic here is
                # accommodates this modular feature.
                # todo need to add the feature for multiple funds of each bucket (i.e. Generator type)
                if fund_characteristics[i_fund]['Generator'] == 'risk_free':
                    temp_val_post_rebalance = ((i_row[1]['AV Post-Death Claims'] *
                                                self.actuarial_assumptions['Fixed Allocation Funds Automatic '
                                                                           'Rebalancing Target']) if
                                               (i_row[1]['Rebalance Indicator'] == 1) else temp_val_post_death)
                    temp_val_risk_free = temp_val_risk_free + temp_val_post_rebalance
                elif fund_characteristics[i_fund]['Generator'] == 'random_normal':
                    # assumes only two funds, one of risk-free (fixed-income like) and the other is random_normal or
                    # risk-on (equity-like)
                    temp_val_post_rebalance = i_row[1]['AV Post-Charges'] - temp_val_risk_free

                list_col_name_by_fund = ['Pre-Withdrawal', 'Post-Withdrawal', 'Post-Charges', 'Post-Death Claims',
                                         'Post-Rebalance']
                list_col_name_by_fund = [' '.join([s1, i_fund]) for s1 in list_col_name_by_fund]
                i_row[1][list_col_name_by_fund] = [temp_val_pre_withdrawal, temp_val_post_withdrawal,
                                                   temp_val_post_charges, temp_val_post_death, temp_val_post_rebalance]
            lag_i_row = i_row[1]
            self.actuarial_cfs.loc[i_row[0], :] = i_row[1]

    def pv_actuarial_cfs(self):
        # calculate the present values
        temp_pv_db_claim = np.multiply.reduce([self.actuarial_cfs['DF'].values,
                                               self.actuarial_cfs['NAR Death Claims'].values]).sum()
        temp_pv_wb_claim = np.multiply.reduce([self.actuarial_cfs['DF'].values,
                                               self.actuarial_cfs['Withdrawal Claims'].values]).sum()
        temp_pv_rc = np.multiply.reduce([self.actuarial_cfs['DF'].values,
                                         self.actuarial_cfs['Rider Charge'].values]).sum()
        self.actuarial_pv.loc[len(self.actuarial_pv)] = [temp_pv_db_claim, temp_pv_wb_claim, temp_pv_rc]

    def create_check_output_table(self, check_outputs=False,
                                  benchmark_output_filename='Test VA Withdrawal_hardcodeFundReturns.xlsx',
                                  round_decimals=2):
        if check_outputs:
            # check outputs with the benchmark output
            excel_benchmark = pd.read_excel(benchmark_output_filename, header=None)
            bench_df = create_df(df=excel_benchmark, columns_list=self.actuarial_cfs.columns, min_row=14,
                                 max_row=None).fillna(0)
            duplicate_columns = bench_df.columns[bench_df.columns.duplicated()]
            bench_df = bench_df.drop(columns=duplicate_columns).round(decimals=round_decimals).reset_index()
            output_df = self.actuarial_cfs.round(decimals=round_decimals).fillna(0)
            common_columns = [col for col in bench_df.columns if col in output_df.columns]
            common_columns.remove('Anniversary')
            filter_bench_df = bench_df[common_columns]
            filter_output_df = output_df[common_columns].round(decimals=round_decimals)

            # Compare the Excel data with the data frame
            if filter_bench_df.equals(filter_output_df):
                print("********* SUCCESSFUL: The Excel file and the output dataframe are identical *********")
            else:
                print("********* CHECK FOR DIFFERENCES: The Excel file and the output dataframe are different. Here are the "
                      "difference for further examination *********")
                # Compare the dataframes and show only the differences
                differences = filter_bench_df.where(filter_bench_df != filter_output_df).stack()
                differences = differences.reset_index(level=1)
                differences.columns = ['Column', 'Difference']
                print(differences)
        # todo bug in original Excel file since the Age >= was not applied like stated in policy, so for instance at
        #  80yrs the max annual withdrawal rate was 6% instead of 7%, see differences above (saved benchmark fix also)

        # create the Excel output with three sheets: fund returns, actuarial cashflow, and actuarial present values
        writer = pd.ExcelWriter('variable_annuity_output.xlsx', engine='xlsxwriter')
        self.fund_returns.to_excel(writer, sheet_name='Fund Returns')
        self.actuarial_cfs.to_excel(writer, sheet_name='Variable Annuity Cashflows')
        self.actuarial_pv.to_excel(writer, sheet_name='Present Values')
        writer.save()
