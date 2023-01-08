from variable_annuity_processor import variableAnnuity
import pandas as pd

if __name__ == '__main__':
    print("********* RUN: Executing the variable annuity calculator ********* ")
    VA = variableAnnuity(file_name='Test VA Withdrawal.xlsx',
                         inputs_max_row=15,
                         output_col_names=['Year', 'Anniversary', 'Age', 'Contribution', 'AV Pre-Fee',
                                           'Pre-Fee by Fund', 'M&E/Fund Fees', 'AV Pre-Withdrawal',
                                           'Pre-Withdrawal by Fund', 'Withdrawal Amount', 'AV Post-Withdrawal',
                                           'Post-Withdrawal by Fund', 'Rider Charge', 'AV Post-Charges',
                                           'Post-Charges by Fund', 'Death Payments', 'AV Post-Death Claims',
                                           'Post-Death Claims by Fund', 'Post-Rebalance by Fund', 'ROP Death Base',
                                           'NAR Death Claims', 'Death Benefit Base', 'Withdrawal Base',
                                           'Cumulative Withdrawal', 'Maximum Annual Withdrawal',
                                           'Maximum Annual Withdrawal Rate', 'Eligible Step-Up', 'Growth Phase',
                                           'Withdrawal Phase', 'Automatic Periodic Benefit Status', 'Last Death',
                                           'Rebalance Indicator', 'DF', 'Withdrawal Claims'],
                         list_pvs_names=['PV_DB_Claim', 'PV_WB_Claim', 'PV_RC'])
    # To structure the investments in the case that we want to model with more than just the two-fund approach,
    # this attribution of a returns generator for a risk-free (i.e., fixed income fund) and random-normal (i.e., equity-
    # like fund, other distributions could be added to the returns' generator, and multivariate properties too).
    # ASSUMPTIONS (room for future improvements to this calculator):
    # Currently only modelling two-funds but as per the policy (see page 2 of the PDF)
    # Assumes
    fund_characteristics = {'Fund1': {'Generator': 'risk_free', 'Start Allocation': 0.16},
                            'Fund2': {'Generator': 'random_normal', 'Start Allocation': 0.64}}
    # Choose whether to run test, uses fixed fund returns
    run_test = True
    # STEP 1: use a simple data generating process for returns with a random and fixed return process by fund
    VA.generate_fund_returns(generate_funds=fund_characteristics,
                             years=40,
                             from_excel=run_test)

    # STEP 2: run the iterative and vectorized when possible calculations for the actuarial values of the annuity
    # policy given the returns generated in the prior method
    VA.incremental_actuarial_cfs(plan_beneficiary={'Anniversary': pd.to_datetime('8/01/2016').date(), 'Start Age': 60,
                                                   'Annual Contribution': 0},
                                 fund_characteristics=fund_characteristics,
                                 years=40)

    # STEP 3: calculate the present value of the Death Benefit Claims, Withdrawal Claims and Rider Charges
    VA.pv_actuarial_cfs()

    print("********* RUN: Completed the run of the variable annuity calculator ********* ")
    # STEP 4: produce the Excel output and check outputs if run_test is set to True
    print("********* TEST: Produce the output and check for any differences with the Excel benchmark ********* ")
    VA.create_check_output_table(check_outputs=run_test,
                                 benchmark_output_filename='Test VA Withdrawal_hardcodeFundReturns_fix.xlsx')
