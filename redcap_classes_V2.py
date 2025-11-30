
import pandas as pd
import numpy as np
import re
from redcap import Project



class RedcapProcessor:
    def __init__(self, api_url, api_key):
        # Store API connection info
        self.api_url = api_url
        self.api_key = api_key
        self.project = Project(api_url, api_key)

        self.records = {}  # populated later
        self.df = pd.DataFrame()

        # ---- Column replacement ----
        self.replacement_dict = {
            ("patient_id", "record_id"): "StudyID",
            ("demo_age", "bl_age","baseline_age"): "Age",
            ("demo_sex", "bl_sex","baseline_sex"): "Sex",
            ("bmi_calc", "bl_bmi_calc","baseline_bmi","bl_bmi"): "BMI",
            ("bloodwork_hemoglobin", "lp_hemoglobin","blood_work_hemoglobin","lp_hemoglobin","teg_hgb",): "Hemoglobin",
            ("bloodwork_creatinine", "lp_creatinine","teg_creatinine"): "Creatinine",
            ("bloodwork_teg_crt_r", "rteg_crt_rvalue","crt_rvalue","teg_crt_r",): "R_time",
            ("bloodwork_teg_crt_k", "rteg_crt_ktime","crt_ktime","teg_crt_k"): "K_time",
            ("bloodwork_teg_crt_ang", "rteg_crt_aangle","crt_alpha","teg_crt_aangle"): "Alpha_Angle",
            ("bloodwork_teg_crt_ma", "rteg_crt_ma","crt_ma","teg_crt_ma",): "MA",
            ("bloodwork_teg_crt_ly30", "rteg_crt_ly30","crt_ly30","teg_crt_ly30"): "LY30",
            ("bloodwork_teg_crt_act", "rteg_crt_tegact","crt_act","teg_crt_tegact"): "ACT",
            ("bloodwork_teg_adp_agg","pm_adp_aggregation","teg_adp_agg","pm_adp_agg",): "ADP-agg",
            ("bloodwork_teg_adp_inh","pm_adp_inhibition","teg_adp_inh","pm_adp_inh",): "ADP-inh",
            ("bloodwork_teg_adp_ma","pm_adp_ma","teg_adp_ma","pm_adp_ma"): "ADP-ma",
            ("bloodwork_teg_aa_agg","pm_aa_aggregation","teg_aa_agg","pm_aa_agg"): "AA-agg",
            ("bloodwork_teg_aa_inh","pm_aa_inhibition","teg_aa_inh","pm_aa_inh"): "AA-inh",
            ("bloodwork_teg_aa_ma","pm_aa_ma","teg_aa_ma",'pm_aa_ma',): "AA-ma",
            ("blood_rbc",): "blood_rbc",
            ("lab_rteg_timepoint","bloodwork_timepoint","blood_work_timepoint","rteg_timepoint"):'Time',
            ('date_time_injury','adm_injury_date','date_injury'):'Injury_date',
            ('admission_date_time','adm_er_date','adm_date'): "Admission_date", 
            ('surgery_date_time','intra_op_date','intraop_date_surg','postop_dt_surg'):'Surgery_date',
            ('teg_date_time','lab_dt_blood_draw','teg_date','teg_bd_date','time_teg',):'Draw_date',
            ('teg_time','teg_bd_time'):'teg_time', 
            ('teg_time_lab_panel',):'lab_time',
            ('aoota_classification','inj_aoota',):'AO_OTA',
            ('comp_dvt_yn','complication_dvt','outcomes_vte_type___1'):'DVT',
            ('comp_pe_yn','complication_pe','outcomes_vte_type___2'):'PE',
            ('reason_withdrawal','wd_reason','study_wd_reason'): 'Withdrawn',
            ('outcomes_outcome_type___4','complication_death','comp_death_yn'):'comp_death',
            ('diabetes','bl_comorbidity_check___1','bl_comorbidities___1',):'comorbidty_diabetes',
            ('cancer','bl_comorbidity_check___2','bl_comorbidities___2',):'comorbidty_cancer',
            ('cardiovascular','bl_comorbidity_check___3','bl_comorbidities___3',):'comorbidty_cardiovascular',
            ('pulmonary','bl_comorbidity_check___4','bl_comorbidities___4',):'comorbidty_pulmonary',
            ('prior_stroke','bl_comorbidity_check___5','bl_comorbidities___5',):'comorbidty_stroke',
            ('current_smoker',):'comorbidty_current_smoker',
            ('pulmonary_yesno','comp_pulmonary','comp_pulmonary_yn'): 'complication_pulmonary',
            ('cardio_yesno','comp_cardio','comp_cardio_yn',): 'complication_cardiovascular',
            ('infection_yesno','comp_infection','comp_infection_yn',): 'complication_infection',
            ('cas_score',):'CAS',
            ('cas_timepoint',):'cas_timepoint',
        }

        # ---- Timepoint dictionary ----
        self.timepoint_dict = {
            "Admission" : ['Admission', 'Admission/Pre-Op', 'admission', 'Emergency Admission', 'emergency admission', 'admission/pre-operative', 'admission/ pre-operative', 'Admit',
                           'admission/pre-op', 'admisssion', 'pre op/admission', 'admit', 'admission/post-fracture day 1','Admission/ Pre-Operative','Admission/Pre-Operative','Pre op/admission','Admission/Pre-op','Admisssion'],

            "Pre-Op": ['Pre-Op','Pre Op','pre op','PRE OP','Pre Operative','Pre op','Pre-Operative Day',
                    'Pre-OP','pre-op','Pre-operative','Pre-Operative Day 1/OR Day','Pre-Op/OR Day','Preop',
                    'Pre-Operative','Pre-op','1 hour pre-op','1hr pre-op','pre-op 1 hour','pre-op (unsch. day 5)','1hr Pre-Op',
                    'preop 1 hour','ex-fix pod 4/preop','preop', 'Ex-fix POD 4/Preop','PREOP','Pre-Op (unsch. day 5)', '1 Hour Pre-Op','Pre-OP-Stage 2-2','Pre-procedure 1','Pre-procedure 2','4 hr Pre-OP'],

            "POST_OP": ['reaming', 'intraoperative', 'post-operative', 'post-op', '1h post-op', '1 hour post-op', 'post op', '1hr post-op', '1 hour post op','1hr Post-op',
                    '1 hour po', '1 hour post ream', 'post reaming', 'post ream operation 1', 'post ream', '1 hour post-ream', '1 hour post reaming', 'postream',
                    'po reaming', 'post-ream', 'post reaming ','Post-Operative', 'Post-Op', 'Post-operative', 'Post-op', '1 Hour Post-Op', '1 Hour Post Op', 'POST REAMING', 'PO REAMING', '1 hour PO',
                    'Reaming ', 'Post Reaming ','Post-Ream','POSTREAM','Post- REAM', 'Post-REAM', '1hr Post-Op','POST-REAM','Post ream operation 1','1 HOUR POST REAM','1 Hour PO', 'Post Op'],


            "PFD1": ['PFD1','PFD 1','Post Frac/Pre-Op','Post Fracture Day 1','Day 1 post #','Day 1 Post #','postfractureday1',
                     'POSTFRACTUREDAY1','POST FRACTURE DAY 1','Post-Fracture Day 1','post fracture day 1',
                     'Preop/Post fracture Day 1','Post-fracture day 1','Post fracture D1','Post Fracture Day1',
                     'Day 1 - Post Fracture','Post frac Day 1 - Pre-op','Post Frac Day 1','Day 1', 'admission/post-fracture day 1','Post-fracture #1/ Pre-op','postfracture day 1', 'post-fracture day 1'],

            "PFD2": ['PFD2','PFD 2','Day 2 post #','postfractureday2','POSTFRACTUREDAY2','POST FRACTURE DAY 2',
                     'Post Fracture Day 2','Post Frac Day 2','Day 2 Post fracture/Pre-Op','Day 2 Post #','post fracture day 2','Ex-Fix POD 2','Unscheduled post ex-fix','PT-ST1d2'],

            "PFD3": ['Day 3 post #','Post Frac Day 3','Ex-Fix POD 3','PTST1d3'],

            "PFD4": ['Post Frac Day 4'],

            "POD1" : ['post op day 1', 'POD1', 'Day 1 post-op', 'PO Day 1', 'Day 1 Post-Op', 'POD 1', 'POD 1 ', 'Day 1 post op', 'Post Operative Day 1', 'Day 1 Post-op',
                       'Day 1 post o', 'Postoperative Day 1', 'Pod 1', '24h Post-Op', '24hrs post-op', 'post operative day 1', '24hr Post-Op', '24 Hours Post-Op', 
                       '24h post-op', '24h post=op', 'pod 1', 'po day 1', '24 hour po', '24h post op', 'post operative day 1', '24hrs post-op', '24hr post-op', 'post op day 1', 
                       'po day 1/24hrs po', 'pod1', '24 hours po', 'po day1/24hrs po', '24 hours post-op','24h Post=Op', '24h Post-op', 'PO Day1/24hrs PO', 'Post operative Day 1','24 hours PO','24 hour PO','PO Day 1/24hrs PO','Day 1 post Stage 1','Post-Operative 1'],


            "POD2": ['POD 2','POD2','Day 2 post-op','Day 2 Post-Op','Day 2 post op','PO Day 2','Day 2 Post-op',
                    'Post Operative Day 2','Day  2 post-op','48h Post-Op (Discharge)','48h Post-op','48hrs post-op','POD2',
                    'post operative day 2','48hr Post-Op','48 Hours Post-Op','48h post op','48 hours post-op','48hrs post-op',
                    'po day 2/48 hours po','po day 2/48hrs po','48h post-op','48h post-op','48h post-op','po day 2',
                    'post operative day 2','48 hour po','48h post-op (discharge)','pod 2','48hr post-op','pod2','48 hours po', '48h Post-Op','48 hours PO','PO Day 2/48 hours PO','48 hour PO','48hr Post-op','PO Day 2/48hrs PO'],


            "POD3": ['POD3','Day 3 post-op','POD 3','Day 3 Post-Op','PO Day 3','Day 3 post op','Post Operative Day 3',
                    'Day 3 Post-op','Day 3 pot-op','Day 3 PO','POD3','PO Day 3/72hrs PO','72 Hours Post-Op (Discharge)',
                    '72h Post-op','72h post-op','72hr Post-Op','pod 3 and pod 1','72h post-op (discharge)','72 hours post-op (discharge)',
                    '72hrs post-op','72 hour po','72 hours po','pod 3','po day 3','post operative day 3','72 hours post-op','72h post op','pod3','pod 3','po day 3/72hrs po','72h Post-Op (Discharge)',
                    '72h Post-Op', '72 Hours Post-Op', '72 hours PO', '72 hour PO'],


            "POD4": ['POD 4','Day 4 post-op','Day 4 Post-Op','PO Day 4','Day 4 post op','Day 4 Post-op','POD4',
                'post operative day 4','96hr Post-Op','pod 4 and pod 2','96h post-op (discharge)','96h post-op','96h post op', '96 hour PO','PO Day 4/96hrs PO','PO Day 4/92hrs PO',
                'po day 4/96hrs po','po day 4/92hrs po','pod 4','pod4','96h post-op (discharge)','96hr post-op','96 hours po',
                '96 hour po','po day 4','96 hours post-op','postoperative day 4','96hrs post-op','96h Post-Op (Discharge)', '96h Post-Op', '96h Post-op', '96 Hours Post-Op', '96 hours PO', 
                'Post Operative Day 4', 'Postoperative Day 4', '96hr Post-op','Pre-OP Stage 2'],


            "POD5": ['pod 5','POD5','POD 5','Day 5 post-op','PO Day 5','Day 5 Post-Op','Day 5 post op','Day 5 Post-op',
                     'Day 5 po','Post Operative Day 5','POD 5 '],

            "POD6":['pod 3 - surgery 2'],

            "POD7": ['POD7','POD 7','PO Day 7','Post Operative Day 7','pod 7', 'POD7','Day 7 post-op'],

            "Week2": ['2 week','2 Week FU','2 weeks follow up','2-Week','2 weeks','2 Week','2weeks','2 Week F/U',
                        '2 week F/U','2 week follow-up','2-week','2 Week PO','2 week post#','2 weeks ','Week2','2week',
                        '2 week f/u','2 week fu',' 2 week follow up','2 weeks follow up not done','pt was in three hills hospital',
                        '2 weeks follow up','2 weeks','2weeks','2 weeks follow up not done, pt was in three hills hospital',
                        'pt was at the three hills hospital at 2 weeks','2 weeks ','2 week follow up','2week'],


            "Week4": ['4 week','4 Week FU','4 weeks','4 weeks follow up','4-Week','4weeks','4 WEEKS FOLLOW UP',
                '4 weeks f/u','4 week follow up','4 week ','4week','Week4','4 weeks','4 week fu','4week','4 week F/U','Unscheduled 4 Week F/U'],


            "Week6": ['6 week','6 weeks follow up','6 Week FU','6weeks','6 weeks','6-Week','6 Week','6 Week F/U',
                        '6 weeek f/u','6 week f/u','6 week fu', '6 week F/U', '6 Weeek F/U','6-week','6 week post op','6 week follow up', '6 Week Follow Up', '6 weeks '],

                      
            "Month3": ['3 month','3 Month Follow Up','3 months follow up','3 months','3 Month FU','3months',
                       '3 month follow up','3-Month','3 Month F/U','3 Month','3  month','3-month','3 month f/u',
                       'unscheduled 3 months follow up','12 weeks','3 month F/U'],

            "Month6": ['6 months']
        }

         # ---- PRE_OP_medication ----
        self.medications_preop = {
            **dict.fromkeys(
                    ['HPA-001', 'HPA-004', 'HPA-008', 'HPA-009', 'HPA-010',
                    'HPA-012', 'HPA-014', 'HPA-015', 'HPA-016', 'HPA-017', 'HPA-019',
                    'HPA-020', 'HPA-021', 'HPA-022', 'HPA-024', 'HPA-026', 'HPA-028',
                    'HPA-029', 'HPA-030', 'HPA-032', 'HPA-033', 'HPA-035', 'HPA-036',
                    'HPA-038', 'HPA-039', 'HPA-042', 'HPA-043','HPA-048', 'HPA-050' ,
                    'HPA-051','HPA-052','TH-162', 'TH-170', 'TH-198', 'TH-212', 'TH-217', 
                    'TH-225', 'TH-227', 'TH-236', 'TH-240','TH-255', 'TH-262', 'TH-267', 'TH-274', 'TH-284','TH-286',
                    'TH-302', 'TF-121','TF-128', 'TPA-058','TPA-082','TPA-093' ], "DOAC"),
            **dict.fromkeys(['TH-244'], 'Warfarin'),
            **dict.fromkeys(['TH-006','TH-008','TH-011','TH-013','TH-023','TH-025','TH-026','TH-028','TH-031','TH-035','TH-038','TH-041','TH-046','TH-059','TH-066','TH-072','TH-082','TH-086','TH-090','TH-092','TH-093','TH-100',
                             'TH-102','TH-105','TH-116','TH-126','TH-127','TH-128','TH-133','TH-139','TH-185','TH-258','TH-301','TH-305','TF-005','TF-038','TF-059','TF-071','TF-109','TF-121','TF-136','TPA-019', 'TPA-085'],'ASA'),
            **dict.fromkeys(['TH-004','TH-010','TH-075','TH-110'], 'MIS')
        }

         # ---- POST_OP_medication ----
        # TF 042, TF 128 'TPA-010','TPA-058','TPA-082','TPA-093',TPA-100' are on DOAC post-operatively.
        # TF 027, TF 039, TF 042, TF 059, TF 073, TF 120, TF 128,  'TPA-055','TPA-056','TPA-058','TPA-073','TPA-082','TPA-089','TPA-095','TPA-097' are on DOAC at follow-up. 
        self.medications_postop = {
            **dict.fromkeys(['TF-042','TF-128','TF-027','TF-039','TF-059','TF-073','TF-120',
                             
                             'TPA-010','TPA-055','TPA-056','TPA-058','TPA-073','TPA-082','TPA-089','TPA-095','TPA-097','TPA-093','TPA-100',

                             'HPA-001', 'HPA-004', 'HPA-008', 'HPA-009', 'HPA-010',
                             'HPA-012', 'HPA-014', 'HPA-015', 'HPA-016', 'HPA-017', 'HPA-019',
                             'HPA-020', 'HPA-021', 'HPA-022', 'HPA-024', 'HPA-026', 'HPA-028',
                             'HPA-029', 'HPA-030', 'HPA-032', 'HPA-033', 'HPA-035', 'HPA-036',
                             'HPA-038', 'HPA-039', 'HPA-042', 'HPA-043','HPA-048', 'HPA-050' ,
                             'HPA-051','HPA-052','HPA-053'

                             'TH-162','TH-170',' TH-198', 'TH-212', 'TH-217', 'TH-225', 'TH-227', 'TH-236', 'TH-240', 'TH-244','TH-255', 'TH-262', 'TH-267','TH-271' ,'TH-274', 'TH-284', 'TH-302'], "DOAC"),

            **dict.fromkeys(['TH-036','TH-042','TH-043','TH-044','TH-062','TH-068','TH-073','TH-083','TH-135','TH-170','TH-180','TH-215','TH-221','TH-273','TH-299'],'ASA'),
            **dict.fromkeys(['TH-133', 'TH-246', 'TH-258', 'TH-276', 'TH-305', 'TH-259'],'ASA+LMWH'),}



        self.arth_fix = {
            **dict.fromkeys([
                'Hemi-arthroplasty (monopolar, bipolar)/Other',
                'total arthroplasty direct anterior approach depuy pinnacle actis',
                'direct anterior approach: primary right hip arthroplasty - depuy actis',
                'revision arthroplasty, hip, depuy corail',
                'Total Hip Arthroplasty',
                'Hemi-arthroplasty (monopolar, bipolar)',
                'Hemi-arthroplasty'
            ], 'Arthoplasty'),

            **dict.fromkeys([
                'Long cephalomedullary nail',
                'Dynamic Hip Screw/Other',
                'Short cephalomedullary nail',
                'Dynamic Hip Screw',
                'Cannulated Screws',
                'Other',
                'Short cephalomedullary nail/Dynamic Hip Screw',
                'Short cephalomedullary nail/Other',
                'Cannulated Screws/Short cephalomedullary nail'
            ], 'Fixation')
        }
        

        # ---- Columns to be added to demographic or blood analyses
        self.demographic_cols = ['Study','StudyID','Death','Withdrawn','Age','Sex','BMI','Injury_date','Admission_date','Surgery_date','AO_OTA','Treatment','DVT','PE','VTE_type','VTE','comorbidty_diabetes','comorbidty_cancer',
                              'comorbidty_cardiovascular','comorbidty_pulmonary','comorbidty_stroke','complication_pulmonary', 'complication_cardiovascular','complication_infection','Pre_op_med','time_injury_to_surgery_hours',
                              'total_blood_rbc','blood_rbc_yn']
        
        
        self.lab_cols = ['Study','StudyID', 'Time','CAS','cas_timepoint','VTE_type','VTE', 'Hemoglobin', 'Creatinine', 'R_time', 'K_time','Alpha_Angle', 'MA', 'LY30', 'ACT','ADP-agg', 'ADP-inh','ADP-ma','AA-agg','AA-inh','AA-ma',  'Draw_date_lab', 'Draw_date_teg','Pre_op_med','injury_to_lab_hrs']
        
        # Placeholder for processed DataFrame
        self.df = None

    # ----------------------------------------------------------
    # MASTER FUNCTION (calls all modular steps)
    # ----------------------------------------------------------
    def fetch_and_process(self):
        self._fetch_records()
        self._clean_data()
        self._drop_irrelevant_columns()
        self._replace_column_names()
        self._clean_studyids()
        self._filter_studies()
        self._filter_screening_status()
        self._process_vte_flags()
        self._replace_missing_values()
        self._process_comorbidities_complications()
        self._assign_timepoints()
        # # self._compute_cas()
        # self._fill_missing_values()
        self._build_records()

        return self.df

    # ----------------------------------------------------------
    # STEP 1: Fetch records
    # ----------------------------------------------------------
    def _fetch_records(self):
        records_data = self.project.export_records(raw_or_label='label')
        self.df = pd.DataFrame(records_data)
        print('Step1')
        

    # ----------------------------------------------------------
    # STEP 2: Clean basic data
    # ----------------------------------------------------------
    def _clean_data(self):
        # Replace empty strings with NaN
        self.df = self.df.replace(r'^\s*$', np.nan, regex=True)
        print('Step2')

    # ----------------------------------------------------------
    # STEP 3: Drop irrelevant columns
    # ----------------------------------------------------------
    def _drop_irrelevant_columns(self):

        if 'comp_cardio_yn' in self.df.columns and 'comp_cardio' in self.df.columns:
            self.df  = self.df.drop(columns=['comp_cardio'])

        if 'comp_pulmonary_yn' in self.df.columns and 'comp_pulmonary' in self.df.columns:
            self.df = self.df.drop(columns=['comp_pulmonary_yn'])

        if 'comp_infection_yn' in self.df.columns and 'comp_infection' in self.df.columns:
            self.df = self.df.drop(columns=['comp_infection_yn'])

        if 'screen_patient_id' in self.df.columns:
            self.df = self.df.rename(columns={'screen_patient_id': 'StudyID', 'record_id': 'index'})
            print('We have index')

        print('Step3')

    # ----------------------------------------------------------
    # STEP 4: Replace columns with replacement dictionary
    # ----------------------------------------------------------
    def _replace_column_names(self):
        col_mapping = {}
        for keys, standard_name in self.replacement_dict.items():
            for k in keys:
                if k in self.df.columns:
                    col_mapping[k] = standard_name
        self.df = self.df.rename(columns=col_mapping)
        print("Step4")

    # ----------------------------------------------------------
    # STEP 5: Clean StudyID, handle withdrawals, ffill/bfill
    # ----------------------------------------------------------
    def _clean_studyids(self):
        if 'StudyID' not in self.df.columns:
            return
        
        self.df = self.df.replace({'Participant Withdrawn': np.nan})
        self.df['StudyID'] = self.df['StudyID'].astype(str).str.strip()
        print('Step5')

    # ----------------------------------------------------------
    # STEP 6: Filter unwanted studies
    # ----------------------------------------------------------
    def _filter_studies(self):
        if 'StudyID' not in self.df.columns:
            return

        to_remove_reasons = {
            'TH-226': 'Treated non-operatively',
            'TF-070': 'Multiple Surgery Patient_bilateral femur fracture',
            'TF-084': 'Multiple Surgery Patient_bilateral femur fracture',
            'TF-115': 'Multiple Surgery Patient_bilateral femur fracture',
            'TPA-019': 'Two stage surgeries related to pelvis/acetabulum - 30 April 2021',
            'TPA-028': 'Two stage surgeries related to pelvis/acetabulum - 6th and 11th April 2021',
            'TPA-035': 'Multiple Surgery Patient: both pelvic and femur surgery on October 19, and another pelvic surgery on October 26, 2021',
            'TPA-043': 'Two stage surgeries related to pelvis/acetabulum - 4th and 6th April 2022',
            'TPA-048': 'Two stage surgeries related to pelvis/acetabulum - 6th and 10th May 2022',
            'TPA-079': 'Two stage surgeries related to pelvis/acetabulum - 6th and 12th Feb 2024'
        }

        to_remove = list(to_remove_reasons.keys())
        present_to_remove = self.df['StudyID'].isin(to_remove) | self.df['StudyID'].str.startswith('TPANO')

        if present_to_remove.any():
            found_ids = self.df.loc[present_to_remove, 'StudyID'].unique()
            print("Removing the following StudyIDs from dataset:")
            for sid in found_ids:
                reason = to_remove_reasons.get(sid, 'Excluded - Non-Operative Arm')
                print(f" - {sid}: {reason}")
            self.df = self.df[~present_to_remove].copy()
        
        print('Step6')

    # ----------------------------------------------------------
    # STEP 7: Filter by screening_status
    # ----------------------------------------------------------
    def _filter_screening_status(self):
        if 'index' in self.df.columns:
            self.df['index'] = self.df['index'].astype(str).str.strip()
            self.df['StudyID'] = self.df['StudyID'].replace('nan', np.nan)
            self.df['StudyID'] = self.df.groupby('index')['StudyID'].ffill().bfill()
        
        print("Step7")
    

    # ----------------------------------------------------------
    # STEP 8: Process VTE flags
    # ----------------------------------------------------------
    def _process_vte_flags(self):
        for col in ['DVT', 'PE']:
            # Convert to boolean: True if "Yes", False if "No" or missing
            self.df[col + '_bool'] = self.df[col].str.strip().str.lower().isin(['yes', 'checked'])
            
            # Group by StudyID and check if any True exists
            self.df[col] = self.df.groupby('StudyID')[col + '_bool'].transform('any')
            
            # Convert back to "Yes"/"No"
            self.df[col] = self.df[col].map({True: 'Yes', False: 'No'})
            
            # Drop temporary column
            self.df.drop(columns=[col + '_bool'], inplace=True)


         # ---- Construct VTE_type dynamically ----
        conditions = [(self.df['DVT'] == 'Yes') & (self.df['PE'] == 'Yes'),
                      (self.df['DVT'] == 'Yes') & (self.df['PE'] != 'Yes'),
                      (self.df['DVT'] != 'Yes') & (self.df['PE'] == 'Yes')
                    ]

        # Define corresponding values
        choices = ['Both', 'DVT', 'PE']

        # Apply to create VTE_type column
        self.df['VTE_type'] = np.select(conditions, choices, default=None)

        # Optional: VTE summary
        self.df['VTE'] = np.where(self.df['VTE_type'].notnull(), 'Yes', 'No')
        print('Step8')

    # ----------------------------------------------------------
    # STEP 9: Missing Values
    # ----------------------------------------------------------
    def _replace_missing_values(self):
        """Convert common REDCap missing codes to NaN"""
        missing_values = ['None', '-999', '', 'NaN','Not applicable', None] #'Not applicable'
        self.df = self.df.replace(missing_values, np.nan)
        
        print('Step 9')
    # ----------------------------------------------------------
    # STEP 10: Process comorbidities
    # ----------------------------------------------------------
    def _process_comorbidities_complications(self):
        comorb_cols = ['comorbidty_diabetes', 'comorbidty_cancer', 'comorbidty_cardiovascular',
                    'comorbidty_pulmonary', 'comorbidty_stroke']  
        comp_cols = ['complication_pulmonary', 'complication_cardiovascular', 'complication_infection']
        
        replace_map = {
            'Checked': 'Yes',
            'Unchecked': 'No',
            'Yes*': 'Yes',
            'Other': np.nan,
            'Yes': 'Yes',
            'No': 'No',
            None: 'No'
        }

        # Normalize all columns first
        for col in comorb_cols + comp_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).replace('nan', np.nan).str.strip()
                self.df[col] = self.df[col].replace(replace_map)
            else:
                self.df[col] = np.nan

        # Now join 'Yes' columns
        self.df['Comorbidities'] = self.df[comorb_cols].apply(
            lambda x: ','.join([c for c in x.index if x[c] == 'Yes']), axis=1
        )

        self.df['Complications'] = self.df[comp_cols].apply(
            lambda x: ','.join([c for c in x.index if x[c] == 'Yes']), axis=1
        )
        print('Step10')
    # ----------------------------------------------------------
    # STEP 11: Process timepoints
    # ----------------------------------------------------------
    def _assign_timepoints(self):
        if 'Time' in self.df.columns:
            # Create a lowercase mapping for all variations
            timepoint_dict_lower = {k.lower(): v for v_list in self.timepoint_dict.values() for k in v_list for v in [list(self.timepoint_dict.keys())[list(self.timepoint_dict.values()).index(v_list)]]}

            # Map using lowercase column values
            self.df['Time'] = self.df['Time'].astype(str).str.lower().map(timepoint_dict_lower)
    
        print('Step11')

    # ----------------------------------------------------------
    # STEP 11: Get CAS-score
    # ----------------------------------------------------------
    # def _compute_cas(self):
    #     # Example CAS computation combining multiple frailty scales
    #     cfs_cols = ['very_fit','well','managing_well','vulnerable','mildly_frail','moderately_frail',
    #                 'severely_frail','very_severely_frail','terminally_ill']
    #     self.df['CAS'] = self.df[cfs_cols].apply(lambda x: next((val for val in x if pd.notna(val)), np.nan), axis=1)

    #     print('Step11')
    
    


    # ----------------------------------------------------------
    # STEP 12: Build records
    # ----------------------------------------------------------
    def _build_records(self):
        """
        Build Record objects (not DataFrames).
        Later, you can convert them to DataFrames with record.to_dataframe().
        """
        self.records = {}

        for study_id, rows in self.df.groupby("StudyID"):

            # ------------------------------
            # 1. EXTRACT DEMOGRAPHICS
            # ------------------------------
            demo = {}
            for col in self.demographic_cols:
                if col in rows.columns:
                    values = rows[col].dropna()
                    demo[col] = values.iloc[0] if len(values) > 0 else None

            # ------------------------------
            # 2. BUILD BLOOD DRAW OBJECTS
            # ------------------------------
            blood_draws = []

            for _, row in rows.iterrows():
                labs = {col: row[col] for col in self.lab_cols if col in row.index}

                # skip rows where all labs are NA
                if not any(pd.notna(v) for v in labs.values()):
                    continue

                # draw_id = row.get("Draw_ID", None)
                draw_id = "_".join(
                    str(row.get(col, "")) 
                    for col in ["redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance"]
                ).strip("_")

                blood_draws.append(BloodDraw(draw_id=draw_id, **labs))

            # ------------------------------
            # 3. CREATE RECORD OBJECT
            # ------------------------------
            record = Record(
                record_id=study_id,
                demographics=demo,
                blood_draws=blood_draws
            )

            # compute time differences inside the record
            record.add_time_differences()

            self.records[study_id] = record

        print("Step12: Records Successfully Built!")


##########################
    def get_all_demographics(self):
        rows = []
        for sid, rec in self.records.items():
            row = rec.demographics.copy()
            row["StudyID"] = sid
            rows.append(row)
        return pd.DataFrame(rows)


    # ---------------------------------------------------
    # ⭐ ALL LABS-ONLY (all records) as one DataFrame
    # ---------------------------------------------------
    def get_all_labs(self):
        dfs = [rec.to_lab_dataframe() for rec in self.records.values()]
        dfs = [df for df in dfs if not df.empty]
        if len(dfs) == 0:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)


    # ---------------------------------------------------
    # ⭐ FULL MERGED DF (labs + demographics)
    # ---------------------------------------------------
    def get_full_dataframe(self):
        dfs = [rec.to_dataframe() for rec in self.records.values()]
        dfs = [df for df in dfs if not df.empty]
        if len(dfs) == 0:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    

   


#     # --------------------------------------------------------------------------------------------------OLD CODE WORKED
   

       

#         if 'Draw_date' in df.columns:
#             # --- Step 1: Parse Draw_date safely ---
#             parsed_draw = pd.to_datetime(df['Draw_date'], errors='coerce')

#             # --- Step 2: Identify if Draw_date has a time (non-midnight) ---
#             has_time = parsed_draw.dt.time.astype(str) != "00:00:00"

#             # --- Step 3: Ensure time columns exist ---
#             if 'teg_time' not in df.columns:
#                 df['teg_time'] = pd.NA
#             teg_exists = True

#             # If lab_time missing entirely, mark flag
#             lab_time_exists = 'lab_time' in df.columns
#             if not lab_time_exists:
#                 df['lab_time'] = pd.NA  # create column if missing

#             # --- Step 4: Replace missing times with midnight ---
#             df['teg_time'] = df['teg_time'].fillna('00:00').astype(str)

#             if lab_time_exists:
#                 df['lab_time'] = df['lab_time'].astype(str)
#                 # Replace only rows where teg_time is '00:00'
#                 df.loc[df['teg_time'] == '00:00', 'teg_time'] = df['lab_time']
#             else:
#                 df['lab_time'] = df['teg_time'].fillna('00:00').astype(str)


#             # --- Step 5: Define fallback date (lab_date_visit if Draw_date missing) ---
#             if 'lab_date_visit' in df.columns:
#                 fallback_dates = pd.to_datetime(df['lab_date_visit'], errors='coerce')
#             else:
#                 fallback_dates = pd.Series([pd.NaT] * len(df), index=df.index)

#             # --- Step 6: Build Draw_date_teg ---
#             df['Draw_date_teg'] = np.where(
#                 has_time,
#                 parsed_draw.astype(str),
#                 (
#                     fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
#                     + ' '
#                     + df['teg_time']
#                 )
#             )

#             # --- Step 7: Build Draw_date_lab ---
#             if lab_time_exists:
#                 # If lab_time exists, only use it; leave NaT if missing
#                 df['Draw_date_lab'] = np.where(
#                     has_time,
#                     parsed_draw.astype(str),
#                     np.where(
#                         df['lab_time'].notna() & (df['lab_time'] != 'NaT'),
#                         fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d') + ' ' + df['lab_time'],
#                         np.nan  # leave missing as NaT
#                     )
#                 )
#             else:
#                 # If lab_time column missing entirely → fallback to teg_time
#                 df['Draw_date_lab'] = np.where(
#                     has_time,
#                     parsed_draw.astype(str),
#                     (
#                         fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
#                         + ' '
#                         + df['teg_time']
#                     )
#                 )
#             # --- Step 8: Hemoglobin consistency rules ---

#             # Ensure Draw_date_lab and Draw_date_teg are proper datetimes
#             df['Draw_date_lab'] = pd.to_datetime(df['Draw_date_lab'], errors='coerce')
#             df['Draw_date_teg'] = pd.to_datetime(df['Draw_date_teg'], errors='coerce')

#             # If Hemoglobin is missing → Draw_date_lab should be missing too
#             df.loc[df['Hemoglobin'].isna(), 'Draw_date_lab'] = pd.NaT

#             # If Hemoglobin is present but Draw_date_lab is missing → copy from Draw_date_teg
#             df.loc[
#                 (~df['Hemoglobin'].isna()) & (df['Draw_date_lab'].isna()),
#                 'Draw_date_lab'
#             ] = df.loc[
#                 (~df['Hemoglobin'].isna()) & (df['Draw_date_lab'].isna()),
#                 'Draw_date_teg']


#         if 'adm_injury_time' in df.columns and 'Injury_date' in df.columns:
#             df['Injury_date'] = pd.to_datetime(df['Injury_date'].astype(str) + ' ' + df['adm_injury_time'].astype(str),
#             errors='coerce')

#         if 'time_injury' in df.columns and 'Injury_date' in df.columns:
#             df['Injury_date'] = pd.to_datetime(df['Injury_date'].astype(str) + ' ' + df['time_injury'].astype(str),
#             errors='coerce')

#         if 'intraop_time_surg' in df.columns and 'Surgery_date' in df.columns:
#             df['Surgery_date'] = pd.to_datetime(df['Surgery_date'].astype(str) + ' ' + df['intraop_time_surg'].astype(str),
#             errors='coerce')

#         if {'ota_type_61', 'ota_type_62'}.issubset(df.columns):
#             df['AO_OTA'] = (df[['ota_type_61', 'ota_type_62']].apply(lambda x: '/'.join(x.dropna().astype(str)), axis=1).replace('', np.nan))

#         if 'complication_dvt' in df.columns:
#             df['DVT'] = np.where(df['complication_dvt']=='Yes','DVT','No')


#         if 'complication_pe' in df.columns:
#             df['PE'] = np.where(df['complication_pe']=='Yes','PE','No')


#         # Merge multiple timepoint columns into a single 'Time' column
#         timepoint_cols = ['teg_preop_tp', 'teg_postop_tp1','teg_postop_tp2', 'teg_fu_tp','teg_timepoint']
#         existing_timepoint_cols = [col for col in timepoint_cols if col in df.columns]

#         if existing_timepoint_cols:
#             # Take the first non-null value across the columns
#             df['Time'] = df[existing_timepoint_cols].bfill(axis=1).iloc[:, 0]


#         # --- Merge surgery date columns into a single 'Surgery_date' column ---
#         surg_date_cols = ['surg_date_pelvis', 'surg_date_ant_acet', 'surg_date_post_acet']
#         existing_surg_cols = [col for col in surg_date_cols if col in df.columns]

#         if existing_surg_cols:
#             # Merge into one column using first non-null value
#             df['Surgery_date'] = df[existing_surg_cols].bfill(axis=1).iloc[:, 0]

#             # Optional: ensure datetime type
#             df['Surgery_date'] = pd.to_datetime(df['Surgery_date'], errors='coerce')

#             # Fill the same Surgery_date across all rows for each StudyID
#             df['Surgery_date'] = df.groupby('StudyID')['Surgery_date'].transform(lambda x: x.ffill().bfill())
           


#         # ---- Withdrawn/Death from main Withdrawn column ----
#         if 'Withdrawn' in df.columns:
#             withdrew_values = {'patient withdrew consent', 'other reason', 'lost to follow up'}

#             # Initialize Death column
#             df['Death'] = 'No'

#             # Normalize Withdrawn column
#             df['Withdrawn_norm'] = df['Withdrawn'].astype(str).str.strip().str.lower()

#             # Death rows
#             death_mask = df['Withdrawn_norm'] == 'death'
#             df.loc[death_mask, ['Death', 'Withdrawn']] = ['Yes', 'No']

#             # Withdrawn rows
#             withdrew_mask = df['Withdrawn_norm'].isin(withdrew_values) & ~death_mask
#             df.loc[withdrew_mask, 'Withdrawn'] = 'Yes'

#             # All other rows
#             df.loc[~death_mask & ~withdrew_mask, 'Withdrawn'] = 'No'

#             df.drop(columns=['Withdrawn_norm'], inplace=True)
#         else:
#             df['Withdrawn'] = 'No'
#             df['Death'] = 'No'

#         # ---- Override Death if outcomes_outcome_type indicates mortality ----
#         if 'comp_death' in df.columns:
    
            
#             comp_death_series = df['comp_death']
#             if isinstance(comp_death_series, pd.DataFrame):
#                 # Take the first column if somehow multiple columns
#                 comp_death_series = comp_death_series.iloc[:, 0]

#             # Normalize text
#             comp_death_series = comp_death_series.astype(str).str.strip().str.lower()

#             # Mark mortality if 'checked' or 'no'
#             mortality_ids = df.loc[comp_death_series.isin(['checked', 'yes']), 'StudyID'].unique()
#             df.loc[df['StudyID'].isin(mortality_ids), 'Death'] = 'Yes'
#             df.loc[df['StudyID'].isin(mortality_ids), 'Withdrawn'] = 'No'

#         # ---- Ensure Withdrawn/Death consistent per StudyID ----
#         for study_id, group in df.groupby('StudyID'):
#             if (group['Death'] == 'Yes').any():
#                 df.loc[df['StudyID'] == study_id, ['Death', 'Withdrawn']] = ['Yes', 'No']
#             elif (group['Withdrawn'] == 'Yes').any():
#                 df.loc[df['StudyID'] == study_id, 'Withdrawn'] = 'Yes'

        
#         treatment_map = {
#             #Hip
#             'intra_treatment___1': 'Hemi-arthroplasty (monopolar, bipolar)',
#             'intra_treatment___2': 'Total Hip Arthroplasty',
#             'intra_treatment___3': 'Cannulated Screws',
#             'intra_treatment___4': 'Short cephalomedullary nail',
#             'intra_treatment___5': 'Long cephalomedullary nail',
#             'intra_treatment___6': 'Dynamic Hip Screw',
#             'intra_treatment___7': 'Other',

#             #Pathway
#             'intraop_treatment___1': 'Hemi-arthroplasty (monopolar, bipolar)',
#             'intraop_treatment___2': 'Total Hip Arthroplasty',
#             'intraop_treatment___3': 'Cannulated Screws',
#             'intraop_treatment___4': 'Short cephalomedullary nail',
#             'intraop_treatment___5': 'Long cephalomedullary nail',
#             'intraop_treatment___6': 'Dynamic Hip Screw',
#             'intraop_treatment___7': 'Other'
#         }
#         def map_treatments(df, prefix):
#             cols = [c for c in df.columns if c.startswith(prefix)]
#             if not cols:
#                 return pd.Series([np.nan] * len(df))
#             def mapper(row):
#                 checked = [treatment_map[col] for col in cols if row.get(col) == 'Checked']
#                 return '/'.join(checked) if checked else np.nan
#             return df.apply(mapper, axis=1)

#         df['intra_treatment'] = map_treatments(df, 'intra_treatment___')
#         df['intraop_treatment'] = map_treatments(df, 'intraop_treatment___')

#         # Fill both within each StudyID
#         df['intra_treatment'] = df.groupby('StudyID')['intra_treatment'].ffill().bfill()
#         df['intraop_treatment'] = df.groupby('StudyID')['intraop_treatment'].ffill().bfill()

#         # Prefer intra_treatment, but fallback to intraop_treatment
#         df['Treatment'] = df['intra_treatment'].combine_first(df['intraop_treatment'])
#         df['Treatment'] = df['Treatment'].map(self.arth_fix)


#         # Medications
#         df['Pre_op_med'] = df['StudyID'].map(self.medications_preop)
#         df['Pre_op_med'] = df['Pre_op_med'].replace({np.nan: 'LMWH'})



#          # Step 5: Standardize timepoints
#         if 'Time' in df.columns:
#             df['Time'] = df['Time'].apply(self._map_timepoint)

    

#         # Step 6: Ensure all dempgraphic columns exist
#         for col in self.dempgraphic_cols:
#             if col not in df.columns:
#                 df[col] = np.nan
        




#         # ---- Comorbidity and complications
        
#         if 'bl_tobacco' in df.columns:
#             df['comorbidty_current_smoker']=np.where(df['bl_tobacco']==1, 'Yes','No')
        
#         df['comorbidty_current_smoker'] = df.groupby('StudyID')['comorbidty_current_smoker'].ffill().bfill()
        

#         df['StudyID'] = df['StudyID'].astype(str).str.strip().str.upper()
#         df = df.sort_values('StudyID').reset_index(drop=True)

#         COMORBIDITY_COMPLICATIONS = [
#             'comorbidty_diabetes','comorbidty_cancer','comorbidty_cardiovascular',
#             'comorbidty_pulmonary','comorbidty_stroke',
#             'complication_pulmonary','complication_cardiovascular','complication_infection'
#         ]

#         # Replace map
#         replace_map = {
#             'Checked': 'Yes',
#             'Unchecked': 'No',
#             'Yes*': 'Yes',
#             'Other': np.nan,
#             'Yes': 'Yes',
#             'No': 'No',
#             None: 'No'
#         }

#         for col in COMORBIDITY_COMPLICATIONS:
#             if col in df.columns:
               
#                df.loc[:, col] = df.loc[:, col].astype(str).replace('nan', np.nan)  # convert to string, handle NaN
#                df.loc[:, col] = df.loc[:, col].str.strip()            # string operations
#                df.loc[:, col] = df.loc[:, col].replace(replace_map)              
#             else:
#                 df.loc[:, col] = np.nan



#         # Count 'Yes' per StudyID and set all accordingly
#         for col in COMORBIDITY_COMPLICATIONS:
#             df[col] = df.groupby('StudyID')[col].transform(lambda x: 'Yes' if (x == 'Yes').sum() >= 1 else 'No')

#         for col in COMORBIDITY_COMPLICATIONS:            
#             df[col]=df[col].replace(np.nan,'No')

    
#         df = df.sort_values('StudyID').reset_index(drop=True)

#         ######
#         df['Age'] = df.groupby('StudyID')['Age'].transform(lambda x: x.ffill().bfill()) 
#         df['Sex'] = df.groupby('StudyID')['Sex'].transform(lambda x: x.ffill().bfill()) 
#         df['BMI'] = df.groupby('StudyID')['BMI'].transform(lambda x: x.ffill().bfill()) 
#         df['Injury_date'] = df.groupby('StudyID')['Injury_date'].transform(lambda x: x.ffill().bfill()) 
#         df['Admission_date'] = df.groupby('StudyID')['Admission_date'].transform(lambda x: x.ffill().bfill()) 
#         df['Surgery_date'] = df.groupby('StudyID')['Surgery_date'].transform(lambda x: x.ffill().bfill()) 
#         df['AO_OTA'] = df.groupby('StudyID')['AO_OTA'].transform(lambda x: x.ffill().bfill()) 


#     # ---- Time calculations ----
#         df['Injury_date'] = pd.to_datetime(df['Injury_date'], errors="coerce")
#         df['Surgery_date'] = pd.to_datetime(df['Surgery_date'], errors="coerce")

#         df['time_injury_to_surgery_hours'] = (
#             (df['Surgery_date'] - df['Injury_date']).dt.total_seconds() / 3600
#         )

#     # ---- Blood Transfusions ----
#         df['blood_rbc'] = pd.to_numeric(df['blood_rbc'], errors='coerce').fillna(0)
#         total_blood_transfusions = df.groupby('StudyID')['blood_rbc'].sum().to_dict()
        
#         df['total_blood_rbc'] = df['StudyID'].map(total_blood_transfusions)
#         df['blood_rbc_yn'] = np.where(df['total_blood_rbc']==0, 'No','Yes')
        
#     # Create a study column with study names
#         df['Study'] = df['StudyID'].str.extract(r'^(TH|HPA|TF|TPA)').replace({
#                 'TH': 'Hip',
#                 'HPA': 'Pathway',
#                 'TF': 'Femur',
#                 'TPA': 'Pelvis'
#             })
#         if 'CAS' not in df.columns:
#             df['CAS'] = np.nan

#         if 'cas_timepoint' in df.columns:
#                     df['cas_timepoint']=df['cas_timepoint'].apply(self._map_timepoint)
#                     # Ensure CAS is numeric
#                     df['CAS'] = pd.to_numeric(df['CAS'], errors='coerce')

#                     df_cas = df.dropna(subset=['CAS'])[['StudyID','cas_timepoint','CAS']].rename(columns={'cas_timepoint':'Time'})
                    

#                     df = df.drop(columns=['CAS','cas_timepoint']).merge(df_cas, on=['StudyID', 'Time'], how='left')


        


#         # Step 7: Save the processed DataFrame
#         print('=================================================================================')
#         print('Total Number of Patients Included:',df['StudyID'].nunique())

#         # Step 8: Build Record objects
#         self._build_records()


#         self.df = df

#         return self.df


  

#     # # -----------------------
#     # # Map timepoints
#     # # -----------------------
#     # def _map_timepoint(self, tp):
#     #     if pd.isna(tp):
#     #         return tp
#     #     for standard, variants in self.timepoint_dict.items():
#     #         if tp in variants:
#     #             return standard
#     #     return tp
    
#     # -----------------------
#     # Build records
#     # # -----------------------
#     # def _build_records(self):
#     #     self.records = {}
#     #     for study_id, rows in self.df.groupby("StudyID"):

#     #         # Demographics
#     #         demo = rows[self.dempgraphic_cols].apply(
#     #             lambda col: col.dropna().iloc[0] if col.dropna().any() else None
#     #         )
#     #         demo_dict = demo.to_dict()

#     #         blood_draws = []

#     #         lab_cols_available = [c for c in self.lab_cols if c in rows.columns]
#     #         # for _, row in rows.iterrows():
#     #         #     if pd.notnull(row.get("Draw_date")):
#     #         #         bd_data = row[self.lab_cols].to_dict()
#     #         for _, row in rows.iterrows():
#     #             if pd.notnull(row.get("Draw_date")):
#     #                 bd_data = row[lab_cols_available].to_dict()
#     #                 bd_data["Draw_date"] = row["Draw_date"]
#     #                 blood_draws.append(BloodDraw(row["Draw_date"], **bd_data))


#     #         # Save Record
#     #         # self.records[study_id] = Record(study_id, demographics=demo_dict, blood_draws=blood_draws)
#     #         rec = Record(study_id,demographics=demo_dict, blood_draws=blood_draws) #######
#     #         rec.add_time_differences()  # <-- calculate once per patient
#     #         self.records[study_id] = rec
            

#     # ------------------------------------------------------------------------------
#     # Get demographics for a patient
#     # ------------------------------------------------------------------------------
#     def get_patient_demographics(self, study_id):
#         """Display just the demographics for a single patient with StudyID as header, return None."""
#         df_demo = self.get_all_demographics()
#         patient_demo = df_demo[df_demo['StudyID'] == study_id].copy()

#         if not patient_demo.empty:
#             # Take first row and transpose
#             patient_row = patient_demo.iloc[[0]].transpose()

#             # Set the StudyID as the column header
#             patient_row.columns = [study_id]

#             # Drop the StudyID row to avoid repetition
#             patient_row = patient_row.drop('StudyID')

#             display(patient_row)
        
#         return None

    
#     # ------------------------------------------------------------------------------
#     # Get all demographics for all patients
#     # ------------------------------------------------------------------------------
#     def get_all_demographics(self):
#         if self.df is None:
#             raise ValueError("Data not processed. Run fetch_and_process() first.")

#         # Keep only demographic columns that exist in the df
#         cols = [c for c in self.dempgraphic_cols if c in self.df.columns]

#         df_demo = self.df[cols].copy()

#         # Remove duplicates per StudyID
#         df_demo = df_demo.drop_duplicates(subset='StudyID', keep='first')

#         return df_demo
    
#     # ------------------------------------------------------------------------------
#     # Get patient blood draws
#     # ------------------------------------------------------------------------------
#     def get_patient_blood_draws(self, study_id):
#         """Display just the blooddraws for a single patient with StudyID as header, return None."""
#         df_blood_draws = self.get_all_blood_draws()
#         patient_blood_draws = df_blood_draws[df_blood_draws['StudyID'] == study_id].copy()
#         display(patient_blood_draws)
        
#         return None

    
#     # ------------------------------------------------------------------------------
#     # Get all blood draws for all patients
#     # ------------------------------------------------------------------------------
#     def get_all_blood_draws(self):
#         all_draws = []

#         for rec in self.records.values():

#             # --- Patient-level info ---
#             demo = rec.get_demographics()
#             injury_date  = pd.to_datetime(demo.get('Injury_date',  pd.NA), errors='coerce')
#             surgery_date = pd.to_datetime(demo.get('Surgery_date', pd.NA), errors='coerce')

#             dvt_flag = demo.get('DVT', 'No')
#             pe_flag  = demo.get('PE',  'No')

#             # Loop through blood draws for this record
#             for bd in rec.blood_draws:

#                 row = {"StudyID": rec.study_id}
#                 row.update(bd.labs)

#                 # Ensure keys exist
#                 row['Draw_date_lab'] = row.get('Draw_date_lab', pd.NA)
#                 row['Draw_date_teg'] = row.get('Draw_date_teg', pd.NA)

#                 # Convert dates
#                 draw_lab = pd.to_datetime(row['Draw_date_lab'], errors='coerce')
#                 draw_teg = pd.to_datetime(row['Draw_date_teg'], errors='coerce')

#                 # --- Time deltas ---
#                 row['injury_to_lab_hrs'] = (
#                     (draw_lab - injury_date).total_seconds() / 3600
#                     if pd.notnull(draw_lab) and pd.notnull(injury_date)
#                     else pd.NA
#                 )

#                 row['injury_to_teg_hrs'] = (
#                     (draw_teg - injury_date).total_seconds() / 3600
#                     if pd.notnull(draw_teg) and pd.notnull(injury_date)
#                     else pd.NA
#                 )

#                 row['surgery_to_lab_hrs'] = (
#                     (draw_lab - surgery_date).total_seconds() / 3600
#                     if pd.notnull(draw_lab) and pd.notnull(surgery_date)
#                     else pd.NA
#                 )

#                 row['surgery_to_teg_hrs'] = (
#                     (draw_teg - surgery_date).total_seconds() / 3600
#                     if pd.notnull(draw_teg) and pd.notnull(surgery_date)
#                     else pd.NA
#                 )

#                 # Patient-level info
#                 row["Pre_op_med"] = self.medications_preop.get(rec.study_id, 'No')
#                 row["Injury_date"] = injury_date
#                 row["Surgery_date"] = surgery_date
#                 row["DVT"] = dvt_flag
#                 row["PE"] = pe_flag

#                 # Save row
#                 all_draws.append(row)

#         # Convert to DataFrame AFTER building all rows
#         df_all = pd.DataFrame(all_draws)

#         # --- Exclusions ---
#         mask = (df_all['StudyID'].isin(['TPA-046', 'TPA-073'])) & (df_all['Time'] == 'Pre-Op')
#         df_all = df_all[~mask].copy()

#         # --- Study name mapping ---
#         df_all['Study'] = (
#             df_all['StudyID']
#             .str.extract(r'^(TH|HPA|TF|TPA)')[0]
#             .replace({
#                 'TH': 'Hip',
#                 'HPA': 'Pathway',
#                 'TF': 'Femur',
#                 'TPA': 'Pelvis'
#             })
#         )

#         return df_all

    
    
# -----------------------
# BloodDraw and Record classes
# -----------------------

class BloodDraw:
    def __init__(self, draw_id=None, **labs):
        self.draw_id = draw_id

        # store all lab values dynamically
        for k, v in labs.items():
            setattr(self, k, v)


class Record:
    def __init__(self, record_id, demographics, blood_draws):
        self.record_id = record_id
        self.demographics = demographics
        self.blood_draws = blood_draws

    # Your existing methods:
    def add_time_differences(self):
        pass

    def get_demographics(self):
        return self.demographics

    def get_all_labs(self):
        return self.blood_draws

    # ---------------------------------------------------
    # ⭐ NEW: Return demographics as a DataFrame
    # ---------------------------------------------------
    def to_demographics_dataframe(self):
        df = pd.DataFrame([self.demographics])
        df.insert(0, "StudyID", self.record_id)
        return df

    # ---------------------------------------------------
    # ⭐ NEW: Return labs-only as a DataFrame
    # ---------------------------------------------------
    def to_lab_dataframe(self):
        """Each blood draw = one row; no demographics."""
        if len(self.blood_draws) == 0:
            return pd.DataFrame()

        rows = []
        for draw in self.blood_draws:
            row = {"StudyID": self.record_id, "Draw_ID": draw.draw_id}
            for k, v in draw.__dict__.items():
                if k != "draw_id":   # avoid duplicating
                    row[k] = v
            rows.append(row)

        return pd.DataFrame(rows)

    # ---------------------------------------------------
    # ⭐ NEW: Full flattened DF = demographics + labs
    # ---------------------------------------------------
    def to_dataframe(self):
        df_labs = self.to_lab_dataframe()
        if df_labs.empty:
            return pd.DataFrame()

        for col, val in self.demographics.items():
            df_labs[col] = val

        return df_labs

