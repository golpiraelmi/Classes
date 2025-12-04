
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
            ('teg_time','teg_bd_time',):'teg_time', 
            ('teg_time_lab_panel',):'lab_time',
            ('aoota_classification','inj_aoota',):'AO_OTA',
            ('comp_dvt_yn','complication_dvt','outcomes_vte_type___1'):'DVT',
            ('comp_pe_yn','complication_pe','outcomes_vte_type___2'):'PE',
            ('reason_withdrawal','wd_reason','study_wd_reason'): 'Withdrawn',
            ('outcomes_outcome_type___4','complication_death','comp_death_yn'):'comp_death',
            ('diabetes','bl_comorbidity_check___1','bl_comorbidities___1',):'comorb_diabetes',
            ('cancer','bl_comorbidity_check___2','bl_comorbidities___2',):'comorb_cancer',
            ('cardiovascular','bl_comorbidity_check___3','bl_comorbidities___3',):'comorb_cardiovascular',
            ('pulmonary','bl_comorbidity_check___4','bl_comorbidities___4',):'comorb_pulmonary',
            ('prior_stroke','bl_comorbidity_check___5','bl_comorbidities___5',):'comorb_stroke',
            ('current_smoker',):'comorbidty_current_smoker',
            ('pulmonary_yesno','comp_pulmonary','comp_pulmonary_yn'): 'comp_pulmonary',
            ('cardio_yesno','comp_cardio','comp_cardio_yn',): 'comp_cardiovascular',
            ('infection_yesno','comp_infection','comp_infection_yn',): 'comp_infection',
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
                    'HPA-051','HPA-052','HPA-053','THB-HPA-007','TH-162', 'TH-170', 'TH-198', 'TH-212', 'TH-217', 
                    'TH-225', 'TH-227', 'TH-236', 'TH-240','TH-255', 'TH-262', 'TH-267', 'TH-274', 'TH-284','TH-286',
                    'TH-302', 'TF-121','TF-128', 'TPA-058','TPA-082','TPA-093' ], "DOAC"),
            **dict.fromkeys(['TH-244'], 'Warfarin'),
            **dict.fromkeys(['TH-006','TH-008','TH-011','TH-013','TH-023','TH-025','TH-026','TH-028','TH-031','TH-035','TH-038','TH-041','TH-046','TH-059','TH-066','TH-072','TH-082','TH-086','TH-090','TH-092','TH-093','TH-100',
                             'TH-102','TH-105','TH-116','TH-126','TH-127','TH-128','TH-133','TH-139','TH-185','TH-258','TH-301','TH-305','TF-005','TF-038','TF-059','TF-071','TF-109','TF-121','TF-136','TPA-019', 'TPA-085'],'ASA'),
            **dict.fromkeys(['TH-004','TH-010','TH-075','TH-110'], 'No Info')
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
        self.demographic_cols = ['Study','StudyID','Death','Withdrawn','Age','Sex','BMI','Injury_date','Admission_date','Surgery_date','AO_OTA','Treatment','DVT','PE','VTE_type','VTE','comorb_diabetes','comorb_cancer',
                              'comorb_cardiovascular','comorb_pulmonary','comorb_stroke','comp_pulmonary', 'comp_cardiovascular','comp_infection','Pre_op_med','time_injury_to_surgery_hours',
                              'total_blood_rbc','blood_rbc_yn']
        
        
        self.lab_cols = ['Study','StudyID', 'Time','CAS','VTE_type','VTE', 'Hemoglobin', 'Creatinine', 'R_time', 'K_time','Alpha_Angle', 'MA', 'LY30', 'ACT','ADP-agg', 'ADP-inh','ADP-ma','AA-agg','AA-inh','AA-ma','Draw_date_lab', 'Draw_date_teg','Pre_op_med','time_injury_to_surgery_hours']
        
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
        self._filter_patients()
        self._filter_screening_status()
        self._process_vte_flags()
        self._replace_missing_values()
        self._process_comorbidities_complications()
        self._assign_timepoints()
        self._process_surgery_injury_dates()
        self._times_to_analyses()
        self._Process_AO_OTA()
        self._Process_medication()
        self._Process_Death_Withdrawals()
        self._process_treatment()
        self._add_blood_transfusion()
        self._add_study_names()
        self._compute_cas()
        self._build_records()


        return self.df

    # ----------------------------------------------------------
    # STEP 1: Fetch records
    # ----------------------------------------------------------
    
    def _fetch_records(self):
        records_data = self.project.export_records(raw_or_label='label')
        self.df = pd.DataFrame(records_data)
        print('Data Fetched')

    

    # ----------------------------------------------------------
    # STEP 2: Clean basic data
    # ----------------------------------------------------------
    def _clean_data(self):
        # Replace empty strings with NaN
        self.df = self.df.replace(r'^\s*$', np.nan, regex=True)

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
  
    # ----------------------------------------------------------
    # STEP 5: Clean StudyID, handle withdrawals
    # ----------------------------------------------------------
    def _clean_studyids(self):
        if 'StudyID' not in self.df.columns:
            return
        
        self.df = self.df.replace({'Participant Withdrawn': np.nan})
        self.df['StudyID'] = self.df['StudyID'].astype(str).str.strip()

    # ----------------------------------------------------------
    # STEP 6: Filter unwanted patients
    # ----------------------------------------------------------
    def _filter_patients(self):
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
            'TPA-079': 'Two stage surgeries related to pelvis/acetabulum - 6th and 12th Feb 2024',
            'TPA-104': 'Two stage surgeries related to pelvis/acetabulum - Recent'
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
        
    # ----------------------------------------------------------
    # STEP 7: Filter by screening_status
    # ----------------------------------------------------------
    def _filter_screening_status(self):

        # --- Create StudyID and index if screening columns exist ---
        if 'screen_patient_id' in self.df.columns:
            self.df = self.df.rename(columns={
                'screen_patient_id': 'StudyID',
                'record_id': 'index'
            })

        # --- Fill StudyID if index exists ---
        if 'index' in self.df.columns:
            self.df['index'] = self.df['index'].astype(str).str.strip()
            self.df['StudyID'] = self.df['StudyID'].replace('nan', np.nan)
            self.df['StudyID'] = self.df.groupby('index')['StudyID'].ffill().bfill()

        # --- Handle screening_status ---
        if 'screening_status' in self.df.columns:
            self.df['StudyID'] = self.df['StudyID'].astype(str)
            self.df['StudyID'] = self.df.groupby('index')['StudyID'].ffill().bfill()
            self.df['screening_status'] = self.df.groupby('index')['screening_status'].ffill().bfill()

            # keep only eligible → enrolled
            self.df = self.df[
                self.df['screening_status'].astype(str).str.strip() == 'Eligible → enrolled'
            ]
    # ----------------------------------------------------------
    # STEP 8: Process VTE flags
    # ----------------------------------------------------------
    def _process_vte_flags(self):
        if 'complication_dvt' in self.df.columns:
            self.df['DVT'] = np.where(self.df['complication_dvt']=='Yes','DVT','No')


        if 'complication_pe' in self.df.columns:
            self.df['PE'] = np.where(self.df['complication_pe']=='Yes','PE','No')


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
      
    # ----------------------------------------------------------
    # STEP 9: Missing Values
    # ----------------------------------------------------------
    def _replace_missing_values(self):
        """Convert common REDCap missing codes to NaN"""
        missing_values = ['None', '-999', '', 'NaN','Not applicable','-2997', None] #'Not applicable'
        self.df = self.df.replace(missing_values, np.nan)
       
    # ----------------------------------------------------------
    # STEP 10: Process comorbidities
    # ----------------------------------------------------------
    def _process_comorbidities_complications(self):

        self.df['StudyID'] = self.df['StudyID'].astype(str).str.strip().str.upper()
        self.df = self.df.sort_values('StudyID').reset_index(drop=True)

        COMORBIDITY_COMPLICATIONS = [
            'comorb_diabetes','comorb_cancer','comorb_cardiovascular',
            'comorb_pulmonary','comorb_stroke',
            'comp_pulmonary','comp_cardiovascular','comp_infection'
        ]

        # Replace map
        replace_map = {
            'Checked': 'Yes',
            'Unchecked': 'No',
            'Yes*': 'Yes',
            'Other': np.nan,
            'Yes': 'Yes',
            'No': 'No',
            None: 'No'
        }

        for col in COMORBIDITY_COMPLICATIONS:
            if col in self.df.columns:
               
               self.df.loc[:, col] = self.df.loc[:, col].astype(str).replace('nan', np.nan)  # convert to string, handle NaN
               self.df.loc[:, col] = self.df.loc[:, col].str.strip()            # string operations
               self.df.loc[:, col] = self.df.loc[:, col].replace(replace_map)              
            else:
                self.df.loc[:, col] = np.nan



        # Count 'Yes' per StudyID and set all accordingly
        for col in COMORBIDITY_COMPLICATIONS:
            self.df[col] = self.df.groupby('StudyID')[col].transform(lambda x: 'Yes' if (x == 'Yes').sum() >= 1 else 'No')

        for col in COMORBIDITY_COMPLICATIONS:            
            self.df[col]=self.df[col].replace(np.nan,'No')
        
    # ----------------------------------------------------------
    # STEP 11: Process timepoints
    # ----------------------------------------------------------
    def _assign_timepoints(self):

        # Merge multiple timepoint columns into a single 'Time' column (Pelvis)
        timepoint_cols = ['teg_preop_tp', 'teg_postop_tp1','teg_postop_tp2', 'teg_fu_tp','teg_timepoint']
        existing_timepoint_cols = [col for col in timepoint_cols if col in self.df.columns]

        if existing_timepoint_cols:
            # Take the first non-null value across the columns
            self.df['Time'] = self.df[existing_timepoint_cols].bfill(axis=1).iloc[:, 0]


        if 'Time' in self.df.columns:
            # Create a lowercase mapping for all variations
            timepoint_dict_lower = {k.lower(): v for v_list in self.timepoint_dict.values() for k in v_list for v in [list(self.timepoint_dict.keys())[list(self.timepoint_dict.values()).index(v_list)]]}

            # Map using lowercase column values
            self.df['Time'] = self.df['Time'].astype(str).str.lower().map(timepoint_dict_lower)

    # ----------------------------------------------------------
    # STEP 12: Process Surgery/Injury date columns
    # ----------------------------------------------------------
    def _process_surgery_injury_dates(self):
        if any(col in self.df.columns for col in ['surg_date_pelvis','surg_date_ant_acet','surg_date_post_acet']):

            surg_date_cols = ['surg_date_pelvis', 'surg_date_ant_acet', 'surg_date_post_acet']
            existing_surg_cols = [col for col in surg_date_cols if col in self.df.columns]

            if existing_surg_cols:
                # Merge into one column using first non-null value
                self.df['Surgery_date'] = self.df[existing_surg_cols].bfill(axis=1).iloc[:, 0]
                self.df['Surgery_date'] = pd.to_datetime(self.df['Surgery_date'], errors='coerce')



        if 'adm_injury_time' in self.df.columns and 'Injury_date' in self.df.columns:
            self.df['Injury_date'] = pd.to_datetime(self.df['Injury_date'].astype(str) + ' ' + self.df['adm_injury_time'].astype(str),
            errors='coerce')

        if 'time_injury' in self.df.columns and 'Injury_date' in self.df.columns:
            self.df['Injury_date'] = pd.to_datetime(self.df['Injury_date'].astype(str) + ' ' + self.df['time_injury'].astype(str),
            errors='coerce')

        if 'intraop_time_surg' in self.df.columns and 'Surgery_date' in self.df.columns:
            self.df['Surgery_date'] = pd.to_datetime(self.df['Surgery_date'].astype(str) + ' ' + self.df['intraop_time_surg'].astype(str),
            errors='coerce')


    # ----------------------------------------------------------
    # STEP 13: Process times to blood draw
    # ----------------------------------------------------------
    def _times_to_analyses(self):
        if 'teg_time'in self.df.columns:
            self.df['teg_time'] = self.df['teg_time'].fillna('00:00').astype(str) ########## ADDED DEC 4

        if 'lab_time'in self.df.columns:
            self.df['lab_time'] = self.df['lab_time'].fillna('00:00').astype(str) ########## ADDED DEC 4

        if 'Draw_date' in self.df.columns:
            # Parse Draw_date safely
            parsed_draw = pd.to_datetime(self.df['Draw_date'], errors='coerce')

            # Identify if Draw_date has a time (non-midnight)
            has_time = parsed_draw.dt.time.astype(str) != "00:00:00"

            # Ensure time columns exist
            if 'teg_time' not in self.df.columns:
                self.df['teg_time'] = pd.NA
            teg_exists = True

            # If lab_time missing entirely, mark flag
            lab_time_exists = 'lab_time' in self.df.columns
            if not lab_time_exists:
                self.df['lab_time'] = pd.NA  # create column if missing
                self.df['lab_time'] = self.df['lab_time'].fillna('00:00').astype(str) ########## ADDED DEC 4

            # Replace missing times with midnight
            self.df['teg_time'] = self.df['teg_time'].fillna('00:00').astype(str)

            if lab_time_exists:
                self.df['lab_time'] = self.df['lab_time'].astype(str)
                # Replace only rows where teg_time is '00:00'
                self.df.loc[self.df['teg_time'] == '00:00', 'teg_time'] = self.df['lab_time']
            else:
                self.df['lab_time'] = self.df['teg_time'].fillna('00:00').astype(str)

            # Define fallback date (lab_date_visit if Draw_date missing)
            if 'lab_date_visit' in self.df.columns:
                fallback_dates = pd.to_datetime(self.df['lab_date_visit'], errors='coerce')
            else:
                fallback_dates = pd.Series([pd.NaT] * len(self.df), index=self.df.index)

            # Build Draw_date_teg
            self.df['Draw_date_teg'] = np.where(
                has_time,
                parsed_draw.astype(str),
                (
                    fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
                    + ' '
                    + self.df['teg_time']
                )
            )

            # Build Draw_date_lab
            if lab_time_exists:
                # If lab_time exists, only use it; leave NaT if missing
                self.df['Draw_date_lab'] = np.where(
                    has_time,
                    parsed_draw.astype(str),
                    np.where(
                        self.df['lab_time'].notna() & (self.df['lab_time'] != 'NaT'),
                        fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d') + ' ' + self.df['lab_time'],
                        np.nan  # leave missing as NaT
                    )
                )
            else:
                # If lab_time column missing entirely → fallback to teg_time
                self.df['Draw_date_lab'] = np.where(
                    has_time,
                    parsed_draw.astype(str),
                    (
                        fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
                        + ' '
                        + self.df['teg_time']
                    )
                )
            
            # Hemoglobin consistency rules

            # Ensure Draw_date_lab and Draw_date_teg are proper datetimes
            self.df['Draw_date_lab'] = pd.to_datetime(self.df['Draw_date_lab'], errors='coerce')
            self.df['Draw_date_teg'] = pd.to_datetime(self.df['Draw_date_teg'], errors='coerce')

            # If Hemoglobin is missing → Draw_date_lab should be missing too
            self.df.loc[self.df['Hemoglobin'].isna(), 'Draw_date_lab'] = pd.NaT

            # If Hemoglobin is present but Draw_date_lab is missing → copy from Draw_date_teg
            self.df.loc[
                (~self.df['Hemoglobin'].isna()) & (self.df['Draw_date_lab'].isna()),
                'Draw_date_lab'
            ] = self.df.loc[
                (~self.df['Hemoglobin'].isna()) & (self.df['Draw_date_lab'].isna()),
                'Draw_date_teg']
            
       
    # ----------------------------------------------------------
    # STEP 14: Process AO_OTA
    # ----------------------------------------------------------
    def _Process_AO_OTA(self):
        if {'ota_type_61', 'ota_type_62'}.issubset(self.df.columns):
            self.df['AO_OTA'] = (self.df[['ota_type_61', 'ota_type_62']].apply(lambda x: '/'.join(x.dropna().astype(str)), axis=1).replace('', np.nan))
     
    # ----------------------------------------------------------
    # STEP 15: Process Medication
    # ----------------------------------------------------------
    def _Process_medication(self):
        self.df['Pre_op_med'] = self.df['StudyID'].map(self.medications_preop)
        self.df['Pre_op_med'] = self.df['Pre_op_med'].replace({np.nan: 'LMWH'})
      
    # ----------------------------------------------------------
    # STEP 16: Process Death/Withdrawals
    # ----------------------------------------------------------
    def _Process_Death_Withdrawals(self):
        # ---- Withdrawn/Death from main Withdrawn column ----
        if 'Withdrawn' in self.df.columns:
            withdrew_values = {'patient withdrew consent', 'other reason', 'lost to follow up'}

            # Initialize Death column
            self.df['Death'] = 'No'

            # Normalize Withdrawn column
            self.df['Withdrawn_norm'] = self.df['Withdrawn'].astype(str).str.strip().str.lower()

            # Death rows
            death_mask = self.df['Withdrawn_norm'] == 'death'
            self.df.loc[death_mask, ['Death', 'Withdrawn']] = ['Yes', 'No']

            # Withdrawn rows
            withdrew_mask = self.df['Withdrawn_norm'].isin(withdrew_values) & ~death_mask
            self.df.loc[withdrew_mask, 'Withdrawn'] = 'Yes'

            # All other rows
            self.df.loc[~death_mask & ~withdrew_mask, 'Withdrawn'] = 'No'

            self.df.drop(columns=['Withdrawn_norm'], inplace=True)
        else:
            self.df['Withdrawn'] = 'No'
            self.df['Death'] = 'No'

        # ---- Override Death if outcomes_outcome_type indicates mortality ----
        if 'comp_death' in self.df.columns:

            comp_death_series = self.df['comp_death']
            if isinstance(comp_death_series, pd.DataFrame):
                # Take the first column if somehow multiple columns
                comp_death_series = comp_death_series.iloc[:, 0]

            # Normalize text
            comp_death_series = comp_death_series.astype(str).str.strip().str.lower()

            # Mark mortality if 'checked' or 'yes'
            mortality_ids = self.df.loc[comp_death_series.isin(['checked', 'yes']), 'StudyID'].unique()
            self.df.loc[self.df['StudyID'].isin(mortality_ids), 'Death'] = 'Yes'
            self.df.loc[self.df['StudyID'].isin(mortality_ids), 'Withdrawn'] = 'No'

        # ---- Ensure Withdrawn/Death consistent per StudyID ----
        for study_id, group in self.df.groupby('StudyID'):
            if (group['Death'] == 'Yes').any():
                self.df.loc[self.df['StudyID'] == study_id, ['Death', 'Withdrawn']] = ['Yes', 'No']
            elif (group['Withdrawn'] == 'Yes').any():
                self.df.loc[self.df['StudyID'] == study_id, 'Withdrawn'] = 'Yes'

    # ----------------------------------------------------------
    # STEP 17: Process Treatment
    # ----------------------------------------------------------
    def _process_treatment(self):
        treatment_map = {
            # Hip
            'intra_treatment___1': 'Hemi-arthroplasty (monopolar, bipolar)',
            'intra_treatment___2': 'Total Hip Arthroplasty',
            'intra_treatment___3': 'Cannulated Screws',
            'intra_treatment___4': 'Short cephalomedullary nail',
            'intra_treatment___5': 'Long cephalomedullary nail',
            'intra_treatment___6': 'Dynamic Hip Screw',
            'intra_treatment___7': 'Other',

            # Pathway
            'intraop_treatment___1': 'Hemi-arthroplasty (monopolar, bipolar)',
            'intraop_treatment___2': 'Total Hip Arthroplasty',
            'intraop_treatment___3': 'Cannulated Screws',
            'intraop_treatment___4': 'Short cephalomedullary nail',
            'intraop_treatment___5': 'Long cephalomedullary nail',
            'intraop_treatment___6': 'Dynamic Hip Screw',
            'intraop_treatment___7': 'Other'
        }

        def process(prefix):
            # find columns starting with the prefix
            cols = [c for c in self.df.columns if c.startswith(prefix)]
            if not cols:
                return pd.Series([np.nan] * len(self.df), index=self.df.index)

            def mapper(row):
                checked = [
                    treatment_map[col]
                    for col in cols
                    if str(row.get(col, "")).strip().lower() in ("checked", "1", "true")
                ]
                return "/".join(checked) if checked else np.nan

            return self.df.apply(mapper, axis=1)

        # Process hip & pathway treatments separately
        self.df['intra_treatment'] = process('intra_treatment')
        self.df['intraop_treatment'] = process('intraop_treatment')

        # Preferred: hip → pathway
        self.df['Treatment'] = self.df['intra_treatment'].combine_first(
            self.df['intraop_treatment']
        )

        # Final classification via arth_fix map (if you have it)
        if hasattr(self, 'arth_fix'):
            self.df['Treatment'] = self.df['Treatment'].map(self.arth_fix).fillna(
                self.df['Treatment']
            )

    # ----------------------------------------------------------
    # STEP 18: Blood Transfusion
    # ----------------------------------------------------------

    def _add_blood_transfusion(self):
        self.df['blood_rbc'] = pd.to_numeric(self.df['blood_rbc'], errors='coerce').fillna(0)
        total_blood_transfusions = self.df.groupby('StudyID')['blood_rbc'].sum().to_dict()
        
        self.df['total_blood_rbc'] = self.df['StudyID'].map(total_blood_transfusions)
        self.df['blood_rbc_yn'] = np.where(self.df['total_blood_rbc']==0, 'No','Yes')

    # ----------------------------------------------------------
    # STEP 19: Study Names
    # ----------------------------------------------------------
    def _add_study_names(self):
        self.df['Study'] = self.df['StudyID'].str.extract(r'^(TH|HPA|TF|TPA)').replace({
                'TH': 'Hip',
                'HPA': 'Pathway',
                'TF': 'Femur',
                'TPA': 'Pelvis'
            })

    # ----------------------------------------------------------
    # STEP 20: CAS-score
    # ----------------------------------------------------------
    def _compute_cas(self):
        if 'CAS' not in self.df.columns:
            self.df['CAS'] = np.nan

        if 'cas_timepoint' in self.df.columns:
                    timepoint_dict_lower = {k.lower(): v for v_list in self.timepoint_dict.values() for k in v_list for v in [list(self.timepoint_dict.keys())[list(self.timepoint_dict.values()).index(v_list)]]}
                    self.df['cas_timepoint'] = self.df['cas_timepoint'].astype(str).str.lower().map(timepoint_dict_lower)
                    # self.df['cas_timepoint']=self.df['cas_timepoint'].apply(self._map_timepoint)
                    self.df['CAS'] = pd.to_numeric(self.df['CAS'], errors='coerce')
                    df_cas = self.df.dropna(subset=['CAS'])[['StudyID','cas_timepoint','CAS']].rename(columns={'cas_timepoint':'Time'})
                

                    self.df = self.df.drop(columns=['CAS','cas_timepoint']).merge(df_cas, on=['StudyID', 'Time'], how='left')

        

    # ----------------------------------------------------------
    # Build records
    # ----------------------------------------------------------
    def _build_records(self):
        """
        Build Record objects (not DataFrames).
        Later, you can convert them to DataFrames with record.to_dataframe().
        """
        self.records = {}

        for study_id, rows in self.df.groupby("StudyID"):

            # EXTRACT DEMOGRAPHICS

            demo = {}
            for col in self.demographic_cols:
                if col in rows.columns:
                    values = rows[col].dropna()
                    demo[col] = values.iloc[0] if len(values) > 0 else None

            # BUILD BLOOD DRAW OBJECTS

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


            # CREATE RECORD OBJECT

            record = Record(
                record_id=study_id,
                demographics=demo,
                blood_draws=blood_draws,
            )

            # compute time differences inside the record
            record.add_time_differences()
            

            self.records[study_id] = record

        print("Records Successfully Built!")
        print(f"********* This study has {self.df['StudyID'].nunique()} patients ************")


    ##### HELPER METHODS #####
    ### ALL PATIENTS
    def get_all_demographics(self):
        rows = []
        for sid, rec in self.records.items():
            row = rec.demographics.copy()
            row["StudyID"] = sid
            rows.append(row)
        return pd.DataFrame(rows)

    


    def get_all_labs(self):

        # Collect all dataframe rows
        dfs = [rec.to_lab_dataframe() for rec in self.records.values()]
        dfs = [df for df in dfs if not df.empty]
        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)

        teg_cols = ['Time','R_time','K_time','Alpha_Angle','MA','LY30','ACT','ADP-agg','ADP-inh','ADP-ma','AA-agg','AA-inh','AA-ma','Draw_date_teg','time_injury_teg_hours']
        lab_cols = ['Time','Hemoglobin', 'Creatinine','Draw_date_lab','time_injury_lab_hours']

       
        if "Draw_ID" in out.columns:
            out["draw_group"] = out["Draw_ID"].astype(str).str.split(r"[_/]").str[0]
        else:
            out["draw_group"] = np.nan
        
    
        # Build TEG-only frame
        teg_rows = out.dropna(subset=teg_cols, how="all").copy()
        teg_rows = teg_rows[['StudyID','draw_group'] + teg_cols].drop_duplicates()

        # Build LAB-only frame
        lab_rows = out.dropna(subset=lab_cols, how="all").copy()
        # lab_rows["Time"] = lab_rows["draw_group"]
        lab_rows = lab_rows[['StudyID','draw_group'] + lab_cols].drop_duplicates()
        lab_rows['Time'] = lab_rows['Time'].fillna(lab_rows['draw_group'])

        # Merge LAB + TEG
        merged = pd.merge(teg_rows, lab_rows, on=["StudyID","draw_group",'Time'], how="outer")

        merged = pd.merge(
            teg_rows, 
            lab_rows, 
            on=["StudyID","draw_group",'Time'], 
            how="outer", 
            suffixes=('_teg','_lab')
            )
        
        merged = merged.groupby(['StudyID','draw_group','Time'], as_index=False).agg(lambda x: x.dropna().iloc[0] if x.notna().sum() > 0 else np.nan)
        
     

        # Attach the rest of data
        metadata_cols = ['Study','CAS','VTE_type','VTE','Pre_op_med','draw_group']
        metadata = out[['StudyID','Time'] + metadata_cols].drop_duplicates()

        merged = pd.merge(
            merged, metadata,
            on=["StudyID",'draw_group',"Time"],
            how="inner"
        )


        # Fix order
        order = ['Study','StudyID','draw_group','Pre_op_med','Time','CAS','VTE','VTE_type','Draw_date_lab','time_injury_lab_hours','Hemoglobin','Creatinine','Draw_date_teg','time_injury_teg_hours','R_time','K_time','Alpha_Angle','MA','LY30','ACT','ADP-agg','ADP-inh','ADP-ma','AA-agg','AA-inh','AA-ma']

        final_order = [c for c in order if c in merged.columns]

        merged = merged[final_order]
      
        return merged


        

      




    def get_full_dataframe(self):
        dfs = [rec.to_dataframe() for rec in self.records.values()]
        dfs = [df for df in dfs if not df.empty]
        if len(dfs) == 0:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    

    def get_demo(self, StudyID):
        """
        Return the demographics stored for a single patient as a DataFrame.
        Read-only, no extra columns added.
        """
        if StudyID not in self.records:
            return pd.DataFrame()  
        return pd.DataFrame([self.records[StudyID].demographics.copy()]).transpose()
    

    def get_draws(self, StudyID):
        """
        Return all blood-draw rows for a single patient by filtering the
        DataFrame returned by self.get_all_labs().
        The returned DataFrame is a copy (read-only for callers).
        """
        # Get the combined table you already wrote
        all_labs = self.get_all_labs()
        if all_labs.empty:
            return pd.DataFrame()

        # Filter by StudyID and return a copy
        patient_df = all_labs[all_labs.get("StudyID") == StudyID].copy()

        # If you'd rather return an empty DataFrame when not found:
        if patient_df.empty:
            return pd.DataFrame()

        # Optional: reset index and sort by Time if present
        if "Time" in patient_df.columns:
            patient_df = patient_df.sort_values(by=["Time"]).reset_index(drop=True)
        else:
            patient_df = patient_df.reset_index(drop=True)

        return patient_df
    
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

    def add_time_differences(self):
        injury = self.demographics.get('Injury_date')
        surgery = self.demographics.get('Surgery_date')

        if injury is not None and surgery is not None:
            injury_dt = pd.to_datetime(injury, errors='coerce')
            surgery_dt = pd.to_datetime(surgery, errors='coerce')
            if pd.notna(injury_dt) and pd.notna(surgery_dt):
                self.demographics['time_injury_to_surgery_hours'] = (surgery_dt - injury_dt).total_seconds() / 3600
            else:
                self.demographics['time_injury_to_surgery_hours'] = None
        else:
            self.demographics['time_injury_to_surgery_hours'] = None

    def get_demographics(self):
        return self.demographics

    def get_all_labs(self):
        return self.blood_draws


    # ---------------------------------------------------
    # ⭐ Return demographics as a DataFrame
    # ---------------------------------------------------
    def to_demographics_dataframe(self):
        df = pd.DataFrame([self.demographics])
        df.insert(0, "StudyID", self.record_id)
        return df

    # ---------------------------------------------------
    # ⭐ Return labs-only as a DataFrame
    # ---------------------------------------------------
    def to_lab_dataframe(self):
        """Each blood draw = one row; includes time from injury to lab/TEG."""
        if len(self.blood_draws) == 0:
            return pd.DataFrame()

        # get injury date from demographics
        injury_date = self.demographics.get("Injury_date")
        if injury_date is not None:
            # ensure it is a pandas datetime
            injury_date = pd.to_datetime(injury_date)

        rows = []
        for draw in self.blood_draws:
            # Build Draw_ID (you can also include redcap_event_name etc.)
            draw_id = getattr(draw, "draw_id", None) or "_".join(
                str(draw.__dict__.get(col, "")) 
                for col in ["redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance"]
            ).strip("_")

            row = {"StudyID": self.record_id, "Draw_ID": draw_id}

            # Copy all lab fields
            for k, v in draw.__dict__.items():
                if k not in ["draw_id", "redcap_event_name", "redcap_repeat_instrument", "redcap_repeat_instance"]:
                    row[k] = v

            # Compute time differences if possible
            # Lab
            draw_date_lab = getattr(draw, "Draw_date_lab", None)
            if injury_date is not None and draw_date_lab is not None:
                draw_date_lab = pd.to_datetime(draw_date_lab)
                row["time_injury_lab_hours"] = (draw_date_lab - injury_date).total_seconds() / 3600
            else:
                row["time_injury_lab_hours"] = None

            # TEG
            draw_date_teg = getattr(draw, "Draw_date_teg", None)
            if injury_date is not None and draw_date_teg is not None:
                draw_date_teg = pd.to_datetime(draw_date_teg)
                row["time_injury_teg_hours"] = (draw_date_teg - injury_date).total_seconds() / 3600
            else:
                row["time_injury_teg_hours"] = None

            rows.append(row)

        return pd.DataFrame(rows)


    # ---------------------------------------------------
    # ⭐ Full flattened DF = demographics + labs
    # ---------------------------------------------------
    def to_dataframe(self):
        df_labs = self.to_lab_dataframe()
        if df_labs.empty:
            return pd.DataFrame()

        for col, val in self.demographics.items():
            df_labs[col] = val

        return df_labs

