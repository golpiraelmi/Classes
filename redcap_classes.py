
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
        self.records = {}   # <-- will store Record objects

        # ---- Column replacement dictionary ----
        self.replacement_dict = {
            ("patient_id", "record_id"): "StudyID",
            ("demo_age", "bl_age","baseline_age"): "Age",
            ("demo_sex", "bl_sex","baseline_sex"): "Sex",
            ("bmi_calc", "bl_bmi_calc","baseline_bmi","bl_bmi"): "BMI",
            ("bloodwork_hemoglobin", "lp_hemoglobin","blood_work_hemoglobin","lp_hemoglobin","teg_hgb",): "Hemoglobin",
            ("bloodwork_creatinine", "lp_creatinine","blood_work_creatinine","lp_creatinine","teg_creatinine"): "Creatinine",
            ("bloodwork_teg_crt_r", "rteg_crt_rvalue","blood_work_teg_crt_r","crt_rvalue","teg_crt_r",): "R_time",
            ("bloodwork_teg_crt_k", "rteg_crt_ktime","blood_work_teg_crt_k","crt_ktime","teg_crt_k"): "K_time",
            ("bloodwork_teg_crt_ang", "rteg_crt_aangle","blood_work_teg_crt_ang","crt_alpha","teg_crt_aangle"): "Alpha_Angle",
            ("bloodwork_teg_crt_ma", "rteg_crt_ma","blood_work_teg_crt_ma","crt_ma","teg_crt_ma",): "MA",
            ("bloodwork_teg_crt_ly30", "rteg_crt_ly30","blood_work_teg_crt_ly30","crt_ly30","teg_crt_ly30"): "LY30",
            ("bloodwork_teg_crt_act", "rteg_crt_tegact","blood_work_teg_crt_act","crt_act","teg_crt_tegact"): "ACT",
            ("blood_work_teg_adp_agg","pm_adp_aggregation","teg_adp_agg"): "ADP-agg",
            ("blood_work_teg_adp_inh","pm_adp_inhibition","teg_adp_inh"): "ADP-inh",
            ("blood_work_teg_adp_ma","pm_adp_ma","teg_adp_ma"): "ADP-ma",
            ("blood_work_teg_aa_agg","pm_aa_aggregation","teg_aa_agg"): "AA-agg",
            ("blood_work_teg_aa_inh","pm_aa_inhibition","teg_aa_inh"): "AA-inh",
            ("blood_work_teg_aa_ma","pm_aa_ma","teg_aa_ma"): "AA-ma",
            ("blood_products_rbc","blood_rbc",): "rbc",
            ("lab_rteg_timepoint","bloodwork_timepoint","blood_work_timepoint","rteg_timepoint"):'Time',
            ('date_time_injury','adm_injury_date','date_injury'):'Injury_date',
            ('admission_date_time','adm_er_date','adm_date'): "Admission_date", 
            ('surgery_date_time','intra_op_date','intraop_date_surg','postop_dt_surg'):'Surgery_date',
            ('teg_date_time','lab_dt_blood_draw','teg_date','dt_blood_drawn','teg_run_date'):'Draw_date',
            ('teg_run_time','teg_time',):'teg_time',
            ('teg_time_lab_panel',):'lab_time',
            ('aoota_classification','inj_aoota',):'AO_OTA',
            ('comp_dvt_yn','complication_dvt',):'DVT',
            ('comp_pe_yn','complication_pe',):'PE',
            ('reason_withdrawal','wd_reason','study_wd_reason'): 'Withdrawn',
            ('outcomes_outcome_type___4','complication_death','comp_death_yn'):'comp_death'
            
           
            
            
        }

        # ---- Timepoint dictionary ----
        self.timepoint_dict = {
            "Admission" : ['Admission', 'Admission/Pre-Op', 'admission', 'Emergency Admission', 'emergency admission', 'admission/pre-operative', 'admission/ pre-operative', 
                           'admission/pre-op', 'admisssion', 'pre op/admission', 'admit', 'admission/post-fracture day 1','Admission/ Pre-Operative','Admission/Pre-Operative','Pre op/admission','Admission/Pre-op'],

            "Pre-Op": ['Pre-Op','Pre Op','pre op','PRE OP','Pre Operative','Pre op','Pre-Operative Day',
                    'Pre-OP','pre-op','Pre-operative','Pre-Operative Day 1/OR Day','Pre-Op/OR Day','Preop',
                    'Pre-Operative','Pre-op','1 hour pre-op','1hr pre-op','pre-op 1 hour','pre-op (unsch. day 5)',
                    'preop 1 hour','ex-fix pod 4/preop','preop', 'Ex-fix POD 4/Preop','PREOP'],

            "POST_OP": ['reaming', 'intraoperative', 'post-operative', 'post-op', '1h post-op', '1 hour post-op', 'post op', '1hr post-op', '1 hour post op',
                    '1 hour po', '1 hour post ream', 'post reaming', 'post ream operation 1', 'post ream', '1 hour post-ream', '1 hour post reaming', 'postream',
                    'po reaming', 'post-ream', 'post reaming ','Post-Operative', 'Post-Op', 'Post-operative', 'Post-op', '1 Hour Post-Op', '1 Hour Post Op', 'POST REAMING', 'PO REAMING', '1 hour PO',
                    'Reaming ', 'Post Reaming ','Post-Ream','POSTREAM','Post- REAM', 'Post-REAM', '1hr Post-Op','POST-REAM','Post ream operation 1'],


            "PFD1": ['PFD1','PFD 1','Post Frac/Pre-Op','Post Fracture Day 1','Day 1 post #','Day 1 Post #','postfractureday1',
                     'POSTFRACTUREDAY1','POST FRACTURE DAY 1','Post-Fracture Day 1','post fracture day 1',
                     'Preop/Post fracture Day 1','Post-fracture day 1','Post fracture D1','Post Fracture Day1',
                     'Day 1 - Post Fracture','Post frac Day 1 - Pre-op','Post Frac Day 1','Day 1', 'admission/post-fracture day 1','Post-fracture #1/ Pre-op','postfracture day 1', 'post-fracture day 1'],

            "PFD2": ['PFD2','PFD 2','Day 2 post #','postfractureday2','POSTFRACTUREDAY2','POST FRACTURE DAY 2',
                     'Post Fracture Day 2','Post Frac Day 2','Day 2 Post fracture/Pre-Op','Day 2 Post #','post fracture day 2','Ex-Fix POD 2','Unscheduled post ex-fix'],

            "PFD3": ['Day 3 post #','Post Frac Day 3','Ex-Fix POD 3'],

            "PFD4": ['Post Frac Day 4'],

            "POD1" : ['post op day 1', 'POD1', 'Day 1 post-op', 'PO Day 1', 'Day 1 Post-Op', 'POD 1', 'POD 1 ', 'Day 1 post op', 'Post Operative Day 1', 'Day 1 Post-op',
                       'Day 1 post o', 'Postoperative Day 1', 'Pod 1', '24h Post-Op', '24hrs post-op', 'post operative day 1', '24hr Post-Op', '24 Hours Post-Op', 
                       '24h post-op', '24h post=op', 'pod 1', 'po day 1', '24 hour po', '24h post op', 'post operative day 1', '24hrs post-op', '24hr post-op', 'post op day 1', 
                       'po day 1/24hrs po', 'pod1', '24 hours po', 'po day1/24hrs po', '24 hours post-op','24h Post=Op', '24h Post-op', 'PO Day1/24hrs PO', 'Post operative Day 1'],


            "POD2": ['POD 2','POD2','Day 2 post-op','Day 2 Post-Op','Day 2 post op','PO Day 2','Day 2 Post-op',
                    'Post Operative Day 2','Day  2 post-op','48h Post-Op (Discharge)','48h Post-op','48hrs post-op','POD2',
                    'post operative day 2','48hr Post-Op','48 Hours Post-Op','48h post op','48 hours post-op','48hrs post-op',
                    'po day 2/48 hours po','po day 2/48hrs po','48h post-op','48h post-op','48h post-op','po day 2',
                    'post operative day 2','48 hour po','48h post-op (discharge)','pod 2','48hr post-op','pod2','48 hours po', '48h Post-Op','48 hours PO','PO Day 2/48 hours PO'],


            "POD3": ['POD3','Day 3 post-op','POD 3','Day 3 Post-Op','PO Day 3','Day 3 post op','Post Operative Day 3',
                    'Day 3 Post-op','Day 3 pot-op','Day 3 PO','POD3','PO Day 3/72hrs PO','72 Hours Post-Op (Discharge)',
                    '72h Post-op','72h post-op','72hr Post-Op','pod 3 and pod 1','72h post-op (discharge)','72 hours post-op (discharge)',
                    '72hrs post-op','72 hour po','72 hours po','pod 3','po day 3','post operative day 3','72 hours post-op','72h post op','pod3','pod 3','po day 3/72hrs po','72h Post-Op (Discharge)',
                    '72h Post-Op', '72 Hours Post-Op', '72 hours PO', '72 hour PO'],


            "POD4": ['POD 4','Day 4 post-op','Day 4 Post-Op','PO Day 4','Day 4 post op','Day 4 Post-op','POD4',
                'post operative day 4','96hr Post-Op','pod 4 and pod 2','96h post-op (discharge)','96h post-op','96h post op',
                'po day 4/96hrs po','po day 4/92hrs po','pod 4','pod4','96h post-op (discharge)','96hr post-op','96 hours po',
                '96 hour po','po day 4','96 hours post-op','postoperative day 4','96hrs post-op','96h Post-Op (Discharge)', '96h Post-Op', '96h Post-op', '96 Hours Post-Op', '96 hours PO', 
                'Post Operative Day 4', 'Postoperative Day 4'],


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
                '4 weeks f/u','4 week follow up','4 week ','4week','Week4','4 weeks','4 week fu','4week','4 week F/U',],


            "Week6": ['6 week','6 weeks follow up','6 Week FU','6weeks','6 weeks','6-Week','6 Week','6 Week F/U',
                        '6 weeek f/u','6 week f/u','6 week fu', '6 week F/U', '6 Weeek F/U','6-week'],

                      
            "Month3": ['3 month','3 Month Follow Up','3 months follow up','3 months','3 Month FU','3months',
                       '3 month follow up','3-Month','3 Month F/U','3 Month','3  month','3-month','3 month f/u',
                       'unscheduled 3 months follow up','12 weeks','3 month F/U'],

            "Month6": ['6 months']
        }

        ### PRE_OP_DOAC
        self.medications = {
            **dict.fromkeys(
                ['HPA-001', 'HPA-004', 'HPA-008', 'HPA-009', 'HPA-010',
                    'HPA-012', 'HPA-014', 'HPA-015', 'HPA-016', 'HPA-017', 'HPA-019',
                    'HPA-020', 'HPA-021', 'HPA-022', 'HPA-024', 'HPA-026', 'HPA-028',
                    'HPA-029', 'HPA-030', 'HPA-032', 'HPA-033', 'HPA-035', 'HPA-036',
                    'HPA-038', 'HPA-039', 'HPA-042', 'HPA-043','HPA-048', 'HPA-050' ,
                    'TH-162', 'TH-170', 'TH-198', 'TH-212', 'TH-217', 
                    'TH-225', 'TH-227', 'TH-236', 'TH-240', 'TH-244', 
                    'TH-255', 'TH-262', 'TH-267', 'TH-274', 'TH-284','TH-286',
                    'TH-302', 'TF-121','TF-128', 'TPA-058','TPA-082','TPA-093' ], "Yes"),
                     **dict.fromkeys(['TH-110'], np.nan)
        }


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
        


        # self.vte_type_map = {**dict.fromkeys(
        #     ["TH-003", "TH-201", "TH-227", "TH-253", "TH-264", "TH-301","TF-069", "TF-073", "TF-085", "TF-120","TPA-016", "TPA-030", "TPA-061", "TPA-095"], "DVT"),
        #     **dict.fromkeys(["TH-082","TH-088","TH-271","TH-292","HPA-028","TF-022","TPA-010", "TPA-011", "TPA-019", "TPA-021","TPA-026", "TPA-036", "TPA-081"], "PE"),
        #     **dict.fromkeys(["TH-261", "TH-279","TPA-001", "TPA-073", "TPA-097", "TPA-100"], "Both")
        # }

        # ---- Define metadata and lab columns ----
        self.metadata_cols = ['StudyID','Age','Sex','BMI','Injury_date','Admission_date','Surgery_date','AO_OTA','Treatment','DVT','PE']
        self.lab_cols = ['StudyID', 'Time', 'Hemoglobin', 'Creatinine', 'R_time', 'K_time','Alpha_Angle', 'MA', 'LY30', 'ACT', 'Injury_date', 'Draw_date_lab', 'Draw_date_teg']

        
        # Placeholder for processed DataFrame
        self.df = None


    # -----------------------
    # Fetch and process REDCap data
    # -----------------------

    def fetch_and_process(self):
        
        # Step 1: Export REDCap records
        records_data = self.project.export_records(raw_or_label='label')
        df = pd.DataFrame(records_data)

        # Step 2: Replace empty strings with NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # Step 3: Rename columns using replacement_dict
        if 'screen_patient_id' in df.columns:
            df=df.rename(columns= {'screen_patient_id':'StudyID', 'record_id':'index'})
        
        col_mapping = {}
        for keys, standard_name in self.replacement_dict.items():
            for k in keys:
                if k in df.columns:
                    col_mapping[k] = standard_name
        df = df.rename(columns=col_mapping)

        if 'StudyID' in df.columns:
            studyid_upper = df['StudyID'].astype(str).str.upper()

            to_remove_reasons = {
                'TH-226': 'Treated non-operatively',
                # 'HPA-048': 'Excluded just to match my numbers to past reports',
                # 'HPA-050': 'Excluded just to match my numbers to past reports',
                'TF-070': 'Multiple Surgery Patient_TIMEPOINT_ISSUE',
                'TF-084': 'Multiple Surgery Patient_TIMEPOINT_ISSUE',

                'TPA-019': 'Multiple Surgery Patient',
                'TPA-028': 'Multiple Surgery Patient',
                'TPA-043': 'Multiple Surgery Patient',
                'TPA-048': 'Multiple Surgery Patient',
                'TPA-056': 'Multiple Surgery Patient',
                'TPA-079': 'Multiple Surgery Patient'
            }

            to_remove = list(to_remove_reasons.keys())

            present_to_remove = df['StudyID'].isin(to_remove) | df['StudyID'].str.startswith('TPANO')

            if present_to_remove.any():
                found_ids = df.loc[present_to_remove, 'StudyID'].unique()
                
                print("Removing the following StudyIDs from dataset:")
                for sid in found_ids:
                    reason = to_remove_reasons.get(sid, 'Excluded - Non-Operative Arm')
                    print(f" - {sid}: {reason}")
                
                df = df[~present_to_remove].copy()


        



        if 'index' in df.columns:
            df['index'] = df['index'].astype(str).str.strip()
            df['StudyID'] = df['StudyID'].replace('nan', np.nan)
            df['StudyID'] = df.groupby('index')['StudyID'].ffill().bfill()

        # Step 4: Filter by screening_status (if column exists)
        if 'screening_status' in df.columns:
            df['StudyID'] = df['StudyID'].astype(str)
            df['StudyID'] = df.groupby('index')['StudyID'].ffill().bfill()
            df['screening_status'] = df.groupby('index')['screening_status'].ffill().bfill()
            df = df[df['screening_status'].astype(str).str.strip() == 'Eligible → enrolled']

        


        # --- Step 4: Assign to self.df and replacing missing values ---
        self.df = df

        self._replace_missing_values()

        if 'Draw_date' in df.columns:
            # --- Step 1: Parse Draw_date safely ---
            parsed_draw = pd.to_datetime(df['Draw_date'], errors='coerce')

            # --- Step 2: Identify if Draw_date has a time (non-midnight) ---
            has_time = parsed_draw.dt.time.astype(str) != "00:00:00"

            # --- Step 3: Ensure time columns exist ---
            if 'teg_time' not in df.columns:
                df['teg_time'] = pd.NA
            teg_exists = True

            # If lab_time missing entirely, mark flag
            lab_time_exists = 'lab_time' in df.columns
            if not lab_time_exists:
                df['lab_time'] = pd.NA

            # --- Step 4: Replace missing times with midnight ---
            df['teg_time'] = df['teg_time'].fillna('00:00').astype(str)
            if lab_time_exists:
                df['lab_time'] = df['lab_time'].astype(str)
            else:
                df['lab_time'] = df['lab_time'].fillna('00:00').astype(str)

            # --- Step 5: Define fallback date (lab_date_visit if Draw_date missing) ---
            if 'lab_date_visit' in df.columns:
                fallback_dates = pd.to_datetime(df['lab_date_visit'], errors='coerce')
            else:
                fallback_dates = pd.Series([pd.NaT] * len(df), index=df.index)

            # --- Step 6: Build Draw_date_teg ---
            df['Draw_date_teg'] = np.where(
                has_time,
                parsed_draw.astype(str),
                (
                    fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
                    + ' '
                    + df['teg_time']
                )
            )

            # --- Step 7: Build Draw_date_lab ---
            if lab_time_exists:
                # If lab_time exists, only use it; leave NaT if missing
                df['Draw_date_lab'] = np.where(
                    has_time,
                    parsed_draw.astype(str),
                    np.where(
                        df['lab_time'].notna() & (df['lab_time'] != 'NaT'),
                        fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d') + ' ' + df['lab_time'],
                        np.nan  # leave missing as NaT
                    )
                )
            else:
                # If lab_time column missing entirely → fallback to teg_time
                df['Draw_date_lab'] = np.where(
                    has_time,
                    parsed_draw.astype(str),
                    (
                        fallback_dates.combine_first(parsed_draw).dt.strftime('%Y-%m-%d')
                        + ' '
                        + df['teg_time']
                    )
                )

            # --- Step 8: Convert both to datetime ---
            df['Draw_date_teg'] = pd.to_datetime(df['Draw_date_teg'], errors='coerce')
            df['Draw_date_lab'] = pd.to_datetime(df['Draw_date_lab'], errors='coerce')


        if 'adm_injury_time' in df.columns and 'Injury_date' in df.columns:
            df['Injury_date'] = pd.to_datetime(df['Injury_date'].astype(str) + ' ' + df['adm_injury_time'].astype(str),
            errors='coerce')

        if 'time_injury' in df.columns and 'Injury_date' in df.columns:
            df['Injury_date'] = pd.to_datetime(df['Injury_date'].astype(str) + ' ' + df['time_injury'].astype(str),
            errors='coerce')

        if 'intraop_time_surg' in df.columns and 'Surgery_date' in df.columns:
            df['Surgery_date'] = pd.to_datetime(df['Surgery_date'].astype(str) + ' ' + df['intraop_time_surg'].astype(str),
            errors='coerce')

        if {'ota_type_61', 'ota_type_62'}.issubset(df.columns):
            df['AO_OTA'] = (df[['ota_type_61', 'ota_type_62']].apply(lambda x: '/'.join(x.dropna().astype(str)), axis=1).replace('', np.nan))

        if 'complication_dvt' in df.columns:
            df['DVT'] = np.where(df['complication_dvt']=='Yes','DVT','No')

        if 'complication_pe' in df.columns:
            df['PE'] = np.where(df['complication_pe']=='Yes','PE','No')


        # Merge multiple timepoint columns into a single 'Time' column
        timepoint_cols = ['teg_preop_tp', 'teg_postop_tp2', 'teg_fu_tp']
        existing_timepoint_cols = [col for col in timepoint_cols if col in df.columns]

        if existing_timepoint_cols:
            # Take the first non-null value across the columns
            df['Time'] = df[existing_timepoint_cols].bfill(axis=1).iloc[:, 0]


        # --- Merge surgery date columns into a single 'Surgery_date' column ---
        surg_date_cols = ['surg_date_pelvis', 'surg_date_ant_acet', 'surg_date_post_acet']
        existing_surg_cols = [col for col in surg_date_cols if col in df.columns]

        if existing_surg_cols:
            # Merge into one column using first non-null value
            df['Surgery_date'] = df[existing_surg_cols].bfill(axis=1).iloc[:, 0]

            # Optional: ensure datetime type
            df['Surgery_date'] = pd.to_datetime(df['Surgery_date'], errors='coerce')

            # Fill the same Surgery_date across all rows for each StudyID
            df['Surgery_date'] = df.groupby('StudyID')['Surgery_date'].transform(lambda x: x.ffill().bfill())


        # ---- Withdrawn/Death from main Withdrawn column ----
        if 'Withdrawn' in df.columns:
            withdrew_values = {'patient withdrew consent', 'other reason', 'lost to follow up'}

            # Initialize Death column
            df['Death'] = 'No'

            # Normalize Withdrawn column
            df['Withdrawn_norm'] = df['Withdrawn'].astype(str).str.strip().str.lower()

            # Death rows
            death_mask = df['Withdrawn_norm'] == 'death'
            df.loc[death_mask, ['Death', 'Withdrawn']] = ['Yes', 'No']

            # Withdrawn rows
            withdrew_mask = df['Withdrawn_norm'].isin(withdrew_values) & ~death_mask
            df.loc[withdrew_mask, 'Withdrawn'] = 'Withdrew'

            # All other rows
            df.loc[~death_mask & ~withdrew_mask, 'Withdrawn'] = 'No'

            df.drop(columns=['Withdrawn_norm'], inplace=True)
        else:
            df['Withdrawn'] = 'No'
            df['Death'] = 'No'

        # ---- Override Death if outcomes_outcome_type indicates mortality ----
        if 'comp_death' in df.columns:
    
            
            comp_death_series = df['comp_death']
            if isinstance(comp_death_series, pd.DataFrame):
                # Take the first column if somehow multiple columns
                comp_death_series = comp_death_series.iloc[:, 0]

            # Normalize text
            comp_death_series = comp_death_series.astype(str).str.strip().str.lower()

            # Mark mortality if 'checked' or 'no'
            mortality_ids = df.loc[comp_death_series.isin(['checked', 'yes']), 'StudyID'].unique()
            df.loc[df['StudyID'].isin(mortality_ids), 'Death'] = 'Yes'
            df.loc[df['StudyID'].isin(mortality_ids), 'Withdrawn'] = 'No'

        # ---- Ensure Withdrawn/Death consistent per StudyID ----
        for study_id, group in df.groupby('StudyID'):
            if (group['Death'] == 'Yes').any():
                df.loc[df['StudyID'] == study_id, ['Death', 'Withdrawn']] = ['Yes', 'No']
            elif (group['Withdrawn'] == 'Withdrew').any():
                df.loc[df['StudyID'] == study_id, 'Withdrawn'] = 'Withdrew'



        treatment_map = {
            'intra_treatment___1': 'Hemi-arthroplasty (monopolar, bipolar)',
            'intra_treatment___2': 'Total Hip Arthroplasty',
            'intra_treatment___3': 'Cannulated Screws',
            'intra_treatment___4': 'Short cephalomedullary nail',
            'intra_treatment___5': 'Long cephalomedullary nail',
            'intra_treatment___6': 'Dynamic Hip Screw',
            'intra_treatment___7': 'Other'
        }

        checkbox_cols = list(treatment_map.keys())

        if set(checkbox_cols).issubset(df.columns):
            def map_checked_values(row):
                # Only include the treatment if the checkbox is checked (1)
                checked = [treatment_map[col] for col in checkbox_cols if row[col] == 'Checked']
                return '/'.join(checked) if checked else np.nan
            
            df['intra_treatment'] = df.apply(map_checked_values, axis=1)
            df['intra_treatment'] = df.groupby('StudyID')['intra_treatment'].ffill().bfill()
            df['Treatment'] = df['intra_treatment'].map(self.arth_fix)
        
        
        # Step 5: Standardize timepoints
        if 'Time' in df.columns:
            df['Time'] = df['Time'].apply(self._map_timepoint)

        # Step 6: Ensure all metadata columns exist
        for col in self.metadata_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Step 7: Save the processed DataFrame
        print(df['StudyID'].nunique())

        # Step 8: Build Record objects
        self._build_records()

        return self.df


    

    # -----------------------
    # Replace Missing Values
    # -----------------------
    def _replace_missing_values(self):
        """Convert common REDCap missing codes to NaN"""
        missing_values = ['None', '-999', '', 'NaN', None]
        self.df.replace(missing_values, np.nan, inplace=True)


    # -----------------------
    # Map timepoints
    # -----------------------
    def _map_timepoint(self, tp):
        if pd.isna(tp):
            return tp
        for standard, variants in self.timepoint_dict.items():
            if tp in variants:
                return standard
        return tp
    
    # -----------------------
    # Build records
    # -----------------------
    def _build_records(self):
        self.records = {}
        for study_id, rows in self.df.groupby("StudyID"):

            # Demographics
            demo = rows[self.metadata_cols].apply(
                lambda col: col.dropna().iloc[0] if col.dropna().any() else None
            )
            demo_dict = demo.to_dict()

            




            blood_draws = []
            for _, row in rows.iterrows():
                if pd.notnull(row.get("Draw_date")):
                    bd_data = row[self.lab_cols].to_dict()
                    bd_data["Draw_date"] = row["Draw_date"]
                    blood_draws.append(BloodDraw(row["Draw_date"], **bd_data))


            # Save Record
            # self.records[study_id] = Record(study_id, demographics=demo_dict, blood_draws=blood_draws)
            rec = Record(study_id, demographics=demo_dict, blood_draws=blood_draws)
            rec.add_time_differences()  # <-- calculate once per patient
            self.records[study_id] = rec

    # ------------------------------------------------------------------------------
    # Get demographics for a patient
    # ------------------------------------------------------------------------------
    def get_patient_demographics(self, patient_id):
        if self.df is None:
            raise ValueError("Data not loaded. Run fetch_and_process() first.")

        patient_rows = self.df[self.df['StudyID'] == patient_id]
        if patient_rows.empty:
            return None

        # Take first non-null value for each metadata column
        patient_demo = patient_rows[self.metadata_cols].apply(
            lambda col: col.dropna().iloc[0] if col.dropna().any() else None
        )

        # --- Pre-op DOAC ---
        medication = self.medications.get(patient_demo["StudyID"], None)
        patient_demo["Pre_op_doac"] = medication if medication is not None else 'No'
        if patient_demo['Pre_op_doac'] == 'NoData':
            patient_demo['Pre_op_doac'] = np.nan

        # Dates
        patient_demo['Injury_date'] = pd.to_datetime(patient_demo['Injury_date'], errors="coerce")
        patient_demo['Surgery_date'] = pd.to_datetime(patient_demo['Surgery_date'], errors="coerce")


        # ---- Determine DVT/PE/SVT ----
        def flag_event(column_name):
            if column_name in patient_rows.columns:
                checked = patient_rows[column_name].astype(str).str.strip().str.lower() == 'checked'
                return 'Yes' if checked.any() else 'No'
            return 'No'

        patient_demo['DVT'] = 'DVT' if flag_event('outcomes_vte_type___1') == 'Yes' else 'No'
        patient_demo['PE']  = 'PE'  if flag_event('outcomes_vte_type___2') == 'Yes' else 'No'
        patient_demo['SVT'] = 'SVT' if flag_event('outcomes_vte_type___3') == 'Yes' else 'No'

        # ---- Determine Withdrawn / Death status ----
        if {'Withdrawn', 'Death'}.issubset(patient_rows.columns):
            # Take the first non-null value for each
            patient_demo['Death'] = patient_rows['Death'].dropna().iloc[0] if patient_rows['Death'].notna().any() else 'No'
            patient_demo['Withdrawn'] = patient_rows['Withdrawn'].dropna().iloc[0] if patient_rows['Withdrawn'].notna().any() else 'No'
        else:
            patient_demo['Death'] = 'No'
            patient_demo['Withdrawn'] = 'No'

        


        

        # ---- Construct VTE_type dynamically ----
        if patient_demo['DVT'] == 'DVT' and patient_demo['PE'] == 'PE':
            patient_demo['VTE_type'] = 'Both'
        elif patient_demo['DVT'] == 'DVT':
            patient_demo['VTE_type'] = 'DVT'
        elif patient_demo['PE'] == 'PE':
            patient_demo['VTE_type'] = 'PE'
        else:
            patient_demo['VTE_type'] = None

        # ---- Add simple Yes/No VTE summary ----
        patient_demo['VTE'] = 'Yes' if patient_demo['VTE_type'] is not None else 'No'

        # ---- Time from injury to surgery ----
        patient_demo['time_injury_to_surgery_hours'] = (
            patient_demo['Surgery_date'] - patient_demo['Injury_date']
        ).total_seconds() / 3600 if pd.notnull(patient_demo['Surgery_date']) and pd.notnull(patient_demo['Injury_date']) else np.nan

        return patient_demo

    
    # ------------------------------------------------------------------------------
    # Get all demographics for all patients
    # ------------------------------------------------------------------------------
    def get_all_demographics(self):
        all_demo = []

        for record in self.records.values():
            demo = record.get_demographics().copy()
            demo['StudyID'] = record.study_id  # Ensure StudyID is included
            demo['Pre_op_doac'] = self.medications.get(record.study_id, None)
            all_demo.append(demo)

        # Build main demographics dataframe
        df_demo = pd.DataFrame(all_demo)

        
        df_demo['Pre_op_doac'] = df_demo['Pre_op_doac'].replace({None: 'No', 'NoData': np.nan})
        
        df_demo['DVT']=np.where(df_demo['DVT'].astype(str).str.strip().str.lower().isin(['yes','checked']), "DVT", 'No')
        df_demo['PE']=np.where(df_demo['PE'].astype(str).str.strip().str.lower().isin(['yes','checked']), "PE", 'No')


        if 'outcomes_vte_type___1' in self.df.columns:
            dvt_flags = (
                self.df.loc[self.df['outcomes_vte_type___1'].astype(str).str.strip().str.lower() == 'checked', 'StudyID']
                .unique()
            )
            df_demo.loc[df_demo['StudyID'].isin(dvt_flags), 'DVT'] = 'DVT'
        
        if 'outcomes_vte_type___2' in self.df.columns:
            pe_flags = (
                self.df.loc[self.df['outcomes_vte_type___2'].astype(str).str.strip().str.lower() == 'checked', 'StudyID']
                .unique()
            )
            df_demo.loc[df_demo['StudyID'].isin(pe_flags), 'PE'] = 'PE'


        if 'outcomes_vte_type___3' in self.df.columns:
            pe_flags = (
                self.df.loc[self.df['outcomes_vte_type___3'].astype(str).str.strip().str.lower() == 'checked', 'StudyID']
                .unique()
            )
            df_demo.loc[df_demo['StudyID'].isin(pe_flags), 'SVT'] = 'SVT'


    

        # ---- Construct VTE_type dynamically ----
        def classify_vte(row):
            if row['DVT'] == 'DVT' and row['PE'] == 'PE':
                return 'Both'
            elif row['DVT'] == 'DVT':
                return 'DVT'
            elif row['PE'] == 'PE':
                return 'PE'
            else:
                return None

        df_demo['VTE_type'] = df_demo.apply(classify_vte, axis=1)

        # ---- Add a simple Yes/No VTE summary column ----
        df_demo['VTE'] = np.where(df_demo['VTE_type'].isnull(), 'No', 'Yes')


       
        # ---- Withdrawn / Death status ----
        if {'Withdrawn', 'Death'}.issubset(self.df.columns):
            # Take the first non-null value for each StudyID
            death_map = (
                self.df.groupby('StudyID')['Death']
                .first()
                .to_dict()
            )
            withdrew_map = (
                self.df.groupby('StudyID')['Withdrawn']
                .first()
                .to_dict()
            )

            df_demo['Death'] = df_demo['StudyID'].map(death_map).fillna('No')
            df_demo['Withdrawn'] = df_demo['StudyID'].map(withdrew_map).fillna('No')
        else:
            df_demo['Death'] = 'No'
            df_demo['Withdrawn'] = 'No'


        # ---- Time calculations ----
        df_demo['Injury_date'] = pd.to_datetime(df_demo['Injury_date'], errors="coerce")
        df_demo['Surgery_date'] = pd.to_datetime(df_demo['Surgery_date'], errors="coerce")

        df_demo['time_injury_to_surgery_hours'] = (
            (df_demo['Surgery_date'] - df_demo['Injury_date']).dt.total_seconds() / 3600
        )

        return df_demo
    
    
    # ------------------------------------------------------------------------------
    # Get patient blood draws
    # ------------------------------------------------------------------------------

    def get_patient_blood_draws(self, study_id):
        rec = self.records.get(study_id)
        if not rec:
            return pd.DataFrame()
        rows = []
        for bd in rec.blood_draws:
            row = {"StudyID": rec.study_id}
            row.update(bd.labs)
            rows.append(row)
        return pd.DataFrame(rows)
       

    # ------------------------------------------------------------------------------
    # Get all blood draws for all patients
    # ------------------------------------------------------------------------------
    def get_all_blood_draws(self):
        all_draws = []

        for rec in self.records.values():
            demo = rec.get_demographics()
            injury_date = pd.to_datetime(demo.get('Injury_date', pd.NA), errors='coerce')
            surgery_date = pd.to_datetime(demo.get('Surgery_date', pd.NA), errors='coerce')
            dvt_flag = demo.get('DVT', 'No')
            pe_flag  = demo.get('PE', 'No')

            for bd in rec.blood_draws:
                row = {"StudyID": rec.study_id}
                row.update(bd.labs)

                # Ensure draw date columns exist
                row['Draw_date_lab'] = row.get('Draw_date_lab', pd.NA)
                row['Draw_date_teg'] = row.get('Draw_date_teg', pd.NA)

                # Convert to datetime
                draw_lab = pd.to_datetime(row['Draw_date_lab'], errors='coerce')
                draw_teg = pd.to_datetime(row['Draw_date_teg'], errors='coerce')

                # Compute hours from injury
                row['injury_to_lab_hrs'] = ((draw_lab - injury_date).total_seconds()/3600) if pd.notnull(draw_lab) and pd.notnull(injury_date) else pd.NA
                row['injury_to_teg_hrs'] = ((draw_teg - injury_date).total_seconds()/3600) if pd.notnull(draw_teg) and pd.notnull(injury_date) else pd.NA

                # Compute hours from surgery
                row['surgery_to_lab_hrs'] = ((draw_lab - surgery_date).total_seconds()/3600) if pd.notnull(draw_lab) and pd.notnull(surgery_date) else pd.NA
                row['surgery_to_teg_hrs'] = ((draw_teg - surgery_date).total_seconds()/3600) if pd.notnull(draw_teg) and pd.notnull(surgery_date) else pd.NA

                # Add patient-level info
                row["Pre_op_doac"] = self.medications.get(rec.study_id, 'No')
                row["Injury_date"] = injury_date
                row["Surgery_date"] = surgery_date

                # VTE type
                if dvt_flag == 'DVT' and pe_flag == 'PE':
                    row['VTE_type'] = 'Both'
                elif dvt_flag == 'DVT':
                    row['VTE_type'] = 'DVT'
                elif pe_flag == 'PE':
                    row['VTE_type'] = 'PE'
                else:
                    row['VTE_type'] = 'No'

                # Simple Yes/No VTE
                row['VTE'] = 'Yes' if row['VTE_type'] in ['DVT', 'PE', 'Both'] else 'No'

                all_draws.append(row)

            df_all = pd.DataFrame(all_draws)
            return df_all
    
# -----------------------
# BloodDraw and Record classes
# -----------------------
class BloodDraw:
    def __init__(self, draw_id, **labs):
        self.draw_id = draw_id
        self.labs = labs

    def __repr__(self):
        return f"BloodDraw(draw_id={self.draw_id}, labs={self.labs})"



class Record:
    def __init__(self, record_id, demographics=None, blood_draws=None):
        self.study_id = record_id
        self.demographics = demographics or {}
        self.blood_draws = blood_draws or []

    def get_demographics(self):
        return self.demographics

    def add_blood_draw(self, blood_draw):
        self.blood_draws.append(blood_draw)

    def add_time_differences(self):
        """Compute hours from injury/surgery to TEG and LAB draw times."""
    
        # Convert demographic dates
        injury_date = pd.to_datetime(self.demographics.get("Injury_date", None), errors="coerce")
        surgery_date = pd.to_datetime(self.demographics.get("Surgery_date", None), errors="coerce")

        for bd in self.blood_draws:
            # Convert draw times if they exist
            draw_date_lab = pd.to_datetime(bd.labs.get("Draw_date_lab"), errors="coerce")
            draw_date_teg = pd.to_datetime(bd.labs.get("Draw_date_teg"), errors="coerce")

            # Injury → LAB
            if pd.notnull(injury_date) and pd.notnull(draw_date_lab):
                bd.labs["injury_to_lab_hrs"] = (draw_date_lab - injury_date).total_seconds() / 3600
            else:
                bd.labs["injury_to_lab_hrs"] = np.nan

            # Injury → TEG
            if pd.notnull(injury_date) and pd.notnull(draw_date_teg):
                bd.labs["injury_to_teg_hrs"] = (draw_date_teg - injury_date).total_seconds() / 3600
            else:
                bd.labs["injury_to_teg_hrs"] = np.nan

            # Surgery → LAB
            if pd.notnull(surgery_date) and pd.notnull(draw_date_lab):
                bd.labs["surgery_to_lab_hrs"] = (draw_date_lab - surgery_date).total_seconds() / 3600
            else:
                bd.labs["surgery_to_lab_hrs"] = np.nan

            # Surgery → TEG
            if pd.notnull(surgery_date) and pd.notnull(draw_date_teg):
                bd.labs["surgery_to_teg_hrs"] = (draw_date_teg - surgery_date).total_seconds() / 3600
            else:
                bd.labs["surgery_to_teg_hrs"] = np.nan


    def __repr__(self):
        return f"<Record {self.study_id}: {len(self.blood_draws)} blood draws>"