
import pandas as pd
import numpy as np
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
            ("patient_id", "screen_patient_id"): "StudyID",
            ("demo_age", "bl_age","baseline_age"): "Age",
            ("demo_sex", "bl_sex","baseline_sex"): "Sex",
            ("bmi_calc", "bl_bmi_calc","baseline_bmi"): "BMI",
            ("bloodwork_hemoglobin", "lp_hemoglobin","blood_work_hemoglobin"): "Hemoglobin",
            ("bloodwork_creatinine", "lp_creatinine","blood_work_creatinine"): "Creatinine",
            ("bloodwork_teg_crt_r", "rteg_crt_rvalue","blood_work_teg_crt_r"): "R_time",
            ("bloodwork_teg_crt_k", "rteg_crt_ktime","blood_work_teg_crt_k"): "K_time",
            ("bloodwork_teg_crt_ang", "rteg_crt_aangle","blood_work_teg_crt_ang"): "Alpha_Angle",
            ("bloodwork_teg_crt_ma", "rteg_crt_ma","blood_work_teg_crt_ma"): "MA",
            ("bloodwork_teg_crt_ly30", "rteg_crt_ly30","blood_work_teg_crt_ly30"): "LY30",
            ("bloodwork_teg_crt_act", "rteg_crt_tegact","blood_work_teg_crt_act"): "ACT",
            ("blood_work_teg_adp_agg",): "ADP-agg",
            ("blood_work_teg_adp_inh",): "ADP-inh",
            ("blood_work_teg_adp_ma",): "ADP-ma",
            ("blood_work_teg_aa_agg",): "AA-agg",
            ("blood_work_teg_aa_inh",): "AA-inh",
            ("blood_work_teg_aa_ma",): "AA-ma",
            ("blood_products_rbc",): "rbc",
            ("lab_rteg_timepoint","bloodwork_timepoint","blood_work_timepoint"):'Time',
            ('date_time_injury','adm_injury_date'):'Injury_date',
            ('admission_date_time',): "Admission_date", 
            ('surgery_date_time','intra_op_date','intraop_date_surg'):'Surgery_date',
            



            ('teg_date_time','lab_dt_blood_draw','teg_date'):'Draw_date',
        }

        # ---- Timepoint dictionary ----
        self.timepoint_dict = {
            "Admission": ['Admission','Admission/Pre-Op','admission'],
            "Pre-Op": ['Pre-Op','Pre Op','pre op','PRE OP','Pre Operative','Pre op','Pre-Operative Day',
                       'Pre-OP','pre-op','Pre-operative','Pre-Operative Day 1/OR Day','Pre-Op/OR Day','Preop'],
            "PFD1": ['PFD1','PFD 1','Post Frac/Pre-Op','Post Fracture Day 1','Day 1 post #','Day 1 Post #','postfractureday1',
                     'POSTFRACTUREDAY1','POST FRACTURE DAY 1','Post-Fracture Day 1','post fracture day 1',
                     'Preop/Post fracture Day 1','Post-fracture day 1','Post fracture D1','Post Fracture Day1',
                     'Day 1 - Post Fracture','Post frac Day 1 - Pre-op','Post Frac Day 1','Day 1'],
            "PFD2": ['PFD2','PFD 2','Day 2 post #','postfractureday2','POSTFRACTUREDAY2','POST FRACTURE DAY 2',
                     'Post Fracture Day 2','Post Frac Day 2','Day 2 Post fracture/Pre-Op','Day 2 Post #'],
            "PFD3": ['Day 3 post #','Post Frac Day 3'],
            "PFD4": ['Post Frac Day 4'],
            "POD1": ['post op day 1','POD1','Day 1 post-op','PO Day 1','Day 1 Post-Op','POD 1','Day 1 post op',
                     'Post Operative Day 1','Day 1 Post-op','Day 1 post o','Postoperative Day 1','Pod 1'],
            "POD2": ['POD 2','POD2','Day 2 post-op','Day 2 Post-Op','Day 2 post op','PO Day 2','Day 2 Post-op',
                     'Post Operative Day 2','Day  2 post-op'],
            "POD3": ['POD3','Day 3 post-op','POD 3','Day 3 Post-Op','PO Day 3','Day 3 post op','Post Operative Day 3',
                     'Day 3 Post-op','Day 3 pot-op','Day 3 PO'],
            "POD4": ['POD 4','Day 4 post-op','Day 4 Post-Op','PO Day 4','Day 4 post op','Day 4 Post-op'],
            "POD5": ['pod 5','POD5','POD 5','Day 5 post-op','PO Day 5','Day 5 Post-Op','Day 5 post op','Day 5 Post-op',
                     'Day 5 po','Post Operative Day 5'],
            "POD7": ['POD7','POD 7','PO Day 7','Post Operative Day 7','pod 7'],
            "Week2": ['2 week','2 Week FU','2 weeks follow up','2-Week','2 weeks','2 Week','2weeks','2 Week F/U',
                      '2 week F/U','2 week follow-up','2-week','2 Week PO','2 week post#'],
            "Week4": ['4 week','4 Week FU','4 weeks','4 weeks follow up','4-Week','4weeks','4 WEEKS FOLLOW UP',
                      '4 weeks f/u','4 week follow up'],
            "Week6": ['6 week','6 weeks follow up','6 Week FU','6weeks','6 weeks','6-Week','6 Week','6 Week F/U',
                      '6 week F/U','6-week','6 week post op','6 week follow up','6 Week Follow Up'],
            "Month3": ['3 month','3 Month Follow Up','3 months follow up','3 months','3 Month FU','3months',
                       '3 month follow up','3-Month','3 Month F/U','3 Month','3  month','3-month','3 month f/u',
                       'unscheduled 3 months follow up','12 weeks'],
            "Month6": ['6 months']
        }

        # ---- Define metadata and lab columns ----
        self.metadata_cols = ['StudyID','Age','Sex','BMI','Injury_date','Admission_date','Surgery_date']
        self.lab_cols = ['StudyID','Time','Hemoglobin', 'Creatinine', 'R_time', 'K_time','Alpha_Angle', 'MA','LY30', 'ACT']
        
        # Placeholder for processed DataFrame
        self.df = None


    # -----------------------
    # Fetch and process REDCap data
    # -----------------------

    def fetch_and_process(self):
        # Step 1: Export REDCap records
        records_data = self.project.export_records()
        df = pd.DataFrame(records_data)

        # Step 2: Replace empty strings with NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # Step 3: Rename columns using replacement_dict
        col_mapping = {}
        for keys, standard_name in self.replacement_dict.items():
            for k in keys:
                if k in df.columns:
                    col_mapping[k] = standard_name
        df = df.rename(columns=col_mapping)

        # Step 4: Filter by screening_status (if column exists)
        if 'screening_status' in df.columns:
            
            df['StudyID'] = df['StudyID'].astype(str)
            df['StudyID'] = df.groupby('record_id')['StudyID'].ffill().bfill()
            df['screening_status'] = df.groupby('record_id')['screening_status'].ffill().bfill()

            if 'record_id' in df.columns:
                df['record_id'] = df['record_id'].astype(str).str.strip()
                df['StudyID'] = df['StudyID'].replace('nan', np.nan)
                df['StudyID'] = df.groupby('record_id')['StudyID'].ffill().bfill()
        
            df = df[df['screening_status'].astype(str).str.strip() == '1']

        # Step 5: Standardize timepoints
        if 'Time' in df.columns:
            df['Time'] = df['Time'].apply(self._map_timepoint)

        # Step 6: Ensure all metadata columns exist
        for col in self.metadata_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Step 7: Save the processed DataFrame
        self.df = df

        # Step 8: Build Record objects
        self._build_records()

        return self.df



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
    # Get demographics for a patient
    # -----------------------
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
        return patient_demo
    
    # -----------------------
    # Get patient blood draws
    # -----------------------

    def get_patient_blood_draws(self, patient_id):
        if self.df is None:
            raise ValueError("Data not loaded. Run fetch_and_process() first.")

        patient_rows = self.df[self.df['StudyID'] == patient_id]

        if patient_rows.empty:
            return pd.DataFrame()  # return empty DataFrame if patient not found

        # Only keep metadata + lab columns
        cols = ['Draw_date'] + self.lab_cols
        # Some Draw_date values may be NaN, drop those rows
        blood_draws = patient_rows[cols].dropna(subset=['Draw_date'])

        # Optional: reset index
        blood_draws = blood_draws.reset_index(drop=True)
        return blood_draws
    

    # -----------------------
    # Get all blood draws for all patients
    # -----------------------
    def get_all_blood_draws(self):
        if self.df is None:
            raise ValueError("Data not loaded. Run fetch_and_process() first.")

        # Only keep StudyID, Draw_date, and lab columns
        cols = ['StudyID', 'Draw_date'] + self.lab_cols
        blood_draws = self.df[cols].dropna(subset=['Draw_date']).reset_index(drop=True)

        return blood_draws
    
    # -----------------------
    # Get all demographics for all patients
    # -----------------------
    def get_all_demographics(self):
    
        all_demo = []
        for record in self.records.values():
            demo = record.get_demographics().copy()
            demo['StudyID'] = record.study_id  # Ensure StudyID is included
            all_demo.append(demo)

        return pd.DataFrame(all_demo)
    



    def _build_records(self):
        self.records = {}
        for study_id, rows in self.df.groupby("StudyID"):
            # Demographics
            demo = rows[self.metadata_cols].apply(
                lambda col: col.dropna().iloc[0] if col.dropna().any() else None
            )
            demo_dict = demo.to_dict()

            # Blood draws
            blood_draws = []
            for _, row in rows.iterrows():
                bd_data = row[self.lab_cols].dropna().to_dict()
                if bd_data:
                    blood_draws.append(BloodDraw(bd_data))

            # Save Record
            self.records[study_id] = Record(study_id, demographics=demo_dict, blood_draws=blood_draws)

    
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

    def __repr__(self):
        return f"<Record {self.study_id}: {len(self.blood_draws)} blood draws>"

