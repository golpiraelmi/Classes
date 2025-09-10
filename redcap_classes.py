
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
            "POD1": ['post op day 1','POD1','Day 1 post-op','PO Day 1','Day 1 Post-Op','POD 1','POD 1 ','Day 1 post op',
                     'Post Operative Day 1','Day 1 Post-op','Day 1 post o','Postoperative Day 1','Pod 1'],
            "POD2": ['POD 2','POD2','Day 2 post-op','Day 2 Post-Op','Day 2 post op','PO Day 2','Day 2 Post-op',
                     'Post Operative Day 2','Day  2 post-op'],
            "POD3": ['POD3','Day 3 post-op','POD 3','Day 3 Post-Op','PO Day 3','Day 3 post op','Post Operative Day 3',
                     'Day 3 Post-op','Day 3 pot-op','Day 3 PO'],
            "POD4": ['POD 4','Day 4 post-op','Day 4 Post-Op','PO Day 4','Day 4 post op','Day 4 Post-op'],
            "POD5": ['pod 5','POD5','POD 5','Day 5 post-op','PO Day 5','Day 5 Post-Op','Day 5 post op','Day 5 Post-op',
                     'Day 5 po','Post Operative Day 5','POD 5 '],
            "POD7": ['POD7','POD 7','PO Day 7','Post Operative Day 7','pod 7'],
            "Week2": ['2 week','2 Week FU','2 weeks follow up','2-Week','2 weeks','2 Week','2weeks','2 Week F/U',
                      '2 week F/U','2 week follow-up','2-week','2 Week PO','2 week post#','2 weeks '],
            "Week4": ['4 week','4 Week FU','4 weeks','4 weeks follow up','4-Week','4weeks','4 WEEKS FOLLOW UP',
                      '4 weeks f/u','4 week follow up','4 week '],
            "Week6": ['6 week','6 weeks follow up','6 Week FU','6weeks','6 weeks','6-Week','6 Week','6 Week F/U',
                      '6 week F/U','6-week','6 week post op','6 week follow up','6 Week Follow Up','6 weeks '],
            "Month3": ['3 month','3 Month Follow Up','3 months follow up','3 months','3 Month FU','3months',
                       '3 month follow up','3-Month','3 Month F/U','3 Month','3  month','3-month','3 month f/u',
                       'unscheduled 3 months follow up','12 weeks'],
            "Month6": ['6 months']
        }


        self.medications = {
            **dict.fromkeys(
                ['HPA-001', 'HPA-004', 'HPA-008', 'HPA-009', 'HPA-010', 'HPA-012',
            'HPA-014', 'HPA-015', 'HPA-016', 'HPA-017', 'HPA-019', 'HPA-020',
            'HPA-021', 'HPA-022', 'HPA-024', 'HPA-026', 'HPA-028', 'HPA-029',
            'HPA-030', 'HPA-032', 'HPA-033', 'HPA-035', 'HPA-036', 'HPA-038',
            'HPA-039', 'HPA-042', 'HPA-043', 'TH-004', 'TH-010', 'TH-075',
            'TH-088', 'TH-101', 'TH-110', 'TH-162', 'TH-170', 'TH-194',
            'TH-198', 'TH-212', 'TH-217', 'TH-225', 'TH-226', 'TH-227','TH-236',
            'TH-240', 'TH-255', 'TH-262', 'TH-267', 'TH-274',
            'TH-284', 'TH-286', 'TH-290', 'TH-302'], "OAC")
        }


        self.vte_type_map = {**dict.fromkeys(
            ["TH-003","TH-201","TH-227","TH-253","TH-264","TH-301"], "DVT"),
            **dict.fromkeys(["TH-082","TH-088","TH-271","TH-292","HPA-028"], "PE"),
            **dict.fromkeys(["TH-261","TH-279"], "Both")
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
        records_data = self.project.export_records(raw_or_label='label')
        df = pd.DataFrame(records_data)
        # display(df)

        # Step 2: Replace empty strings with NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # Step 3: Rename columns using replacement_dict
        col_mapping = {}
        for keys, standard_name in self.replacement_dict.items():
            for k in keys:
                if k in df.columns:
                    col_mapping[k] = standard_name
        df = df.rename(columns=col_mapping)

        # --- Step 4: Assign to self.df BEFORE replacing missing values ---
        self.df = df


        self._replace_missing_values()

        # Step 4: Filter by screening_status (if column exists)
        if 'screening_status' in df.columns:
            
            df['StudyID'] = df['StudyID'].astype(str)
            df['StudyID'] = df.groupby('record_id')['StudyID'].ffill().bfill()
            df['screening_status'] = df.groupby('record_id')['screening_status'].ffill().bfill()

            if 'record_id' in df.columns:
                df['record_id'] = df['record_id'].astype(str).str.strip()
                df['StudyID'] = df['StudyID'].replace('nan', np.nan)
                df['StudyID'] = df.groupby('record_id')['StudyID'].ffill().bfill()
        
            df = df[df['screening_status'].astype(str).str.strip() == 'Eligible → enrolled']

     
        # if 'Draw_date' in df.columns:
        #     # Ensure teg_time exists; if not, create a column of NaNs
        #     if 'teg_time' not in df.columns:
        #         df['teg_time'] = pd.NA

        #     # Fill missing teg_time with midnight
        #     df['teg_time'] = df['teg_time'].fillna('00:00').astype(str)

        #     # Only combine when Draw_date is not missing
        #     df['Draw_date'] = pd.to_datetime(
        #         df['Draw_date'].astype(str) + ' ' + df['teg_time'],
        #         errors='coerce'
        #     )
        if 'Draw_date' in df.columns:
            # Ensure teg_time exists; if not, create a column of NaNs
            if 'teg_time' not in df.columns:
                df['teg_time'] = pd.NA

            # Fill missing teg_time with midnight
            df['teg_time'] = df['teg_time'].fillna('00:00').astype(str)

            # Use Draw_date if available, otherwise fall back to lab_date_visit (or NA)
            draw_dates = df['Draw_date'].combine_first(
                df['lab_date_visit'] if 'lab_date_visit' in df.columns else pd.Series([pd.NA]*len(df))
            )

            # Convert to datetime
            df['Draw_date'] = pd.to_datetime(draw_dates.astype(str) + ' ' + df['teg_time'], errors='coerce')


        if 'adm_injury_time' in df.columns and 'Injury_date' in df.columns:
            df['Injury_date'] = pd.to_datetime(df['Injury_date'].astype(str) + ' ' + df['adm_injury_time'].astype(str),
            errors='coerce')

        if 'intraop_time_surg' in df.columns and 'Surgery_date' in df.columns:
            df['Surgery_date'] = pd.to_datetime(df['Surgery_date'].astype(str) + ' ' + df['intraop_time_surg'].astype(str),
            errors='coerce')



        # Step 5: Standardize timepoints
        if 'Time' in df.columns:
            df['Time'] = df['Time'].apply(self._map_timepoint)


        # if 'Sex' in df.columns:
        #     df['Sex'] = df['Sex'].map(self.sex_dict).astype(object)
            
            

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

        
        # --- Add Pre_op_doac ---
        medication = self.medications.get(patient_demo["StudyID"], None)
        patient_demo["Pre_op_doac"] = medication


        patient_demo["VTE_type"] = self.vte_type_map.get(patient_demo["StudyID"], None)

        

        

        return patient_demo
    
    # -----------------------
    # Get patient blood draws
    # -----------------------

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
       

    # -----------------------
    # Get all blood draws for all patients
    # -----------------------
    def get_all_blood_draws(self):
        all_draws = []
        for rec in self.records.values():
            for bd in rec.blood_draws:
                row = {"StudyID": rec.study_id}
                row.update(bd.labs)
                all_draws.append(row)
        return pd.DataFrame(all_draws)
    
    # -----------------------
    # Get all demographics for all patients
    # -----------------------
    def get_all_demographics(self):
    
        all_demo = []
        for record in self.records.values():
            demo = record.get_demographics().copy()
            demo['StudyID'] = record.study_id  # Ensure StudyID is included
            demo['Pre_op_doac'] = self.medications.get(record.study_id, None)
            demo["VTE_type"] = self.vte_type_map.get(record.study_id, None)
            all_demo.append(demo)

            df_demo = pd.DataFrame(all_demo)
            df_demo['VTE'] = np.where(df_demo['VTE_type'].isnull(), 'No', 'Yes')
            df_demo=df_demo.replace({'Participant Withdrawn': np.nan})
            df_demo['Pre_op_doac']=df_demo['Pre_op_doac'].replace({None: 'Non_DOAC'})


        return df_demo
    



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
        """Attach time from injury → draw for each blood draw, safely."""
        injury_date = pd.to_datetime(self.demographics.get("Injury_date", None), errors="coerce")

        if pd.notnull(injury_date):
            for bd in self.blood_draws:
                draw_date = pd.to_datetime(bd.labs.get("Draw_date", None), errors="coerce")
                if pd.notnull(draw_date):
                    delta = draw_date - injury_date
                    bd.labs["time_from_injury_to_draw_hours"] = delta.total_seconds() / 3600
                    


    def __repr__(self):
        return f"<Record {self.study_id}: {len(self.blood_draws)} blood draws>"

