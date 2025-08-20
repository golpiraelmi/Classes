import numpy as np
import pandas as pd

class ListBasedStudyLoader:
    def __init__(self):
        # ---- Hard-coded replacement dictionary ----
        self.replacement_dict = {
            ("patient_id", "screen_patient_id"): "StudyID",
            ("demo_age", "bl_age"): "Age",
            ("demo_sex", "bl_sex"): "Sex",
            ("bmi_calc", "bl_bmi_calc"): "BMI",
            ("bloodwork_hemoglobin", "lp_hemoglobin"): "Hemoglobin",
            ("bloodwork_creatinine", "lp_creatinine"): "Creatinine",
            ("bloodwork_teg_crt_r", "rteg_crt_rvalue"): "R_time",
            ("bloodwork_teg_crt_k", "rteg_crt_ktime"): "K_time",
            ("bloodwork_teg_crt_ang", "rteg_crt_aangle"): "Alpha_Angle",
            ("bloodwork_teg_crt_ma", "rteg_crt_ma"): "MA",
            ("bloodwork_teg_crt_ly30", "rteg_crt_ly30"): "LY30",
            ("bloodwork_teg_crt_act", "rteg_crt_tegact"): "ACT",
            ("lab_rteg_timepoint","bloodwork_timepoint"):'Time',
            ('date_time_injury',):'Injury_date',
            ('admission_date_time',):'Admission_date',
            ('surgery_date_time',):'Surgery_date',
            ('teg_date_time','lab_dt_blood_draw'):'Draw_date',
            
            
        }

        # ---- Hard-coded timepoint dictionary ----
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
                       'unscheduled 3 months follow up']
        }

        # ---- Hard-coded medications (Pre_operative OACS) ----
        self.medications = {
            **dict.fromkeys(
        ["TH-162","TH-170","TH-198","TH-212","TH-217","TH-225","TH-240","TH-255","TH-262",
         "TH-274","TH-284","TH-302","HPA-001","HPA-008","HPA-009","HPA-010","HPA-012","HPA-014","HPA-016","HPA-017",
         "HPA-021","HPA-022","HPA-026","HPA-028","HPA-030","HPA-033","HPA-035","HPA-036","HPA-039","HPA-042","HPA-043"], "Apixaban"),

            **dict.fromkeys(["TH-227","TH-267","HPA-038"], "Dabigatran"),

            **dict.fromkeys(["TH-236","HPA-004","HPA-015","HPA-019","HPA-020","HPA-024","HPA-029","HPA-032"], "Rivaroxaban")
        }

        self.vte_type_map = {**dict.fromkeys(
            ["TH-003","TH-201","TH-227","TH-253","TH-264","TH-301","HPA-010","HPA-015","HPA-019","HPA-021"], "DVT"),
            **dict.fromkeys(["TH-082","TH-088","TH-271","TH-292","HPA-022","HPA-042"], "PE"),
            **dict.fromkeys(["TH-261","TH-279","HPA-001","HPA-004","HPA-032"], "Both")
        }
        

    def standardize_columns(self, df):
        # Combine date + time columns
        if 'adm_injury_date' in df.columns and 'adm_injury_time' in df.columns:
            df['date_time_injury'] = pd.to_datetime(
                df['adm_injury_date'].astype(str) + ' ' + df['adm_injury_time'].astype(str),
                errors='coerce'
            )
        if 'intraop_date_surg' in df.columns and 'intraop_time_surg' in df.columns:
            df['surgery_date_time'] = pd.to_datetime(
                df['intraop_date_surg'].astype(str) + ' ' + df['intraop_time_surg'].astype(str),
                errors='coerce'
            )
        if 'adm_er_date' in df.columns and 'adm_er_time' in df.columns:
            df['admission_date_time'] = pd.to_datetime(
                df['adm_er_date'].astype(str) + ' ' + df['adm_er_time'].astype(str),
                errors='coerce'
            )
        if 'teg_date' in df.columns and 'teg_time' in df.columns:
            df['teg_date_time'] = pd.to_datetime(
                df['teg_date'].astype(str) + ' ' + df['teg_time'].astype(str),
                errors='coerce'
            )

            # Drop original columns
        columns_to_drop = ['adm_injury_date', 'adm_injury_time',
                           'intraop_date_surg', 'intraop_time_surg',
                           'adm_er_date','adm_er_time','teg_time','teg_date']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

       

        
        

        # Rename columns
        col_map = {}
        for col in df.columns:
            replaced = False
            for key_tuple, standard in self.replacement_dict.items():
                if col in key_tuple:
                    col_map[col] = standard
                    replaced = True
                    break
            if not replaced:
                col_map[col] = col
        df = df.rename(columns=col_map)

        # Forward-fill StudyID for repeating instruments
        if "StudyID" in df.columns:
            df["StudyID"] = df["StudyID"].ffill()

        # Fill Age, Sex, BMI per StudyID
        
        fill_cols = ["Age", "Sex", "BMI", "Injury_date","Surgery_date","Admission_date",'complication_dvt','complication_pe','bl_comorb_vte']
        if "StudyID" in df.columns:
            for col in fill_cols:
                if col in df.columns:
                    df[col] = df.groupby("StudyID")[col].transform(lambda x: x.ffill().bfill())

        # Add Medication column
        if "StudyID" in df.columns:
            df["Medication"] = df["StudyID"].map(self.medications)
            df["DOAC_status"] = np.where(df["Medication"].isnull(), "Non_OAC", "OAC")

        # Add VTE column
        if "StudyID" in df.columns:
            df["VTE"] = df["StudyID"].map(self.vte_type_map)



        # Convert date columns to datetime
        for col in ["Injury_date", "Admission_date", "Surgery_date", "Draw_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Calculate hours_to_surgery & hours_from_injury_to_bloodDraw
        if "Admission_date" in df.columns and "Surgery_date" in df.columns:
            df["hours_to_surgery"] = ((df["Surgery_date"] - df["Admission_date"]).dt.total_seconds()/3600)
        if "Injury_date" in df.columns and "Draw_date" in df.columns:
            df["hours_from_injury_to_bloodDraw"] = ((df["Draw_date"] - df["Injury_date"]).dt.total_seconds()/3600)

        # Standardize Time column
        if "Time" in df.columns:
            df["Time"] = df["Time"].astype(str).str.strip().apply(self._map_timepoint)
            df = self._fix_preop_to_admission(df)
            

        return df

    def _map_timepoint(self, raw_value):
        if pd.isna(raw_value):
            return raw_value
        val = raw_value.strip().lower()
        for standard, variants in self.timepoint_dict.items():
            if val in [v.lower() for v in variants]:
                return standard
        return raw_value

    def _fix_preop_to_admission(self, df):
        if "StudyID" not in df.columns or "Time" not in df.columns:
            return df
        has_admission = df.groupby("StudyID")["Time"].transform(lambda x: x.eq("Admission").any())
        has_admission = has_admission.fillna(False)
        df.loc[(df["Time"] == "Pre-Op") & (~has_admission), "Time"] = "Admission"
        return df

    
    

    
    
    # --- Add first_valid_draw as a method ---
    '''If "Admission" exists with a Hemoglobin value, it is first_draw.
        If "Admission" is missing or has missing Hemoglobin, pick the earliest timepoint with non-missing Hemoglobin among your relevant list (Pre-draw, POD1, etc.)
        If hours_from_injury_to_bloodDraw is missing then go by labels '''
    
    def first_valid_draw(self, group, relevant_timepoints=None):
        # Use provided timepoints or default to your list
        if relevant_timepoints is None:
            relevant_timepoints = ['Admission','Pre-Op','Pre-draw','PFD1','PFD2','PFD3','PFD4','4 hr Pre-OP']

        # Filter to relevant timepoints with non-missing Hemoglobin
        valid = group[(group['Time'].isin(relevant_timepoints)) & (group['Hemoglobin'].notna())]

        if valid.empty:
            return pd.Series([False]*len(group), index=group.index)

        # Case 1: Admission exists
        if 'Admission' in valid['Time'].values:
            first_idx = valid[valid['Time']=='Admission'].index
        else:
            # Case 2: Pick by hours_from_injury_to_bloodDraw if available
            if valid['hours_from_injury_to_bloodDraw'].notna().any():
                min_hour = valid['hours_from_injury_to_bloodDraw'].min()
                first_idx = valid[valid['hours_from_injury_to_bloodDraw']==min_hour].index
            else:
                # Case 3: Fall back to earliest relevant timepoint
                for tp in relevant_timepoints:
                    if tp in valid['Time'].values:
                        first_idx = valid[valid['Time']==tp].index
                        break

        # Create boolean series
        is_first = pd.Series(False, index=group.index)
        is_first.loc[first_idx] = True
        return is_first

    # --- Add calculate_hemoglobin_drop as a method ---
    def calculate_hemoglobin_drop(self, df, timepoint_start="first_draw", timepoint_end="POD1"):
        # Ensure we have a 'blood_draw_label' column
        if 'blood_draw_label' not in df.columns:
            raise ValueError("DataFrame must have 'blood_draw_label' column indicating timepoints.")

        # Filter to relevant timepoints
        df_filtered = df[df['blood_draw_label'].isin([timepoint_start, timepoint_end])]

        # Pivot wide with DOAC_status retained in index
        df_wide = df_filtered.pivot(index=['StudyID','DOAC_status'], 
                                    columns='blood_draw_label', 
                                    values='Hemoglobin')

        # Ensure both timepoints exist
        for tp in [timepoint_start, timepoint_end]:
            if tp not in df_wide.columns:
                raise ValueError(f"Missing hemoglobin data for {tp}")

        # Rename columns
        df_wide = df_wide.rename(columns={
            timepoint_start: f"{timepoint_start}_Hgb",
            timepoint_end: f"{timepoint_end}_Hgb"
        })

        # Calculate drop
        df_wide['Hb_Drop'] = df_wide[f"{timepoint_end}_Hgb"] - df_wide[f"{timepoint_start}_Hgb"]

        # Reset index for clean dataframe
        return df_wide.reset_index()
    

    def compute_hb_drop(self, df, relevant_timepoints=None, timepoint_end="POD1"):
        if relevant_timepoints is None:
            relevant_timepoints = ['Admission','Pre-Op','Pre-draw','PFD1','PFD2','PFD3','PFD4','4 hr Pre-OP']

        # Step 1: Compute first_draw
        df['first_draw'] = df.groupby('StudyID', group_keys=False).apply(
            lambda g: self.first_valid_draw(g, relevant_timepoints)
        )

        # Step 2: Create blood_draw_label
        df['blood_draw_label'] = df['Time']
        df.loc[df['first_draw'], 'blood_draw_label'] = 'first_draw'

        # Step 3: Prepare subset for hemoglobin drop calculation
        df_subset = df[['StudyID','DOAC_status','Hemoglobin','blood_draw_label']].drop_duplicates(
            subset=['StudyID','blood_draw_label']
        )

        # Step 4: Calculate Hb drop using existing method
        hb_drop_df = self.calculate_hemoglobin_drop(df_subset, timepoint_start='first_draw', timepoint_end=timepoint_end)

        # Step 5: Drop rows where Hb_Drop is missing
        hb_drop_df = hb_drop_df.dropna(subset=['Hb_Drop'])

        return hb_drop_df
