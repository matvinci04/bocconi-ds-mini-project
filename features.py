FEATURES = [
    "study_hours","lecture_attendance","prev_gpa","part_time_job",
    "social_hours","sleep_hours","practice_tests"]
TARGET = "exam_score"

def split_features_target(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X,y
