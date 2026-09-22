from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import date, timedelta
from functools import lru_cache

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SUMMARY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "attendance_data.xlsx"
)

ATTENDANCE_LOG_FILE = os.path.join(
    BASE_DIR,
    "data",
    "attendance_logs.xlsx"
)

LAST_WORKING_DAY = date(2026, 11, 28)

PERIODS = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7"
]

LEAVE_THRESHOLD = 4


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_register(value):
    return str(value).strip().replace(".0", "")


def safe_number(value, default=0.0):
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


# ============================================================
# LOAD STUDENT SUMMARY
# ============================================================

@lru_cache(maxsize=1)
def load_students():

    data = pd.read_excel(
        SUMMARY_FILE,
        sheet_name="Cohort Compliance Audit",
        dtype={"Register No": str}
    )

    data["Register No"] = (
        data["Register No"]
        .apply(clean_register)
    )

    data = data.fillna("")

    return data


df = load_students()


# ============================================================
# LOAD DAILY ATTENDANCE LOGS
# ============================================================

@lru_cache(maxsize=1)
def load_attendance_logs():

    if not os.path.exists(ATTENDANCE_LOG_FILE):
        return pd.DataFrame()

    logs = pd.read_excel(
        ATTENDANCE_LOG_FILE
    )

    required_columns = [
        "Date",
        "Day",
        "Reg No",
        "Student Name",
        "Status",
        *PERIODS
    ]

    missing = [
        column
        for column in required_columns
        if column not in logs.columns
    ]

    if missing:
        raise ValueError(
            "attendance_logs.xlsx is missing columns: "
            + ", ".join(missing)
        )

    # Keep only columns actually used by the application.
    logs = logs[required_columns].copy()

    logs["Date"] = pd.to_datetime(
        logs["Date"],
        errors="coerce"
    )

    logs["Reg No"] = (
        logs["Reg No"]
        .apply(clean_register)
    )

    logs["Status"] = (
        logs["Status"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    logs = logs[
        logs["Status"] == "active"
    ].copy()

    # Normalize period values once when the file is loaded.
    # This avoids repeating expensive .astype/.str operations
    # every time the prediction page is opened.
    for period in PERIODS:
        logs[period] = (
            logs[period]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    logs["period_absences"] = 0

    for period in PERIODS:
        logs["period_absences"] += (
            logs[period] == "A"
        ).astype("int8")

    logs["daily_leave"] = (
        logs["period_absences"]
        >= LEAVE_THRESHOLD
    ).astype("int8")

    return logs

# ============================================================
# PREPARE PERIOD-LEVEL TRAINING DATA
# ============================================================

def prepare_training_data(logs):

    if logs.empty:
        return pd.DataFrame()

    long_data = logs.melt(
        id_vars=[
            "Date",
            "Day",
            "Reg No",
            "Student Name"
        ],
        value_vars=PERIODS,
        var_name="Period",
        value_name="Attendance"
    )

    long_data["Attendance"] = (
        long_data["Attendance"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    long_data = long_data[
        long_data["Attendance"].isin(
            ["P", "A"]
        )
    ].copy()

    long_data["Absent"] = (
        long_data["Attendance"] == "A"
    ).astype(int)

    long_data["weekday"] = (
        long_data["Date"].dt.dayofweek
    )

    long_data["period_num"] = (
        long_data["Period"]
        .str.extract(r"(\d+)")
        .astype(int)
    )

    long_data = long_data.sort_values(
        [
            "Reg No",
            "Date",
            "period_num"
        ]
    )

    group = long_data.groupby(
        "Reg No"
    )

    long_data["prior_events"] = (
        group.cumcount()
    )

    long_data["prior_absences"] = (
        group["Absent"].cumsum()
        - long_data["Absent"]
    )

    global_rate = float(
        long_data["Absent"].mean()
    )

    long_data["hist_abs_rate"] = (
        long_data["prior_absences"]
        /
        long_data["prior_events"]
        .replace(0, pd.NA)
    )

    long_data["hist_abs_rate"] = (
        long_data["hist_abs_rate"]
        .fillna(global_rate)
        .astype(float)
    )

    long_data["recent_abs_rate"] = (
        long_data
        .groupby("Reg No")["Absent"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=35,
                min_periods=7
            )
            .mean()
        )
    )

    long_data["recent_abs_rate"] = (
        long_data["recent_abs_rate"]
        .fillna(
            long_data["hist_abs_rate"]
        )
        .fillna(global_rate)
    )

    return long_data


# ============================================================
# EXISTING AI FUTURE ABSENCE FORECAST
# ============================================================

@lru_cache(maxsize=1)
def build_ai_forecast():

    logs = load_attendance_logs()

    training = prepare_training_data(
        logs
    )

    if (
        training.empty
        or training["Reg No"].nunique() < 2
    ):

        return {
            "available": False,
            "message":
                "Daily attendance logs are required "
                "for the AI prediction model."
        }

    features = [
        "Reg No",
        "weekday",
        "period_num",
        "hist_abs_rate",
        "recent_abs_rate"
    ]

    X = training[features]

    y = training["Absent"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                [
                    "Reg No",
                    "weekday",
                    "period_num"
                ]
            ),
            (
                "numeric",
                StandardScaler(),
                [
                    "hist_abs_rate",
                    "recent_abs_rate"
                ]
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                )
            )
        ]
    )

    unique_dates = sorted(
        training["Date"]
        .dropna()
        .dt.date
        .unique()
    )

    accuracy = None
    auc = None

    if len(unique_dates) >= 10:

        split_index = int(
            len(unique_dates) * 0.80
        )

        split_date = pd.Timestamp(
            unique_dates[split_index]
        )

        train_part = training[
            training["Date"] < split_date
        ]

        test_part = training[
            training["Date"] >= split_date
        ]

        if (
            not train_part.empty
            and not test_part.empty
            and test_part["Absent"].nunique() >= 2
        ):

            test_model = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000
                        )
                    )
                ]
            )

            test_model.fit(
                train_part[features],
                train_part["Absent"]
            )

            test_prob = (
                test_model
                .predict_proba(
                    test_part[features]
                )[:, 1]
            )

            test_pred = (
                test_prob >= 0.50
            ).astype(int)

            accuracy = round(
                accuracy_score(
                    test_part["Absent"],
                    test_pred
                ) * 100,
                1
            )

            auc = round(
                roc_auc_score(
                    test_part["Absent"],
                    test_prob
                ),
                3
            )

    model.fit(
        X,
        y
    )

    latest_log_date = (
        training["Date"]
        .max()
        .date()
    )

    forecast_start = (
        latest_log_date
        + timedelta(days=1)
    )

    if forecast_start > LAST_WORKING_DAY:

        return {
            "available": True,
            "message":
                "The forecast period has ended.",
            "accuracy": accuracy,
            "auc": auc,
            "latest_log_date":
                latest_log_date.strftime(
                    "%d-%m-%Y"
                ),
            "forecast_start":
                forecast_start.strftime(
                    "%d-%m-%Y"
                ),
            "forecast_end":
                LAST_WORKING_DAY.strftime(
                    "%d-%m-%Y"
                ),
            "future_sessions": 0,
            "future_dates": 0,
            "active_students": 0,
            "expected_absences": 0,
            "projected_below_75": 0,
            "projected_below_85": 0,
            "projected_below_90": 0,
            "class_projected_attendance": 0,
            "student_forecasts": [],
            "date_forecasts": []
        }

    future_dates = []

    current = forecast_start

    while current <= LAST_WORKING_DAY:

        if current.weekday() != 6:
            future_dates.append(
                current
            )

        current += timedelta(
            days=1
        )

    student_stats = (
        training
        .groupby("Reg No")
        .agg(
            student_name=(
                "Student Name",
                "first"
            ),
            total_events=(
                "Absent",
                "size"
            ),
            historical_absences=(
                "Absent",
                "sum"
            ),
            current_absence_rate=(
                "Absent",
                "mean"
            )
        )
        .reset_index()
    )

    recent_rates = (
        training
        .sort_values(
            [
                "Reg No",
                "Date",
                "period_num"
            ]
        )
        .groupby("Reg No")["Absent"]
        .apply(
            lambda s:
            s.tail(35).mean()
        )
    )

    student_stats[
        "recent_absence_rate"
    ] = (
        student_stats["Reg No"]
        .map(recent_rates)
        .fillna(
            student_stats[
                "current_absence_rate"
            ]
        )
    )

    prediction_rows = []

    for _, student in student_stats.iterrows():

        for future_date in future_dates:

            for period_num in range(1, 8):

                prediction_rows.append(
                    {
                        "Reg No":
                            str(
                                student["Reg No"]
                            ),

                        "Date":
                            future_date,

                        "weekday":
                            future_date.weekday(),

                        "period_num":
                            period_num,

                        "hist_abs_rate":
                            float(
                                student[
                                    "current_absence_rate"
                                ]
                            ),

                        "recent_abs_rate":
                            float(
                                student[
                                    "recent_absence_rate"
                                ]
                            ),

                        "Student Name":
                            student[
                                "student_name"
                            ]
                    }
                )

    future_data = pd.DataFrame(
        prediction_rows
    )

    future_data[
        "absence_probability"
    ] = (
        model
        .predict_proba(
            future_data[features]
        )[:, 1]
    )

    student_forecast = (
        future_data
        .groupby("Reg No")
        .agg(
            expected_absences=(
                "absence_probability",
                "sum"
            ),
            highest_probability=(
                "absence_probability",
                "max"
            )
        )
        .reset_index()
    )

    student_forecast = (
        student_forecast
        .merge(
            student_stats[
                [
                    "Reg No",
                    "student_name",
                    "total_events",
                    "historical_absences"
                ]
            ],
            on="Reg No",
            how="left"
        )
    )

    future_sessions = (
        len(future_dates) * 7
    )

    student_forecast[
        "projected_attendance"
    ] = (
        (
            student_forecast[
                "total_events"
            ]
            -
            student_forecast[
                "historical_absences"
            ]
            +
            future_sessions
            -
            student_forecast[
                "expected_absences"
            ]
        )
        /
        (
            student_forecast[
                "total_events"
            ]
            +
            future_sessions
        )
        * 100
    )

    def risk_label(projected):

        if projected < 75:
            return "Critical"

        if projected < 85:
            return "High"

        if projected < 90:
            return "Medium"

        return "Low"

    student_forecast["risk"] = (
        student_forecast[
            "projected_attendance"
        ].apply(risk_label)
    )

    student_forecast = (
        student_forecast
        .sort_values(
            [
                "projected_attendance",
                "expected_absences"
            ]
        )
    )

    student_forecast_rows = []

    for _, row in student_forecast.iterrows():

        student_forecast_rows.append(
            {
                "register_no":
                    str(row["Reg No"]),

                "student_name":
                    str(
                        row["student_name"]
                    ),

                "expected_absences":
                    round(
                        float(
                            row[
                                "expected_absences"
                            ]
                        ),
                        1
                    ),

                "highest_probability":
                    round(
                        float(
                            row[
                                "highest_probability"
                            ]
                        ) * 100,
                        1
                    ),

                "projected_attendance":
                    round(
                        float(
                            row[
                                "projected_attendance"
                            ]
                        ),
                        1
                    ),

                "risk":
                    row["risk"]
            }
        )

    date_forecast = (
        future_data
        .groupby("Date")
        .agg(
            expected_absences=(
                "absence_probability",
                "sum"
            ),
            highest_probability=(
                "absence_probability",
                "max"
            )
        )
        .reset_index()
    )

    date_forecast = (
        date_forecast
        .sort_values(
            "expected_absences",
            ascending=False
        )
    )

    date_forecast_rows = []

    for _, row in date_forecast.head(12).iterrows():

        date_forecast_rows.append(
            {
                "date":
                    row["Date"].strftime(
                        "%d-%m-%Y"
                    ),

                "day":
                    row["Date"].strftime(
                        "%A"
                    ),

                "expected_absences":
                    round(
                        float(
                            row[
                                "expected_absences"
                            ]
                        ),
                        1
                    ),

                "highest_probability":
                    round(
                        float(
                            row[
                                "highest_probability"
                            ]
                        ) * 100,
                        1
                    )
            }
        )

    return {

        "available": True,

        "accuracy": accuracy,

        "auc": auc,

        "latest_log_date":
            latest_log_date.strftime(
                "%d-%m-%Y"
            ),

        "forecast_start":
            forecast_start.strftime(
                "%d-%m-%Y"
            ),

        "forecast_end":
            LAST_WORKING_DAY.strftime(
                "%d-%m-%Y"
            ),

        "future_sessions":
            future_sessions,

        "future_dates":
            len(future_dates),

        "active_students":
            len(student_stats),

        "expected_absences":
            round(
                float(
                    student_forecast[
                        "expected_absences"
                    ].sum()
                ),
                1
            ),

        "projected_below_75":
            int(
                (
                    student_forecast[
                        "projected_attendance"
                    ] < 75
                ).sum()
            ),

        "projected_below_85":
            int(
                (
                    student_forecast[
                        "projected_attendance"
                    ] < 85
                ).sum()
            ),

        "projected_below_90":
            int(
                (
                    student_forecast[
                        "projected_attendance"
                    ] < 90
                ).sum()
            ),

        "class_projected_attendance":
            round(
                float(
                    student_forecast[
                        "projected_attendance"
                    ].mean()
                ),
                1
            ),

        "student_forecasts":
            student_forecast_rows,

        "date_forecasts":
            date_forecast_rows
    }


# ============================================================
# NEW AI: DAILY LEAVE DATA
# ============================================================

def prepare_daily_leave_data(logs):

    if logs.empty:
        return pd.DataFrame()

    data = logs.copy()

    for period in PERIODS:

        data[period] = (
            data[period]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    data["period_absences"] = 0

    for period in PERIODS:

        data["period_absences"] += (
            data[period] == "A"
        ).astype(int)

    data["daily_leave"] = (
        data["period_absences"]
        >= LEAVE_THRESHOLD
    ).astype(int)

    data = data[
        [
            "Date",
            "Reg No",
            "Student Name",
            "period_absences",
            "daily_leave"
        ]
    ].copy()

    data = data.dropna(
        subset=["Date"]
    )

    data = data.sort_values(
        [
            "Reg No",
            "Date"
        ]
    )

    return data


# ============================================================
# NEW AI: SELECTED DATE LEAVE PREDICTION
# ============================================================

@lru_cache(maxsize=31)
def build_selected_date_leave_prediction(
    prediction_date
):

    logs = load_attendance_logs()

    if logs.empty:
        return None, (
            "Daily attendance data is not available."
        )

    latest_log_date = (
        logs["Date"]
        .max()
        .date()
    )

    if prediction_date <= latest_log_date:
        return None, (
            "Please select a date after "
            f"{latest_log_date.strftime('%d-%m-%Y')}."
        )

    if prediction_date > LAST_WORKING_DAY:
        return None, (
            "Prediction date cannot be after "
            f"{LAST_WORKING_DAY.strftime('%d-%m-%Y')}."
        )

    if prediction_date.weekday() == 6:
        return None, (
            "Sunday is not a working day. "
            "Please select another date."
        )

    # --------------------------------------------------------
    # DAILY HISTORY
    # --------------------------------------------------------

    daily_data = (
        logs[
            [
                "Date",
                "Reg No",
                "Student Name",
                "period_absences",
                "daily_leave"
            ]
        ]
        .dropna(subset=["Date"])
        .sort_values(["Reg No", "Date"])
        .copy()
    )

    daily_data["weekday"] = (
        daily_data["Date"].dt.dayofweek
    )

    # All students shown in the dashboard are included.
    student_list = (
        df["Register No"]
        .astype(str)
        .apply(clean_register)
        .drop_duplicates()
        .tolist()
    )

    # --------------------------------------------------------
    # Efficient historical features
    # --------------------------------------------------------

    global_leave_rate = float(
        daily_data["daily_leave"].mean()
    )

    # Per-student period absence rate.
    period_rates = (
        logs.groupby("Reg No")["period_absences"]
        .sum()
        .div(
            logs.groupby("Reg No").size() * len(PERIODS)
        )
        .fillna(0.0)
    )

    # Previous-day features.
    grouped = daily_data.groupby(
        "Reg No",
        sort=False
    )

    daily_data["previous_leave"] = (
        grouped["daily_leave"].shift(1)
    )

    daily_data["historical_leave_rate"] = (
        grouped["daily_leave"]
        .transform(
            lambda s:
            s.shift(1)
            .expanding()
            .mean()
        )
        .fillna(global_leave_rate)
        .astype(float)
    )

    daily_data["recent_leave_rate"] = (
        grouped["daily_leave"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                window=10,
                min_periods=1
            )
            .mean()
        )
        .fillna(daily_data["historical_leave_rate"])
        .fillna(global_leave_rate)
        .astype(float)
    )

    # Same-weekday historical leave rate, excluding the current row.
    weekday_group = daily_data.groupby(
        ["Reg No", "weekday"],
        sort=False
    )["daily_leave"]

    same_weekday_count = weekday_group.cumcount()
    same_weekday_sum = (
        weekday_group.cumsum()
        - daily_data["daily_leave"]
    )

    daily_data["same_weekday_rate"] = (
        same_weekday_sum
        / same_weekday_count.replace(0, pd.NA)
    )

    daily_data["same_weekday_rate"] = (
        daily_data["same_weekday_rate"]
        .fillna(daily_data["historical_leave_rate"])
        .fillna(global_leave_rate)
        .astype(float)
    )

    # Calculate consecutive leave streak before each day.
    streak_values = []
    for _, group in daily_data.groupby(
        "Reg No",
        sort=False
    ):
        streak = 0

        for value in group["daily_leave"].tolist():
            streak_values.append(streak)

            if int(value) == 1:
                streak += 1
            else:
                streak = 0

    daily_data["leave_streak"] = streak_values

    daily_data["period_absence_rate"] = (
        daily_data["Reg No"]
        .map(period_rates)
        .fillna(0.0)
        .astype(float)
    )

    # --------------------------------------------------------
    # Target-date features
    # --------------------------------------------------------

    target_weekday = prediction_date.weekday()

    feature_rows = []

    for register_no in student_list:

        history = daily_data[
            daily_data["Reg No"] == register_no
        ]

        if history.empty:
            historical_rate = global_leave_rate
            recent_rate = global_leave_rate
            same_weekday_rate = global_leave_rate
            previous_leave = 0
            leave_streak = 0
        else:
            historical_rate = float(
                history["daily_leave"].mean()
            )

            recent_rate = float(
                history["daily_leave"]
                .tail(10)
                .mean()
            )

            weekday_history = history[
                history["weekday"] == target_weekday
            ]

            if weekday_history.empty:
                same_weekday_rate = historical_rate
            else:
                same_weekday_rate = float(
                    weekday_history["daily_leave"].mean()
                )

            previous_leave = int(
                history.iloc[-1]["daily_leave"]
            )

            leave_streak = int(
                history.iloc[-1]["leave_streak"]
            )

        feature_rows.append(
            {
                "Reg No": register_no,
                "weekday": target_weekday,
                "historical_leave_rate": historical_rate,
                "recent_leave_rate": recent_rate,
                "same_weekday_rate": same_weekday_rate,
                "previous_leave": previous_leave,
                "leave_streak": leave_streak,
                "period_absence_rate": float(
                    period_rates.get(
                        register_no,
                        0.0
                    )
                )
            }
        )

    feature_data = pd.DataFrame(feature_rows)

    # --------------------------------------------------------
    # Historical training rows
    # --------------------------------------------------------

    training_data = daily_data[
        [
            "Reg No",
            "weekday",
            "historical_leave_rate",
            "recent_leave_rate",
            "same_weekday_rate",
            "previous_leave",
            "leave_streak",
            "period_absence_rate",
            "daily_leave"
        ]
    ].copy()

    # The first historical row for a student has no previous-day
    # information, so it is not useful as a supervised example.
    training_data = training_data[
        training_data["previous_leave"].notna()
    ].copy()

    feature_columns = [
        "Reg No",
        "weekday",
        "historical_leave_rate",
        "recent_leave_rate",
        "same_weekday_rate",
        "previous_leave",
        "leave_streak",
        "period_absence_rate"
    ]

    if (
        training_data.empty
        or training_data["daily_leave"].nunique() < 2
    ):
        return None, (
            "There is not enough historical leave variation "
            "to train the selected-date prediction model."
        )

    # --------------------------------------------------------
    # Lightweight AI model
    # --------------------------------------------------------
    #
    # Logistic regression is intentionally used here instead of
    # a 300-tree Random Forest. Both provide probability-based
    # classification, but LogisticRegression uses substantially
    # less RAM on Render's free instance.
    #

    categorical_features = [
        "Reg No",
        "weekday"
    ]

    numeric_features = [
        "historical_leave_rate",
        "recent_leave_rate",
        "same_weekday_rate",
        "previous_leave",
        "leave_streak",
        "period_absence_rate"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        training_data[feature_columns],
        training_data["daily_leave"]
    )

    feature_data["probability"] = (
        model
        .predict_proba(
            feature_data[feature_columns]
        )[:, 1]
    )

    feature_data["prediction"] = (
        feature_data["probability"] >= 0.50
    ).map(
        {
            True: "YES",
            False: "NO"
        }
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    for _, row in feature_data.iterrows():

        results.append(
            {
                "roll_no":
                    str(row["Reg No"]),

                "prediction":
                    row["prediction"],

                "probability":
                    round(
                        float(
                            row["probability"]
                        ) * 100,
                        1
                    )
            }
        )

    results = sorted(
        results,
        key=lambda x: (
            0 if x["prediction"] == "YES"
            else 1,
            -x["probability"]
        )
    )

    leave_roll_numbers = [
        item["roll_no"]
        for item in results
        if item["prediction"] == "YES"
    ]

    prediction_result = {
        "prediction_date":
            prediction_date.strftime(
                "%d-%m-%Y"
            ),

        "students_checked":
            len(results),

        "predicted_leave":
            len(leave_roll_numbers),

        "predicted_present":
            len(results)
            - len(leave_roll_numbers),

        "results":
            results,

        "leave_roll_numbers":
            leave_roll_numbers
    }

    return prediction_result, None

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# STUDENT PORTAL
# ============================================================

@app.route("/student")
def student_search():

    return render_template(
        "student_search.html"
    )


@app.route(
    "/student/result",
    methods=["POST"]
)
def student_result():

    register_no = clean_register(
        request.form.get(
            "register_no",
            ""
        )
    )

    student_data = df[
        df["Register No"]
        .astype(str)
        .str.lower()
        ==
        register_no.lower()
    ]

    if student_data.empty:

        return render_template(
            "student_search.html",
            error=(
                "Register Number not found. "
                "Please check and try again."
            )
        )

    student = (
        student_data
        .iloc[0]
        .to_dict()
    )

    return render_template(
        "student_profile.html",
        student=student
    )


@app.route(
    "/student/<register_no>"
)
def student_profile(
    register_no
):

    register_no = clean_register(
        register_no
    )

    student_data = df[
        df["Register No"]
        .astype(str)
        .str.lower()
        ==
        register_no.lower()
    ]

    if student_data.empty:

        return render_template(
            "student_search.html",
            error=(
                "Register Number not found. "
                "Please check and try again."
            )
        )

    student = (
        student_data
        .iloc[0]
        .to_dict()
    )

    return render_template(
        "student_profile.html",
        student=student
    )


# ============================================================
# STAFF PORTAL
# ============================================================

@app.route("/staff")
def staff_search():

    return render_template(
        "staff_search.html"
    )


@app.route("/staff/dashboard")
def staff_dashboard():

    # --------------------------------------------------------
    # CLASS VALIDATION
    # --------------------------------------------------------

    class_name = request.args.get(
        "class_name",
        ""
    ).strip().upper()

    if class_name != "CSE B":

        return render_template(
            "staff_search.html",
            error=(
                "Invalid class. "
                "Only CSE B is currently available."
            )
        )

    # --------------------------------------------------------
    # BASIC CLASS STATISTICS
    # --------------------------------------------------------

    total_students = len(df)

    attendance_values = pd.to_numeric(
        df["Overall Attendance %"],
        errors="coerce"
    )

    average_attendance = round(
        attendance_values.mean(),
        2
    )

    safe_students = len(
        df[
            df["Risk Tier"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "safe"
        ]
    )

    risk_students = (
        total_students
        - safe_students
    )

    below_75 = int(
        (
            attendance_values < 75
        ).sum()
    )

    between_75_84 = int(
        (
            (attendance_values >= 75)
            &
            (attendance_values < 85)
        ).sum()
    )

    between_85_89 = int(
        (
            (attendance_values >= 85)
            &
            (attendance_values < 90)
        ).sum()
    )

    above_90 = int(
        (
            attendance_values >= 90
        ).sum()
    )

    highest_attendance = round(
        float(
            attendance_values.max()
        ),
        2
    )

    lowest_attendance = round(
        float(
            attendance_values.min()
        ),
        2
    )

    # --------------------------------------------------------
    # CLASS HEALTH
    # --------------------------------------------------------

    if average_attendance >= 90:

        class_health = (
            "Excellent class attendance. "
            "The class is maintaining a strong "
            "attendance level."
        )

        health_status = "Excellent"

    elif average_attendance >= 85:

        class_health = (
            "Good overall class attendance. "
            "Most students are maintaining "
            "healthy attendance levels."
        )

        health_status = "Good"

    elif average_attendance >= 75:

        class_health = (
            "Moderate attendance level. "
            "Some students require attention "
            "to prevent attendance shortage."
        )

        health_status = "Moderate"

    else:

        class_health = (
            "Attendance requires immediate "
            "attention. Several students may "
            "be at risk."
        )

        health_status = "Critical"

    # --------------------------------------------------------
    # EXISTING AI
    # --------------------------------------------------------

    try:

        ai_forecast = (
            build_ai_forecast()
        )

    except Exception as error:

        ai_forecast = {
            "available": False,
            "message": str(error)
        }

    # --------------------------------------------------------
    # NEW SELECTED-DATE AI
    # --------------------------------------------------------

    selected_date_prediction = None

    prediction_error = None

    prediction_date_text = (
        request.args.get(
            "prediction_date",
            ""
        ).strip()
    )

    if prediction_date_text:

        try:

            prediction_date = (
                date.fromisoformat(
                    prediction_date_text
                )
            )

            (
                selected_date_prediction,
                prediction_error
            ) = (
                build_selected_date_leave_prediction(
                    prediction_date
                )
            )

        except ValueError:

            prediction_error = (
                "Invalid prediction date."
            )

        except Exception as error:

            prediction_error = (
                "Prediction could not be generated: "
                + str(error)
            )

    # --------------------------------------------------------
    # LATEST LOG DATE FOR CALENDAR
    # --------------------------------------------------------

    latest_log_date = ""

    try:

        logs = load_attendance_logs()

        if not logs.empty:

            latest_date = (
                logs["Date"]
                .dropna()
                .max()
                .date()
            )

            # Calendar should start AFTER
            # the latest attendance date.
            next_date = (
                latest_date
                + timedelta(days=1)
            )

            latest_log_date = (
                next_date.isoformat()
            )

    except Exception:

        latest_log_date = ""

    # --------------------------------------------------------
    # STUDENTS
    # --------------------------------------------------------

    students = df.to_dict(
        orient="records"
    )

    # --------------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "staff_dashboard.html",

        students=students,

        class_name=class_name,

        total_students=total_students,

        average_attendance=
            average_attendance,

        safe_students=
            safe_students,

        risk_students=
            risk_students,

        below_75=
            below_75,

        between_75_84=
            between_75_84,

        between_85_89=
            between_85_89,

        above_90=
            above_90,

        highest_attendance=
            highest_attendance,

        lowest_attendance=
            lowest_attendance,

        class_health=
            class_health,

        health_status=
            health_status,

        ai_forecast=
            ai_forecast,

        selected_date_prediction=
            selected_date_prediction,

        prediction_error=
            prediction_error,

        latest_log_date=
            latest_log_date
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )