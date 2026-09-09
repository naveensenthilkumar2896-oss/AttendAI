from flask import Flask, render_template, request
import pandas as pd
import os


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)


# ==================================================
# EXCEL FILE LOCATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "attendance_data.xlsx"
)


# ==================================================
# LOAD STUDENT DATA
# ==================================================

def load_students():

    data = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Cohort Compliance Audit",
        dtype={"Register No": str}
    )

    # Clean Register Numbers
    data["Register No"] = (
        data["Register No"]
        .astype(str)
        .str.strip()
        .str.replace(".0", "", regex=False)
    )

    # Replace empty cells
    data = data.fillna("")

    return data


# ==================================================
# LOAD EXCEL DATA
# ==================================================

df = load_students()


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==================================================
# STUDENT PORTAL
# ==================================================

@app.route("/student")
def student_search():

    return render_template(
        "student_search.html"
    )


# ==================================================
# STUDENT SEARCH RESULT
# ==================================================

@app.route("/student/result", methods=["POST"])
def student_result():

    register_no = request.form.get(
        "register_no",
        ""
    ).strip()

    register_no = register_no.replace(
        ".0",
        ""
    )

    student_data = df[
        df["Register No"]
        .astype(str)
        .str.lower()
        ==
        register_no.lower()
    ]


    # STUDENT NOT FOUND
    # Stay on the same Student Portal page
    if student_data.empty:

        return render_template(
            "student_search.html",
            error="Register Number not found. Please check and try again."
        )


    # STUDENT FOUND
    student = student_data.iloc[0].to_dict()

    return render_template(
        "student_profile.html",
        student=student
    )


# ==================================================
# STUDENT PROFILE
# ==================================================

@app.route("/student/<register_no>")
def student_profile(register_no):

    register_no = str(
        register_no
    ).strip()

    register_no = register_no.replace(
        ".0",
        ""
    )

    student_data = df[
        df["Register No"]
        .astype(str)
        .str.lower()
        ==
        register_no.lower()
    ]


    # If invalid register number is entered in URL,
    # return to Student Portal instead of a new page
    if student_data.empty:

        return render_template(
            "student_search.html",
            error="Register Number not found. Please check and try again."
        )


    student = student_data.iloc[0].to_dict()

    return render_template(
        "student_profile.html",
        student=student
    )


# ==================================================
# STAFF PORTAL
# ==================================================

@app.route("/staff")
def staff_search():

    return render_template(
        "staff_search.html"
    )


# ==================================================
# STAFF DASHBOARD
# ==================================================

@app.route("/staff/dashboard")
def staff_dashboard():

    class_name = request.args.get(
        "class_name",
        "Attendance Dashboard"
    )


    # ----------------------------------------------
    # BASIC CLASS STATISTICS
    # ----------------------------------------------

    total_students = len(df)


    attendance_values = pd.to_numeric(
        df["Overall Attendance %"],
        errors="coerce"
    )


    average_attendance = round(
        attendance_values.mean(),
        2
    )


    # ----------------------------------------------
    # SAFE AND AT-RISK STUDENTS
    # ----------------------------------------------

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
        -
        safe_students
    )


    # ----------------------------------------------
    # ATTENDANCE RANGE ANALYTICS
    # ----------------------------------------------

    below_75 = int(
        (attendance_values < 75).sum()
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
        (attendance_values >= 90).sum()
    )


    # ----------------------------------------------
    # HIGHEST ATTENDANCE VALUE
    # ----------------------------------------------

    highest_attendance = round(
        float(attendance_values.max()),
        2
    )


    # ----------------------------------------------
    # LOWEST ATTENDANCE VALUE
    # ----------------------------------------------

    lowest_attendance = round(
        float(attendance_values.min()),
        2
    )


    # ----------------------------------------------
    # CLASS HEALTH ANALYSIS
    # ----------------------------------------------

    if average_attendance >= 90:

        class_health = (
            "Excellent class attendance. "
            "The class is maintaining a very strong "
            "attendance level."
        )

        health_status = "Excellent"


    elif average_attendance >= 85:

        class_health = (
            "Good overall class attendance. "
            "Most students are maintaining healthy "
            "attendance levels."
        )

        health_status = "Good"


    elif average_attendance >= 75:

        class_health = (
            "Moderate attendance level. "
            "Some students require attention to prevent "
            "attendance shortage."
        )

        health_status = "Moderate"


    else:

        class_health = (
            "Attendance requires immediate attention. "
            "A significant number of students may be "
            "at risk."
        )

        health_status = "Critical"


    # ----------------------------------------------
    # CONVERT STUDENT DATA
    # ----------------------------------------------

    students = df.to_dict(
        orient="records"
    )


    # ----------------------------------------------
    # SEND DATA TO DASHBOARD
    # ----------------------------------------------

    return render_template(

        "staff_dashboard.html",

        students=students,

        class_name=class_name,

        total_students=total_students,

        average_attendance=average_attendance,

        safe_students=safe_students,

        risk_students=risk_students,

        below_75=below_75,

        between_75_84=between_75_84,

        between_85_89=between_85_89,

        above_90=above_90,

        highest_attendance=highest_attendance,

        lowest_attendance=lowest_attendance,

        class_health=class_health,

        health_status=health_status
    )


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )