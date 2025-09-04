import os
import pickle
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# ✅ Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "svc_clf.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    ans = None
    if request.method == "POST":
        try:
            # Convert inputs to float safely
            n  = float(request.form.get("n") or 0)
            p  = float(request.form.get("p") or 0)
            k  = float(request.form.get("k") or 0)
            t  = float(request.form.get("t") or 0)
            h  = float(request.form.get("h") or 0)
            ph = float(request.form.get("ph") or 0)
            rf = float(request.form.get("rf") or 0)

            # ✅ Make prediction
            ans = model.predict([[n, p, k, t, h, ph, rf]])[0]

            return redirect(url_for("res", ans=str(ans)))

        except Exception as e:
            print("Error:", e)
            ans = "Invalid input"

    return render_template("index.html", ans=ans)


@app.route("/result")
def res():
    ans = request.args.get("ans", default="Try again", type=str)
    return render_template("results.html", ans=ans)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
