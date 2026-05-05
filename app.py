import json
import os
import threading

from flask import Flask, jsonify, redirect, render_template, url_for

app = Flask(__name__)

# ──────────────────────────────────────────────
# Shared training state (protected by a lock)
# ──────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "is_training": False,
    "progress":    "",
    "log":         [],        # list of status messages
    "error":       None,
    "complete":    False,
}


def _update(msg: str, error: str = None, complete: bool = False):
    with _lock:
        _state["progress"] = msg
        _state["log"].append(msg)
        if error is not None:
            _state["error"] = error
        if complete:
            _state["complete"] = complete


def _run_training():
    with _lock:
        _state.update({"is_training": True, "complete": False,
                        "error": None, "log": [], "progress": "Starting…"})
    try:
        from train_models import train_and_evaluate
        train_and_evaluate(status_cb=_update)
        _update("Training complete!", complete=True)
    except Exception as exc:
        _update(f"Error: {exc}", error=str(exc))
    finally:
        with _lock:
            _state["is_training"] = False


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route("/")
def index():
    trained = os.path.exists("static/results.json")
    return render_template("index.html", trained=trained)


@app.route("/train", methods=["POST"])
def train():
    with _lock:
        if _state["is_training"]:
            return jsonify({"success": False, "message": "Already training."})

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return jsonify({"success": True})


@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_state))


@app.route("/results")
def results():
    if not os.path.exists("static/results.json"):
        return redirect(url_for("index"))

    with open("static/results.json") as f:
        data = json.load(f)

    models_order = ["knn", "decision_tree", "naive_bayes"]
    models = [data[k] for k in models_order]
    best_key = data["best_model"]
    best_label = data[best_key]["label"]

    return render_template(
        "results.html",
        data=data,
        models=models,
        models_order=models_order,
        best_key=best_key,
        best_label=best_label,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
