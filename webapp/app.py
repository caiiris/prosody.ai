"""
Poetry era classifier webapp.
Paste a poem → get closest time period match with feature explanations.
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONC_PATH  = os.path.join(BASE_DIR, "data", "concreteness_ratings.csv")
TRAIN_CSV  = os.path.join(BASE_DIR, "data", "splits", "3class_train.csv")
MODEL_PKL  = os.path.join(BASE_DIR, "data", "model_3class.pkl")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import compute_features

ERA_ORDER = ["Pre-1800", "1800-1900", "Post-1900"]
ERA_META  = {
    "Pre-1800": {
        "label":       "Pre-Romantic / Early Modern",
        "years":       "before 1800",
        "description": "Formal verse, archaic diction (thee, thou, hath), consistent rhyme and line-initial capitalisation.",
        "color":       "#5c4033",
    },
    "1800-1900": {
        "label":       "Romantic / Victorian",
        "years":       "1800 – 1900",
        "description": "Lyrical intensity and emotional depth; structured verse; moderate archaic language; strong punctuation.",
        "color":       "#2d4a3e",
    },
    "Post-1900": {
        "label":       "Modernist & Contemporary",
        "years":       "after 1900",
        "description": "Free verse; conversational diction; minimal punctuation and rhyme; sparse archaic vocabulary.",
        "color":       "#1a3a5c",
    },
}

FEATURE_META = {
    "cap_rate":                {"label": "Line capitalisation",       "unit": "%",       "scale": 100},
    "punct_rate":              {"label": "Line-end punctuation",      "unit": "%",       "scale": 100},
    "colon_density":           {"label": "Colon / semicolon density", "unit": "/100 words",  "scale": 1},
    "concrete_abstract_ratio": {"label": "Concrete-to-abstract ratio","unit": "",        "scale": 1},
    "adv_verb_ratio":          {"label": "Adverb-to-verb ratio",      "unit": "",        "scale": 1},
    "rhyme_rate":              {"label": "Rhyme rate",                "unit": "%",       "scale": 100},
    "archaic_density":         {"label": "Archaic word density",      "unit": "/100 words",  "scale": 1},
}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Resource loading ───────────────────────────────────────────────────────────

def load_concreteness(path: str) -> dict:
    lex = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                word, score = line.rsplit(",", 1)
                lex[word.lower()] = float(score)
    return lex


def load_nltk_pos():
    import nltk
    for pkg in ["averaged_perceptron_tagger_eng", "punkt", "punkt_tokenizer"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return nltk.pos_tag


def load_cmu():
    import nltk
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    return cmudict.dict()


# ── Model ──────────────────────────────────────────────────────────────────────

from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder


class BayesianHurdleNB:
    HURDLE_FEATURES = {
        "rhyme_rate":       {"gate_is_upper": False, "threshold": 0,    "dist": "beta"},
        "colon_density":    {"gate_is_upper": False, "threshold": 0,    "dist": "gamma"},
        "cap_rate":         {"gate_is_upper": True,  "threshold": 0.95, "dist": "beta"},
        "punct_rate":       {"gate_is_upper": True,  "threshold": 0.90, "dist": "beta"},
        "archaic_density":  {"gate_is_upper": False, "threshold": 0,    "dist": "gamma"},
    }
    NON_HURDLE = ["concrete_abstract_ratio", "adv_verb_ratio"]

    def __init__(self, uniform_prior=True):
        self.uniform_prior = uniform_prior
        self.classes_ = ERA_ORDER
        self.log_priors_ = {}
        self.hurdle_params_ = {}
        self.non_hurdle_params_ = {}

    def fit(self, df, y_col="era"):
        if self.uniform_prior:
            for c in self.classes_:
                self.log_priors_[c] = np.log(1.0 / len(self.classes_))
        else:
            cc = df[y_col].value_counts(); total = len(df)
            for c in self.classes_:
                self.log_priors_[c] = np.log(cc.get(c, 1) / total)

        for feat, cfg in self.HURDLE_FEATURES.items():
            self.hurdle_params_[feat] = {}
            for c in self.classes_:
                vals = df.loc[df[y_col] == c, feat].dropna().values.astype(float)
                if len(vals) == 0:
                    self.hurdle_params_[feat][c] = None; continue
                if cfg["gate_is_upper"]:
                    p_gate = np.mean(vals >= cfg["threshold"])
                    cond   = vals[vals < cfg["threshold"]]
                else:
                    p_gate = np.mean(vals > cfg["threshold"])
                    cond   = vals[vals > cfg["threshold"]]
                p_gate = np.clip(p_gate, 1e-6, 1 - 1e-6)
                dp = None
                if len(cond) >= 2:
                    try:
                        if cfg["dist"] == "beta":
                            a, b, _, _ = scipy_stats.beta.fit(np.clip(cond, 1e-6, 1-1e-6), floc=0, fscale=1)
                            dp = ("beta", a, b)
                        else:
                            sh, _, sc = scipy_stats.gamma.fit(np.maximum(cond, 1e-6), floc=0)
                            dp = ("gamma", sh, sc)
                    except Exception:
                        pass
                self.hurdle_params_[feat][c] = {"p_gate": p_gate, "gate_is_upper": cfg["gate_is_upper"],
                                                 "threshold": cfg["threshold"], "dist_params": dp}

        for feat in self.NON_HURDLE:
            self.non_hurdle_params_[feat] = {}
            for c in self.classes_:
                vals = df.loc[df[y_col] == c, feat].dropna().values.astype(float)
                if len(vals) < 2:
                    self.non_hurdle_params_[feat][c] = None; continue
                try:
                    sh, _, sc = scipy_stats.gamma.fit(np.maximum(vals, 1e-6), floc=0)
                    self.non_hurdle_params_[feat][c] = ("gamma", sh, sc)
                except Exception:
                    self.non_hurdle_params_[feat][c] = None
        return self

    def _ll_hurdle(self, x, p):
        if p is None: return 0.0
        in_gate = (x >= p["threshold"]) if p["gate_is_upper"] else (x > p["threshold"])
        if in_gate: return np.log(p["p_gate"])
        ll = np.log(1 - p["p_gate"])
        if p["dist_params"]:
            dt = p["dist_params"][0]
            if dt == "beta":
                ll += scipy_stats.beta.logpdf(np.clip(x, 1e-6, 1-1e-6), p["dist_params"][1], p["dist_params"][2])
            else:
                ll += scipy_stats.gamma.logpdf(max(x, 1e-6), p["dist_params"][1], loc=0, scale=p["dist_params"][2])
        return ll

    def _ll_nh(self, x, p):
        if p is None or np.isnan(x): return 0.0
        return scipy_stats.gamma.logpdf(max(x, 1e-6), p[1], loc=0, scale=p[2])

    def _log_posts(self, row):
        lps = {}
        for c in self.classes_:
            lp = self.log_priors_[c]
            for feat in self.HURDLE_FEATURES:
                x = row.get(feat, float("nan"))
                if not np.isnan(x):
                    lp += self._ll_hurdle(x, self.hurdle_params_[feat][c])
            for feat in self.NON_HURDLE:
                lp += self._ll_nh(row.get(feat, float("nan")), self.non_hurdle_params_[feat][c])
            lps[c] = lp
        return lps

    def explain(self, row: dict, force_era: str = None, force_era_means: dict = None) -> dict:
        """
        Compute per-feature log-likelihood contributions.
        If force_era is given, generate reasons relative to that era
        (useful when RF makes the final prediction but BH-NB explains it).
        """
        all_feats = list(self.HURDLE_FEATURES.keys()) + self.NON_HURDLE
        contribs = {feat: {} for feat in all_feats}
        log_posts = {}
        for c in self.classes_:
            lp = self.log_priors_[c]
            for feat in self.HURDLE_FEATURES:
                x = row.get(feat, float("nan"))
                ll = self._ll_hurdle(x, self.hurdle_params_[feat][c]) if not np.isnan(x) else 0.0
                contribs[feat][c] = ll; lp += ll
            for feat in self.NON_HURDLE:
                ll = self._ll_nh(row.get(feat, float("nan")), self.non_hurdle_params_[feat][c])
                contribs[feat][c] = ll; lp += ll
            log_posts[c] = lp

        predicted = force_era if force_era in self.classes_ else max(log_posts, key=log_posts.get)
        log_arr = np.array([log_posts[c] for c in self.classes_])
        log_arr -= log_arr.max(); probs = np.exp(log_arr); probs /= probs.sum()

        reasons = []
        for feat in all_feats:
            ll_pred   = contribs[feat][predicted]
            ll_others = np.mean([contribs[feat][c] for c in self.classes_ if c != predicted])
            reasons.append((feat, ll_pred - ll_others, row.get(feat, float("nan"))))
        reasons.sort(key=lambda t: abs(t[1]), reverse=True)

        top = []
        for feat, adv, val in reasons[:3]:
            meta         = FEATURE_META.get(feat, {"label": feat, "unit": "", "scale": 1})
            scale        = meta["scale"]
            unit         = meta["unit"]
            name         = meta["label"]
            disp         = val * scale
            era_means    = force_era_means or {}
            pred_mean    = era_means.get(predicted, {}).get(feat, None)
            other_means  = [era_means.get(c, {}).get(feat, None)
                            for c in self.classes_ if c != predicted]
            other_means  = [m for m in other_means if m is not None]

            if pred_mean is not None and other_means:
                avg_other = np.mean(other_means)
                pred_disp = pred_mean * scale
                other_disp = avg_other * scale
                if unit:
                    top.append(
                        f"{name}: {disp:.1f}{unit} "
                        f"(avg {predicted}: {pred_disp:.1f}{unit}, "
                        f"other eras: {other_disp:.1f}{unit})"
                    )
                else:
                    top.append(
                        f"{name}: {disp:.2f} "
                        f"(avg {predicted}: {pred_disp:.2f}, "
                        f"other eras: {other_disp:.2f})"
                    )
            else:
                direction = "consistent with" if adv > 0 else "atypical for"
                top.append(f"{name}: {disp:.1f}{unit} — {direction} {predicted}")

        return {
            "predicted": predicted,
            "probs": {c: float(probs[i]) for i, c in enumerate(self.classes_)},
            "top_reasons": top,
        }


FEATURE_COLS_RF = [
    "cap_rate", "punct_rate", "colon_density",
    "concrete_abstract_ratio", "adv_verb_ratio", "rhyme_rate",
    "archaic_density",
    "uses_rhyme", "uses_colons", "caps_all", "high_punct", "uses_archaism",
]


def build_hurdle_row(feat: dict) -> dict:
    row = dict(feat)
    row["uses_rhyme"]    = int(feat["rhyme_rate"] > 0)
    row["uses_colons"]   = int(feat["colon_density"] > 0)
    row["caps_all"]      = int(feat["cap_rate"] >= 0.95)
    row["high_punct"]    = int(feat["punct_rate"] >= 0.90)
    row["uses_archaism"] = int(feat["archaic_density"] > 0)
    return row


def train_or_load_models(train_csv: str, pkl_path: str):
    if os.path.exists(pkl_path):
        log.info("Loading cached models from %s", pkl_path)
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except (AttributeError, pickle.UnpicklingError) as e:
            log.warning("Pickle load failed (%s), retraining…", e)
            os.remove(pkl_path)

    log.info("Training models on %s …", train_csv)
    df = pd.read_csv(train_csv)
    if "era_3class" in df.columns:
        df["era"] = df["era_3class"]

    # BH-NB
    bnb = BayesianHurdleNB(uniform_prior=True)
    bnb.fit(df)

    # RF
    X = df[FEATURE_COLS_RF].fillna(0).values
    y = df["era"].values
    le = LabelEncoder(); le.fit(ERA_ORDER); y_enc = le.transform(y)
    class_counts = np.bincount(y_enc)
    sw = np.array([len(y_enc) / (len(ERA_ORDER) * class_counts[yy]) for yy in y_enc])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)

    scaler = StandardScaler()
    scaler.fit(X)

    # Era means (for feature display context)
    era_means = {}
    for era in ERA_ORDER:
        sub = df[df["era"] == era]
        era_means[era] = {f: float(sub[f].mean()) for f in
                          ["cap_rate","punct_rate","colon_density",
                           "concrete_abstract_ratio","adv_verb_ratio",
                           "rhyme_rate","archaic_density"]}

    bundle = {"bnb": bnb, "rf": rf, "scaler": scaler, "le": le, "era_means": era_means}
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Models saved to %s", pkl_path)
    return bundle


# ── App startup ────────────────────────────────────────────────────────────────

app = Flask(__name__)

log.info("Loading NLP resources…")
LEX     = load_concreteness(CONC_PATH)
POS_TAG = load_nltk_pos()
CMU     = load_cmu()
BUNDLE  = train_or_load_models(TRAIN_CSV, MODEL_PKL)
log.info("Ready.")

# Warm up NLTK tagger with a dummy call so first request is fast
try:
    POS_TAG(["warm"])
except Exception:
    pass


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "poem" not in data:
        return jsonify({"error": "Missing 'poem' in request body"}), 400
    poem = data["poem"].strip()
    if not poem:
        return jsonify({"error": "Poem text is empty"}), 400
    if len(poem.split()) < 8:
        return jsonify({"error": "Paste a longer poem (at least a few lines) for a meaningful result."}), 400

    try:
        feat = compute_features(poem, LEX, CMU, POS_TAG)
    except Exception as e:
        log.exception("Feature error")
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    # Replace NaN with 0 for model input
    feat_clean = {k: (0.0 if np.isnan(v) else v) for k, v in feat.items()}
    hurdle_row = build_hurdle_row(feat_clean)

    # RF prediction
    rf    = BUNDLE["rf"]
    X_row = np.array([[hurdle_row.get(c, 0.0) for c in FEATURE_COLS_RF]])
    rf_probs_arr = rf.predict_proba(X_row)[0]
    rf_classes   = rf.classes_
    rf_probs = {c: float(rf_probs_arr[list(rf_classes).index(c)]) for c in ERA_ORDER}
    rf_pred  = max(rf_probs, key=rf_probs.get)

    # BH-NB explanations — anchored to RF's predicted era, with era means for context
    bnb_result = BUNDLE["bnb"].explain(feat_clean, force_era=rf_pred,
                                       force_era_means=BUNDLE["era_means"])

    # Feature display: value + context vs era means
    era_means = BUNDLE["era_means"]
    feature_display = []
    for key, meta in FEATURE_META.items():
        val = feat_clean.get(key, 0.0)
        disp_val = round(val * meta["scale"], 2)
        predicted_mean  = era_means[rf_pred].get(key, 0.0)
        # find highest-mean era for this feature
        all_means = {e: era_means[e].get(key, 0.0) for e in ERA_ORDER}
        max_era = max(all_means, key=all_means.get)
        # direction: higher than predicted mean or lower?
        if predicted_mean > 0:
            ratio = val / predicted_mean
            if ratio > 1.4:
                context = "high"
            elif ratio < 0.6:
                context = "low"
            else:
                context = "moderate"
            else:
                context = "none" if val == 0.0 else "moderate"
        feature_display.append({
            "key":   key,
            "label": meta["label"],
            "value": disp_val,
            "unit":  meta["unit"],
            "context": context,
        })

    return jsonify({
        "era":         rf_pred,
        "era_meta":    ERA_META[rf_pred],
        "probabilities": [
            {"era": e, "label": ERA_META[e]["label"],
             "years": ERA_META[e]["years"],
             "prob": round(rf_probs[e], 4)}
            for e in ERA_ORDER
        ],
        "top_reasons": bnb_result["top_reasons"],
        "features":    feature_display,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
