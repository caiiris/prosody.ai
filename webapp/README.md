# Poetry Era webapp

Paste a poem and see its closest time period match.

## Run locally

From the project root:

```bash
pip install -r requirements.txt
cd webapp && python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Current behavior

- The **Analyze** button sends the pasted poem to `POST /analyze` and shows the result.
- The backend currently returns a **placeholder** response (e.g. "Contemporary", 72% confidence). When you add your trained model, update `app.py`: load the saved pipeline and call it in `analyze()` instead of `_placeholder_predict()`.
- **Clear** resets the text area and hides the result.
- Short poems (under ~10 words) get a message asking for more text.

## Wiring in the real model

In `app.py`, replace the body of `_placeholder_predict(poem)` (or the logic inside `analyze()` that calls it) with:

1. Load your saved model/pipeline once at startup (e.g. in a global or `current_app`).
2. Run your feature extraction on `poem`.
3. Call `model.predict()` / pipeline and map the predicted class to an era name.
4. Return `{"era": "...", "confidence": ..., "alternatives": [...]}` in the same shape as the placeholder so the existing frontend keeps working.
