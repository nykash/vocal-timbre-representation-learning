# From representation to vocal coach

The research prototype was connected to a small product to test whether the representation could support an interaction, not only an offline notebook.

## Inference path

1. The browser records a vocal clip as WAV.
2. A FastAPI endpoint accepts the upload.
3. Audio is resampled to 22.05 kHz and divided into six-second log-mel chunks.
4. The VAE encoder produces one mean vector per chunk.
5. Chunk vectors are averaged into a clip representation.
6. A diagonal Gaussian fitted per reference artist produces a ranked distribution.
7. Artist probabilities are projected onto a small vocabulary of vocal-style tags.

The Gaussian layer is a lightweight supervised probe over a self-supervised representation. Softmax temperature changes presentation confidence; it is not statistical calibration.

## LLM and frontend

The React frontend combines live pitch tracking with the timbre analysis. An LLM acts as an interaction router:

- tuning questions open a pitch-analysis view;
- timbre/style questions request the recorded clip analysis;
- structured ML results are inserted into a follow-up prompt; and
- the response turns measurements into short coaching suggestions.

This separation is deliberate. The audio model computes features and comparisons; the LLM does not invent acoustic measurements. The UI shows the closest references and style tags while the conversational layer explains them.

## Current integration status

The larger application contains all of these paths, but they are not equally active on the current landing page:

- VAE style analysis is active through the recording modal and is fed back to the coaching conversation.
- Exercise recording combines VAE style output with client-side pitch summaries for grading.
- The current main page passes an empty live pitch context and a placeholder tuning summary; complete rolling-pitch wiring remains in the older analyzer page and the sing-along flow.
- Vocal-register inference is exposed by the backend and its own modal, but is not currently inserted into the chat prompt.

This showcase therefore treats the product as a working research integration with known wiring gaps, not as a fully unified production system.

## Limitations

- Artist similarity is a communication device, not an identity claim.
- The downstream probabilities are relative to artists present in the reference set.
- Temperature-scaled scores are not calibrated confidence estimates.
- Accompaniment, microphone, room acoustics, and song choice can affect the embedding.
- Style tags are manually curated and culturally subjective.

For a production system, I would expose embedding stability across multiple clips, reject out-of-distribution audio, calibrate the downstream probe, and phrase results as “similar within this reference set.”
