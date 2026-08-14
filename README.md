# CCATT HALDA-HOI: Trainee-Equipment Interaction Detection

This repo detects which piece of medical equipment a CCATT trainee used,
who used it, and exactly when -- from wide-angle CAM + PAN video recordings
of a training simulation.

## Which folder do I need?

- **[`CCATT_Inference/`](CCATT_Inference/)** -- run the already-trained
  model on your own videos. Start here if you just want results. See its
  [README](CCATT_Inference/README.md) for setup and usage.
- **[`Model training code/`](Model%20training%20code/)** -- train or
  fine-tune the individual models (CDN equipment/action detection, person
  role identification) yourself. Start here only if you're changing the
  models, not just running them.
