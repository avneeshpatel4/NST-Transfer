# AdaIN style transfer

## Deploying to Render

This repository includes `render.yaml`. In Render, create a **Blueprint** from
the repository (or create a Python Web Service manually) and use:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --config gunicorn.conf.py app:app`

Set a random `SECRET_KEY` environment variable in the Render dashboard.

The service intentionally uses one Gunicorn worker and resizes each uploaded
image to a maximum of 256 × 256 before inference. The original 512-pixel CPU
inference can exceed the memory or request-time limit of small Render
instances, leading to a 502. Set `INFERENCE_SIZE=384` or `512` only on an
instance with sufficient memory; 256 is the recommended default for reliable
deployment.

Render storage is ephemeral: uploads and generated images disappear after a
restart or redeploy. Use persistent object storage if results must be retained.
