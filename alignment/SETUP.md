# alignment env / Vertex setup

## Auth (Vertex, not AI Studio)

Vertex uses a GCP **service account**, not an API key. The service account + env file live in
`~/.secrets/` on ccn2 (chmod 600, outside any repo) — never in git.

One-time (GCP), project `hs-hs-langcog-gemini`:

```bash
gcloud iam service-accounts create vlm-headcam --project=hs-hs-langcog-gemini
gcloud projects add-iam-policy-binding hs-hs-langcog-gemini \
  --member="serviceAccount:vlm-headcam@hs-hs-langcog-gemini.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud iam service-accounts keys create vlm-headcam-sa.json \
  --iam-account=vlm-headcam@hs-hs-langcog-gemini.iam.gserviceaccount.com
gcloud services enable aiplatform.googleapis.com --project=hs-hs-langcog-gemini
# scp vlm-headcam-sa.json to ccn2, then:  chmod 700 ~/.secrets; chmod 600 ~/.secrets/*
```

`~/.secrets/vlm-headcam.env` (chmod 600), `source`d before running:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=hs-hs-langcog-gemini
export GOOGLE_CLOUD_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=$HOME/.secrets/vlm-headcam-sa.json
```

The `google-genai` SDK reads exactly these four vars; the scoring script hard-codes nothing.

**Shared-cluster hygiene:** scope the key to `aiplatform.user` on the one project, and revoke it
after a big run (`gcloud iam service-accounts keys delete …`). Don't leave a broad, long-lived
key on a shared filesystem.

## Python

Needs `google-genai`, `pillow`, `pandas`, `pyarrow`. Any recent env works (e.g. the pose
`pose_env`); `pip install google-genai pillow pandas`.
