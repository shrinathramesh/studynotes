# Smart Study Notes Generator (Vercel)

## Deploy
1. Create a free Hugging Face access token (huggingface.co/settings/tokens)
   with permission to call Inference Providers.
2. Install the CLI and deploy from this folder:
       npm i -g vercel
       vercel
       vercel env add HF_TOKEN        # paste the token, select all environments
       vercel --prod
   (Or push to GitHub, import the repo in Vercel, and add HF_TOKEN under
   Settings -> Environment Variables.)

## Local test
       vercel dev

## Optional env vars
- HF_MODEL : default facebook/bart-large-cnn
- HF_URL   : override the full inference endpoint if Hugging Face changes it

## Structure
- public/index.html   -> the web UI
- api/summarize.py    -> serverless summarizer (stdlib only, no requirements.txt)
- vercel.json         -> 60 s function timeout
