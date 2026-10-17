from huggingface_hub import snapshot_download

for repo in ['facebook/dinov2-small',
             'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf']:
    print(snapshot_download(repo,allow_patterns=['*.json','*.safetensors'],
                            cache_dir='/tmp/earshot-nav/models'),flush=True)
