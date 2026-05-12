import requests
import base64
from src.schemas import ClipSearch, GenerateContent
from pprint import pprint

ray_endpoint = "http://127.0.0.1:8000"
image = base64.b64encode(open("./data/paddington.jpeg", "rb").read()).decode("utf-8")

# test gemma3
gemma_req = requests.post(
    url=f"{ray_endpoint}/gemma3/generate-content",
    json=GenerateContent(
        prompt="Given this image, where is it from?", input_image=image
    ).model_dump(),
)
if gemma_req.status_code != 200:
    print(f"error: {gemma_req.text}")
else:
    pprint(gemma_req.json())

# clip embed
clip_embed = requests.post(
    url=f"{ray_endpoint}/clip/embed-image",
    json=ClipSearch(input_image=image).model_dump(),
)
if clip_embed.status_code != 200:
    print(f"error: {clip_embed.text}")
else:
    pprint(clip_embed.json())


# clip image search
clip_image_search = requests.post(
    url=f"{ray_endpoint}/clip/image-search",
    json=ClipSearch(input_image=image).model_dump(),
)
if clip_image_search.status_code != 200:
    print(f"error: {clip_image_search.text}")
else:
    pprint(clip_image_search.json())
