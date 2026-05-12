import ray
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI

from src.models.gemma3 import Gemma3
from src.models.clip import Clip
from src.schemas import GenerateContent, ClipSearch, load_movies

ray.init()
serve.start()
app = FastAPI()


@serve.deployment
@serve.ingress(app)
class ModelRouter:
    def __init__(self, gemma3: DeploymentHandle, clip: DeploymentHandle):
        self.gemma3 = gemma3
        self.clip = clip

    @app.post("/gemma3/generate-content")
    async def gemma3_inference(self, input: GenerateContent):
        return await self.gemma3.generate_content.remote(input)

    @app.post("/clip/embed-image")
    async def clip_embed_image(self, input: ClipSearch):
        result = await self.clip.embed_image.remote(input.input_image)
        return {"embedding": result.tolist()}

    @app.post("/clip/image-search")
    async def clip_image_search(self, input: ClipSearch):
        return await self.clip.search_with_image.remote(**input.model_dump())


gemma3 = Gemma3.bind()
clip = Clip.bind(documents=load_movies("./data/movies.json"))
rayApp = ModelRouter.bind(gemma3, clip)
