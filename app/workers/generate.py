from app.adapters.inbound.generate_handler import GenerateHandler
from app.container import Container
from app.workers import worker


@worker(
    name="generate",
    queue="generate.request",
    exchange="cognition.exchange",
    routing_key="generate.request",
)
def create_generate_handler(container: Container) -> GenerateHandler:
    return GenerateHandler(publisher=container.publisher)
