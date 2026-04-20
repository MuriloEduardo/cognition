from app.adapters.inbound.amqp.handlers.cognition_handler import CognitionHandler
from app.container import Container
from app.workers import worker


@worker(
    name="cognition",
    queue="cognition.request",
    exchange="cognition.exchange",
    routing_key="cognition.request",
)
def create_cognition_handler(container: Container) -> CognitionHandler:
    return CognitionHandler(
        publisher=container.publisher,
        llm_service=container.llm_service,
        inference_logs=container.inference_logs,
    )
