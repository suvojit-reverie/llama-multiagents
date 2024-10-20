import boto3
print(boto3.__version__)

# Set default AWS region
default_region = "us-west-2"

# Create a Bedrock client in the AWS Region of your choice.
bedrock = boto3.client("bedrock", region_name=default_region)

import rich, json

def print_json(data):
    rich.print_json(json.dumps(data))

# List all models from meta
models = bedrock.list_foundation_models(
    byProvider='Meta'  # comment this line to get all models from all providers
)

# print_json(models)

###Prints all models
for model in models['modelSummaries']:
    print(model['modelId'])


from botocore.exceptions import ClientError

# Set the model ID.
# model_id = "us.meta.llama3-2-3b-instruct-v1:0"
model_id = "meta.llama3-1-70b-instruct-v1:0"

# Set the prompt.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Create a Bedrock Runtime client in the AWS Region you want to use.
bedrock_runtime = boto3.client("bedrock-runtime", region_name=default_region)

# Embed the prompt in Llama 3's instruction format.
# More information: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/
formatted_prompt = f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Format the request payload using the model's native structure.
native_request = {
    "prompt": formatted_prompt,
    "max_gen_len": 512,
    "temperature": 0.5,
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
    
    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["generation"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# print_json(response)
rich.print(response)


###converseAPI
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def stream_conversation(bedrock_client,
                    model_id,
                    messages,
                    system_prompts,
                    inference_config,
                    additional_model_fields):
    """
    Sends messages to a model and streams the response.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        messages (JSON) : The messages to send.
        system_prompts (JSON) : The system prompts to send.
        inference_config (JSON) : The inference configuration to use.
        additional_model_fields (JSON) : Additional model fields to use.

    Returns:
        None
    """

    logger.info("Streaming messages with model %s", model_id)

    response = bedrock_client.converse_stream(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    stream = response.get('stream')
    if stream:
        for event in stream:

            if 'messageStart' in event:
                print(f"\nRole: {event['messageStart']['role']}")

            if 'contentBlockDelta' in event:
                print(event['contentBlockDelta']['delta']['text'], end="")

            if 'messageStop' in event:
                print(f"\nStop reason: {event['messageStop']['stopReason']}")

            if 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata:
                    print("\nToken usage")
                    print(f"Input tokens: {metadata['usage']['inputTokens']}")
                    print(
                        f"Output tokens: {metadata['usage']['outputTokens']}")
                    print(f"Total tokens: {metadata['usage']['totalTokens']}")
                if 'metrics' in event['metadata']:
                    print(
                        f"Latency: {metadata['metrics']['latencyMs']} ms")


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

system_prompt = """You are an app that creates playlists for a radio station
  that plays rock and pop music. Only return song names and the artist."""

# Message to send to the model.
input_text = "Create a list of 3 pop songs."

message = {
    "role": "user",
    "content": [{"text": input_text}]
}
conversation = [message]

# System prompts.
system_prompts = [{"text": system_prompt}]

# inference parameters to use.
temperature = 0.5

# Base inference parameters.
inference_config = {
    "temperature": temperature
}

# Additional model inference parameters.
additional_model_fields = {}

try:
    bedrock_client = boto3.client(service_name='bedrock-runtime')

    stream_conversation(bedrock_client,
                        model_id,
                        conversation,
                        system_prompts,
                        inference_config,
                        additional_model_fields)

except ClientError as err:
    message = err.response['Error']['Message']
    logger.error("A client error occurred: %s", message)
    print("A client error occured: " +
          format(message))

else:
    print(
        f"Finished streaming messages with model {model_id}.")






# from langchain_aws import ChatBedrockConverse
# llm = ChatBedrockConverse(
#     model="us.meta.llama3-2-3b-instruct-v1:0",
#     temperature=0,
#     max_tokens=None,
#     # other params...
# )

# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# llm.invoke(messages)

# for chunk in llm.stream(messages):
#     # print(chunk)
#     try:
#     # if chunk.content !=[] and chunk.content[0]:
#         if chunk.content[0]:
#             print(chunk.content[0]["text"])
#     except Exception as e:
#         pass