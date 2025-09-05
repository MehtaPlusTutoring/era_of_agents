import asyncio
import os
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

async def main():
    # Load environment variables from config.env
    load_dotenv("config.env")
    
    # Get Azure OpenAI credentials from environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Initialize the Kernel
    kernel = Kernel()

    # Add Azure OpenAI chat service
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    
    kernel.add_service(chat_service)
        
    # User prompt
    prompt = "Write a detailed, inspiring message for a new intern on their first day of joining Tabby, a company that creates a cat matching app. Include 5 bullet points of advice."
    
    # Invoke stream
    response = kernel.invoke_prompt_stream(prompt)

    async for chunk in response:
        print(chunk[0].content, end="")

if __name__ == "__main__":
    asyncio.run(main())