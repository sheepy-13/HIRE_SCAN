import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Initialize the kernel
kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
model_id = "gpt-3.5-turbo"
service_id = "chat-gpt"  # Ensure this matches your intended service ID
env_file_path =r"C:\Users\gauth\Hire-scan\.env"  # Use raw string to avoid escaping backslashes
chat_completion = OpenAIChatCompletion(service_id=service_id, env_file_path=env_file_path,ai_model_id=model_id)

# Add OpenAI chat completion service to the kernel
kernel.add_service(chat_completion)

# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 2000
req_settings.temperature = 0.7
req_settings.top_p = 0.8

# Prompt template configuration
prompt = """
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words."""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    execution_settings=req_settings,
)

# Add a function to the kernel based on the prompt template configuration
function = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt_template_config=prompt_template_config,
)

# Create a reusable summarize function
summarize = kernel.add_function(
    function_name="summarize_function",
    plugin_name="summarize_plugin",
    prompt="{{$input}}\n\nOne line TLDR with the fewest words.",
    prompt_template_settings=req_settings,
)

async def main():
    # Invoke the initial TL;DR function for the robot laws
    result = await kernel.invoke(function)
    
    # Print the result from the robot laws summary
    print("Robot Laws Summary:")
    print(result) # => Robots must not harm humans.
    
    # Summarize the laws of thermodynamics
    thermodynamics_summary = await kernel.invoke(summarize, input="""
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy.""")
    print("Thermodynamics Laws Summary:")
    print(thermodynamics_summary)  # => Energy conserved, entropy increases, zero entropy at 0K.

    # Summarize the laws of motion
    motion_summary = await kernel.invoke(summarize, input="""
1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
2. The acceleration of an object depends on the mass of the object and the amount of force applied.
3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first.""")
    print("Laws of Motion Summary:")
    print(motion_summary)  # => Objects move in response to forces.

    # Summarize the law of universal gravitation
    gravitation_summary = await kernel.invoke(summarize, input="""
Every point mass attracts every single other point mass by a force acting along the line intersecting both points.
The force is proportional to the product of the two masses and inversely proportional to the square of the distance between them.""")
    print("Law of Universal Gravitation Summary:")
    print(gravitation_summary)  # => Gravitational force between two point masses is inversely proportional to the square of the distance between them.

if __name__ == "__main__":
    asyncio.run(main())
