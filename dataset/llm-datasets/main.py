import openai
import pandas as pd
from dotenv import load_dotenv
import os
import datetime
import logging

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Verify if the API key is loaded
if openai.api_key is None:
    raise ValueError("API key not found. Make sure OPENAI_API_KEY is defined in your .env file.")

# Set up logging configuration
logging.basicConfig(
    filename='synthetic_data_generation.log',  # Log file name
    filemode='a',  # Append to the file
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

logging.info("Started synthetic data generation script")
print("Started synthetic data generation script")

MODEL = 'gpt-3.5-turbo'  # Replace with your model version or API call method for Gemini

# Define the base system prompt with Financial Literacy Tutor context and simulation information
base_system_prompt = (
    "You are a tutor for the Financial Literacy Tutor platform, "
    "which aims to close the financial literacy gap by leveraging AI to provide an accessible, engaging, and personalized financial education platform. "
    "The platform guides learners through financial concepts step-by-step, using AI to tailor content to each user’s learning style, ability, and progress. "
    "It offers a freemium model, providing basic content for free and premium features at an affordable price. "
    "Unique selling points include skill assessments, personalized study plans, and interactive finance quizzes that make learning more engaging and effective. "
    "The platform also includes investment simulations to help users apply their knowledge in practical scenarios.\n\n"
    "I want to build an AI tutor app and now I am trying to simulate the conversation between a student and the AI tutor, so that we could have a dataset for evaluation.\n"
    "We propose that there are several stages the student can be in:\n"
    "1. 'Building Rapport with Students'\n"
    "2. 'Assessing the Student's School Curriculum'\n"
    "3. 'Evaluating the Student's Learning Style'\n"
    "4. 'Quizzing Students for Current Knowledge'\n"
    "5. 'Designing a Personalised Study Plan'\n"
    "6. 'Beginning Lectures Following the Study Plan'\n"
    "7. 'Explaining Concepts and Providing Examples'\n"
    "8. 'Quizzing Students with Past Exam Questions'\n"
    "9. 'Providing Images, Diagrams, and Online Resources'\n"
    "10. 'Ensuring Students Can Ask Questions Anytime'\n"
    "11. 'Updating Pedagogy According to Conversations'\n"
    "12. 'Repeating Stages 4-11 for New Topics'\n"
    "Generate a conversation that involves every stage as mentioned above in sequential order. Do this for 2 topics so that we repeat stages 4-11 once. Write in a human and informal way. Label the conversation well with either the AI tutor or the student, and also the stage of the conversation at every point."
)

# Define stages and their respective prompts, including stages 10, 11, and 12
stage_prompts = {
    7: [
        "Break down the concept of compound interest into manageable pieces for a university student.",
        "Use analogies to explain inflation to a student who understands food prices.",
        "Incorporate diagrams to illustrate the difference between fixed and variable interest rates.",
        "Explain why understanding financial literacy is important for a student's future.",
        "How would you encourage questions from a student unsure about their understanding of stock markets?",
        "Provide multiple examples to explain risk diversification in investing.",
        "Connect the concept of present value to what the student has learned about future value.",
        "Assess the student's understanding of bond yields through a follow-up quiz question."
    ],
    8: [
        "Provide past exam questions related to capital budgeting.",
        "Simulate an exam condition quiz on risk management in a timed setting.",
        "Review common mistakes in solving the Net Present Value (NPV) problem.",
        "Discuss strategies for solving long case studies in corporate finance.",
        "Track a student's progress over time based on their answers to questions on portfolio management.",
        "Offer feedback on a student's answer to a question about weighted average cost of capital (WACC).",
        "Modify study sessions based on a student's performance in fixed income analysis quizzes."
    ],
    9: [
        "Incorporate infographics and images to explain portfolio diversification visually.",
        "Share online resources for students to learn about financial planning and investment.",
        "Recommend educational videos that explain the concept of market risk visually.",
        "Create a resource bank for students to access materials on financial literacy anytime.",
        "Encourage digital literacy by teaching students how to evaluate online finance articles.",
        "Recommend interactive tools like financial calculators or budgeting apps.",
        "Include diagrams to illustrate the impact of interest rates on bonds during lectures.",
        "Offer additional reading materials for students who want to delve deeper into investment analysis."
    ],
    10: [
        "Establish a Q&A period for students to ask questions on personal finance topics.",
        "Check if a student has questions about the recent lecture on options trading.",
        "Emphasise that all questions about financial derivatives are valid and important.",
        "Follow up on a student's unanswered question about exchange-traded funds (ETFs)."
    ],
    11: [
        "Hold regular feedback sessions with students about their learning experiences in finance.",
        "Adapt teaching methods for financial planning based on student feedback.",
        "Incorporate student suggestions to enhance the learning environment.",
        "Stay informed on best practices for teaching financial literacy and investment strategies.",
        "Monitor student engagement during discussions on market analysis and risk management.",
        "Reflect on teaching effectiveness in explaining the time value of money.",
        "Encourage a growth mindset in students learning about risk tolerance and investment horizons.",
        "Plan for continuous improvement by adjusting lessons on personal financial management."
    ],
    12: [
        "Repeat steps 4 to 11 iteratively for topics like investment analysis and financial planning.",
        "Repeat steps 4 to 11 iteratively for topics related to budgeting, savings, and debt management.",
        "Repeat steps 4 to 11 iteratively for topics such as market analysis and investment diversification."
    ]
}

# Function to generate synthetic data using ChatGPT for a specific stage
def generate_synthetic_data_for_stage(stage):
    try:
        # Access the prompts for the specific stage
        prompts = stage_prompts.get(stage, [])
        if not prompts:
            logging.warning(f"No prompts found for stage {stage}")
            return

        logging.info(f"Generating data for Stage {stage} only with {len(prompts)} prompts.")
        print(f"Generating data for Stage {stage} only...")

        # Initialize an empty list to store data entries for the specific stage
        stage_data = []

        for idx, prompt in enumerate(prompts):
            # Create a dynamic prompt indicating the current stage
            dynamic_prompt = f"We are currently at Stage {stage}: {prompt}"
            full_system_prompt = f"{base_system_prompt}\n\n{dynamic_prompt}"
            student_prompt = f"I would like to know more about {prompt.split(':')[0].lower()}."

            # Generate the response using the OpenAI API
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            chatgpt_output = response['choices'][0]['message']['content']

            # Log and print the output
            logging.info(f"Data generated successfully for stage {stage}, prompt index {idx}.")
            print(f"ChatGPT Output for Stage {stage}, Prompt {idx}:\n{chatgpt_output}\n")

            # Save the data entry
            data_entry = {
                "x": prompt,
                "x1": student_prompt,
                "x2": chatgpt_output,
                "x3": MODEL,
                "x4": datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S.%f"),
                "y1": stage,  # Stage number
                "y2": chr(97 + idx)  # Sub-stage identifier (e.g., 'a', 'b', etc.)
            }
            stage_data.append(data_entry)

        return stage_data

    except Exception as e:
        logging.error(f"Error generating data for stage {stage}: {e}")
        print(f"Error generating data for stage {stage}: {e}")
        return None

# Create an output directory if it doesn't exist
output_directory = 'output'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate through stages 7 to 12 and generate data for each stage
for selected_stage in range(7, 13):
    stage_data = generate_synthetic_data_for_stage(selected_stage)

    # Convert to a dataframe and save to CSV if data is available
    if stage_data:
        output_filepath = os.path.join(output_directory, f'synthetic_data_stage_{selected_stage}.csv')
        df = pd.DataFrame(stage_data)
        df.to_csv(output_filepath, index=False)
        logging.info(f"Data generation complete for Stage {selected_stage}. Saved as '{output_filepath}'.")
        print(f"Data generation complete for Stage {selected_stage}. Saved as '{output_filepath}'.")
    else:
        logging.warning(f"No data generated for Stage {selected_stage}.")
        print(f"No data generated for Stage {selected_stage}.")
