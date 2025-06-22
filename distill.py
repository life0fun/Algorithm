import os
import dspy
import random
import torch
import json
import re
import traceback
from typing import Literal
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from dspy.clients.lm_local import LocalProvider
from dspy import BaseLM 
from dspy.teleprompt import BootstrapFinetune
from dspy.evaluate import Evaluate
from dspy.datasets import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from dspy.adapters.chain_of_thought_adapter import ChainOfThoughtAdapter

#
# Running LLM locally without GPU. https://dspy.ai/learn/programming/language_models/
# 1. ollama run llama3.2:1b; use ollama api/chat. 'ollama_chat/llama3.2:1b'. Exact Match.
# 2. SGLang requires cuda toolkit. apt install nvidia-cuda-toolkit
# 3. SGLang+flashinfer pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/
# 4. uv pip install "sglang[all]>=0.4.7"; uv pip install -U torch transformers==4.48.3 accelerate trl peft
# 5. Running SGLang "No accelerator (CUDA, XPU, HPU) is available".

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

dspy.inspect_history(n=5)

# use aihubmix proxy
# lm = dspy.LM('openai/gpt-4o-mini', api_base='https://aihubmix.com')
# lm = dspy.LM('', api_base='https://aihubmix.com')

# huggingface-cli login --token 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
# export HUGGINGFACE_HUB_TOKEN=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
hf_hub_download(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    filename="config.json",
    use_auth_token=True  # make sure you're logged in
)

# A basic module to distill from teacher. 
# The teacher will generate the answer, and potentially a thought process.
class TeacherQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> thought, answer")

    def forward(self, question):
        return self.generate_answer(question=question)

class StudentQASignature(dspy.Signature):
    question = dspy.InputField(desc="The student's question")
    reasoning = dspy.OutputField(desc="The reasoning answer")
    answer = dspy.OutputField(desc="The short, factual answer")

class StudentQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(StudentQASignature)
        # self.generate_answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        # return dspy.Prediction(answer=question + " answer is hello world")
        return self.generate_answer(question=question)


class HFWrapper:
    def __init__(self, model_name, max_tokens):
        self.model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.auto_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.auto_model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_model.to(self.device)
        self.max_new_tokens = max_tokens

    def generate(self, prompts, max_tokens=None, **kwargs):
        outputs = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.auto_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_tokens or self.max_new_tokens,
                temperature=kwargs.get("temperature", 0.0),
                do_sample=kwargs.get("do_sample", False),
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(text)
        return outputs

    def decode(self, completions, prompts=None):
        decoded = []
        for i, completion in enumerate(completions):
            if prompts:
                prompt = prompts[i]
                if completion.startswith(prompt):
                    completion = completion[len(prompt):].strip()
            print(f"----decode completion start:\n {completion.strip()} \n -----\n")
            decoded.append(completion.strip())
            mock_answer = f"""[[ ## reasoning ## ]] The user is asking for the capital of Germany. This is a factual question that can be answered directly from common knowledge.
[[ ## answer ## ]] The capital of Germany is Berlin. [[ ## completed ## ]]"""
            fields = ChainOfThoughtAdapter().parse(StudentQASignature, mock_answer)
            decoded.append(fields)
        return decoded

    def __call__(self, prompt, **kwargs):
        return self.generate([prompt], **kwargs)[0]

class HFLocalModel(dspy.BaseLM):
    def __init__(self, model_name, max_tokens=2048):
        self.backend = HFWrapper(model_name, max_tokens)
        super().__init__(model=model_name)

    
    def __call__(self, *args, **kwargs):
        # DSPy may pass chat-style inputs like {"messages": [...]}
        prompt = None
        if "messages" in kwargs:
            messages = kwargs["messages"]
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                prompt = user_msgs[-1]["content"]
            else:
                prompt = messages[-1].get("content", "")

        # Fallback to common prompt keys
        for key in ("prompt", "question", "input", "text"):
            if prompt is None and key in kwargs:
                val = kwargs[key]
                if isinstance(val, str):
                    prompt = val
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    prompt = val[0]

        if prompt is None and args:
            prompt = args[0] if isinstance(args[0], str) else None

        if prompt is None:
            raise ValueError(f"No prompt found in __call__. args={args}, kwargs={kwargs}")

        print(f"----- calling LM with prompt \n {prompt} \n----\n")
        completions = self.backend.generate([prompt], **kwargs)
        return completions
    #     decoded = self.backend.decode(completions, prompts=[prompt])
    #     return decoded[0]
        
# Teacher LM: A powerful model like GPT-4o-mini (or ChatGPT equivalent)
# This model will generate high-quality demonstrations for the student.
teacher_lm = dspy.LM('openai/gpt-4o-mini', api_key=openai_api_key, max_tokens=3000)

# ollama run llama3.2:1b; Model name(ollama_chat/llama3.2:1b) must be exact match !!
#student_lm = dspy.LM('ollama_chat/llama3.2:1b', api_base='http://localhost:11434', api_key='')

# Student LM: A smaller, local Hugging Face model to be fine-tuned.
student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
# student_lm_name = "google/flan-t5-small" # super smaller model for local CPU.
# Dspy client LocalProvider uses SGLang under the hood for local model inference.
# student_lm = dspy.LM(model=f"openai/local:{student_lm_name}", provider=LocalProvider(), max_tokens=2000)
# launch the model via sglang, require GPU accelerator.
# student_lm.launch()
# student_lm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#student_lm_name = "google-t5/t5-small"
# student_lm = dspy.LM(model=HFLocalModel(
#     model_name=student_lm_name,
#     max_tokens=512
#     #model_kwargs={"torch_dtype": "auto"}
# ))
# student_lm = dspy.LM(model=HFLocalModel(model_name=student_lm_name, max_tokens=512))
student_lm = HFLocalModel(model_name=student_lm_name, max_tokens=5096)
print(f"\n--- local student local model {student_lm} ---")

# A small labeled development set for evaluation
# In a real scenario, this would be manually labeled data.
evaluate_dataset = [
    dspy.Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    dspy.Example(question="What is the chemical symbol for water?", answer="H2O").with_inputs("question"),
]

# --- 4. Evaluate the Student LM (before fine-tuning) ---
print("\n--- Evaluating Student LM BEFORE Fine-tuning ---")
# Temporarily configure DSPy to use the student_lm for evaluation
dspy.settings.configure(lm=student_lm)

student = StudentQA()
question = "what is the capital of German ?"
answer = student(question=question)
print(f"the answer to question {question} is {answer}")
# for ex in evaluate_dataset:
#     print("Running on:", ex.question)
#     out = student(question=ex.question)
#     print("RAW Prediction:", out)
#     print("Fields in out:", out.__dict__)
#     print("Prediction:", out.answer)

# Define a simple metric for evaluation (e.g., exact match for demonstration)
def metric_em(pred, gold, **kwargs):
    """Simple exact match metric."""
    print(f"metrics gold: {gold} -- prediction: {pred}")
    return pred.answer.strip().lower() == gold.answer.strip().lower()

evaluate_student = Evaluate(devset=evaluate_dataset, metric=metric_em, num_threads=1, display_progress=True, display_table=0)
initial_score = evaluate_student(StudentQA())
print(f"Initial Student LM evaluation score: {initial_score}")


# Convert a subset of the dataset to DSPy Example format
def create_dspy_example(hotpotqa_item):
    question = hotpotqa_item['question']
    answer = hotpotqa_item['answer']

    # Combine sentences from supporting_facts into a single context string
    # For HotPotQA, the 'context' field has 'title' and 'sentences'.
    # We want to flatten the sentences into a list of strings.
    passages = []
    for title, sentences in zip(hotpotqa_item['context']['title'], hotpotqa_item['context']['sentences']):
        # Each sentence is actually a list [title, sentence_text]
        # We want just the sentence_text
        for s_title, s_text in sentences:
            passages.append(s_text)

    # Convert passages to dspy.Passage objects if you want to use them with DSPy's RM directly
    # For now, let's just keep them as strings or simple dicts if passing directly to modules
    # If your DSPy program *retrieves* context, then you just need question and answer for demonstration.
    # If your DSPy program *consumes* pre-provided context, then you'll pass it.

    # Option 1: For demonstration/training if your program *retrieves* context
    # This is for training the *LM* part of your DSPy program
    return dspy.Example(question=question, answer=answer).with_inputs('question')

    # Option 2: If your program expects pre-provided context (e.g., you're not using DSPy's RM)
    # This is more for direct evaluation or if you're building a simpler QA without explicit retrieval.
    # return dspy.Example(question=question, passages=passages, answer=answer).with_inputs('question', 'passages')

# Train Dataset with examples 
# For distillation, you primarily need *unlabeled* data for the teacher
# to generate demonstrations. A small *labeled* devset is good for evaluation.
def load_hotpotqa():
    hotpotqa_dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
    train_set = hotpotqa_dataset['train']
    validation_set = hotpotqa_dataset['validation']
    # test_set = hotpotqa_dataset['test'] # Note: 'test' answers are often hidden for leaderboard submissions
    num_examples = 5 
    dspy_trainset = [create_dspy_example(item) for item in train_set.select(range(num_examples))]
    dspy_valset = [create_dspy_example(item) for item in validation_set.select(range(num_examples // 2))]

    print(f"Created {len(dspy_trainset)} DSPy training examples.")
    print(f"Created {len(dspy_valset)} DSPy validation examples.")
    print("\nSample DSPy Example:")
    print(dspy_trainset[0])

# Example: Create a synthetic unlabeled dataset (replace with your real data)
unlabeled_trainset = [
    dspy.Example(question="What is the capital of France?").with_inputs("question"),
    dspy.Example(question="Who wrote 'To Kill a Mockingbird'?").with_inputs("question"),
    dspy.Example(question="Explain the concept of photosynthesis.").with_inputs("question"),
    dspy.Example(question="What is the main function of the human heart?").with_inputs("question"),
    dspy.Example(question="Name two benefits of regular exercise.").with_inputs("question"),
    dspy.Example(question="What is a black hole in space?").with_inputs("question"),
    dspy.Example(question="Define artificial intelligence.").with_inputs("question"),
    dspy.Example(question="Who was the first person to walk on the moon?").with_inputs("question"),
    dspy.Example(question="What is the largest ocean on Earth?").with_inputs("question"),
    dspy.Example(question="Describe the water cycle.").with_inputs("question"),
]
# For a real application, you'd load your data, e.g.:
# data_loader = DataLoader()
# trainset_raw = data_loader.from_json("your_unlabeled_data.json")
# unlabeled_trainset = [dspy.Example(question=item['question']) for item in trainset_raw]

# Load the Banking77 dataset.
# CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
# kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)


#  Distillation (Bootstrapping and Fine-tuning) ---

# Initialize the BootstrapFinetune optimizer
# This optimizer will use the teacher_lm to create demonstrations
# and then fine-tune the student_lm on these demonstrations.
# `trainset`: the unlabeled data for the teacher to generate examples.
# `student`: the DSPy program instance that will be fine-tuned (using student_lm).
# `teacher`: the DSPy program instance that will act as the teacher (using teacher_lm).
# `config`: fine-tuning parameters. These are passed directly to the `fine-tune` method of the student LM.

# Switch back to teacher for bootstrapping
dspy.configure(lm=teacher_lm)
print("\n--- Starting Bootstrap Finetune (Distillation) ---")
# The metric here guides the teacher during bootstrapping (optional, but good for quality control)
# You might want a more sophisticated metric depending on your task.
# For simplicity, we'll use a basic success metric for the teacher's self-evaluation.
# For BootstrapFinetune, the `metric` primarily influences which teacher-generated examples are kept.
# If you don't have a specific metric to guide the teacher's self-correction, `None` is acceptable.
# However, for true quality control, consider a simpler metric that aligns with your desired output.

# For this example, let's keep it simple without a complex metric for the teacher's self-correction
# as the primary goal is distillation of patterns, not necessarily maximizing a specific metric on the *unlabeled* data.
# The fine-tuning process itself will rely on the generated examples.

# Fine-tuning configuration for the student model
# These are typical small model fine-tuning parameters. Adjust as needed.
finetune_config = dict(
    #num_train_epochs=3,  # Number of training epochs
    #per_device_train_batch_size=4, # Batch size for training
    gradient_accumulation_steps=2, # Accumulate gradients over multiple steps
    learning_rate=5e-5, # Learning rate
    fp16=True, # Use mixed precision for faster training and less memory (if GPU supports)
    # If using bf16 (bfloat16) on supported hardware, use: bf16=True,
    logging_steps=10, # Log training progress every N steps
    save_strategy="epoch", # Save checkpoint after each epoch
    output_dir="./fine_tuned_model", # Directory to save the fine-tuned model
    # You might want to add more specific parameters depending on the HuggingFace model
    # and the `trl` library's `SFTTrainer` (which DSPy uses internally).
)

dspy.settings.experimental = True
teleprompter = BootstrapFinetune(
    metric=None,  # Or define a metric for the teacher to self-correct during bootstrapping
    train_kwargs=finetune_config
)

# Compile the student program using the teacher and unlabeled data
# This step will:
# 1. Use the teacher LM to generate demonstrations on `unlabeled_trainset`.
# 2. Fine-tune the student_lm using these generated demonstrations.
compiled_student_program = teleprompter.compile(
    student=StudentQA(),
    trainset=unlabeled_trainset,
    teacher=TeacherQA() # The teacher program structure should match the student's
)

print("\n--- Distillation and Fine-tuning Complete! ---")
print(f"Fine-tuned model saved to: {finetune_config['output_dir']}")

# --- 6. Evaluate the Fine-tuned Student LM ---

print("\n--- Evaluating Student LM AFTER Fine-tuning ---")
# Now, re-configure DSPy to use the *fine-tuned* student LM
# Note: compiled_student_program already holds the fine-tuned student_lm
# but if you need to load it explicitly from the saved checkpoint:
# fine_tuned_student_lm = dspy.HFModel(checkpoint=finetune_config['output_dir'], model=student_lm_name)
# dspy.configure(lm=fine_tuned_student_lm) # Then configure DSPy with the loaded model

# Evaluate the compiled program, which now uses the fine-tuned student LM
fine_tuned_score = evaluate_student(compiled_student_program)
print(f"Fine-tuned Student LM score: {fine_tuned_score}")

# --- 7. Example Usage of the Fine-tuned Model ---
print("\n--- Example Usage of Fine-tuned Student LM ---")
prediction = compiled_student_program(question="What is the currency of the United States?")
print(f"Question: What is the currency of the United States?")
print(f"Predicted Answer by Fine-tuned Model: {prediction.answer}")

prediction_2 = compiled_student_program(question="Name the largest planet in our solar system.")
print(f"Question: Name the largest planet in our solar system.")
print(f"Predicted Answer by Fine-tuned Model: {prediction_2.answer}")

# --- Optional: Save and Load the fine-tuned model manually (if needed outside DSPy) ---
# The BootstrapFinetune already saves it, but for explicit saving:
# compiled_student_program.lm.save_pretrained("./my_distilled_model")

# To load it later:
# loaded_lm = dspy.HFModel(checkpoint="./my_distilled_model", model=student_lm_name)
# dspy.configure(lm=loaded_lm)
# loaded_program = BasicQA() # Re-instantiate your program
# loaded_program.load("my_distilled_model_dspy_state.json") # If you saved DSPy program state

