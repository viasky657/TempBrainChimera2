# TempBrainChimera2

# Fine-tuning Phi-4-multimodal on CinePile Dataset

This repository contains scripts for fine-tuning the Microsoft Phi-4-multimodal-instruct model on the CinePile dataset, a long-form video question answering dataset.

## CinePile Dataset

CinePile is a question-answering-based, long-form video understanding dataset created using advanced large language models (LLMs) with a human-in-the-loop pipeline. It consists of approximately 300,000 training data points and 5,000 test data points.

Each row in the dataset includes:
- A question about a movie scene
- Five multiple-choice options
- The correct answer key
- Movie metadata (name, year, genre)
- YouTube clip title and link
- Movie scene description with subtitles and visual descriptions

The test split includes additional annotations:
- `hard_split`: Indicates particularly challenging questions
- `visual_reliance`: Indicates whether a question requires visual information

## Requirements

```
pip install transformers datasets torch numpy pillow opencv-python pytube tqdm
```

## Scripts

### 1. Fine-tuning Script: `finetune_phi4_on_cinepile.py`

This script fine-tunes the Phi-4-multimodal model on the CinePile dataset.

```bash
python finetune_phi4_on_cinepile.py \
    --output_dir cinepile_model \
    --num_frames 16 \
    --max_videos 100 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --use_hard_split \
    --visual_reliance
```

#### Arguments:

- `--output_dir`: Path to save the processed dataset and model
- `--num_frames`: Number of frames to sample per video
- `--max_videos`: Maximum number of videos to process (for development/testing)
- `--use_hard_split`: Use only the hard split from the test dataset
- `--visual_reliance`: Use only questions that require visual information
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for training
- `--num_train_epochs`: Number of training epochs
- `--use_hico`: Use Hierarchical Token Compression for long videos

### 2. Inference Script: `inference_phi4_on_cinepile.py`

This script evaluates the fine-tuned model on the CinePile dataset.

```bash
python inference_phi4_on_cinepile.py \
    --model_path cinepile_model \
    --dataset_path cinepile_processed \
    --output_file cinepile_results.json \
    --max_videos 100
```

#### Arguments:

- `--model_path`: Path to the fine-tuned model
- `--dataset_path`: Path to the processed dataset
- `--output_file`: Path to save the results
- `--num_frames`: Number of frames to sample per video
- `--max_videos`: Maximum number of videos to process
- `--use_hard_split`: Use only the hard split from the test dataset
- `--visual_reliance`: Use only questions that require visual information
- `--use_hico`: Use Hierarchical Token Compression for long videos
- `--use_raw_dataset`: Use raw CinePile dataset instead of processed dataset

## Workflow

1. **Data Preparation**: The fine-tuning script downloads videos from YouTube links in the CinePile dataset, extracts frames, and prepares the data for training.

2. **Fine-tuning**: The model is fine-tuned on the prepared data, learning to answer multiple-choice questions about video content.

3. **Evaluation**: The inference script evaluates the fine-tuned model on the test set, calculating accuracy and saving detailed results.

## Using the Hard Split and Visual Reliance Filters

The CinePile dataset includes annotations for particularly challenging questions (`hard_split`) and questions that require visual information (`visual_reliance`). You can use these filters to focus training and evaluation on specific subsets:

- For challenging questions: `--use_hard_split`
- For visually-dependent questions: `--visual_reliance`

## Hierarchical Token Compression (HiCo)

For long videos, you can enable Hierarchical Token Compression with the `--use_hico` flag. This uses adaptive temporal sampling and token merging to efficiently process longer videos.

## Example Usage

### Fine-tuning on visually-dependent questions:

```bash
python finetune_phi4_on_cinepile.py \
    --output_dir cinepile_visual_model \
    --num_frames 16 \
    --max_videos 500 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --visual_reliance
```

### Evaluating on challenging questions:

```bash
python inference_phi4_on_cinepile.py \
    --model_path cinepile_visual_model \
    --output_file cinepile_hard_results.json \
    --use_raw_dataset \
    --use_hard_split \
    --max_videos 100
```

## Notes

- The scripts include a `max_videos` parameter to limit the number of videos processed, which is useful for development and testing.
- For production use, you may want to remove this limit or set it to a higher value.
- Processing videos requires significant computational resources, especially for fine-tuning.
- GPU acceleration is highly recommended for both fine-tuning and inference.



# COCONUT Model Training with CinePile Dataset

This repository contains scripts for fine-tuning the COCONUT class model with Phi-4-multimodal on the CinePile dataset, a long-form video question answering dataset.

## Overview

The COCONUT (COntinuous COgNition Using Transformers) model is enhanced with the Phi-4-multimodal architecture to process video data. The implementation includes:

1. **Continual Propagation**: A mechanism to avoid catastrophic forgetting when training on sequential datasets
2. **Binary Patching**: Dynamic patching of continuous latent states for efficient processing
3. **Hierarchical Token Compression (HiCo)**: Efficient processing of long videos

## Requirements

```bash
pip install transformers datasets torch numpy pillow opencv-python pytube tqdm
```

## Files

- `Phi4CoconutModel.py`: Implementation of the COCONUT model with Phi-4-multimodal
- `finetune_coconut_on_cinepile.py`: Script for fine-tuning the model on CinePile
- `inference_coconut_on_cinepile.py`: Script for evaluating the fine-tuned model

## Fine-tuning

To fine-tune the COCONUT model on the CinePile dataset:

```bash
python finetune_coconut_on_cinepile.py \
    --output_dir coconut_cinepile_model \
    --num_frames 16 \
    --max_videos 100 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --use_hard_split \
    --visual_reliance \
    --use_hico \
    --apply_freq 100 \
    --replacement_rate 1e-5 \
    --maturity_threshold 1000
```

### Arguments:

- `--output_dir`: Path to save the fine-tuned model
- `--num_frames`: Number of frames to sample per video
- `--max_videos`: Maximum number of videos to process (for development/testing)
- `--use_hard_split`: Use only the hard split from the test dataset
- `--visual_reliance`: Use only questions that require visual information
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for training
- `--num_train_epochs`: Number of training epochs
- `--use_hico`: Use Hierarchical Token Compression for long videos
- `--apply_freq`: Frequency (in steps) to apply continual propagation
- `--replacement_rate`: Fraction of mature units to replace during continual propagation
- `--maturity_threshold`: Number of updates before a unit is considered mature and eligible for replacement

## Inference

To evaluate the fine-tuned model on the CinePile dataset:

```bash
python inference_coconut_on_cinepile.py \
    --model_path coconut_cinepile_model \
    --output_file coconut_cinepile_results.json \
    --num_frames 16 \
    --max_videos 100 \
    --use_hard_split \
    --visual_reliance \
    --use_hico
```

### Arguments:

- `--model_path`: Path to the fine-tuned model
- `--output_file`: Path to save the evaluation results
- `--num_frames`: Number of frames to sample per video
- `--max_videos`: Maximum number of videos to process
- `--use_hard_split`: Use only the hard split from the test dataset
- `--visual_reliance`: Use only questions that require visual information
- `--use_hico`: Use Hierarchical Token Compression for long videos
- `--batch_size`: Batch size for inference

## Continual Propagation

The continual propagation mechanism helps maintain plasticity during training by selectively reinitializing low-utility units. This prevents the model from becoming overly specialized to the first dataset it encounters, allowing it to learn from new data without forgetting what it learned from previous datasets.

### How it works:

1. **Utility Tracking**: The model tracks the utility of each unit based on its contribution to the network's output.
2. **Maturity Threshold**: Units must be updated a certain number of times before they are considered mature and eligible for replacement.
3. **Selective Reinitialization**: A small fraction of mature units with the lowest utility are reinitialized periodically during training.

### Parameters:

- `replacement_rate`: Fraction of mature units to replace (typically very small, e.g., 1e-5)
- `maturity_threshold`: Number of updates before a unit is considered mature (e.g., 1000)
- `apply_freq`: Frequency (in steps) to apply continual propagation (e.g., 100)

## Hierarchical Token Compression (HiCo)

For long videos, the Hierarchical Token Compression (HiCo) mechanism efficiently processes the video by:

1. Adaptively sampling frames based on content importance
2. Compressing redundant information
3. Maintaining a hierarchical representation of the video

Enable HiCo with the `--use_hico` flag during fine-tuning and inference.

## Example Workflow

1. **Fine-tune the model**:
   ```bash
   python finetune_coconut_on_cinepile.py --output_dir coconut_model --max_videos 100 --use_hico
   ```

2. **Evaluate the model**:
   ```bash
   python inference_coconut_on_cinepile.py --model_path coconut_model --output_file results.json --max_videos 100 --use_hico
   ```

3. **Analyze the results**:
   ```python
   import json
   with open('results.json', 'r') as f:
       results = json.load(f)
   print(f"Accuracy: {results['accuracy']:.4f}")
   ```

## Notes

- The scripts include a `max_videos` parameter to limit the number of videos processed, which is useful for development and testing.
- For production use, you may want to remove this limit or set it to a higher value.
- Processing videos requires significant computational resources, especially for fine-tuning.
- GPU acceleration is highly recommended for both fine-tuning and inference.

## Continual_propagation_explained

Here's how it works with your sequential datasets:

Maintaining Plasticity: The continual propagation mechanism prevents the model from becoming overly specialized to the first dataset by selectively reinitializing a tiny fraction of less-used units during training. This maintains the model's ability to learn from new data (datasets 2 and 3) without completely forgetting what it learned from the first dataset.

Balance Between Old and New: Only the least useful units (based on their contribution to the network's output) are reinitialized, while the most important units that encode critical information from previous datasets are preserved.

Training Approach: You would train on your datasets sequentially, using the train_with_continual_propagation method for each dataset. The model will gradually adapt to new information while retaining important knowledge from previous datasets.

For your specific case with 3 large datasets, I recommend:

Start with the default hyperparameters: replacement_rate=1e-5 and maturity_threshold=1000
Monitor performance on validation sets from all previous datasets during training
If you notice the model forgetting too much from earlier datasets, decrease the replacement rate
If it's not learning new information well enough, slightly increase the replacement rate
The implementation includes:

Parameters to control the continual propagation process
Utility tracking for each unit based on its contribution
Selective reinitialization of low-utility units
Training integration with activation tracking
Example code showing how to use the feature
This approach should allow your model to effectively learn from all three datasets while maintaining a balance between retaining old information and acquiring new knowledge.

## TOVA Compression Improvements

The Token Omission Via Attention (TOVA) compression mechanism has been enhanced with several optimizations to improve processing speed and memory efficiency in the Phi4COCONUTModel:

### 1. Dynamic Cache Sizing

TOVA compression now supports dynamic cache sizing, which automatically adjusts the KV cache size based on sequence length and memory constraints:

- **Adaptive Memory Usage**: Cache size scales with input sequence length, using a configurable proportion
- **Bounded Adjustments**: Minimum and maximum cache sizes prevent extreme values
- **Efficiency Gains**: Shorter sequences use smaller caches, while longer sequences maintain sufficient context

### 2. Parallel Compression

Compression operations now leverage parallel processing for improved performance:

- **Multi-threaded Execution**: Uses ThreadPoolExecutor to parallelize key and value tensor compression
- **Automatic Optimization**: Only activates for large caches where parallelization benefits outweigh overhead
- **Performance Monitoring**: Tracks speedup metrics to quantify benefits

### 3. Optimized Entropy Calculation

The entropy-based token importance calculation has been optimized for better performance:

- **Vectorized Operations**: Replaces loops with efficient tensor operations
- **Numerical Stability**: Improved normalization with better handling of edge cases
- **Memory Efficiency**: In-place operations reduce memory overhead

### 4. Custom Importance Scoring

A new `compress_with_importance` method allows the model to use custom token importance scores:

- **Multi-signal Integration**: Combines attention patterns, hidden state features, and position-based importance
- **Adaptive Learning**: Importance scores evolve based on historical compression decisions
- **Recency Bias**: Preserves recent tokens that are often more relevant for next-token prediction

### 5. Length Penalty for Reinforcement Learning

A new length penalty mechanism has been added to encourage efficient token usage during reinforcement learning:

- **Threshold-Based Penalty**: Only applies after a specified token count (default: 1000 tokens)
- **Configurable Penalty Functions**: Supports linear, quadratic, or exponential penalties
- **Accuracy Preservation**: Can preserve accuracy components of rewards while penalizing verbosity
- **RL Integration**: Seamlessly integrates with reinforcement learning workflows

The length penalty can be applied during generation or training:

```python
# Apply during text generation
model.generate_text(
    prompt="Summarize this article:",
    apply_length_penalty=True,
    length_penalty_threshold=1000,
    length_penalty_factor=0.001
)

# Apply during reinforcement learning
modified_reward = model.apply_length_penalty_to_reward(
    reward=original_reward,
    sequence_length=sequence_length,
    accuracy_component=accuracy_reward,
    threshold=1000,
    penalty_factor=0.001,
    penalty_type="quadratic"
)
```

### Benefits for Phi4COCONUTModel

These improvements provide several benefits for the Phi4COCONUTModel:

1. **Faster Processing**: Parallel compression and optimized calculations reduce latency
2. **Reduced Memory Usage**: Dynamic sizing prevents over-allocation of memory
3. **Better Context Preservation**: Improved token selection maintains the most relevant context
4. **Adaptive Behavior**: The system learns which tokens are most important based on actual usage patterns
5. **Resource Efficiency**: Length penalties encourage concise outputs while maintaining quality

### Usage

The enhanced TOVA compression is automatically used by the Phi4COCONUTModel when `use_tova=True`. The model has been updated to use all new features with optimized default parameters:

```python
# Enhanced TOVA compressor with adaptive learning and new features
self.tova_compressor = TOVACompression(
    cache_max_size=self.tova_cache_size,
    layer_based=self.tova_layer_wise,
    head_weight_strategy="weighted",  # Use weighted strategy for adaptive head importance
    num_heads=num_heads,
    learning_rate=0.01,
    weight_momentum=0.9,
    entropy_weight=0.3,
    dynamic_sizing=True,  # Enable dynamic cache sizing
    min_cache_size=128,   # Minimum cache size
    max_cache_size=2048,  # Maximum cache size
    cache_size_factor=0.5, # Factor to determine cache size relative to sequence length
    use_parallel=True,     # Enable parallel compression
    num_workers=4,         # Number of worker threads for parallel processing
    optimize_entropy=True  # Enable optimized entropy calculation
)
```

These improvements make TOVA compression more efficient and effective, particularly for long sequences and multimodal inputs where KV cache management is critical for performance.

Usage Example
# During reinforcement learning
modified_reward = model.apply_length_penalty_to_reward(
    reward=original_reward,
    sequence_length=token_count,
    accuracy_component=accuracy_reward,  # This component is preserved
    threshold=1000,
    penalty_factor=0.001
)

# During generation
response = model.generate_text(
    prompt="Explain quantum computing:",
    apply_length_penalty=True,
    length_penalty_threshold=1000,
    length_penalty_factor=0.001
)
This implementation complements TOVA compression - while TOVA efficiently manages which tokens to keep in memory, the length penalty encourages the model to be concise in its outputs. The threshold ensures the model still produces robust and grammatically correct language for reasonable-length outputs, only applying pressure when sequences become excessively long.

# How the whole architecture of the DLLM (Diffusion Language Model) and the COCONUT architecture (LLM) work together to generate a lot of tokens quickly and produce a longer thought sequence than traditional models. 

The user is asking specifically if the diffusion speeds up the latent space generation. Let me look more closely at the code to determine if and how the diffusion process accelerates the latent space generation.

Looking at the code, I need to understand:

How the diffusion process interacts with latent space generation
Whether there's explicit speed improvement from using diffusion
What the code says about performance differences
Let me examine the relevant sections of the code again.

Yes, the diffusion process does speed up the latent space generation. This is one of the key benefits of the diffusion approach in this model. Here's how it works:

Parallel Generation: The diffusion-based approach enables parallel text generation rather than sequential token-by-token generation. This is evident in the generate_parallel method (lines 393-447) which generates multiple tokens simultaneously.

ODE Solver Acceleration: The model specifically implements an ODE (Ordinary Differential Equation) solver method for even faster generation in the generate_with_ode method (lines 449-510). The code comment on line 452 explicitly states: "Generate text using an ODE solver for the diffusion process. This is more efficient than the step-by-step approach."

Direct Performance Comparison: When generating text, the model prioritizes the diffusion-based approach. In the generate_text method (lines 1201-1270), it sets use_ode=True with the comment "Use ODE solver for faster generation" (line 1244).

Error Handling for Non-Diffusion: The model even raises an error if diffusion-based generation fails, with the message: "This model only supports diffusion-based generation" (line 1134), suggesting that diffusion is critical for performance.

Implementation Details: The diffusion process converts discrete tokens to continuous embeddings and refines them through a noise-reduction process. This allows the model to work in a continuous space where it can generate multiple tokens in parallel, rather than having to generate one token at a time as in traditional autoregressive models.

The diffusion approach is particularly beneficial for latent space generation because it allows the model to generate the entire latent space content in parallel rather than sequentially, significantly reducing the time needed to process each latent space.


Yes, the diffusion process significantly speeds up the latent space generation in the Phi4COCONUT model. This acceleration happens in several key ways:

Parallel Token Generation: Traditional autoregressive models generate tokens one at a time. The diffusion approach enables parallel generation of multiple tokens simultaneously, as implemented in the generate_parallel method (lines 393-447).

ODE Solver Optimization: The model implements an even faster generation method using an Ordinary Differential Equation (ODE) solver in the generate_with_ode method. The code explicitly states this is "more efficient than the step-by-step approach" (line 452).

Continuous vs. Discrete Space: By operating in a continuous embedding space rather than discrete token space, the diffusion process can make larger "jumps" in the generation process, effectively denoising multiple tokens at once.

Performance Priority: The model is designed to prioritize diffusion-based generation, with the code even raising an error if non-diffusion generation is attempted: "This model only supports diffusion-based generation" (line 1134).

This acceleration is particularly valuable when processing multiple latent spaces, as the model needs to generate content for each latent space. Without diffusion, the model would need to generate each token sequentially within each latent space, which would be significantly slower for complex reasoning tasks that require multiple latent thinking steps.

# The Model does require autoregressive tokens at the beginning of generation to work 
Yes, the Phi4COCONUT model absolutely requires autoregressive functions to work, despite the error message suggesting it "only supports diffusion-based generation."

The model uses a hybrid architecture where:

Initial Processing is Autoregressive: The forward_continuous_thinking method (lines 736-1008) uses the base Phi-4 model's autoregressive functions to process inputs and latent tokens. This is essential for the model to work.

Latent Token Processing is Autoregressive: When processing each latent token, the model:

Uses autoregressive processing to generate hidden states
Replaces latent token embeddings with these autoregressive hidden states
Continues autoregressive processing to the next latent token
Two-Stage Pipeline: The model operates as a pipeline:

First stage: Autoregressive processing to handle inputs and latent tokens
Second stage: Diffusion-based generation for the final output
Error Message Context: The error message on line 1134 is specifically about the final generation phase. It means the model doesn't support pure autoregressive generation for the final output - it requires diffusion for that stage. However, the diffusion process itself relies on embeddings that were produced by the autoregressive component.

The model is fundamentally a hybrid that requires both components to function properly. The autoregressive component is essential for processing the input and latent tokens, while the diffusion component accelerates the final generation phase. Neither can work without the other in this architecture

#Deep Sleep (Synchorization Between Episodic Memory (higher executive functions) and base mind (lower consciousness functions) to prevent Sleep Paralysis Like state in LLM by having the higher executive function wake up first before the rest of its mind)

enhanced the deep sleep and awakening methods in the COCONUTWLatentThinking.py file to improve synchronization between the phi4 model and episodic memory module. The improvements prevent sleep paralysis by ensuring both components transition between states in a coordinated manner.

Key enhancements to the enter_deep_sleep method:

Added tracking of episodic memory sleep state alongside phi4 model state
Implemented controlled rate transitions (episodic memory at 80% of phi4's rate)
Added synchronization level monitoring and verification
Added safety checks before full shutdown
Implemented forced synchronization if components get out of sync
Key enhancements to the awaken method:

Created a 3-stage approach for emergency awakening
Made episodic memory wake up slightly ahead of phi4 (110% rate) to prevent sleep paralysis
Added sleep paralysis risk detection with automatic corrective actions
Added penalties for sleep paralysis risk in the reward calculation
Added final synchronization verification before completing awakening
These changes ensure the full model can safely sleep and wake up without experiencing sleep paralysis, which occurs when consciousness wakes up before the rest of the mind.

#CinePile Dataset Reinforcement Learning - How it works

The RL component for the CinePile dataset training in the COCONUTWLatentThinking.py file. The implementation:

Imports necessary functions from cinepile_RL.py including normalize_string, evaluate_semantic_similarity, eval_response, ans_key_map, format_question_and_options, print_qa, and get_prompt.

Enhances the train_cinepile_with_rl_rewards method to:

Accept video embeddings from the CinePile dataset
Use normalize_string from cinepile_RL.py to extract predicted answers
Apply a +1.00 reward when answers are correct
Apply a neutral (0) reward when answers are incorrect
Incorporate these rewards into the loss function during training
Uses the proper answer evaluation logic from cinepile_RL.py to determine correctness.

The implementation now properly rewards the model during training on the CinePile dataset, which should help improve its performance on video-based question answering tasks.
