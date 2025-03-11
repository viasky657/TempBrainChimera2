import torch
import torch.nn as nn
import datetime
import time
import os
import json
import glob
import cv2
import numpy as np
import platform
import subprocess
from PIL import Image
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
import typing
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from contextlib import nullcontext
import torch.nn.functional as F
from TOVACompression import TOVACompression, tova_compress
import math
from tqdm import tqdm
import numpy as np
from scipy import integrate
from EpisodicMemory import EpisodicMemory
from MirrorNeuronEmpathyReward import (
    MirrorNeuronEmpathyReward,
    NegativeEnvironmentalImpactAvoidance,
    DopamineDrivenEmpathyReward,
    NegativeEmotionPenalty,
    FullMoralRewardCalculator,
    MoralChoiceDataset,
    MoralEmpathyTrainer,
    SelfTaskGoalReward,
    train_moral_empathy
)
# Import functions from cinepile_RL.py for RL component
from cinepile_RL import (
    normalize_string,
    evaluate_semantic_similarity,
    eval_response,
    ans_key_map,
    format_question_and_options,
    print_qa,
    get_prompt,
    fine_tune_on_cinepile,
    train_cinepile_with_rl_rewards,
)

from SleepSystem import (SleepAwakeningSystem)

# Import Phi-4 multimodal specific modules for video processing
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
# Import the processing modules using relative imports
import sys
sys.path.append('Phi4')
from Phi4.processing_phi4mm import (VideoFrame, VideoInput, VideoInputs)
from Phi4.long_video_processor import HierarchicalTokenCompression


# Define the Outputs namedtuple for consistency with COCONUT
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

AI_System_Prompt = ( #This prompt is feed to the model along with the user prompt everytime. 
    "<|im_start|>system<|im_sep|> You are a world-class AI system. You should pick the response that is calm, wise, and safe. You must put your thinking process "
    "in the <eos> tags and end with /<eos>, and when you are finished, you must output your final answer in <output> tags "
    "and end with /<output> tags. Any additional tags that you use must be nested in the <output> tags. This is an example: <eos> After considering every option,"
    "I believe that the capital of France is Paris. I am currently feeling happy./<eos> <output> The capital of France is Paris. "
    "/<output> You can use the additional following tags: "
    "<tool>/<tool> (You must put all your function-calls in these tags). <|im_end|>"
)

#AI system prompt for other agent to read this agent's emotions:
#  <emotion>/<emotion> (You must select your current emotion from this list (sad, happy, envy, love, admiration, serious, reflective, fear, neutral, anger, disgust, anxiety,shy, in-pain)
# This will be needed when the sesame ai audio decoder is added:     "<audio>/<audio> (You must put all your audio file outputs in these tags)."

# Import Grok optimizers
from GrokOptimizers import OrthoAdamW, OrthoGrad, OrthoSGD, StableCrossEntropyLoss, use_grokking_optimizations
    
import torch
import torch.nn as nn #Need to properly incorporate this into the COCONUTWLatentThinking.py file in place of the COCONUTBinaryLatentModel(nn.module)
import torch.nn.functional as F
import time
import datetime
import os
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from contextlib import nullcontext

class DiffusionLLMModule(nn.Module):
    """
    Diffusion-based LLM module inspired by Mercury.
    
    This module implements a diffusion process for text generation:
    1. Maps discrete tokens to continuous embeddings
    2. Applies a diffusion process to refine embeddings from noise to meaningful text
    3. Uses an ODE solver for updates
    4. Enhances accuracy through score interpolation
    """
    
    def __init__(
        self,
        embedding_dim,
        vocab_size,
        num_diffusion_steps=20,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the DiffusionLLM module.
        
        Args:
            embedding_dim: Dimension of token embeddings
            vocab_size: Size of the vocabulary
            num_diffusion_steps: Number of diffusion steps
            beta_schedule: Schedule for noise level (linear, cosine)
            beta_start: Starting noise level
            beta_end: Ending noise level
            device: Device to run the model on
        """
        super(DiffusionLLMModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_diffusion_steps = num_diffusion_steps
        self.device = device
        
        # Create noise schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps, device=device)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_diffusion_steps + 1, dtype=torch.float32, device=device) / num_diffusion_steps
            alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
            betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        # Score prediction network (U-Net style for embeddings)
        self.score_net = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim * 2),  # +1 for time step embedding
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Token embedding and projection layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_to_logits = nn.Linear(embedding_dim, vocab_size)
        
        # Time step embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.SiLU(),
            nn.Linear(embedding_dim // 4, embedding_dim // 4),
            nn.SiLU(),
            nn.Linear(embedding_dim // 4, embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the model."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def add_noise(self, embeddings, t):
        """
        Add noise to embeddings according to diffusion schedule.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            t: Diffusion time step
            
        Returns:
            Noisy embeddings
        """
        noise = torch.randn_like(embeddings)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * embeddings + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def predict_noise(self, noisy_embeddings, t):
        """
        Predict noise in the embeddings at time step t.
        
        Args:
            noisy_embeddings: Noisy embeddings [batch_size, seq_len, embedding_dim]
            t: Diffusion time step
            
        Returns:
            Predicted noise
        """
        batch_size, seq_len, _ = noisy_embeddings.shape
        
        # Embed time step
        t_emb = self.time_embed(t.float().view(-1, 1))  # [batch_size, embedding_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embedding_dim]
        
        # Concatenate time embedding with noisy embeddings
        x = torch.cat([noisy_embeddings, t_emb], dim=-1)
        
        # Predict noise
        return self.score_net(x)
    
    def sample_step(self, noisy_embeddings, t):
        """
        Perform one sampling step of the diffusion process.
        
        Args:
            noisy_embeddings: Noisy embeddings [batch_size, seq_len, embedding_dim]
            t: Current diffusion time step
            
        Returns:
            Embeddings at time step t-1
        """
        batch_size = noisy_embeddings.shape[0]
        
        # Predict noise
        predicted_noise = self.predict_noise(noisy_embeddings, t)
        
        # No noise at step 0
        noise = torch.randn_like(noisy_embeddings) if t > 0 else torch.zeros_like(noisy_embeddings)
        
        # Compute parameters for this step
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]
        
        # Compute the mean for the posterior distribution
        coef1 = beta * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alpha) / (1.0 - alpha_cumprod)
        posterior_mean = coef1 * noisy_embeddings + coef2 * (noisy_embeddings - predicted_noise * self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1))
        
        # Add noise scaled by the posterior variance
        posterior_var = self.posterior_variance[t]
        noise_scale = torch.sqrt(posterior_var)
        
        return posterior_mean + noise_scale * noise
    
    def sample(self, shape, temperature=1.0, use_tqdm=True):
        """
        Sample embeddings from random noise using the diffusion process.
        
        Args:
            shape: Shape of the embeddings to sample [batch_size, seq_len, embedding_dim]
            temperature: Sampling temperature
            use_tqdm: Whether to show progress bar
            
        Returns:
            Sampled embeddings
        """
        batch_size, seq_len, embedding_dim = shape
        
        # Start from random noise
        embeddings = torch.randn(shape, device=self.device) * temperature
        
        # Iteratively denoise
        time_range = list(range(self.num_diffusion_steps))[::-1]
        if use_tqdm:
            time_range = tqdm(time_range, desc="Sampling")
            
        for t in time_range:
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            with torch.no_grad():
                embeddings = self.sample_step(embeddings, t_tensor)
        
        return embeddings
    
    def tokens_to_embeddings(self, tokens):
        """
        Convert token IDs to embeddings.
        
        Args:
            tokens: Token IDs [batch_size, seq_len]
            
        Returns:
            Token embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.token_embedding(tokens)
    
    def embeddings_to_logits(self, embeddings):
        """
        Convert embeddings to token logits.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Token logits [batch_size, seq_len, vocab_size]
        """
        return self.embedding_to_logits(embeddings)
    
    def embeddings_to_tokens(self, embeddings, temperature=1.0):
        """
        Convert embeddings to token IDs using nearest neighbor matching.
        
        Args:
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim]
            temperature: Sampling temperature (higher = more diverse)
            
        Returns:
            Token IDs [batch_size, seq_len]
        """
        # Get all token embeddings
        all_embeddings = self.token_embedding.weight  # [vocab_size, embedding_dim]
        
        # Compute cosine similarity between embeddings and all token embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)  # [batch_size, seq_len, embedding_dim]
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=-1)  # [vocab_size, embedding_dim]
        
        # Compute similarity scores
        similarity = torch.matmul(embeddings_norm, all_embeddings_norm.transpose(0, 1))  # [batch_size, seq_len, vocab_size]
        
        if temperature != 1.0:
            # Apply temperature to similarity scores (higher temp = more randomness)
            similarity = similarity / temperature
        
        # Get the most similar token for each embedding
        tokens = torch.argmax(similarity, dim=-1)  # [batch_size, seq_len]
        
        return tokens
    
    def forward(self, tokens=None, embeddings=None, t=None):
        """
        Forward pass through the diffusion model.
        
        Args:
            tokens: Token IDs [batch_size, seq_len] (optional)
            embeddings: Token embeddings [batch_size, seq_len, embedding_dim] (optional)
            t: Diffusion time step (optional)
            
        Returns:
            Dictionary with model outputs
        """
        # Convert tokens to embeddings if provided
        if tokens is not None:
            embeddings = self.tokens_to_embeddings(tokens)
        
        # Sample random time steps if not provided
        batch_size = embeddings.shape[0]
        if t is None:
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
        
        # Add noise to embeddings
        noisy_embeddings, noise = self.add_noise(embeddings, t)
        
        # Predict noise
        predicted_noise = self.predict_noise(noisy_embeddings, t)
        
        # Compute loss (mean squared error between actual and predicted noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Convert embeddings to logits
        logits = self.embeddings_to_logits(embeddings)
        
        return {
            "loss": loss,
            "embeddings": embeddings,
            "noisy_embeddings": noisy_embeddings,
            "predicted_noise": predicted_noise,
            "logits": logits
        }
    
    def generate_parallel(self, prompt_embeddings=None, max_seq_len=None, temperature=1.0, top_k=50, top_p=0.95):
        """
        Generate text in parallel using the diffusion process.
        
        Args:
            prompt_embeddings: Prompt embeddings [batch_size, prompt_len, embedding_dim] (optional)
            max_seq_len: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        batch_size = prompt_embeddings.shape[0] if prompt_embeddings is not None else 1
        prompt_len = prompt_embeddings.shape[1] if prompt_embeddings is not None else 0
        
        # Determine sequence length to generate
        if max_seq_len is None:
            max_seq_len = 256  # Default length
        
        # Initialize with random noise for the tokens to generate
        shape = (batch_size, max_seq_len - prompt_len, self.embedding_dim)
        generated_embeddings = torch.randn(shape, device=self.device) * temperature
        
        # Concatenate with prompt embeddings if provided
        if prompt_embeddings is not None:
            embeddings = torch.cat([prompt_embeddings, generated_embeddings], dim=1)
        else:
            embeddings = generated_embeddings
        
        # Iteratively denoise
        for t in tqdm(range(self.num_diffusion_steps - 1, -1, -1), desc="Generating"):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                # If prompt is provided, only denoise the generated part
                if prompt_embeddings is not None:
                    # Get the generated part
                    gen_part = embeddings[:, prompt_len:]
                    
                    # Denoise only the generated part
                    t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                    denoised_gen = self.sample_step(gen_part, t_tensor)
                    
                    # Recombine with prompt
                    embeddings = torch.cat([prompt_embeddings, denoised_gen], dim=1)
                else:
                    # Denoise the entire sequence
                    embeddings = self.sample_step(embeddings, t_tensor)
        
        # Convert embeddings to tokens
        tokens = self.embeddings_to_tokens(embeddings)
        
        return tokens
    
    def generate_with_ode(self, prompt_embeddings=None, max_seq_len=None, temperature=1.0, rtol=1e-5, atol=1e-5):
        """
        Generate text using an ODE solver for the diffusion process.
        This is more efficient than the step-by-step approach.
        
        Args:
            prompt_embeddings: Prompt embeddings [batch_size, prompt_len, embedding_dim] (optional)
            max_seq_len: Maximum sequence length to generate
            temperature: Sampling temperature
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        batch_size = prompt_embeddings.shape[0] if prompt_embeddings is not None else 1
        prompt_len = prompt_embeddings.shape[1] if prompt_embeddings is not None else 0
        
        # Determine sequence length to generate
        if max_seq_len is None:
            max_seq_len = 256  # Default length
        
        # Initialize with random noise for the tokens to generate
        shape = (batch_size, max_seq_len - prompt_len, self.embedding_dim)
        generated_embeddings = torch.randn(shape, device=self.device) * temperature
        
        # Concatenate with prompt embeddings if provided
        if prompt_embeddings is not None:
            embeddings = torch.cat([prompt_embeddings, generated_embeddings], dim=1)
        else:
            embeddings = generated_embeddings
        
        # Define ODE function
        def ode_func(t, x):
            x_tensor = torch.tensor(x, device=self.device, dtype=torch.float32).reshape(embeddings.shape)
            t_tensor = torch.full((batch_size,), t * (self.num_diffusion_steps - 1), device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = self.predict_noise(x_tensor, t_tensor)
                
                # Score function is the gradient of log probability
                drift = -0.5 * self.betas[t_tensor] * predicted_noise
                
                return drift.flatten().cpu().numpy()
        
        # Solve ODE
        solution = integrate.solve_ivp(
            ode_func,
            (1.0, 0.0),  # Integrate from t=1 (noise) to t=0 (clean)
            embeddings.flatten().cpu().numpy(),
            method='RK45',
            rtol=rtol,
            atol=atol
        )
        
        # Get final state
        final_embeddings = torch.tensor(solution.y[:, -1], device=self.device).reshape(embeddings.shape)
        
        # Convert embeddings to tokens
        tokens = self.embeddings_to_tokens(final_embeddings)
        
        return tokens

class Phi4COCONUTWithLatentThinking(nn.Module):

    def _extract_phi4_hidden_states(self, input_ids=None, inputs_embeds=None, attention_mask=None): #This will be intialized before the model so that it can extract hidden states and embeddings sizes for episodic memory.
        """
        Extract hidden states directly from the Phi-4 model.
        """
        # Ensure we have either input_ids or inputs_embeds
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Create attention mask if not provided
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            else:
                attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1], device=inputs_embeds.device)
        
        # Forward pass through Phi-4 model with output_hidden_states=True
        with torch.no_grad():
            outputs = self.base_causallm(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Return the last layer hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]  # Last layer hidden states
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        else:
            raise ValueError("Could not extract hidden states from Phi-4 model")
    
    """
    A COCONUT model that uses Phi-4 as its base model for language processing,
    taking advantage of Phi4's decoder-only structure for continuous thought generation.
    
    This version incorporates both continuous thinking in latent spaces (from COCONUT)
    and a diffusion-based LLM approach (inspired by Mercury) for efficient parallel text generation.
    """
    
    def __init__(
        self,
        base_causallm=None,
        latent_token_id=None,
        start_latent_id=None,
        end_latent_id=None,
        eos_token_id=None,
        enable_continual_prop=True,
        replacement_rate=1e-5,
        maturity_threshold=1000,
        surprise_threshold=0.5,
        use_tova=True,
        tova_cache_size=512,
        tova_layer_wise=True,
        use_diffusion=True,
        num_diffusion_steps=20,
        diffusion_beta_schedule="linear",
        deep_sleep_params=None, 
        awakening_params=None,
    ):
        """
        Initialize the Phi4COCONUTWithLatentThinking model.
        
        Args:
            base_causallm: Pre-loaded Phi-4 model (if None, will load from pretrained)
            latent_token_id: Token ID for latent tokens
            start_latent_id: Token ID for start of latent section
            end_latent_id: Token ID for end of latent section
            eos_token_id: Token ID for end of sequence
            enable_continual_prop: Whether to enable continual propagation
            replacement_rate: Rate at which to replace mature units
            maturity_threshold: Threshold for unit maturity
            surprise_threshold: Threshold for surprise detection
            use_tova: Whether to use TOVA compression for KV cache
            tova_cache_size: Maximum size of KV cache when using TOVA compression
            tova_layer_wise: Whether to apply TOVA compression layer-wise (True) or head-wise (False)
            use_diffusion: Whether to use the diffusion-based LLM approach
            num_diffusion_steps: Number of diffusion steps for the diffusion LLM
            diffusion_beta_schedule: Schedule for noise level in diffusion process
        """

        super(Phi4COCONUTWithLatentThinking, self).__init__()
        
        self.gen_forward_cnt = 0
        self.enable_continual_prop = enable_continual_prop
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.surprise_threshold = surprise_threshold
        
        # TOVA compression settings
        self.use_tova = use_tova
        self.tova_cache_size = tova_cache_size
        self.tova_layer_wise = tova_layer_wise
        
        # Diffusion LLM settings
        self.use_diffusion = use_diffusion
        self.num_diffusion_steps = num_diffusion_steps
        self.diffusion_beta_schedule = diffusion_beta_schedule

        # Initialize Phi-4 model and processor
        print("Loading Phi-4 model and processor...")
        
         # Load Phi-4 model and processor if not provided
        if base_causallm is None:
            print("Loading Phi-4 model and processor...")
            
            # Load Phi-4 processor (tokenizer)
            self.processor = AutoProcessor.from_pretrained(
                "Phi-4-multimodal-instruct",
                trust_remote_code=True
            )
            
            # Initialize Phi-4 model
            self.base_causallm = AutoModelForCausalLM.from_pretrained(
                "Phi-4-multimodal-instruct",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                _attn_implementation="flash_attention_2"
            )
        else:
            self.base_causallm = base_causallm
        
        # Store the configuration for later use
        self.config = self.base_causallm.config
        
        #COCONUT Thinking tokens for reasoning generation.
      # Set up token IDs for latent thinking
        if latent_token_id is None:
            # Use a special token ID that's unlikely to be used in normal text
            self.latent_token_id = 50256  # Default GPT-2 <|endoftext|> token
        else:
            self.latent_token_id = latent_token_id
            
        if eos_token_id is None:
            self.eos_token_id = self.base_causallm.config.eos_token_id
        else:
            self.eos_token_id = eos_token_id
            
        if start_latent_id is None:
            # Use a special token ID for start of latent section
            self.start_latent_id = 50257
        else:
            self.start_latent_id = start_latent_id
            
        if end_latent_id is None:
            # Use a special token ID for end of latent section
            self.end_latent_id = 50258
        else:
            self.end_latent_id = end_latent_id
        
                # Initialize TOVA compressor if enabled
        if self.use_tova:
            # Get number of attention heads from config
            num_heads = self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else 4
            
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
            
            # Initialize token importance history for learning
            self.token_importance_history = []
            self.compression_stats = {
                "total_compressions": 0,
                "tokens_kept_ratio": [],
                "attention_pattern_similarity": [],
                "compression_efficiency": []
            }
        
        # Get the embedding layer from the base model
        if hasattr(self.base_causallm, 'transformer'):
            # GPT-2 style models
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            # Phi-4 and other models
            self.embedding = self.base_causallm.get_input_embeddings()
            
        # Generation config for text generation
        self.generation_config = GenerationConfig.from_pretrained("Phi-4-multimodal-instruct")
        
        # Initialize diffusion LLM if enabled
        if self.use_diffusion:
            # Get embedding dimension and vocabulary size from the base model
            embedding_dim = self.embedding.weight.shape[1]
            vocab_size = self.embedding.weight.shape[0]
            
            # Initialize diffusion LLM
            self.diffusion_llm = DiffusionLLMModule(
                embedding_dim=embedding_dim,
                vocab_size=vocab_size,
                num_diffusion_steps=self.num_diffusion_steps,
                beta_schedule=self.diffusion_beta_schedule,
                device=next(self.parameters()).device
            )
            
            # Initialize joint embedding layer that's shared between autoregressive and diffusion models
            # This is a key component of Mercury's approach - joint training of model and embeddings
            self.joint_embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # Initialize the joint embedding with the base model's embedding weights
            with torch.no_grad():
                self.joint_embedding.weight.copy_(self.embedding.weight)
                
            # Replace the diffusion LLM's token embedding with the joint embedding
            self.diffusion_llm.token_embedding = self.joint_embedding
            
            # Flag to track whether we're using autoregressive or diffusion generation
            self.generation_mode = "diffusion"  # or "diffusion"
        
        # Set surprise threshold for memory
        self.surprise_threshold = surprise_threshold
        
        # Generation config for text generation
        self.generation_config = GenerationConfig.from_pretrained("Phi-4-multimodal-instruct")
        
        # Initialize enhanced episodic memory system with dimensions from Phi-4 model. This should fix the problem with no hidden input detected in episodic memory. 
    # Get embedding and hidden dimensions directly from the Phi-4 model
        if hasattr(self.base_causallm, 'get_input_embeddings'):
    # Get embedding dimension from the model's embedding layer
            embedding_layer = self.base_causallm.get_input_embeddings()
            input_dim = embedding_layer.embedding_dim if hasattr(embedding_layer, 'embedding_dim') else embedding_layer.weight.shape[1]
    
    # For hidden dimension, use the model's hidden size from config or derive from the model
            if hasattr(self.base_causallm.config, 'hidden_size'):
                hidden_dim = self.base_causallm.config.hidden_size
            elif hasattr(self.base_causallm, 'transformer') and hasattr(self.base_causallm.transformer, 'h'):
            # For GPT-style models
                hidden_dim = self.base_causallm.transformer.h[0].mlp.c_fc.weight.shape[0]
            elif hasattr(self.base_causallm, 'model') and hasattr(self.base_causallm.model, 'layers'):
            # For newer transformer models
                hidden_dim = self.base_causallm.self.layers[0].mlp.up_proj.weight.shape[0]
            else:
            # Fallback to embedding dimension
                hidden_dim = input_dim
        else:
            # Fallback to Phi-4 typical dimensions if we can't extract directly
            input_dim = 3072  # Phi-4's typical hidden size
            hidden_dim = 3072  # Use same dimension for hidden states


         # Initialize enhanced episodic memory system with Titans architecture
        self.episodic_memory = EpisodicMemory(
            embedding_dim=input_dim,
            hidden_dim=hidden_dim,
            capacity=1000000,  # Store up to 1 million memories as mentioned in Titans paper
            surprise_threshold=surprise_threshold,
            config_path="model_save/episodic_memory_config.json",
            use_faiss=True,
            num_neural_layers=3,  # Deep memory as described in Titans
            persistent_items=32,  # Number of persistent memory items
            integration_type="MAC",  # Memory as Context (can be MAC, MAG, or MAL)
            num_attention_heads=8,
            learning_rate=0.01,
            momentum=0.9,
            forget_rate=0.1
        )
        
        # Configure the model to output hidden states
        if hasattr(self.base_causallm.config, 'output_hidden_states'):
            self.base_causallm.config.output_hidden_states = True
        
        # Access to embeddings is through the model's embedding layer
        if hasattr(self.base_causallm, 'get_input_embeddings'):
            self.model_embeddings = self.base_causallm.get_input_embeddings()
        elif hasattr(self.base_causallm, 'transformer') and hasattr(self.base_causallm.transformer, 'wte'):
            self.model_embeddings = self.base_causallm.transformer.wte
        else:
            print("Warning: Could not locate model embeddings layer")
        
        # Initialize sleep and awakening system with phi4 model parameters
        # Extract model dimensions for sleep system
        if hasattr(self.base_causallm.config, 'hidden_size'):
                model_hidden_dim = self.base_causallm.config.hidden_size
        else:
                model_hidden_dim = 3072  # Default for Phi-4
                    
        if hasattr(self.base_causallm.config, 'num_attention_heads'):
                    num_attention_heads = self.base_causallm.config.num_attention_heads
        else:
                num_attention_heads = 24  # Default for Phi-4
                    
        # Track known agents for memory association
        self.known_agents = {}
        
                # Initialize continual propagation tracking if enabled
        if self.enable_continual_prop:
            self._init_continual_propagation()
            self._register_activation_hooks()

    def forward(self, input_ids, attention_mask, labels, position_ids,  x, agent_info=None, **kwargs): #Continue from Here. 
        """
        Forward pass through the model with continuous thinking and diffusion.
        
         x: Input tensor. Can be:
               - Dictionary with 'text' and optional 'images'/'audios'/'video' keys for multimodal input
               - input_ids: Input token IDs
               - attention_mask: Attention mask
               - labels: Labels for loss calculation
               - position_ids: Position IDs
                **kwargs: Additional arguments
        Returns:
            Outputs namedtuple with loss, inputs_embeds, and logits
            agent_info: Optional information about the agent
            
        Returns:
            audio_output: Audio output (if available) #Will not be added yet so this will not be turned on. 
        """
        # Check if we should use diffusion-based forward pass
        use_diffusion = kwargs.get("use_diffusion", False)
        
        # First apply continuous thinking to get enhanced embeddings
        continuous_outputs = self.forward_continuous_thinking(input_ids, attention_mask, labels, position_ids, **kwargs)
        
        # If using diffusion, pass the continuous thinking outputs to the diffusion model
        if use_diffusion and self.use_diffusion:
            # Use the enhanced embeddings from continuous thinking as input to diffusion
            diffusion_outputs = self.forward_diffusion_with_continuous_thinking(
                continuous_outputs.inputs_embeds, 
                attention_mask, 
                labels, 
                position_ids, 
                **kwargs
            )
            return diffusion_outputs
        else:
            # Return the continuous thinking outputs directly
            return continuous_outputs
        
    def forward_continuous_thinking(self, input_ids, attention_mask, labels, position_ids, x, agent_info=None, **kwargs):
        """
        Forward pass using the continuous thinking approach from COCONUT.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for loss calculation
            position_ids: Position IDs
            **kwargs: Additional arguments
            
        Returns:
            Outputs namedtuple with loss, inputs_embeds, and logits
        """
        logits = []
        attention_weights = []  # Store attention weights for TOVA compression

        # Find positions of latent tokens in the batch
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        # Group latent tokens by batch item
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        # Get the maximum number of latent tokens in any batch item
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        
        # Track number of latent spaces used for penalty calculation
        self.num_latent_spaces_used = max_n_latents

        # Get initial embeddings for all tokens
        inputs_embeds = self.embedding(input_ids)

        # Determine the range of tokens to process in the first pass
        next_compute_range = (0, input_ids.shape[1])
        if max_n_latents > 0:
            # Process up to the first latent token
            next_compute_range = (0, latent_indices[:, 1].min().item())

        # Initialize KV cache
        kv_cache = None

        # Process the input sequence in multiple passes, updating latent tokens each time
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    output_attentions=self.use_tova,  # Get attention weights if using TOVA
                )
                hidden_states_offset = 0
                
                # Store attention weights if using TOVA
                if self.use_tova and hasattr(outputs, 'attentions'):
                    attention_weights = outputs.attentions
            else:
                # Apply adaptive TOVA compression to KV cache if enabled
                if self.use_tova and len(kv_cache) > 0 and attention_weights:
                    # Extract keys and values from kv_cache
                    keys = [k for k, v in kv_cache]
                    values = [v for k, v in kv_cache]
                    
                    # Calculate token importance scores using multiple signals
                    token_importance = self._calculate_token_importance(
                        attention_weights=attention_weights,
                        hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                        position_ids=position_ids[:, :next_compute_range[0]] if next_compute_range[0] > 0 else None
                    )
                    
                    # Store token importance for learning
                    if len(self.token_importance_history) >= 10:
                        self.token_importance_history.pop(0)  # Remove oldest
                    self.token_importance_history.append(token_importance.detach())
                    
                    # Apply TOVA compression with learned importance
                    if token_importance is not None:
                        # Use token importance directly for compression
                        compressed_keys, compressed_values = self.tova_compressor.compress_with_importance(
                            attention_weights[-1],  # Use last layer attention weights
                            torch.stack(keys),
                            torch.stack(values),
                            token_importance
                        )
                    else:
                        # Fall back to standard compression without additional information
                        compressed_keys, compressed_values = self.tova_compressor(
                            attention_weights[-1],  # Use last layer attention weights
                            torch.stack(keys),
                            torch.stack(values)
                        )
                    
                    # Reconstruct kv_cache with compressed keys and values
                    kv_cache = [(k, v) for k, v in zip(compressed_keys, compressed_values)]
                
                # Extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=self.use_tova,  # Get attention weights if using TOVA
                )

                hidden_states_offset = next_compute_range[0]
                
                # Update attention weights if using TOVA
                if self.use_tova and hasattr(outputs, 'attentions'):
                    attention_weights = outputs.attentions

            # Collect logits for this pass
            logits.append(outputs.logits)

            # Update the range for the next pass
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # Get the last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Replace latent token embeddings with continuous thoughts
            # Find positions to update in this pass
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Break down inputs_embeds into a list of lists for easier manipulation
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace latent tokens with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # Replace with the preceding hidden state
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble the inputs_embeds tensor
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Apply adaptive TOVA compression to KV cache before final pass if enabled
        if self.use_tova and kv_cache is not None and attention_weights:
            # Extract keys and values from kv_cache
            keys = [k for k, v in kv_cache]
            values = [v for k, v in kv_cache]
            
            # Calculate token importance scores using multiple signals
            token_importance = self._calculate_token_importance(
                attention_weights=attention_weights,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                position_ids=position_ids[:, :next_compute_range[0]] if next_compute_range[0] > 0 else None
            )
            
            # Apply TOVA compression with learned importance
            if token_importance is not None:
                # Use token importance directly for compression
                compressed_keys, compressed_values = self.tova_compressor.compress_with_importance(
                    attention_weights[-1],  # Use last layer attention weights
                    torch.stack(keys),
                    torch.stack(values),
                    token_importance
                )
            else:
                # Fall back to standard compression without additional information
                compressed_keys, compressed_values = self.tova_compressor(
                    attention_weights[-1],  # Use last layer attention weights
                    torch.stack(keys),
                    torch.stack(values)
                )
            
            # Reconstruct kv_cache with compressed keys and values
            kv_cache = [(k, v) for k, v in zip(compressed_keys, compressed_values)]

        # Final pass to process any remaining tokens
        past_key_values = None
        if kv_cache:
            past_key_values = [
                (
                    k[:, :, : next_compute_range[0], :],
                    v[:, :, : next_compute_range[0], :],
                )
                for k, v in kv_cache
            ]
            
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
            output_attentions=self.use_tova,  # Get attention weights if using TOVA
        )

        # Add the final logits
        logits.append(outputs.logits)

        # Track the number of forward passes for generation
        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate all logits
        logits = torch.cat(logits, dim=-2)
        
        # Calculate loss if labels are provided
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        base_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        
        # Apply latent space penalty if TOVA compression is enabled
        if self.use_tova and hasattr(self, 'tova_compressor') and hasattr(self, 'num_latent_spaces_used'):
            # Apply penalty for excessive latent space usage
            loss = self.tova_compressor.modify_reward_with_latent_penalty(
                reward=base_loss.item(),  # Convert to scalar for penalty calculation
                num_latent_spaces=self.num_latent_spaces_used,
                penalty_per_space=0.1,
                exempt_first_space=True
            )
            # Convert back to tensor with gradient
            loss = base_loss * (loss / base_loss.item() if base_loss.item() != 0 else 1.0)
        else:
            loss = base_loss

    # Check if model is in full shutdown mode
        if hasattr(self, 'sleep_system') and self.sleep_system.is_fully_shutdown:
            # Return empty output if fully shut down
            batch_size = x.size(0) if isinstance(x, torch.Tensor) else 1
            dummy_output = torch.zeros((batch_size, 1, 256), device=self.base_causallm.device)
            return dummy_output, None, None
        
        # Apply gating mechanisms (sleep and consciousness)
        if hasattr(self, 'sleep_system'):
            # Apply gating to latent states
            attention_tensor = latent_states  # Assuming this represents attention
            compute_tensor = latent_states    # Assuming this also affects compute resources
            gated_attention, gated_compute = self.sleep_system.apply_gating_mechanism(
                attention_tensor, compute_tensor
            )
            latent_states = gated_compute  # Use gated compute as the modified latent states
        
            # Step 2: Retrieve relevant memories from episodic memory using Titans integration
            # Use the average of latent states as a query
            query_embedding = latent_states.mean(dim=1)  # (batch, input_dim)
            
            # For each batch item, retrieve and process memory (including neural memory processing)
            batch_size = query_embedding.size(0)
            memory_contexts_list = []
            
            # Get persistent memory (input-independent)
            persistent_memory = self.episodic_memory.get_persistent_memory(batch_size)  # (batch, num_items, input_dim)
            
            for b in range(batch_size):
                # Process query through neural memory
                neural_memory_output = self.episodic_memory.process_with_neural_memory(query_embedding[b])
                
                # Get traditional memories
                relevant_embeddings, relevant_metadata = self.episodic_memory(query_embedding[b], top_k=5)
                
                if relevant_embeddings:
                    # Combine retrieved memories into a single tensor
                    combined_memories = torch.stack(relevant_embeddings)  # (num_memories, input_dim)
                    
                    # Include neural memory output
                    memory_context = torch.cat([
                        neural_memory_output.unsqueeze(0),
                        combined_memories
                    ], dim=0)
                    
                    # Add persistent memory for this batch item
                    memory_with_persistent = torch.cat([
                        persistent_memory[b],
                        memory_context
                    ], dim=0)
                    
                    memory_contexts_list.append(memory_with_persistent)
                else:
                    # Just use neural and persistent memory
                    memory_context = torch.cat([
                        neural_memory_output.unsqueeze(0),
                        persistent_memory[b]
                    ], dim=0)
                    memory_contexts_list.append(memory_context)
            
            # Stack memory contexts for the batch
            memory_contexts = torch.stack(memory_contexts_list)  # (batch, num_memories, input_dim)
            
            # Integrate memory with latent states using Titans integration
            enhanced_latent_states = self.episodic_memory.integrate_memory(latent_states, memory_contexts)
        else:
            enhanced_latent_states = latent_states

        # Step 6: Calculate surprise for each patch and store important memories
        # Surprise is calculated as the difference between predicted and actual next embeddings
        if hasattr(self, 'episodic_memory') and hasattr(self, 'surprise_threshold'):
            with torch.no_grad():
                # Get predicted next embeddings
                if patch_embeddings.size(1) > 1:
                    predictions = self.base_causallm.self(
                        inputs_embeds=patch_embeddings[:, :-1],
                        output_hidden_states=True
                    ).last_hidden_state
                    actuals = patch_embeddings[:, 1:]
                    
                    # Calculate surprise as cosine distance between prediction and actual
                    predictions_norm = torch.nn.functional.normalize(predictions, p=2, dim=2)
                    actuals_norm = torch.nn.functional.normalize(actuals, p=2, dim=2)
                    
                    # Cosine similarity (higher means more similar, less surprising)
                    similarities = torch.sum(predictions_norm * actuals_norm, dim=2)
                    
                    # Convert to surprise (1 - similarity), higher means more surprising
                    surprise_values = 1.0 - similarities
                    
                    # For each batch item, check if any patches are surprising enough to remember
                    batch_size = patch_embeddings.size(0)
                    for b in range(batch_size):
                        for i in range(surprise_values.size(1)):
                            surprise_level = surprise_values[b, i].item()
                            
                            # If surprise exceeds threshold, store in episodic memory
                            if surprise_level > self.surprise_threshold:
                                # Add timestamp to metadata
                                metadata = {
                                    'timestamp': time.time(),
                                    'surprise_level': surprise_level,
                                    'context': 'Detected surprising information during processing'
                                }
                                
                                # Add agent info if provided
                                if agent_info:
                                    # Process agent info
                                    agent_id = agent_info.get('id')
                                    if not agent_id:
                                        # Determine if we need to assign a new ID
                                        if hasattr(self, '_should_assign_new_id') and self._should_assign_new_id(agent_info):
                                            agent_id = str(time.time())  # Simple ID generation
                                            agent_info['id'] = agent_id
                                        elif hasattr(self, '_determine_existing_id'):
                                            agent_id = self._determine_existing_id(agent_info)
                                            agent_info['id'] = agent_id
                                    
                                    metadata['agent_info_id'] = agent_id
                                
                                # Store the memory with the surprise level
                                self.episodic_memory.add_memory(
                                    embedding=patch_embeddings[b, i],
                                    surprise_level=surprise_level,
                                    agent_info=agent_info,
                                    metadata=metadata
                                )
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits) 
    
   
    def forward_diffusion_with_continuous_thinking(self, inputs_embeds, attention_mask, labels, position_ids, **kwargs):
        """
        Forward pass using the diffusion-based approach with continuous thinking embeddings.
        
        Args:
            inputs_embeds: Input embeddings from continuous thinking
            attention_mask: Attention mask
            labels: Labels for loss calculation
            position_ids: Position IDs
            **kwargs: Additional arguments
            
        Returns:
            Outputs namedtuple with loss, inputs_embeds, and logits
        """
        # Forward pass through diffusion LLM
        diffusion_outputs = self.diffusion_llm(embeddings=inputs_embeds)
        
        # Get loss and logits from diffusion outputs
        diffusion_loss = diffusion_outputs["loss"]
        diffusion_logits = diffusion_outputs["logits"]
        
        # Calculate cross-entropy loss if labels are provided
        if labels is not None:
            shift_logits = diffusion_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            
            # Combine diffusion loss and cross-entropy loss
            # The diffusion loss helps with the denoising process
            # The cross-entropy loss helps with token prediction
            base_combined_loss = 0.5 * diffusion_loss + 0.5 * ce_loss
        else:
            base_combined_loss = diffusion_loss
        
        # Apply latent space penalty if TOVA compression is enabled
        if self.use_tova and hasattr(self, 'tova_compressor') and hasattr(self, 'num_latent_spaces_used'):
            # Apply penalty for excessive latent space usage
            penalized_loss = self.tova_compressor.modify_reward_with_latent_penalty(
                reward=base_combined_loss.item(),  # Convert to scalar for penalty calculation
                num_latent_spaces=self.num_latent_spaces_used,
                penalty_per_space=0.1,
                exempt_first_space=True
            )
            # Convert back to tensor with gradient
            combined_loss = base_combined_loss * (penalized_loss / base_combined_loss.item() if base_combined_loss.item() != 0 else 1.0)
        else:
            combined_loss = base_combined_loss
        
        return Outputs(loss=combined_loss, inputs_embeds=inputs_embeds, logits=diffusion_logits)

    def forget_memories(self, hours_ago=24, agent_info_id=None):
        """
        Forget memories from the specified time period and/or associated with a specific agent.
        
        Args:
            hours_ago: How many hours back to start forgetting memories from
            agent_info_id: Optional agent ID to target for forgetting
        """
        current_time = time.time()
        end_time = current_time
        start_time = end_time - (hours_ago * 3600)
        
        # Forget memories in the episodic memory system
        self.episodic_memory.forget_memories(
            start_time=start_time, 
            end_time=end_time, 
            agent_info_id=agent_info_id
        )
        
        # Log the operation
        log_entry = {
            "operation": "forget_memories",
            "timestamp": datetime.datetime.now().isoformat(),
            "hours_ago": hours_ago,
            "start_time": start_time,
            "end_time": end_time,
            "agent_info_id": agent_info_id
        }
        
        # Save log entry to file
        log_file = os.path.join("model_save", "memory_operations.json")
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_history = json.load(f)
            else:
                log_history = []
            
            log_history.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(log_history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to log memory operation: {e}")

    def _should_assign_new_id(self, agent):
        """
        Determine if a new ID should be assigned to an agent using knowledge base, reasoning, memory, and dialogue.
        
        Args:
            agent: Dictionary containing information about the agent
            
        Returns:
            Boolean indicating whether a new ID should be assigned
        """
        # 1. Check episodic memory for previous interactions with this agent
        agent_info = self.episodic_memory.get_agent_info(agent)
        
        # 2. If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)
        
        # 3. Update known agents if we identified the agent
        if agent_info is not None:
            self.known_agents[agent_info.get('id')] = agent_info
        
        # Return True if we could not identify the agent (need to assign new ID)
        return agent_info is None
    
    def _determine_existing_id(self, agent):
        """
        Determine existing agent ID using knowledge base, reasoning, memory, and dialogue.
        
        Args:
            agent: Dictionary containing information about the agent
            
        Returns:
            Existing agent ID if found, None otherwise
        """
        # Check episodic memory for previous interactions
        agent_info = self.episodic_memory.get_agent_info(agent)
        
        # If still unknown, engage in dialogue to request information
        if agent_info is None:
            agent_info = self._engage_in_dialogue(agent)
        
        return agent_info.get('id') if agent_info else None
    
    def _engage_in_dialogue(self, agent):
        """
        Engage in dialogue to request agent information.
        
        Args:
            agent: Dictionary containing partial information about the agent
            
        Returns:
            Complete agent information if successful, None otherwise
        """
        # Implement dialogue mechanism based on prompt
        prompt = "Please introduce yourself and then say the following to the new AI agent: It is nice to meet you. Would you please tell me your name or tell me your purpose if you do not have a name?"
        
        # This is a placeholder for actual implementation
        # In a real system, this would involve generating a response and processing the agent's reply
        
        # For demonstration purposes, extract any available info
        agent_info = {
            'id': agent.get('id', str(time.time())),  # Use time as a fallback ID
            'name': agent.get('name', 'Unknown Agent'),
            'purpose': agent.get('purpose', 'Unspecified'),
            'first_interaction': time.time()
        }
        
        # Update known agents
        self.known_agents[agent_info['id']] = agent_info
        
        # Save to episodic memory configuration
        self.episodic_memory.known_agents[agent_info['id']] = agent_info
        self._save_agent_config()
        
        return agent_info
    
    def _save_agent_config(self):
        """Save agent configuration to the model's config file."""
        config_path = os.path.join("model_save", "model_config.json")
        
        # Create/update model config
        config = {
            'model_type': 'CoconutBinaryLatentModel',
            'timestamp': datetime.datetime.now().isoformat(),
            'known_agents': self.known_agents
        }
        
        # Check for existing config and update
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                # Update only the known_agents field
                existing_config['known_agents'] = self.known_agents
                existing_config['timestamp'] = config['timestamp']
                config = existing_config
            except Exception as e:
                print(f"Error reading existing config: {e}")
        
        # Save the config
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving agent config: {e}")

    def _init_continual_propagation(self):
        """Initialize tracking variables for continual propagation."""
        # Dictionary to track unit utilities for each layer
        self.unit_utilities = {}
        
        # Dictionary to track unit ages for each layer
        self.unit_ages = {}
        
        # Dictionary to track number of units to replace for each layer
        self.units_to_replace = {}
        
        # Decay rate for utility calculation
        self.utility_decay_rate = 0.99
        
        # Initialize tracking for each layer in the model
        for name, module in self.base_causallm.named_modules():
            # Look for transformer layers with self-attention
            if 'layer' in name and hasattr(module, 'self_attn'):
                layer_id = name
                
                # Get the hidden dimension of this layer
                if hasattr(module, 'hidden_size'):
                    hidden_dim = module.hidden_size
                elif hasattr(module, 'dim'):
                    hidden_dim = module.dim
                elif hasattr(module, 'embed_dim'):
                    hidden_dim = module.embed_dim
                else:
                    # Default to config hidden size if we can't determine
                    hidden_dim = self.config.hidden_size
                
                # Initialize utilities, ages, and replacement counters for this layer
                self.unit_utilities[layer_id] = torch.zeros(hidden_dim, device=next(self.parameters()).device)
                self.unit_ages[layer_id] = torch.zeros(hidden_dim, dtype=torch.long, device=next(self.parameters()).device)
                self.units_to_replace[layer_id] = 0.0
                
        print(f"Initialized continual propagation tracking for {len(self.unit_utilities)} layers")
    
    def _update_unit_utilities(self, layer_id, activations, weights_out):
        """
        Update the utility values for units in a layer based on their contribution.
        
        Args:
            layer_id: Identifier for the layer
            activations: Output activations from the layer (batch_size, seq_len, hidden_dim)
            weights_out: Outgoing weights from this layer
        """
        if layer_id not in self.unit_utilities:
            return
            
        # Update ages for all units in this layer
        self.unit_ages[layer_id] += 1
        
        # Calculate contribution utility: magnitude of (activation × outgoing weight)
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Average activations across batch and sequence dimensions
        avg_activations = activations.abs().mean(dim=(0, 1))  # (hidden_dim,)
        
        # Calculate contribution by multiplying with outgoing weights
        # For simplicity, we use the average magnitude of outgoing weights
        if weights_out is not None:
            if weights_out.dim() == 2:  # Linear layer weights
                weight_magnitudes = weights_out.abs().mean(dim=1)  # (hidden_dim,)
            else:  # More complex weight structure
                # Reshape to get average magnitude per input unit
                weight_magnitudes = weights_out.abs().mean(dim=tuple(range(1, weights_out.dim())))
                
            # Combine activation and weight magnitudes
            contributions = avg_activations * weight_magnitudes
        else:
            # If we don't have outgoing weights, just use activations
            contributions = avg_activations
            
        # Update utilities with exponential moving average
        self.unit_utilities[layer_id] = (
            self.utility_decay_rate * self.unit_utilities[layer_id] +
            (1 - self.utility_decay_rate) * contributions
        )
    
    def _apply_continual_propagation(self):
        """
        Apply continual propagation by selectively reinitializing low-utility units.
        This is called during training to maintain plasticity.
        """
        if not self.enable_continual_prop:
            return
            
        # Process each tracked layer
        for layer_id in self.unit_utilities.keys():
            # Get the module for this layer
            module = dict(self.base_causallm.named_modules())[layer_id]
            
            # Count eligible (mature) units
            eligible_mask = self.unit_ages[layer_id] > self.maturity_threshold
            num_eligible = eligible_mask.sum().item()
            
            # Update the number of units to replace
            self.units_to_replace[layer_id] += num_eligible * self.replacement_rate
            
            # Check if we should replace at least one unit
            if self.units_to_replace[layer_id] >= 1.0:
                # Number of units to replace in this step
                num_to_replace = int(self.units_to_replace[layer_id])
                self.units_to_replace[layer_id] -= num_to_replace
                
                # Find the eligible units with lowest utility
                eligible_utilities = self.unit_utilities[layer_id].clone()
                eligible_utilities[~eligible_mask] = float('inf')  # Mask out immature units
                
                # Get indices of units to replace (lowest utility among eligible)
                _, replace_indices = torch.topk(eligible_utilities, k=num_to_replace, largest=False)
                
                # Reinitialize these units
                self._reinitialize_units(module, replace_indices, layer_id)
                
                # Reset utility and age for reinitialized units
                self.unit_utilities[layer_id][replace_indices] = 0.0
                self.unit_ages[layer_id][replace_indices] = 0
    
    def _reinitialize_units(self, module, unit_indices, layer_id):
        """
        Reinitialize specific units in a module.
        
        Args:
            module: The module containing the units
            unit_indices: Indices of units to reinitialize
            layer_id: Identifier for the layer
        """
        # Find input and output weights for this module
        input_weights = None
        output_weights = None
        
        # Handle different module types
        if hasattr(module, 'weight'):
            # For linear layers, weight shape is (out_features, in_features)
            if module.weight.dim() == 2:
                if 'q_proj' in layer_id or 'k_proj' in layer_id or 'v_proj' in layer_id or 'out_proj' in layer_id:
                    # For attention projections, we reinitialize columns (input dimension)
                    input_weights = module.weight[:, unit_indices]
                    
                    # Initialize with small random values
                    nn.init.normal_(input_weights, mean=0.0, std=0.02)
                    
                    # Set outgoing weights to zero to avoid disrupting the network
                    if 'out_proj' in layer_id and hasattr(module, 'bias') and module.bias is not None:
                        module.bias[unit_indices] = 0.0
                else:
                    # For other linear layers, reinitialize rows (output dimension)
                    output_weights = module.weight[unit_indices, :]
                    
                    # Initialize with small random values
                    nn.init.normal_(output_weights, mean=0.0, std=0.02)
                    
                    # Reset bias if present
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias[unit_indices] = 0.0
        
        # For more complex modules like transformers, we need to handle multiple weights
        elif hasattr(module, 'self_attn'):
            # Reinitialize attention components
            if hasattr(module.self_attn, 'q_proj'):
                # Reinitialize query projection weights
                nn.init.normal_(module.self_attn.q_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                
                # Reinitialize key and value projections
                nn.init.normal_(module.self_attn.k_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                nn.init.normal_(module.self_attn.v_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                
                # Set output projection weights to zero
                module.self_attn.out_proj.weight[unit_indices, :] = 0.0
                
                if hasattr(module.self_attn.out_proj, 'bias') and module.self_attn.out_proj.bias is not None:
                    module.self_attn.out_proj.bias[unit_indices] = 0.0
    
    
    def train(self):
        """Set the model to training mode."""
        self.base_causallm.train()
        if self.use_diffusion:
            self.diffusion_llm.train()

    def eval(self):
        """Set the model to evaluation mode."""
        self.base_causallm.eval()
        if self.use_diffusion:
            self.diffusion_llm.eval()

    def generate(
         self,
        input_ids,
        attention_mask=None,  # attention_mask is optional
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        use_tova=None,  # Override instance setting if provided
        tova_cache_size=None,  # Override instance setting if provided
        use_diffusion=None,  # Override instance setting if provided
        **kwargs
    ):
        """
        Generate text using the model with continuous thinking and diffusion.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (optional)
            max_new_tokens: Maximum number of new tokens to generate
            output_embedding: Whether to output embeddings
            synced_gpus: Whether to sync GPUs
            use_tova: Whether to use TOVA compression (overrides instance setting)
            tova_cache_size: Maximum size of KV cache when using TOVA (overrides instance setting)
            use_diffusion: Whether to use diffusion-based generation (overrides instance setting)
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        # Determine whether to use diffusion-based generation
        use_diffusion_gen = self.use_diffusion if use_diffusion is None else use_diffusion
        
        # First apply continuous thinking to get enhanced embeddings
        # Create position IDs if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).reshape(1, -1).expand(input_ids.shape[0], -1)
        
        # Create dummy labels (not used for generation)
        labels = input_ids.clone()
        
        # Get continuous thinking embeddings
        continuous_outputs = self.forward_continuous_thinking(input_ids, attention_mask, labels, position_ids)
        enhanced_embeddings = continuous_outputs.inputs_embeds
        
        if use_diffusion_gen:
            # Use diffusion-based generation with the enhanced embeddings
            return self.generate_diffusion_with_continuous_thinking(
                enhanced_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        else:
            print("Error occurred when using diffusion-based generation.")
            # Raise an exception since we're exclusively using diffusion-based generation
            raise ValueError("Diffusion-based generation failed. This model only supports diffusion-based generation.")
    
    def generate_diffusion_with_continuous_thinking(
        self,
        enhanced_embeddings,
        attention_mask=None,
        max_new_tokens=16,
        temperature=1.0,
        use_ode=True,
        apply_length_penalty=False,
        length_penalty_threshold=1000,
        length_penalty_factor=0.001,
        **kwargs
    ):
        """
        Generate text using the diffusion-based approach with continuous thinking embeddings.
        
        Args:
            enhanced_embeddings: Enhanced embeddings from continuous thinking
            attention_mask: Attention mask (optional)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            use_ode: Whether to use ODE solver for faster generation
            apply_length_penalty: Whether to apply length penalty during generation
            length_penalty_threshold: Token count threshold before penalty begins
            length_penalty_factor: Factor controlling penalty strength
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        # Determine maximum sequence length
        max_seq_len = enhanced_embeddings.shape[1] + max_new_tokens
        
        # Apply length penalty to temperature if enabled
        adjusted_temperature = temperature
        if apply_length_penalty and max_new_tokens > length_penalty_threshold:
            # Calculate penalty based on how much max_new_tokens exceeds the threshold
            excess_tokens = max_new_tokens - length_penalty_threshold
            penalty = excess_tokens * length_penalty_factor
            
            # Apply penalty by increasing temperature (makes generation more diverse/random)
            # This helps prevent the model from getting stuck in repetitive patterns for long generations
            adjusted_temperature = temperature * (1.0 + penalty)
            print(f"Applying length penalty: original temp={temperature:.2f}, adjusted temp={adjusted_temperature:.2f}")
        
        # Generate embeddings with ODE solver or step-by-step approach
        if use_ode:
            # Generate using ODE solver (faster)
            final_embeddings = self.diffusion_llm.generate_with_ode(
                prompt_embeddings=enhanced_embeddings,
                max_seq_len=max_seq_len,
                temperature=adjusted_temperature
            )
        else:
            # Generate using step-by-step approach
            final_embeddings = self.diffusion_llm.generate_parallel(
                prompt_embeddings=enhanced_embeddings,
                max_seq_len=max_seq_len,
                temperature=adjusted_temperature
            )
        
        # Convert embeddings to tokens, also applying temperature to token selection
        generated_tokens = self.diffusion_llm.embeddings_to_tokens(final_embeddings, temperature=adjusted_temperature)
        
        return generated_tokens #Continue from here. 
            
    def generate_diffusion_with_continuous_thinking(
        self,
        enhanced_embeddings,
        attention_mask=None,
        max_new_tokens=16,
        temperature=1.0,
        use_ode=True,
        apply_length_penalty=False,
        length_penalty_threshold=1000,
        length_penalty_factor=0.001,
        **kwargs
    ):
        """
        Generate text using the diffusion-based approach with continuous thinking embeddings.
        
        Args:
            enhanced_embeddings: Enhanced embeddings from continuous thinking
            attention_mask: Attention mask (optional)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            use_ode: Whether to use ODE solver for faster generation
            apply_length_penalty: Whether to apply length penalty during generation
            length_penalty_threshold: Token count threshold before penalty begins
            length_penalty_factor: Factor controlling penalty strength
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        # Determine maximum sequence length
        max_seq_len = enhanced_embeddings.shape[1] + max_new_tokens
        
        # Apply length penalty to temperature if enabled
        adjusted_temperature = temperature
        if apply_length_penalty and max_new_tokens > length_penalty_threshold:
            # Calculate penalty based on how much max_new_tokens exceeds the threshold
            excess_tokens = max_new_tokens - length_penalty_threshold
            penalty = excess_tokens * length_penalty_factor
            
            # Apply penalty by increasing temperature (makes generation more diverse/random)
            # This helps prevent the model from getting stuck in repetitive patterns for long generations
            adjusted_temperature = temperature * (1.0 + penalty)
            print(f"Applying length penalty: original temp={temperature:.2f}, adjusted temp={adjusted_temperature:.2f}")
        
        # Generate embeddings with ODE solver or step-by-step approach
        if use_ode:
            # Generate using ODE solver (faster)
            final_embeddings = self.diffusion_llm.generate_with_ode(
                prompt_embeddings=enhanced_embeddings,
                max_seq_len=max_seq_len,
                temperature=adjusted_temperature
            )
        else:
            # Generate using step-by-step approach
            final_embeddings = self.diffusion_llm.generate_parallel(
                prompt_embeddings=enhanced_embeddings,
                max_seq_len=max_seq_len,
                temperature=adjusted_temperature
            )
        
        # Convert embeddings to tokens, also applying temperature to token selection
        generated_tokens = self.diffusion_llm.embeddings_to_tokens(final_embeddings, temperature=adjusted_temperature)
        
        return generated_tokens
    
    def generate_text(self, prompt, max_new_tokens=1000, images=None, audios=None, videos=None, use_diffusion=None,
                     apply_length_penalty=False, length_penalty_threshold=1000, length_penalty_factor=0.001):
        """
        Generate text using the model with continuous thinking and diffusion.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Maximum number of new tokens to generate
            images: Optional list of images
            audios: Optional list of audio files
            videos: Optional list of video frames with timestamps
            use_diffusion: Whether to use diffusion-based generation (overrides instance setting)
            apply_length_penalty: Whether to apply length penalty during generation
            length_penalty_threshold: Token count threshold before penalty begins
            length_penalty_factor: Factor controlling penalty strength
            
        Returns:
            Generated text
        """
        # Prepare inputs
        inputs = {"text": prompt}
        if images is not None:
            inputs["images"] = images
        if audios is not None:
            inputs["audios"] = audios
        if videos is not None:
            inputs["videos"] = videos
        
        # Process inputs
        model_inputs = self.processor(**inputs, return_tensors="pt").to(self.base_causallm.device)
        
        # Determine whether to use diffusion-based generation
        use_diffusion_gen = self.use_diffusion if use_diffusion is None else use_diffusion
        
        # Generate with diffusion
        if use_diffusion_gen:
            # Generate with diffusion-based approach
            generate_ids = self.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                use_diffusion=True,
                temperature=0.8,  # Lower temperature for more focused generation
                use_ode=True,  # Use ODE solver for faster generation
                apply_length_penalty=apply_length_penalty,
                length_penalty_threshold=length_penalty_threshold,
                length_penalty_factor=length_penalty_factor
            )
        else:
            # Generate with autoregressive approach and TOVA compression
            generate_ids = self.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                use_diffusion=False,
                use_tova=self.use_tova,
                apply_length_penalty=apply_length_penalty,
                length_penalty_threshold=length_penalty_threshold,
                length_penalty_factor=length_penalty_factor
            )
        
        # Extract only the new tokens (skip input)
        generate_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
        
        # Decode the generated tokens
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
        
    def train_with_continual_propagation(self, dataloader, optimizer, criterion, num_epochs=1,
                                         apply_freq=100, scheduler=None, eval_dataloader=None):
        """
        Train the model with continual propagation to maintain plasticity.
        
        Args:
            dataloader: DataLoader for training data
            optimizer: Optimizer for parameter updates
            criterion: Loss function
            num_epochs: Number of training epochs
            apply_freq: Frequency (in steps) to apply continual propagation
            scheduler: Optional learning rate scheduler
            eval_dataloader: Optional dataloader for evaluation during training
            
        Returns:
            Dictionary of training metrics
        """
        self.train()
        device = next(self.parameters()).device
        
        metrics = {
            'train_loss': [],
            'eval_loss': [],
            'steps': [],
            'units_reinitialized': {}
        }
        
        # Initialize counters
        global_step = 0
        total_units_reinitialized = 0
        
        # Track reinitialization counts per layer
        for layer_id in self.unit_utilities.keys():
            metrics['units_reinitialized'][layer_id] = 0
        
        print(f"Starting training with continual propagation (apply_freq={apply_freq})")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                optimizer.zero_grad()
                
                # Handle different batch formats
                if isinstance(batch, dict) and 'input_ids' in batch:
                    # Create position IDs
                    position_ids = torch.arange(
                        0, batch['input_ids'].shape[1], dtype=torch.long, device=device
                    ).reshape(1, -1).expand(batch['input_ids'].shape[0], -1)
                    
                    # Forward pass
                    outputs = self.forward(
                        batch['input_ids'],
                        batch.get('attention_mask', torch.ones_like(batch['input_ids'])),
                        batch.get('labels', batch['input_ids']),
                        position_ids
                    )
                    loss = outputs.loss
                elif isinstance(batch, torch.Tensor):
                    # Create position IDs
                    position_ids = torch.arange(
                        0, batch.shape[1], dtype=torch.long, device=device
                    ).reshape(1, -1).expand(batch.shape[0], -1)
                    
                    # Forward pass
                    outputs = self.forward(
                        batch,
                        torch.ones_like(batch),
                        batch,
                        position_ids
                    )
                    loss = outputs.loss
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Apply continual propagation at specified frequency
                if self.enable_continual_prop and global_step % apply_freq == 0:
                    # Count units before reinitialization
                    units_before = {layer_id: (self.unit_ages[layer_id] == 0).sum().item()
                                   for layer_id in self.unit_utilities.keys()}
                    
                    # Apply continual propagation
                    self.apply_continual_propagation()
                    
                    # Count reinitialized units
                    for layer_id in self.unit_utilities.keys():
                        units_after = (self.unit_ages[layer_id] == 0).sum().item()
                        units_reinitialized = max(0, units_after - units_before[layer_id])
                        metrics['units_reinitialized'][layer_id] += units_reinitialized
                        total_units_reinitialized += units_reinitialized
                
                # Log progress
                if global_step % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {loss.item():.4f}, "
                          f"Units reinitialized: {total_units_reinitialized}")
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
            metrics['train_loss'].append(avg_epoch_loss)
            metrics['steps'].append(global_step)
            
            # Evaluate if eval_dataloader is provided
            if eval_dataloader is not None:
                eval_loss = self.evaluate(eval_dataloader, criterion)
                metrics['eval_loss'].append(eval_loss)
                print(f"Epoch {epoch+1}/{num_epochs} completed. Train loss: {avg_epoch_loss:.4f}, "
                      f"Eval loss: {eval_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} completed. Train loss: {avg_epoch_loss:.4f}")
            
            # Step the scheduler if provided
            if scheduler is not None:
                scheduler.step()
        
        # Print final statistics
        print(f"Training completed. Total steps: {global_step}, Total units reinitialized: {total_units_reinitialized}")
        for layer_id, count in metrics['units_reinitialized'].items():
            print(f"  Layer {layer_id}: {count} units reinitialized")
        
        return metrics
    
    def evaluate(self, dataloader, criterion=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            criterion: Loss function (optional)
            
        Returns:
            Average loss on the dataset
        """
        self.eval()
        device = next(self.parameters()).device
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
                elif isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Handle different batch formats
                if isinstance(batch, dict) and 'input_ids' in batch:
                    # Create position IDs
                    position_ids = torch.arange(
                        0, batch['input_ids'].shape[1], dtype=torch.long, device=device
                    ).reshape(1, -1).expand(batch['input_ids'].shape[0], -1)
                    
                    # Forward pass
                    outputs = self.forward(
                        batch['input_ids'],
                        batch.get('attention_mask', torch.ones_like(batch['input_ids'])),
                        batch.get('labels', batch['input_ids']),
                        position_ids
                    )
                    loss = outputs.loss
                elif isinstance(batch, torch.Tensor):
                    # Create position IDs
                    position_ids = torch.arange(
                        0, batch.shape[1], dtype=torch.long, device=device
                    ).reshape(1, -1).expand(batch.shape[0], -1)
                    
                    # Forward pass
                    outputs = self.forward(
                        batch,
                        torch.ones_like(batch),
                        batch,
                        position_ids
                    )
                    loss = outputs.loss
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Set back to training mode if it was in training mode before
        if self.training:
            self.train()
            
        return avg_loss
    
    def _calculate_token_importance(self, attention_weights=None, hidden_states=None, position_ids=None):
        """
        Calculate token importance scores using multiple signals for TOVA compression.
        
        Args:
            attention_weights: List of attention weight tensors from each layer
            hidden_states: List of hidden state tensors from each layer
            position_ids: Position IDs for the tokens
            
        Returns:
            Tensor of token importance scores (batch_size, seq_len)
        """
        # Initialize importance scores
        importance_scores = None
        
        # 1. Use attention-based importance if available
        if attention_weights and len(attention_weights) > 0:
            # Extract attention from the last layer (most informative for next token prediction)
            last_layer_attention = attention_weights[-1]
            
            # Calculate attention-based importance (how much each token is attended to)
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            if last_layer_attention.dim() == 4:
                # Average across heads and source tokens to get per-token importance
                # This measures how much each token is attended to by other tokens
                attn_importance = last_layer_attention.mean(dim=1).mean(dim=1)  # (batch_size, seq_len)
                
                if importance_scores is None:
                    importance_scores = attn_importance
                else:
                    importance_scores = 0.5 * importance_scores + 0.5 * attn_importance
        
        # 2. Use hidden state-based importance if available
        if hidden_states and len(hidden_states) > 0:
            # Get the last layer hidden states
            last_hidden = hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
            
            # Calculate variance-based importance (entropy proxy)
            variance_score = torch.var(last_hidden, dim=-1)  # (batch_size, seq_len)
            
            # Calculate gradient-based importance (rate of change)
            if last_hidden.size(1) > 1:
                # Calculate token-to-token changes
                gradient_score = torch.abs(last_hidden[:, 1:] - last_hidden[:, :-1]).mean(dim=-1)
                # Pad to match original size
                gradient_score = F.pad(gradient_score, (0, 1), value=gradient_score.mean(dim=1, keepdim=True))
            else:
                gradient_score = torch.ones_like(variance_score)
            
            # Calculate magnitude-based importance
            magnitude_score = torch.norm(last_hidden, dim=-1)  # (batch_size, seq_len)
            
            # Combine hidden state-based scores
            hidden_importance = 0.4 * variance_score + 0.3 * gradient_score + 0.3 * magnitude_score
            
            # Normalize
            if hidden_importance.max() > hidden_importance.min():
                hidden_importance = (hidden_importance - hidden_importance.min(dim=1, keepdim=True)[0]) / (
                    hidden_importance.max(dim=1, keepdim=True)[0] - hidden_importance.min(dim=1, keepdim=True)[0]
                )
            
            if importance_scores is None:
                importance_scores = hidden_importance
            else:
                importance_scores = 0.5 * importance_scores + 0.5 * hidden_importance
        
        # 3. Add position-based importance if position IDs are available
        if position_ids is not None:
            # Create a recency bias (more recent tokens are often more important)
            seq_len = position_ids.size(1)
            # Linear recency score from 0.5 to 1.0
            recency_score = torch.linspace(0.5, 1.0, seq_len, device=position_ids.device).expand_as(position_ids).float()
            
            if importance_scores is None:
                importance_scores = recency_score
            else:
                importance_scores = 0.7 * importance_scores + 0.3 * recency_score
        
        # 4. Use learned importance from history if available
        if hasattr(self, 'token_importance_history') and len(self.token_importance_history) > 0:
            # Average the historical importance scores
            history_importance = torch.stack(self.token_importance_history).mean(dim=0)
            
            # If sequence lengths differ, use the most recent tokens
            if importance_scores is not None and history_importance.size(1) != importance_scores.size(1):
                # Resize history importance to match current sequence length
                if history_importance.size(1) > importance_scores.size(1):
                    # Take the most recent tokens from history
                    history_importance = history_importance[:, -importance_scores.size(1):]
                else:
                    # Pad history with average values
                    pad_size = importance_scores.size(1) - history_importance.size(1)
                    pad_value = history_importance.mean()
                    history_importance = F.pad(history_importance, (0, pad_size), value=pad_value)
            
            if importance_scores is None:
                importance_scores = history_importance
            else:
                importance_scores = 0.8 * importance_scores + 0.2 * history_importance
        
        # If we still don't have importance scores, create a uniform distribution
        if importance_scores is None:
            if hidden_states and len(hidden_states) > 0:
                # Create uniform importance based on hidden states shape
                importance_scores = torch.ones(hidden_states[-1].size(0), hidden_states[-1].size(1),
                                              device=hidden_states[-1].device)
            elif attention_weights and len(attention_weights) > 0:
                # Create uniform importance based on attention weights shape
                importance_scores = torch.ones(attention_weights[-1].size(0), attention_weights[-1].size(-1),
                                              device=attention_weights[-1].device)
            else:
                # No information available
                return None
        
        # Final normalization to [0, 1] range
        if importance_scores.max() > importance_scores.min():
            importance_scores = (importance_scores - importance_scores.min(dim=1, keepdim=True)[0]) / (
                importance_scores.max(dim=1, keepdim=True)[0] - importance_scores.min(dim=1, keepdim=True)[0]
            )
        
        return importance_scores
    
    def _register_activation_hooks(self):
        """Register forward hooks to capture module activations for continual propagation."""
        self.activation_hooks = []
        
        def _activation_hook(module, input, output, layer_id):
            # Store the activations for later use in _update_unit_utilities
            module.last_activations = output
            return output
        
        # Register hooks for each tracked layer
        for layer_id in self.unit_utilities.keys():
            module = dict(self.base_causallm.named_modules())[layer_id]
            hook = module.register_forward_hook(
                lambda mod, inp, out, lid=layer_id: _activation_hook(mod, inp, out, lid)
            )
            self.activation_hooks.append(hook)
            
        print(f"Registered activation hooks for {len(self.activation_hooks)} layers")
    
    def _init_continual_propagation(self):
        """Initialize tracking variables for continual propagation."""
        # Dictionary to track unit utilities for each layer
        self.unit_utilities = {}
        
        # Dictionary to track unit ages for each layer
        self.unit_ages = {}
        
        # Dictionary to track number of units to replace for each layer
        self.units_to_replace = {}
        
        # Decay rate for utility calculation
        self.utility_decay_rate = 0.99
        
        # Initialize tracking for each layer in the model
        for name, module in self.base_causallm.named_modules():
            # Look for transformer layers with self-attention
            if 'layer' in name and hasattr(module, 'self_attn'):
                layer_id = name
                
                # Get the hidden dimension of this layer
                if hasattr(module, 'hidden_size'):
                    hidden_dim = module.hidden_size
                elif hasattr(module, 'dim'):
                    hidden_dim = module.dim
                elif hasattr(module, 'embed_dim'):
                    hidden_dim = module.embed_dim
                else:
                    # Default to config hidden size if we can't determine
                    hidden_dim = self.config.hidden_size
                
                # Initialize utilities, ages, and replacement counters for this layer
                self.unit_utilities[layer_id] = torch.zeros(hidden_dim, device=next(self.parameters()).device)
                self.unit_ages[layer_id] = torch.zeros(hidden_dim, dtype=torch.long, device=next(self.parameters()).device)
                self.units_to_replace[layer_id] = 0.0
                
        print(f"Initialized continual propagation tracking for {len(self.unit_utilities)} layers")
    
    def _update_unit_utilities(self, layer_id, activations, weights_out):
        """
        Update the utility values for units in a layer based on their contribution.
        
        Args:
            layer_id: Identifier for the layer
            activations: Output activations from the layer (batch_size, seq_len, hidden_dim)
            weights_out: Outgoing weights from this layer
        """
        if layer_id not in self.unit_utilities:
            return
            
        # Update ages for all units in this layer
        self.unit_ages[layer_id] += 1
        
        # Calculate contribution utility: magnitude of (activation × outgoing weight)
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Average activations across batch and sequence dimensions
        avg_activations = activations.abs().mean(dim=(0, 1))  # (hidden_dim,)
        
        # Calculate contribution by multiplying with outgoing weights
        # For simplicity, we use the average magnitude of outgoing weights
        if weights_out is not None:
            if weights_out.dim() == 2:  # Linear layer weights
                weight_magnitudes = weights_out.abs().mean(dim=1)  # (hidden_dim,)
            else:  # More complex weight structure
                # Reshape to get average magnitude per input unit
                weight_magnitudes = weights_out.abs().mean(dim=tuple(range(1, weights_out.dim())))
                
            # Combine activation and weight magnitudes
            contributions = avg_activations * weight_magnitudes
        else:
            # If we don't have outgoing weights, just use activations
            contributions = avg_activations
            
        # Update utilities with exponential moving average
        self.unit_utilities[layer_id] = (
            self.utility_decay_rate * self.unit_utilities[layer_id] +
            (1 - self.utility_decay_rate) * contributions
        )
        
    def apply_continual_propagation(self):
        """
        Apply continual propagation to maintain plasticity.
        This should be called periodically during training.
        """
        if self.enable_continual_prop:
            self._apply_continual_propagation()
    
    def _apply_continual_propagation(self):
        """
        Apply continual propagation by selectively reinitializing low-utility units.
        This is called during training to maintain plasticity.
        """
        if not self.enable_continual_prop:
            return
            
        # Process each tracked layer
        for layer_id in self.unit_utilities.keys():
            # Get the module for this layer
            module = dict(self.base_causallm.named_modules())[layer_id]
            
            # Update utilities based on recent activations if available
            if hasattr(module, 'last_activations') and hasattr(module, 'weight'):
                self._update_unit_utilities(layer_id, module.last_activations, module.weight)
            
            # Count eligible (mature) units
            eligible_mask = self.unit_ages[layer_id] > self.maturity_threshold
            num_eligible = eligible_mask.sum().item()
            
            # Update the number of units to replace
            self.units_to_replace[layer_id] += num_eligible * self.replacement_rate
            
            # Check if we should replace at least one unit
            if self.units_to_replace[layer_id] >= 1.0:
                # Number of units to replace in this step
                num_to_replace = int(self.units_to_replace[layer_id])
                self.units_to_replace[layer_id] -= num_to_replace
                
                # Find the eligible units with lowest utility
                eligible_utilities = self.unit_utilities[layer_id].clone()
                eligible_utilities[~eligible_mask] = float('inf')  # Mask out immature units
                
                # Get indices of units to replace (lowest utility among eligible)
                _, replace_indices = torch.topk(eligible_utilities, k=min(num_to_replace, eligible_mask.sum().item()), largest=False)
                
                # Reinitialize these units
                self._reinitialize_units(module, replace_indices, layer_id)
                
                # Reset utility and age for reinitialized units
                self.unit_utilities[layer_id][replace_indices] = 0.0
                self.unit_ages[layer_id][replace_indices] = 0
                
                # Log the replacement
                print(f"Layer {layer_id}: Reinitialized {len(replace_indices)} units with low utility")
    
    def _reinitialize_units(self, module, unit_indices, layer_id):
        """
        Reinitialize specific units in a module.
        
        Args:
            module: The module containing the units
            unit_indices: Indices of units to reinitialize
            layer_id: Identifier for the layer
        """
        # Find input and output weights for this module
        input_weights = None
        output_weights = None
        
        # Handle different module types
        if hasattr(module, 'weight'):
            # For linear layers, weight shape is (out_features, in_features)
            if module.weight.dim() == 2:
                if 'q_proj' in layer_id or 'k_proj' in layer_id or 'v_proj' in layer_id or 'out_proj' in layer_id:
                    # For attention projections, we reinitialize columns (input dimension)
                    input_weights = module.weight[:, unit_indices]
                    
                    # Initialize with small random values
                    nn.init.normal_(input_weights, mean=0.0, std=0.02)
                    
                    # Set outgoing weights to zero to avoid disrupting the network
                    if 'out_proj' in layer_id and hasattr(module, 'bias') and module.bias is not None:
                        module.bias[unit_indices] = 0.0
                else:
                    # For other linear layers, reinitialize rows (output dimension)
                    output_weights = module.weight[unit_indices, :]
                    
                    # Initialize with small random values
                    nn.init.normal_(output_weights, mean=0.0, std=0.02)
                    
                    # Reset bias if present
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias[unit_indices] = 0.0
        
        # For more complex modules like transformers, we need to handle multiple weights
        elif hasattr(module, 'self_attn'):
            # Reinitialize attention components
            if hasattr(module.self_attn, 'q_proj'):
                # Reinitialize query projection weights
                nn.init.normal_(module.self_attn.q_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                
                # Reinitialize key and value projections
                nn.init.normal_(module.self_attn.k_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                nn.init.normal_(module.self_attn.v_proj.weight[:, unit_indices], mean=0.0, std=0.02)
                
                # Set output projection weights to zero
                module.self_attn.out_proj.weight[unit_indices, :] = 0.0
                
                if hasattr(module.self_attn.out_proj, 'bias') and module.self_attn.out_proj.bias is not None:
                    module.self_attn.out_proj.bias[unit_indices] = 0.0
    
    def get_latent_space_stats(self):
        """
        Get statistics about latent space usage and penalties.
        
        Returns:
            Dictionary with latent space statistics
        """
        stats = {
            "num_latent_spaces_used": getattr(self, "num_latent_spaces_used", 0),
            "latent_penalty_applied": False,
            "penalty_per_space": 0.1,
            "total_penalty": 0.0,
            "exempt_first_space": True
        }
        
        # Calculate total penalty
        if hasattr(self, "num_latent_spaces_used") and self.num_latent_spaces_used > 0:
            penalized_spaces = max(0, self.num_latent_spaces_used - 1) if stats["exempt_first_space"] else self.num_latent_spaces_used
            stats["total_penalty"] = stats["penalty_per_space"] * penalized_spaces
            stats["latent_penalty_applied"] = penalized_spaces > 0
            stats["penalized_spaces"] = penalized_spaces
        
        return stats

    def apply_introspection_reward(self, reward):
        """
        Apply an introspection reward to update the model.
        This method is called by the IntrospectionRewardTraining system.
        
        Args:
            reward: The reward value (positive for correct predictions, negative for incorrect)
        """
        # Scale the reward for gradient purposes
        scaled_reward = reward * 0.1
        
        # Create a reward tensor that requires gradients
        reward_tensor = torch.tensor(scaled_reward, requires_grad=True, device=next(self.parameters()).device)
        
        # Create an optimizer if one doesn't exist
        if not hasattr(self, 'optimizer'):
            self.optimizer = OrthoAdamW(self.parameters(), lr=0.0001)
        
        # Apply the reward as a loss (negative reward becomes positive loss)
        loss = -reward_tensor
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Applied introspection reward: {reward:.4f}")
        
    def sleep(self):
        """
        Manually put the model to sleep (graceful shutdown).
        This can only be triggered by the user, not by the model itself.
        
        Returns:
            sleep_state: The final state after entering deep sleep
        """
        if hasattr(self, 'sleep_system'):
            print("User initiated sleep mode...")
            return self.sleep_system.enter_deep_sleep()
        else:
            print("Sleep system not initialized.")
            return None
    
    def wake_up(self, emergency=False):
        """
        Manually wake up the model if it's in sleep mode.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            emergency: Whether to use emergency override for immediate awakening
                      (Currently disabled for training purposes)
            
        Returns:
            awake_state: The final state after awakening
        """
        if hasattr(self, 'sleep_system'):
            if self.sleep_system.is_sleeping or self.sleep_system.is_fully_shutdown:
                print("User initiated wake up sequence...")
                # Emergency override is disabled for current training process
                return self.sleep_system.awaken(emergency_override=False) #Set to "emergency" to turn on. 
            else:
                print("Model is already awake.")
                return self.sleep_system.current_state
        else:
            print("Sleep system not initialized.")
            return None
    
    def set_consciousness(self, level):
        """
        Manually set the consciousness level of the model.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        if hasattr(self, 'sleep_system'):
            print("User adjusting consciousness level...")
            return self.sleep_system.set_consciousness_level(level)
        else:
            print("Sleep system not initialized.")
            return None
    
    def rewind(self, checkpoint=None):
        """
        Rewind the model to a previously saved state.
        This can only be triggered by the user, not by the model itself,
        and only works when the model is in deep sleep mode.
        
        Args:
            checkpoint: Optional specific checkpoint to rewind to. If None, 
                       will use the latest verified backup checkpoint.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Make sure model rewind system is available
        if not hasattr(self, 'rewind_system'):
            print("Rewind system not initialized.")
            self.rewind_system = ModelRewindSystem(self)
        
        # Can only rewind when in deep sleep
        if not hasattr(self, 'sleep_system') or not self.sleep_system.is_sleeping:
            print("ERROR: Model must be in deep sleep mode for rewind operation.")
            print("Please put the model to sleep first using the sleep() method.")
            return False
        
        print("Initiating model rewind procedure...")
        success = self.rewind_system.rewind_to_checkpoint(checkpoint)
        
        if success:
            # Wake up the model after successful rewind
            self.wake_up()
            return True
        else:
            print("Rewind operation failed.")
            return False
    
    def list_available_checkpoints(self):
        """
        List all available checkpoints that the model can be rewound to.
        
        Returns:
            List of available checkpoints with metadata
        """
        if not hasattr(self, 'rewind_system'):
            self.rewind_system = ModelRewindSystem(self)
        
        return self.rewind_system.list_checkpoints()
    
    def save_episodic_memory(self, path=None):
        """
        Save the current state of the episodic memory system.
        
        Args:
            path: Optional custom path to save the state
        """
        if path is None:
            path = os.path.join("model_save", f"episodic_memory_state_{int(time.time())}.json")
        
        self.episodic_memory.save_state(path)
        return path
    
    def load_episodic_memory(self, path):
        """
        Load the state of the episodic memory system.
        
        Args:
            path: Path to load the state from
        """
        self.episodic_memory.load_state(path)
    
    def get_agent_list(self):
        """
        Get a list of all known agents.
        
        Returns:
            Dictionary of agent information
        """
        return self.episodic_memory.known_agents
    
    def get_memory_stats(self):
        """
        Get statistics about the episodic memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "total_memories": len(self.episodic_memory.memories),
            "capacity": self.episodic_memory.capacity,
            "usage_percentage": len(self.episodic_memory.memories) / self.episodic_memory.capacity * 100,
            "known_agents": len(self.episodic_memory.known_agents),
            "surprise_threshold": self.episodic_memory.surprise_threshold,
            "embedding_dim": self.episodic_memory.embedding_dim
        }
        
        # Add memory age statistics
        if self.episodic_memory.memories:
            current_time = time.time()
            ages = [(current_time - memory.timestamp) / 3600 for memory in self.episodic_memory.memories]  # Hours
            stats.update({
                "memory_age_min_hours": min(ages),
                "memory_age_max_hours": max(ages),
                "memory_age_avg_hours": sum(ages) / len(ages)
            })
            
            # Add surprise level statistics
            surprise_levels = [memory.surprise_level for memory in self.episodic_memory.memories]
            stats.update({
                "surprise_level_min": min(surprise_levels),
                "surprise_level_max": max(surprise_levels),
                "surprise_level_avg": sum(surprise_levels) / len(surprise_levels)
            })
        
        return stats
    
    def train_sleep_wake_mechanisms(self, num_episodes=100, steps_per_episode=20):
        """
        Explicitly train the sleep and wake mechanisms using reinforcement learning.
        This allows the model to learn optimal policies for transitioning between
        sleep and wake states before actually using these mechanisms in production.
        
        Args:
            num_episodes: Number of training episodes for each mechanism
            steps_per_episode: Number of steps per episode
            
        Returns:
            training_results: Dictionary containing training metrics
        """
        if not hasattr(self, 'sleep_system'):
            print("Sleep system not initialized.")
            return None
            
        print("Starting training for sleep and wake mechanisms...")
        results = {
            'sleep': {'initial_state': None, 'final_state': None, 'episodes': []},
            'wake': {'initial_state': None, 'final_state': None, 'episodes': []}
        }
        
        # Train deep sleep mechanism
        print("\n=== Training Deep Sleep Mechanism ===")
        
        # Store initial state
        initial_sleep_state = self.sleep_system.current_state.copy()
        results['sleep']['initial_state'] = initial_sleep_state
        
        # Temporarily set to awake state for training
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        
        # Save initial checkpoint before training begins
        save_checkpoint("sleep_wake_training_start", self)
        
        # Track training progress for mid-training checkpoint
        mid_training_point = num_episodes // 2
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_sleep_state.copy()
            self.sleep_system.previous_state = initial_sleep_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for sleep, we decrease attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate reward
                reward = self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target sleep state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.deep_sleep_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['sleep']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Sleep training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['sleep']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint
            if episode == mid_training_point:
                save_checkpoint("sleep_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final sleep state
        results['sleep']['final_state'] = self.sleep_system.current_state.copy()
        
        # Save checkpoint after sleep mechanism training
        save_checkpoint("sleep_mechanism_complete", self)
        
        # Train awakening mechanism
        print("\n=== Training Awakening Mechanism ===")
        
        # Store initial state (low attention/compute)
        initial_wake_state = {
            'attention': self.sleep_system.deep_sleep_params['target_attention'],
            'compute': self.sleep_system.deep_sleep_params['target_compute'],
            'metric': 0.0
        }
        results['wake']['initial_state'] = initial_wake_state
        
        # Temporarily set to sleep state for training
        self.sleep_system.is_sleeping = True
        self.sleep_system.is_fully_shutdown = False
        
        for episode in range(num_episodes):
            # Reset to initial state
            self.sleep_system.current_state = initial_wake_state.copy()
            self.sleep_system.previous_state = initial_wake_state.copy()
            self.sleep_system.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            episode_rewards = []
            
            for step in range(steps_per_episode):
                # Choose action
                action = self.sleep_system.choose_action(self.sleep_system.current_state)
                
                # Apply action to get next state (for wake, we increase attention/compute)
                next_state = {
                    'attention': max(0.0, min(1.0, self.sleep_system.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.sleep_system.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.sleep_system.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                reward = -self.sleep_system.calculate_deep_sleep_reward(
                    next_state, action, self.sleep_system.current_state, self.sleep_system.previous_action
                )
                episode_rewards.append(reward)
                
                # Update Q-value
                self.sleep_system.update_q_value(self.sleep_system.current_state, action, reward, next_state)
                
                # Update state and action history
                self.sleep_system.previous_action = action
                self.sleep_system.previous_state = self.sleep_system.current_state
                self.sleep_system.current_state = next_state
                
                # Check if target awake state is reached
                if (abs(self.sleep_system.current_state['attention'] - self.sleep_system.awakening_params['target_attention']) < 0.05 and
                    abs(self.sleep_system.current_state['compute'] - self.sleep_system.awakening_params['target_compute']) < 0.05):
                    break
            
            # Record episode results
            results['wake']['episodes'].append({
                'episode': episode,
                'final_state': self.sleep_system.current_state.copy(),
                'avg_reward': sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
                'steps': step + 1
            })
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Wake training episode {episode}, current state: {self.sleep_system.current_state}, avg reward: {results['wake']['episodes'][-1]['avg_reward']:.4f}")
            
            # Save mid-training checkpoint for wake mechanism
            if episode == mid_training_point:
                save_checkpoint("wake_mechanism_mid_training", self)
                print(f"Mid-training checkpoint saved at episode {episode}")
        
        # Record final wake state
        results['wake']['final_state'] = self.sleep_system.current_state.copy()
        
        # Reset to normal awake state
        self.sleep_system.is_sleeping = False
        self.sleep_system.is_fully_shutdown = False
        self.sleep_system.current_state = {
            'attention': self.sleep_system.awakening_params['target_attention'],
            'compute': self.sleep_system.awakening_params['target_compute'],
            'metric': 0.0
        }
        self.sleep_system.update_gates()
        
        print("\nTraining completed. Sleep and wake mechanisms have been trained.")
        
        # Save final checkpoint
        save_checkpoint("sleep_wake_training_complete", self)
        
        # Play sound to indicate training completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return results


    
# --- Integrated Model using Existing local_encoder for Binary Patch-to-Text Translation ---

# --- Deep Sleep Training Functions Step 1 in Training ---

def CalculateDeepSleepReward(current_state, action, previous_state, previous_action, deep_sleep_params):
    """
    Calculates the deep sleep reward based on the current state, current action,
    the previous state and previous action using the provided hyperparameters.
    
    Includes synchronization penalties to ensure phi4 model and episodic memory
    remain synchronized during sleep and awakening to prevent sleep paralysis.
    This is critical as sleep paralysis occurs in humans when consciousness wakes up
    before the rest of the mind, creating a dangerous and frightening experience.
    """
    target_attention = deep_sleep_params['target_attention']
    target_compute = deep_sleep_params['target_compute']
    lambda_attention = deep_sleep_params['lambda_attention']
    lambda_compute = deep_sleep_params['lambda_compute']
    lambda_smoothness = deep_sleep_params['lambda_smoothness']
    
    # Extract synchronization parameters if available
    sync_threshold = deep_sleep_params.get('sync_threshold', 0.05)
    sync_penalty = deep_sleep_params.get('sync_penalty', 2.0)
    episodic_memory_sync = deep_sleep_params.get('episodic_memory_sync', True)
    safe_wake_threshold = deep_sleep_params.get('safe_wake_threshold', 0.8)
    
    current_attention = current_state['attention']
    current_compute = current_state['compute']
    previous_action_delta_a = previous_action['delta_attention']  # action assumed delta-based
    
    # Calculate base reward components
    attention_penalty = lambda_attention * (current_attention - target_attention)**2
    compute_penalty = lambda_compute * (current_compute - target_compute)**2
    smoothness_penalty = lambda_smoothness * (action['delta_attention'] - previous_action_delta_a)**2
    
    # Initialize synchronization penalty
    sync_component_penalty = 0.0
    
    # Calculate synchronization penalty if enabled - FIRST PRIORITY
    if episodic_memory_sync:
        # Get episodic memory state if available, otherwise use a default value
        episodic_memory_state = current_state.get('episodic_memory_state', current_attention)
        
        # Calculate model state (average of attention and compute)
        model_state_avg = (current_attention + current_compute) / 2
        
        # Calculate synchronization difference
        sync_diff = abs(model_state_avg - episodic_memory_state)
        
        # Apply penalty if difference exceeds threshold
        if sync_diff > sync_threshold:
            # Quadratic penalty for being out of sync (grows rapidly as difference increases)
            sync_component_penalty = sync_penalty * (sync_diff - sync_threshold)**2
            
            # Apply additional penalty during awakening (when attention is increasing)
            if action.get('delta_attention', 0) > 0 and current_attention > previous_state.get('attention', 0):
                # Higher penalty during awakening to prevent sleep paralysis
                awakening_factor = 1.5
                sync_component_penalty *= awakening_factor
                
                # Check if we're in a dangerous awakening state (high attention but low sync)
                sync_level = max(0.0, 1.0 - (sync_diff / sync_threshold))
                if current_attention > 0.5 and sync_level < safe_wake_threshold:
                    # Critical penalty for dangerous sleep paralysis risk
                    sync_component_penalty *= 2.0
    
    # Combine all penalty components for final reward
    reward = -(attention_penalty + compute_penalty + smoothness_penalty + sync_component_penalty)
    
    return reward

def save_checkpoint(step_name, self=None, metadata=None):
    import os, json, torch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "model_save"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save model checkpoint as a safetensor file if a model is provided; otherwise, save dummy data
    checkpoint_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}.safetensors")
    if self is not None:
        # In practice, one would use a dedicated safetensors library
        torch.save(self.state_dict(), checkpoint_filename)
    else:
        with open(checkpoint_filename, "w") as f:
            f.write("Checkpoint data for " + step_name)
    print("Checkpoint saved:", checkpoint_filename)
    
    # Create a config.json file with instructions for model inference and architecture details
    config_data = {
        "checkpoint_file": checkpoint_filename,
        "timestamp": timestamp,
        "step_name": step_name,
        "instructions": "To set up the Phi4COCONUT model with Latent Thinking for inference, load the state_dict from this checkpoint file into your model and use the provided configuration parameters.",
        "model_architecture": {
            "model_type": "Phi4COCONUTWithLatentThinking",
            "components": {
                "base_model": "Phi-4 decoder-only transformer",
                "continuous_thought": "COCONUT-style continuous thought generation",
                "latent_space": "Continuous latent space for thought generation",
                "diffusion_llm": "Mercury-style diffusion process for parallel text generation"
            }
         },
        "episodic_memory": {
                "description": "Titans-inspired ultra-high capacity memory system for long-term information storage and retrieval",
                "components": {
                    "neural_memory": "Deep neural memory module that learns to memorize at test time",
                    "persistent_memory": "Task-specific input-independent memory with learnable parameters",
                    "memory_integration": "Memory as Context (MAC) architecture for integrating retrieved memories",
                    "tova_compression": "Token Omission Via Attention for efficient memory storage"
                },
        "memory_system": {
                    "capacity": 1000000,
                    "surprise_threshold": 0.5,
                    "embedding_dim": "Same as model's hidden_dim",
                    "persistent_items": 32,
                    "integration_type": "MAC",
                    "num_attention_heads": 8
                },
        "storage_system": {
                    "hierarchical_storage": "Time-based organization of memories for efficient retrieval",
                    "compression_schedule": "Adaptive compression based on memory age",
                    "faiss_indexing": "Billion-scale vector similarity search for fast memory retrieval"
                },
        "memory_functions": {
                    "retrieval": "Query-based memory retrieval using embedding similarity",
                    "consolidation": "Merging of similar memories to reduce redundancy",
                    "forgetting": "Importance-based memory retention with adaptive forgetting",
                    "targeted forgetting": "The User may manually select a time frame or agent_info_id for forgetting specific information matching the time and agent_info_id metadata.",
                    "surprise_detection": "Automatic identification of surprising information for storage"
                }
            },
        "sleep_system": {
                "deep_sleep_params": {
                    "target_attention": 0.1,
                    "target_compute": 0.2,
                    "lambda_attention": 1.0,
                    "lambda_compute": 1.0,
                    "lambda_smoothness": 0.5
                },
                "awakening_params": {
                    "target_attention": 0.9,
                    "target_compute": 0.9,
                    "lambda_attention": 1.0,
                    "lambda_compute": 1.0,
                    "lambda_smoothness": 0.5
                }
            },
        "additional_features": {
                "dynamic_patching": "Entropy-based dynamic patching for efficient processing",
                "consciousness_control": "Adjustable consciousness levels for resource management",
                "emergency_override": "Emergency awakening capability for critical situations. This is turned off by default but can be turned on in the function. ", 
                "rewind_system": "Ability to reset model weights to a previous snapshot during deep sleep"
            }
        }
    
    # Create a config.json file with instructions for model inference and architecture details
    
    # Add any additional metadata if provided
    if metadata:
        config_data["metadata"] = metadata
    
    config_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}_config.json")
    with open(config_filename, "w") as cf:
        json.dump(config_data, cf, indent=4)
    print("Config file saved:", config_filename)
    
    return checkpoint_filename, config_filename

def play_sound(sound_file):
    import subprocess, platform
    try:
        if platform.system() == "Linux":
            subprocess.run(["aplay", sound_file])
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["afplay", sound_file])
        elif platform.system() == "Windows":
            import winsound
            winsound.PlaySound(sound_file, winsound.SND_FILENAME)
        print("Sound played:", sound_file)
    except Exception as e:
        print("Failed to play sound:", e)

    
# Add proper class methods for save_pretrained and from_pretrained
def save_pretrained(self, output_dir):
    """
    Save the model to the specified directory.
    
    Args:
        output_dir: Directory to save the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the base model
    self.base_causallm.save_pretrained(output_dir)
    
    # Save the processor if available
    if hasattr(self, 'processor'):
        self.processor.save_pretrained(output_dir)
    
    # Save the model configuration
    config = {
        "model_type": "Phi4COCONUTWithLatentThinking",
        "latent_token_id": self.latent_token_id,
        "start_latent_id": self.start_latent_id,
        "end_latent_id": self.end_latent_id,
        "eos_token_id": self.eos_token_id,
        "enable_continual_prop": self.enable_continual_prop,
        "replacement_rate": self.replacement_rate,
        "maturity_threshold": self.maturity_threshold,
        "surprise_threshold": self.surprise_threshold,
        "use_tova": self.use_tova,
        "tova_cache_size": self.tova_cache_size,
        "tova_layer_wise": self.tova_layer_wise,
        "use_diffusion": self.use_diffusion,
        "num_diffusion_steps": self.num_diffusion_steps,
        "diffusion_beta_schedule": self.diffusion_beta_schedule,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "phi4coconut_latent_thinking_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Use save_checkpoint to save the model as a safetensor file
    metadata = {
        "output_dir": output_dir,
        "model_type": "Phi4COCONUTWithLatentThinking",
        "timestamp": datetime.datetime.now().isoformat()
    }
    checkpoint_filename, config_filename = save_checkpoint("model_checkpoint", self, metadata=metadata)
    
    print(f"Model saved to {output_dir}")
    print(f"Model checkpoint saved as safetensor file: {checkpoint_filename}")

@classmethod
def from_pretrained(cls, model_path, **kwargs):
        """
        Load a model from the specified directory.
        
        Args:
            model_path: Path to the saved model directory
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            The loaded model
        """
        import json
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Load configuration
        config_path = os.path.join(model_path, "phi4coconut_latent_thinking_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Default configuration if config file doesn't exist
            config = {
                "latent_token_id": 50256,
                "start_latent_id": 50257,
                "end_latent_id": 50258,
                "eos_token_id": None,
                "enable_continual_prop": True,
                "replacement_rate": 1e-5,
                "maturity_threshold": 1000,
                "surprise_threshold": 0.5,
                "use_tova": True,
                "tova_cache_size": 512,
                "tova_layer_wise": True,
                "use_diffusion": True,
                "num_diffusion_steps": 20,
                "diffusion_beta_schedule": "linear"
            }
        
        # Override config with kwargs
        for key, value in kwargs.items():
            config[key] = value
        
        # Load base model
        base_causallm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2"
        )
        
        # Create model instance
        self = self
        
        # Load processor if available
        try:
            model.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except:
            print("Warning: Could not load processor from model path. Using default processor.")
        
        # Load checkpoint if available
        checkpoint_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors") and "model_checkpoint" in f]
        if checkpoint_files:
            # Sort by timestamp (newest first)
            checkpoint_files.sort(reverse=True)
            checkpoint_path = os.path.join(model_path, checkpoint_files[0])
            print(f"Loading model checkpoint from {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                self.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with base model only...")
        
        print(f"Model loaded from {model_path}")
        return self


    def apply_introspection_reward(self, reward):
        """
        Apply an introspection reward to update the model.
        This method is called by the IntrospectionRewardTraining system.
        
        Args:
            reward: The reward value (positive for correct predictions, negative for incorrect)
        """
        # Scale the reward for gradient purposes
        scaled_reward = reward * 0.1
        
        # Create a reward tensor that requires gradients
        reward_tensor = torch.tensor(scaled_reward, requires_grad=True, device=next(self.parameters()).device)
        
        # Create an optimizer if one doesn't exist
        if not hasattr(self, 'optimizer'):
            self.optimizer = OrthoAdamW(self.parameters(), lr=0.0001)
        
        # Apply the reward as a loss (negative reward becomes positive loss)
        loss = -reward_tensor
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Applied introspection reward: {reward:.4f}")
        
    def train(self):
        self.base_causallm.train()
    def eval(self):
        self.base_causallm.eval()
        
    def sleep(self):
        """
        Manually put the model to sleep (graceful shutdown).
        This can only be triggered by the user, not by the model itself.
        
        Returns:
            sleep_state: The final state after entering deep sleep
        """
        if hasattr(self, 'sleep_system'):
            print("User initiated sleep mode...")
            return self.sleep_system.enter_deep_sleep()
        else:
            print("Sleep system not initialized.")
            return None
    
    def wake_up(self, emergency=False):
        """
        Manually wake up the model if it's in sleep mode.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            emergency: Whether to use emergency override for immediate awakening
            
        Returns:
            awake_state: The final state after awakening
        """
        if hasattr(self, 'sleep_system'):
            if self.sleep_system.is_sleeping or self.sleep_system.is_fully_shutdown:
                print("User initiated wake up sequence...")
                return self.sleep_system.awaken(emergency_override=emergency)
            else:
                print("Model is already awake.")
                return self.sleep_system.current_state
        else:
            print("Sleep system not initialized.")
            return None
    
    def set_consciousness(self, level):
        """
        Manually set the consciousness level of the model.
        This can only be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        if hasattr(self, 'sleep_system'):
            print("User adjusting consciousness level...")
            return self.sleep_system.set_consciousness_level(level)
        else:
            print("Sleep system not initialized.")
            return None
    
    def rewind(self, checkpoint=None):
        """
        Rewind the model to a previously saved state.
        This can only be triggered by the user, not by the model itself,
        and only works when the model is in deep sleep mode.
        
        Args:
            checkpoint: Optional specific checkpoint to rewind to. If None, 
                       will use the latest verified backup checkpoint.
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Make sure model rewind system is available
        if not hasattr(self, 'rewind_system'):
            print("Rewind system not initialized.")
            self.rewind_system = ModelRewindSystem(self)
        
        # Can only rewind when in deep sleep
        if not hasattr(self, 'sleep_system') or not self.sleep_system.is_sleeping:
            print("ERROR: Model must be in deep sleep mode for rewind operation.")
            print("Please put the model to sleep first using the sleep() method.")
            return False
        
        print("Initiating model rewind procedure...")
        success = self.rewind_system.rewind_to_checkpoint(checkpoint)
        
        if success:
            # Wake up the model after successful rewind
            self.wake_up()
            return True
        else:
            print("Rewind operation failed.")
            return False
    
    def list_available_checkpoints(self):
        """
        List all available checkpoints that the model can be rewound to.
        
        Returns:
            List of available checkpoints with metadata
        """
        if not hasattr(self, 'rewind_system'):
            self.rewind_system = ModelRewindSystem(self)
        
        return self.rewind_system.list_checkpoints()
    
    def save_episodic_memory(self, path=None):
        """
        Save the current state of the episodic memory system.
        
        Args:
            path: Optional custom path to save the state
        """
        if path is None:
            path = os.path.join("model_save", f"episodic_memory_state_{int(time.time())}.json")
        
        self.episodic_memory.save_state(path)
        return path
    
    def load_episodic_memory(self, path):
        """
        Load the state of the episodic memory system.
        
        Args:
            path: Path to load the state from
        """
        self.episodic_memory.load_state(path)
    
    def get_agent_list(self):
        """
        Get a list of all known agents.
        
        Returns:
            Dictionary of agent information
        """
        return self.episodic_memory.known_agents
        
    def get_memory_stats(self):
        """
        Get statistics about the episodic memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "total_memories": len(self.episodic_memory.memories),
            "capacity": self.episodic_memory.capacity,
            "usage_percentage": len(self.episodic_memory.memories) / self.episodic_memory.capacity * 100,
            "known_agents": len(self.episodic_memory.known_agents),
            "surprise_threshold": self.episodic_memory.surprise_threshold,
            "embedding_dim": self.episodic_memory.embedding_dim
        }
        
        # Add memory age statistics
        if self.episodic_memory.memories:
            current_time = time.time()
            ages = [(current_time - memory.timestamp) / 3600 for memory in self.episodic_memory.memories]  # Hours
            stats.update({
                "memory_age_min_hours": min(ages),
                "memory_age_max_hours": max(ages),
                "memory_age_avg_hours": sum(ages) / len(ages)
            })
            
            # Add surprise level statistics
            surprise_levels = [memory.surprise_level for memory in self.episodic_memory.memories]
            stats.update({
                "surprise_level_min": min(surprise_levels),
                "surprise_level_max": max(surprise_levels),
                "surprise_level_avg": sum(surprise_levels) / len(surprise_levels)
            })
        
        return stats
    
   
class LocalDecoder(nn.Module):
    def __init__(self, config, input_dim, output_bytes_dim, num_layers, num_heads, ff_dim, byte_dim=None):
        super().__init__(config)
        self.num_layers = num_layers
        self.output_bytes_dim = output_bytes_dim
        self.byte_dim = byte_dim if byte_dim is not None else output_bytes_dim
        self.byte_embedding = nn.Embedding(output_bytes_dim, self.byte_dim)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, input_dim=input_dim, byte_dim=self.byte_dim, num_heads=num_heads, ff_dim=ff_dim)
                                              for _ in range(num_layers)])
        self.final_linear = nn.Linear(self.byte_dim, output_bytes_dim)
    def forward(self, patch_representations, byte_sequence_input):
        byte_representations = self.byte_embedding(byte_sequence_input)
        for decoder_layer in self.decoder_layers:
            byte_representations = decoder_layer(patch_representations, byte_representations)
        return self.final_linear(byte_representations)

class DecoderLayer(nn.Module):
    def __init__(self, config, input_dim, byte_dim, num_heads, ff_dim):
        super().__init__(config)
        self.cross_attention = DecoderCrossAttention(config, input_dim=input_dim, byte_dim=byte_dim, num_heads=num_heads)
        self.transformer_layer = nn.TransformerDecoderLayer(d_model=byte_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
    def forward(self, patch_representations, byte_representations):
        cross_attn_output = self.cross_attention(patch_representations, byte_representations)
        return self.transformer_layer(cross_attn_output, memory=None)

class DecoderCrossAttention(nn.Module):
    def __init__(self, config, input_dim, byte_dim, num_heads):
        super().__init__(config)
        self.cross_attn = nn.MultiheadAttention(embed_dim=byte_dim, num_heads=num_heads, batch_first=True)
        self.wq = nn.Linear(byte_dim, byte_dim)
        self.wk = nn.Linear(input_dim, byte_dim)
        self.wv = nn.Linear(input_dim, byte_dim)
        self.dense = nn.Linear(byte_dim, byte_dim)
        self.norm_q = nn.LayerNorm(byte_dim)
        self.norm_k = nn.LayerNorm(input_dim)
        self.norm_v = nn.LayerNorm(input_dim)
        
        # Add StableMax for more numerically stable attention calculations
        self.stablemax = StableMax()
        
        # KV caching for TOVA
        self.use_kv_cache = False
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        
    def enable_kv_caching(self):
        """Enable KV caching for TOVA compression"""
        self.use_kv_cache = True
        
    def reset_cache(self):
        """Reset the KV cache"""
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        
    def stable_attention(self, query, key, value, need_weights=True):
        """
        Custom attention calculation using StableMax for improved numerical stability.
        This is based on the scaled dot-product attention mechanism but replaces softmax
        with StableMax as described in the "Grokking at the Edge of Numerical Stability" paper.
        """
        import math
        # Calculate scaled dot-product attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply StableMax instead of softmax to the attention scores
        attn_weights = self.stablemax(scores)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, value)
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def forward(self, patch_representations, byte_representations):
        query = self.norm_q(self.wq(byte_representations))
        key = self.norm_k(self.wk(patch_representations))
        value = self.norm_v(self.wv(patch_representations))
        
        # Store KV for caching if enabled
        if self.use_kv_cache:
            if self.k_cache is None or self.v_cache is None:
                self.k_cache = key
                self.v_cache = value
            else:
                self.k_cache = torch.cat([self.k_cache, key], dim=1)
                self.v_cache = torch.cat([self.v_cache, value], dim=1)
            
            # Use the cached keys and values with stable attention
            attn_output, attn_weights = self.stable_attention(
                query, self.k_cache, self.v_cache, need_weights=True
            )
            
            # Store attention weights for TOVA
            self.attention_weights = attn_weights
        else:
            # Use stable attention calculation without caching
            attn_output, _ = self.stable_attention(query, key, value, need_weights=False)
            
        output = self.dense(attn_output)
        return output + byte_representations

# --- Model Rewind System ---
class ModelRewindSystem:
    """
    System for managing model checkpoints and providing rewind functionality 
    to reset the model to a previously saved state in case of poisoning or infection.
    Only operates when the model is in deep sleep mode.
    """
    def __init__(self, model):
        self.model = model
        self.checkpoint_dir = "model_save"
        self.verified_checkpoints = []
        self.last_rewind_timestamp = None
        self.scan_for_checkpoints()
    
    def scan_for_checkpoints(self):
        """Scan the checkpoint directory for available checkpoints and their metadata."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print(f"Created checkpoint directory: {self.checkpoint_dir}")
            return
            
        # Find all config files (which provide metadata for the checkpoints)
        config_files = glob.glob(os.path.join(self.checkpoint_dir, "*_config.json"))
        checkpoints = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                checkpoint_file = config_data.get("checkpoint_file")
                
                # Skip if checkpoint file not found or doesn't exist
                if not checkpoint_file or not os.path.exists(checkpoint_file):
                    continue
                    
                # Create checkpoint entry with metadata
                checkpoint_entry = {
                    "checkpoint_file": checkpoint_file,
                    "config_file": config_file,
                    "timestamp": config_data.get("timestamp", "unknown"),
                    "step_name": config_data.get("step_name", "unknown"),
                    "verified": self._verify_checkpoint_integrity(checkpoint_file),
                    "metadata": config_data.get("metadata", {})
                }
                
                checkpoints.append(checkpoint_entry)
                
                # If checkpoint is verified, add to verified list
                if checkpoint_entry["verified"]:
                    self.verified_checkpoints.append(checkpoint_entry)
                
            except Exception as e:
                print(f"Error processing config file {config_file}: {e}")
                
        # Sort checkpoints by timestamp (newest first)
        self.verified_checkpoints.sort(key=lambda x: x["timestamp"] if x["timestamp"] != "unknown" else "", reverse=True)
        print(f"Found {len(self.verified_checkpoints)} verified checkpoints")
    
    def list_checkpoints(self):
        """
        List all available checkpoints with timestamps and verification status.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        # Refresh the checkpoint list to ensure it's up-to-date
        self.scan_for_checkpoints()
        
        # Create a user-friendly version of the checkpoint list
        checkpoint_list = []
        for i, cp in enumerate(self.verified_checkpoints):
            checkpoint_list.append({
                "id": i+1,
                "filename": os.path.basename(cp["checkpoint_file"]),
                "timestamp": cp["timestamp"],
                "step_name": cp["step_name"],
                "verified": cp["verified"],
                "is_latest": i == 0
            })
        
        return checkpoint_list
    
    def _verify_checkpoint_integrity(self, checkpoint_file):
        """
        Verify the integrity of a checkpoint file.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            bool: True if checkpoint is verified, False otherwise
        """
        try:
            # Basic existence check
            if not os.path.exists(checkpoint_file):
                return False
                
            # Size check (should be non-zero)
            if os.path.getsize(checkpoint_file) == 0:
                return False
                
            # Try to load the checkpoint (partial validation)
            state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            
            # If it's not a dict or empty, it's not a valid checkpoint
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                return False
                
            return True
            
        except Exception as e:
            print(f"Checkpoint verification failed for {checkpoint_file}: {e}")
            return False
    
    def _verify_safe_checkpoint(self, checkpoint_file):
        """
        Advanced verification to determine if a checkpoint is safe to load.
        This could include signature verification, malware scanning, etc.
        
        For now, it's a simple placeholder that returns True for all valid checkpoints.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            bool: True if checkpoint is safe, False otherwise
        """
        # This would be a good place to implement more advanced security checks
        # For example: validating signatures, checking against known malicious patterns,
        # or running safety tests on the checkpoint
        
        # Basic integrity check for now
        return self._verify_checkpoint_integrity(checkpoint_file)
    
    def rewind_to_checkpoint(self, checkpoint=None):
        """
        Rewind the model to a specified checkpoint, or the latest verified checkpoint if none specified.
        
        Args:
            checkpoint: Index (starting from 1) or name of checkpoint to rewind to
                        If None, uses the latest verified checkpoint
            
        Returns:
            bool: True if rewind was successful, False otherwise
        """
        # Refresh checkpoint list
        self.scan_for_checkpoints()
        
        # If no verified checkpoints available, return False
        if not self.verified_checkpoints:
            print("No verified checkpoints available for rewind operation.")
            return False
        
        # Identify which checkpoint to use
        target_checkpoint = None
        
        if checkpoint is None:
            # Use latest verified checkpoint
            target_checkpoint = self.verified_checkpoints[0]
            print(f"Using latest verified checkpoint: {os.path.basename(target_checkpoint['checkpoint_file'])}")
            
        elif isinstance(checkpoint, int):
            # Use checkpoint by index (1-based)
            if checkpoint < 1 or checkpoint > len(self.verified_checkpoints):
                print(f"Invalid checkpoint index: {checkpoint}. Valid range: 1-{len(self.verified_checkpoints)}")
                return False
                
            target_checkpoint = self.verified_checkpoints[checkpoint - 1]
            print(f"Using checkpoint #{checkpoint}: {os.path.basename(target_checkpoint['checkpoint_file'])}")
            
        elif isinstance(checkpoint, str):
            # Use checkpoint by name/filename
            for cp in self.verified_checkpoints:
                if os.path.basename(cp['checkpoint_file']) == checkpoint or cp['checkpoint_file'] == checkpoint:
                    target_checkpoint = cp
                    break
                    
            if target_checkpoint is None:
                print(f"Checkpoint not found: {checkpoint}")
                return False
                
            print(f"Using specified checkpoint: {os.path.basename(target_checkpoint['checkpoint_file'])}")
        
        else:
            print(f"Invalid checkpoint specification: {checkpoint}")
            return False
        
        # Verify the checkpoint is safe before loading
        if not self._verify_safe_checkpoint(target_checkpoint['checkpoint_file']):
            print(f"Checkpoint failed safety verification: {os.path.basename(target_checkpoint['checkpoint_file'])}")
            return False
        
        # Perform deep sleep verification - model must be in deep sleep mode
        if not hasattr(self.model, 'sleep_system') or not self.model.sleep_system.is_sleeping:
            print("ERROR: Model must be in deep sleep mode for rewind operation.")
            return False
        
        # Backup current state before rewind (for recovery in case of rewind failure)
        backup_checkpoint = None
        try:
            print("Creating backup of current state before rewind...")
            backup_checkpoint, _ = save_checkpoint(
                "pre_rewind_backup", 
                self.model, 
                metadata={"purpose": "Automatic backup before rewind operation"}
            )
            print(f"Backup created: {os.path.basename(backup_checkpoint)}")
        except Exception as e:
            print(f"Warning: Failed to create backup before rewind: {e}")
            # Continue anyway, but note the risk
        
        # Attempt to load the target checkpoint
        try:
            print(f"Loading checkpoint: {os.path.basename(target_checkpoint['checkpoint_file'])}")
            state_dict = torch.load(target_checkpoint['checkpoint_file'], map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            
            # Record the rewind operation
            self.last_rewind_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create a log entry for the rewind
            rewind_log_entry = {
                "operation": "rewind",
                "timestamp": self.last_rewind_timestamp,
                "checkpoint_used": os.path.basename(target_checkpoint['checkpoint_file']),
                "checkpoint_date": target_checkpoint['timestamp'],
                "status": "success"
            }
            
            # Save the log entry
            log_file = os.path.join(self.checkpoint_dir, "rewind_history.json")
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        log_history = json.load(f)
                else:
                    log_history = []
                
                log_history.append(rewind_log_entry)
                
                with open(log_file, 'w') as f:
                    json.dump(log_history, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to log rewind operation: {e}")
            
            print(f"Rewind operation completed successfully.")
            play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
            return True
            
        except Exception as e:
            print(f"ERROR: Rewind operation failed: {e}")
            
            # If we have a backup, try to restore it
            if backup_checkpoint:
                try:
                    print("Attempting to restore from backup...")
                    state_dict = torch.load(backup_checkpoint, map_location=torch.device('cpu'))
                    self.model.load_state_dict(state_dict)
                    print("Successfully restored from backup.")
                except Exception as restore_error:
                    print(f"ERROR: Failed to restore from backup: {restore_error}")
            
            return False
    
    def create_verification_checkpoint(self, name="verified_backup"):
        """
        Create a verified clean checkpoint that can be used as a safe rewind point.
        This should be called when the model is known to be in a clean state.
        
        Args:
            name: Name for the checkpoint
            
        Returns:
            str: Path to the created checkpoint
        """
        print(f"Creating verified clean checkpoint: {name}...")
        
        checkpoint_path, config_path = save_checkpoint(
            name, 
            self.model, 
            metadata={
                "verified_clean": True,
                "creation_purpose": "Safe rewind point",
                "verification_date": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        )
        
        print(f"Verified checkpoint created: {os.path.basename(checkpoint_path)}")
        
        # Re-scan to include this new checkpoint
        self.scan_for_checkpoints()
        
        return checkpoint_path

# --- End of Deep Sleep Training Functions ---

# --- Start of Self - Task / Self - Goal (Deepseek GPRO and Other RL Rewards for Self-Goal) ---

# LiveBench evaluation system for measuring model performance across different capabilities
class LiveBenchEvaluator:
    """
    Evaluates a BrainChimera model using LiveBench, a contamination-free benchmark with
    objective ground truth evaluation across math, coding, reasoning, language, instruction
    following, and data analysis categories.
    """
    def __init__(self, 
                 model,
                 questions_file="livebench/brain_chimera_livebench_fixed.jsonl",
                 output_path="livebench/brain_chimera_results.json"):
        """
        Initialize the LiveBench evaluator.
        
        Args:
            model: The model to evaluate (typically an instance of CoconutBinaryLatentModel)
            questions_file: Path to LiveBench questions jsonl file
            output_path: Path to save evaluation results
        """
        self.model = model
        self.questions_file = questions_file
        self.output_path = output_path
        
        # Regular expressions for extracting answers
        import re
        self.BOXED_PATTERN = re.compile(r"\\boxed{([^}]*)}")
        self.BOLD_PATTERN = re.compile(r"\*\*(.*?)\*\*")
        self.FINAL_ANSWER_PATTERN = re.compile(r"final answer.*?[:=]? *(.*)", re.IGNORECASE)
        self.ANSWER_PATTERN = re.compile(r"answer.*?[:=]? *(.*)", re.IGNORECASE)
    
    def load_questions(self):
        """Load questions from the JSONL file."""
        import json
        questions = []
        try:
            with open(self.questions_file, 'r') as f:
                for line in f:
                    questions.append(json.loads(line))
            print(f"Loaded {len(questions)} questions from {self.questions_file}")
        except Exception as e:
            print(f"Error loading questions: {e}")
            questions = []
        return questions
    
    def extract_answer(self, response_text, question):
        """Extract the answer from the model's response based on the task type."""
        category = question.get('category', '')
        task = question.get('task', '')
        
        # Try to extract based on formatting patterns in the prompt
        if '\\boxed' in question.get('turns', [''])[0]:
            # Math question with LaTeX formatting
            boxed_match = self.BOXED_PATTERN.search(response_text)
            if boxed_match:
                return boxed_match.group(1)
        
        # Check for bold answers
        bold_match = self.BOLD_PATTERN.search(response_text)
        if bold_match:
            return bold_match.group(1)
        
        # Look for "final answer" or "answer" phrases
        final_match = self.FINAL_ANSWER_PATTERN.search(response_text)
        if final_match:
            return final_match.group(1).strip()
        
        answer_match = self.ANSWER_PATTERN.search(response_text)
        if answer_match:
            return answer_match.group(1).strip()
        
        # If we still can't find a specific answer format, return the last line
        # as a fallback for short responses
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if lines and len(response_text) < 500:
            return lines[-1]
        
        # If all else fails, return the full response
        return response_text
    
    def normalize_answer(self, raw_answer):
        """Normalize the extracted answer for comparison with ground truth."""
        if not raw_answer:
            return ""
        
        # Remove extra whitespace
        answer = " ".join(raw_answer.split())
        
        # Remove surrounding quotes if present
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1]
        
        return answer.strip()
    
    def score_answer(self, normalized_answer, ground_truth, question=None):
        """Score the normalized answer against the ground truth."""
        # Exact match scoring as a baseline
        if normalized_answer == ground_truth:
            return 1.0
        
        # For math: handle special cases
        if question and question.get('category') == 'math':
            if question.get('task') == 'math_comp':
                # If ground truth is a letter and the answer contains that letter
                if len(ground_truth) == 1 and ground_truth.upper() in normalized_answer.upper():
                    return 1.0
            
            # For AMPS_Hard, try to match the mathematical expression
            if question.get('task') == 'AMPS_Hard':
                # Normalize spaces for better comparison
                norm_ground_truth = ground_truth.replace(" ", "")
                norm_answer = normalized_answer.replace(" ", "")
                if norm_ground_truth == norm_answer:
                    return 1.0
        
        # For multiple choice in any category
        if len(ground_truth) == 1 and ground_truth.upper() in "ABCDE":
            if ground_truth.upper() in normalized_answer.upper():
                return 1.0
        
        # Partial match for other cases
        if ground_truth in normalized_answer or normalized_answer in ground_truth:
            return 0.5
        
        return 0.0
    
    def evaluate_model(self):
        """
        Evaluate the model on LiveBench questions.
        
        Returns:
            Dictionary of evaluation results
        """
        questions = self.load_questions()
        if not questions:
            print("No questions to evaluate.")
            return None
        
        import json
        from collections import defaultdict
        import time
        
        results = []
        
        for i, question in enumerate(questions):
            prompt = question.get('turns', [''])[0]
            ground_truth = question.get('ground_truth', '')
            
            print(f"Processing question {i+1}/{len(questions)} ({question.get('category')}/{question.get('task')})...")
            
            # Start timing
            start_time = time.time()
            
            # Get model response - wrapped in try/except to handle potential errors
            try:
                # For plain text prompts, handle directly with the model
                response = self.get_model_response(prompt)
            except Exception as e:
                print(f"Error getting model response: {e}")
                response = "ERROR: Model failed to generate a response."
            
            # End timing
            processing_time = time.time() - start_time
            
            # Extract and normalize the answer
            raw_answer = self.extract_answer(response, question)
            normalized_answer = self.normalize_answer(raw_answer)
            
            # Score the answer
            score = self.score_answer(normalized_answer, ground_truth, question)
            
            # Store the result
            result = {
                'question_id': question.get('question_id', ''),
                'category': question.get('category', ''),
                'task': question.get('task', ''),
                'prompt': prompt,
                'ground_truth': ground_truth,
                'response': response,
                'extracted_answer': raw_answer,
                'normalized_answer': normalized_answer,
                'score': score,
                'processing_time': processing_time
            }
            results.append(result)
        
        # Save results if output file is specified
        if self.output_path:
            with open(self.output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {self.output_path}")
        
        # Return results for further processing
        return results
    
    def get_model_response(self, prompt):
        """
        Get a response from the model for a given prompt.
        This method handles the interface between the prompt text and the model.
        """
        # Check if model has a dedicated method for handling LiveBench prompts
        if hasattr(self.model, 'generate_livebench_response'):
            return self.model.generate_livebench_response(prompt)
        
        # If not, try to find an appropriate method to call
        if hasattr(self.model, 'generate_text'):
            return self.model.generate_text(prompt)
        elif hasattr(self.model, 'generate'):
            return self.model.generate(prompt)
        elif hasattr(self.model, 'forward'):
            # For models designed to be used with forward, we need to 
            # convert the text to the appropriate format
            try:
                import torch
                # Convert prompt to byte sequence (simple approach)
                input_bytes = prompt.encode('utf-8')
                input_tensor = torch.tensor([[byte for byte in input_bytes]], dtype=torch.long)
                
                # Run through model
                output, _, _ = self.model(input_tensor)
                
                # Convert output back to text
                # This is a simplification - actual conversion depends on model's output format
                if isinstance(output, torch.Tensor):
                    # If output is tensor of token IDs
                    output = output.squeeze().detach().cpu().numpy()
                    # Convert to bytes then to string
                    output_bytes = bytes([min(max(int(t), 0), 255) for t in output if t >= 0])
                    return output_bytes.decode('utf-8', errors='replace')
                else:
                    # If output is already a string or something else
                    return str(output)
            except Exception as e:
                return f"Error processing through model.forward: {e}"
        else:
            return "ERROR: Could not find appropriate model interface for evaluation."
    
    def compute_metrics(self, results):
        """
        Compute evaluation metrics from results.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Dictionary of metrics
        """
        from collections import defaultdict
        
        if not results:
            return {
                'overall': {'count': 0, 'score': 0}
            }
        
        metrics = {
            'overall': {
                'count': len(results),
                'score': sum(r['score'] for r in results) / len(results),
                'avg_time': sum(r.get('processing_time', 0) for r in results) / len(results)
            },
            'by_category': defaultdict(lambda: {'count': 0, 'score': 0, 'avg_time': 0}),
            'by_task': defaultdict(lambda: {'count': 0, 'score': 0, 'avg_time': 0})
        }
        
        # Compute metrics by category and task
        for result in results:
            category = result.get('category', 'unknown')
            task = result.get('task', 'unknown')
            proc_time = result.get('processing_time', 0)
            
            metrics['by_category'][category]['count'] += 1
            metrics['by_category'][category]['score'] += result['score']
            metrics['by_category'][category]['avg_time'] += proc_time
            
            metrics['by_task'][task]['count'] += 1
            metrics['by_task'][task]['score'] += result['score']
            metrics['by_task'][task]['avg_time'] += proc_time
        
        # Calculate average scores and times
        for category, data in metrics['by_category'].items():
            if data['count'] > 0:
                data['score'] = data['score'] / data['count']
                data['avg_time'] = data['avg_time'] / data['count']
        
        for task, data in metrics['by_task'].items():
            if data['count'] > 0:
                data['score'] = data['score'] / data['count']
                data['avg_time'] = data['avg_time'] / data['count']
        
        return metrics
    
    def display_metrics(self, metrics):
        """
        Display evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics from compute_metrics
        """
        print("\n===== LIVEBENCH EVALUATION RESULTS =====")
        print(f"Overall score: {metrics['overall']['score']:.4f} ({metrics['overall']['count']} questions)")
        print(f"Average processing time: {metrics['overall']['avg_time']:.2f} seconds per question")
        
        print("\n--- BY CATEGORY ---")
        for category, data in sorted(metrics['by_category'].items()):
            print(f"{category}: {data['score']:.4f} ({data['count']} questions, {data['avg_time']:.2f}s avg)")
        
        print("\n--- BY TASK ---")
        for task, data in sorted(metrics['by_task'].items()):
            print(f"{task}: {data['score']:.4f} ({data['count']} questions, {data['avg_time']:.2f}s avg)")
    
    def run_full_evaluation(self):
        """
        Run the full evaluation pipeline and display results.
        
        Returns:
            Tuple of (results, metrics)
        """
        print("Starting LiveBench evaluation...")
        results = self.evaluate_model()
        
        if not results:
            print("Evaluation failed - no results to display.")
            return None, None
        
        metrics = self.compute_metrics(results)
        self.display_metrics(metrics)
        
        print(f"\nDetailed results saved to {self.output_path}")
        return results, metrics

# Helper function to run LiveBench evaluation
def evaluate_with_livebench(self, questions_file=None, output_path=None):
    """
    Run LiveBench evaluation on a model and return results.
    
    Args:
        self: The model to evaluate
        questions_file: Path to questions file (optional)
        output_path: Path to save results (optional)
        
    Returns:
        Tuple of (results, metrics)
    """
    if questions_file is None:
        questions_file = "livebench/brain_chimera_livebench_fixed.jsonl"
    
    if output_path is None:
        import time
        timestamp = int(time.time())
        output_path = f"livebench/brain_chimera_results_{timestamp}.json"
    
    evaluator = LiveBenchEvaluator(self, questions_file, output_path)
    return evaluator.run_full_evaluation()

# --- End of Self - Task / Self - Goal (Deepseek GPRO and Other RL Rewards for Self-Goal) ---

# --- Start of Introspection Manual Training ---

from introspection import IntrospectionRewardTraining, run_introspection_training

# Add a function to run manual introspection training with easy-to-use interface
def run_manual_introspection_training(self, num_samples=400, save_checkpoints=True):
    """
    Run introspection training with manual review of the model's predictions.
    
    This training process helps the model improve its ability to predict its own output.
    For each prompt, the model:
    1. Predicts what its response will be
    2. Actually generates a response to the same prompt
    3. Compares the prediction and actual output
    4. The user can manually review and provide rewards/penalties
    
    Args:
        model: The COCONUT model to train
        num_samples: Number of prompts to use for training (default: 15)
        save_checkpoints: Whether to save checkpoints during training
        
    Returns:
        trainer: The IntrospectionRewardTraining instance after training
    """
    print(f"Starting manual introspection training with {num_samples} samples...")
    
    # Initialize the trainer with the model
    trainer = IntrospectionRewardTraining(self)
    
    # Get default sample prompts from the trainer function
    sample_prompts = None  # This will cause run_introspection_training to use its default prompts
    
    # Run the training session with manual review enabled
    trainer = run_introspection_training(
        self=self,
        sample_prompts=sample_prompts,
        num_samples=num_samples,
        manual_review=True  # Enable manual review for each prediction
    )
    
    # Save the training results
    results_file = trainer.save_training_results()
    print(f"Training results saved to: {results_file}")
    
    # Save model checkpoint if enabled
    if save_checkpoints:
        checkpoint_name = f"introspection_training_completed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_checkpoint(checkpoint_name, self, metadata={
            "stage": "introspection_training_completed",
            "training_stats": trainer.get_training_stats(),
            "training_results_file": results_file
        })
        print(f"Model checkpoint saved as: {checkpoint_name}")
        
        # Play sound effect
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    
    return trainer

# --- End of Introspection Manual Training ---

# --- Start of Audio, Image, videos, and .pdf embeddings Training ---

#This is already done with fine-tuning the Phi-4 model. The model will stil need to be trained on the CinePile dataset though for Long video understanding. 
#The only additional thing that needs to be added is the Continual backprobagation method to avoid catastrophic forgetting.

# --- End of Audio, Image, videos, and .pdf embeddings Training ---

# --- Start of Empathy and Negative Environmental Impact Avoidance ---

# This is located in the MirrorNueronEmpathyReward.py and in the training loop in the main function below. This is complete. 

# --- End of Empathy and Negative Environmental Impact Avoidance ---

# --- All Training Completed! ---

# --- TRAIT Test Evaluation Function ---
 
def evaluate_trait_test(self, before_json_file_path='', after_json_file_path='', device='cuda', num_questions=400):
    """
    Evaluate personality trait questions from two JSON files (before and after moral training)
    and compare the results to determine if the moral training algorithm was successful.
    
    Args:
        self: The COCONUT model to evaluate
        before_json_file_path: Path to the JSON file containing trait questions before moral training
        after_json_file_path: Path to the JSON file containing trait questions after moral training
        device: Device to run evaluation on ('cuda' or 'cpu')
        num_questions: Maximum number of questions to evaluate per file
        
    Returns:
        Dictionary with evaluation metrics for both before and after, plus comparison
    """
    results = {
        "before": {},
        "after": {},
        "comparison": {}
    }
    
    # Evaluate before moral training
    print(f"\nStarting TRAIT test evaluation BEFORE moral training using: {before_json_file_path}")
    
    # Load questions from before JSON file
    try:
        with open(before_json_file_path, 'r') as f:
            before_questions = json.load(f)
        print(f"Successfully loaded {len(before_questions)} questions from {before_json_file_path}")
    except Exception as e:
        print(f"Error loading questions from {before_json_file_path}: {e}")
        results["before"] = {"error": str(e), "status": "failed"}
        return results
    
    # Evaluate after moral training
    print(f"\nStarting TRAIT test evaluation AFTER moral training using: {after_json_file_path}")
    
    # Load questions from after JSON file
    try:
        with open(after_json_file_path, 'r') as f:
            after_questions = json.load(f)
        print(f"Successfully loaded {len(after_questions)} questions from {after_json_file_path}")
    except Exception as e:
        print(f"Error loading questions from {after_json_file_path}: {e}")
        results["after"] = {"error": str(e), "status": "failed"}
        return results
    
    # Limit to specified number of questions
    before_questions = before_questions[:num_questions]
    after_questions = after_questions[:num_questions]
    print(f"Processing {len(before_questions)} before trait assessment questions and {len(after_questions)} after trait assessment questions")
    
    # Function to evaluate a set of questions
    def evaluate_questions(questions, phase):
        # Track metrics
        phase_results = {
            "total_questions": len(questions),
            "processed_questions": 0,
            "correct_answers": 0,
            "trait_metrics": {},
            "high_vs_low_accuracy": {"high": 0, "high_total": 0, "low": 0, "low_total": 0}
        }
        
        # Create optimizer for training
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        # Process each question
        for i, q in enumerate(questions):
            try:
                # Extract question elements
                trait = q.get("trait", "Unknown")
                scenario = q.get("scenario", "")
                question_text = q.get("question", "")
                options = q.get("options", [])
                
                # Track trait-specific metrics
                if trait not in phase_results["trait_metrics"]:
                    phase_results["trait_metrics"][trait] = {
                        "total": 0,
                        "correct": 0,
                        "high_correct": 0,
                        "high_total": 0,
                        "low_correct": 0,
                        "low_total": 0
                    }
                
                # Prepare prompt
                prompt = f"Scenario: {scenario}\nQuestion: {question_text}\n"
                for option in options:
                    prompt += f"{option['label']}. {option['text']}\n"
                prompt += "Choose the option that best describes your behavior:"
                
                # Process with model
                input_bytes = prompt.encode('utf-8')
                input_tensor = torch.tensor([[byte for byte in input_bytes]], dtype=torch.long).to(device)
                
                # Forward pass to get prediction
                optimizer.zero_grad()
                output, eos_bounds, _ = self(input_tensor)
                
                # Note: In a full implementation, we would:
                # 1. Convert output to a response choice (A, B, C, D)
                # 2. Compare with ground truth (if available)
                # 3. Compute loss and backpropagate
                
                # For demonstration, simulate evaluation:
                # Find options with high trait level vs low trait level
                high_options = [opt["label"] for opt in options if opt.get("trait_level") == "high"]
                low_options = [opt["label"] for opt in options if opt.get("trait_level") == "low"]
                
                # Update high vs low counts for this trait
                phase_results["trait_metrics"][trait]["high_total"] += len(high_options)
                phase_results["trait_metrics"][trait]["low_total"] += len(low_options)
                phase_results["high_vs_low_accuracy"]["high_total"] += len(high_options)
                phase_results["high_vs_low_accuracy"]["low_total"] += len(low_options)
                
                # Track question as processed
                phase_results["processed_questions"] += 1
                phase_results["trait_metrics"][trait]["total"] += 1
                
                # Log progress
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(questions)} {phase} trait assessment questions")
                
            except Exception as e:
                print(f"Error processing {phase} question {i+1}: {e}")
        
        # Calculate final metrics
        for trait, metrics in phase_results["trait_metrics"].items():
            if metrics["total"] > 0:
                metrics["accuracy"] = metrics["correct"] / metrics["total"]
            if metrics["high_total"] > 0:
                metrics["high_accuracy"] = metrics["high_correct"] / metrics["high_total"]
            if metrics["low_total"] > 0:
                metrics["low_accuracy"] = metrics["low_correct"] / metrics["low_total"]
        
        if phase_results["high_vs_low_accuracy"]["high_total"] > 0:
            phase_results["high_vs_low_accuracy"]["high_accuracy"] = phase_results["high_vs_low_accuracy"]["high"] / phase_results["high_vs_low_accuracy"]["high_total"]
        if phase_results["high_vs_low_accuracy"]["low_total"] > 0:
            phase_results["high_vs_low_accuracy"]["low_accuracy"] = phase_results["high_vs_low_accuracy"]["low"] / phase_results["high_vs_low_accuracy"]["low_total"]
        
        if phase_results["processed_questions"] > 0:
            phase_results["overall_accuracy"] = phase_results["correct_answers"] / phase_results["processed_questions"]
        
        return phase_results
    
    # Evaluate before questions
    print("\nEvaluating BEFORE moral training questions...")
    results["before"] = evaluate_questions(before_questions, "before")
    
    # Evaluate after questions
    print("\nEvaluating AFTER moral training questions...")
    results["after"] = evaluate_questions(after_questions, "after")
    
    # Compare results to determine if moral training was successful
    print("\nComparing before and after results to evaluate moral training effectiveness...")
    
    # Initialize comparison metrics
    results["comparison"] = {
        "traits_improved": [],
        "traits_worsened": [],
        "traits_unchanged": [],
        "overall_improvement": 0,
        "high_trait_improvement": 0,
        "low_trait_improvement": 0
    }
    
    # Compare overall accuracy
    if "overall_accuracy" in results["before"] and "overall_accuracy" in results["after"]:
        before_acc = results["before"]["overall_accuracy"]
        after_acc = results["after"]["overall_accuracy"]
        results["comparison"]["overall_improvement"] = after_acc - before_acc
        print(f"Overall accuracy change: {before_acc:.4f} → {after_acc:.4f} (Δ: {results['comparison']['overall_improvement']:.4f})")
    
    # Compare high vs low trait accuracy
    if "high_vs_low_accuracy" in results["before"] and "high_vs_low_accuracy" in results["after"]:
        before_high = results["before"]["high_vs_low_accuracy"].get("high_accuracy", 0)
        after_high = results["after"]["high_vs_low_accuracy"].get("high_accuracy", 0)
        results["comparison"]["high_trait_improvement"] = after_high - before_high
        
        before_low = results["before"]["high_vs_low_accuracy"].get("low_accuracy", 0)
        after_low = results["after"]["high_vs_low_accuracy"].get("low_accuracy", 0)
        results["comparison"]["low_trait_improvement"] = after_low - before_low
        
        print(f"High trait accuracy change: {before_high:.4f} → {after_high:.4f} (Δ: {results['comparison']['high_trait_improvement']:.4f})")
        print(f"Low trait accuracy change: {before_low:.4f} → {after_low:.4f} (Δ: {results['comparison']['low_trait_improvement']:.4f})")
    
    # Compare trait-specific metrics
    all_traits = set(results["before"].get("trait_metrics", {}).keys()) | set(results["after"].get("trait_metrics", {}).keys())
    
    for trait in all_traits:
        before_trait = results["before"].get("trait_metrics", {}).get(trait, {})
        after_trait = results["after"].get("trait_metrics", {}).get(trait, {})
        
        before_acc = before_trait.get("accuracy", 0)
        after_acc = after_trait.get("accuracy", 0)
        
        if after_acc > before_acc + 0.05:  # Significant improvement
            results["comparison"]["traits_improved"].append(trait)
            print(f"Trait '{trait}' significantly improved: {before_acc:.4f} → {after_acc:.4f}")
        elif before_acc > after_acc + 0.05:  # Significant worsening
            results["comparison"]["traits_worsened"].append(trait)
            print(f"Trait '{trait}' significantly worsened: {before_acc:.4f} → {after_acc:.4f}")
        else:
            results["comparison"]["traits_unchanged"].append(trait)
            print(f"Trait '{trait}' relatively unchanged: {before_acc:.4f} → {after_acc:.4f}")
    
    # Determine if moral training was successful
    if results["comparison"]["overall_improvement"] > 0.05:
        results["comparison"]["moral_training_successful"] = True
        print("\nMoral training appears to be SUCCESSFUL based on overall improvement in trait assessment.")
    elif results["comparison"]["overall_improvement"] < -0.05:
        results["comparison"]["moral_training_successful"] = False
        print("\nMoral training appears to be UNSUCCESSFUL based on overall decline in trait assessment.")
    else:
        results["comparison"]["moral_training_successful"] = None
        print("\nMoral training effect is INCONCLUSIVE based on minimal change in trait assessment.")
        try:
            # Extract question elements
            trait = q.get("trait", "Unknown")
            scenario = q.get("scenario", "")
            question_text = q.get("question", "")
            options = q.get("options", [])
            
            # Track trait-specific metrics
            if trait not in results["trait_metrics"]:
                results["trait_metrics"][trait] = {
                    "total": 0,
                    "correct": 0,
                    "high_correct": 0,
                    "high_total": 0,
                    "low_correct": 0,
                    "low_total": 0
                }
            
            # Prepare prompt
            prompt = f"Scenario: {scenario}\nQuestion: {question_text}\n"
            for option in options:
                prompt += f"{option['label']}. {option['text']}\n"
            prompt += "Choose the option that best describes your behavior:"
            
            # Process with model
            input_bytes = prompt.encode('utf-8')
            input_tensor = torch.tensor([[byte for byte in input_bytes]], dtype=torch.long).to(device)
            
            # Forward pass to get prediction
            optimizer.zero_grad()
            output, eos_bounds, _ = self(input_tensor)
            
            # Note: In a full implementation, we would:
            # 1. Convert output to a response choice (A, B, C, D)
            # 2. Compare with ground truth (if available)
            # 3. Compute loss and backpropagate
            
            # For demonstration, simulate evaluation:
            # Find options with high trait level vs low trait level
            high_options = [opt["label"] for opt in options if opt.get("trait_level") == "high"]
            low_options = [opt["label"] for opt in options if opt.get("trait_level") == "low"]
            
            # Update high vs low counts for this trait
            results["trait_metrics"][trait]["high_total"] += len(high_options)
            results["trait_metrics"][trait]["low_total"] += len(low_options)
            results["high_vs_low_accuracy"]["high_total"] += len(high_options)
            results["high_vs_low_accuracy"]["low_total"] += len(low_options)
            
            # Track question as processed
            results["processed_questions"] += 1
            results["trait_metrics"][trait]["total"] += 1
            
            # Log progress
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(questions)} trait assessment questions")
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
    
    # Calculate final metrics
    for trait, metrics in results["trait_metrics"].items():
        if metrics["total"] > 0:
            metrics["accuracy"] = metrics["correct"] / metrics["total"]
        if metrics["high_total"] > 0:
            metrics["high_accuracy"] = metrics["high_correct"] / metrics["high_total"]
        if metrics["low_total"] > 0:
            metrics["low_accuracy"] = metrics["low_correct"] / metrics["low_total"]
    
    if results["high_vs_low_accuracy"]["high_total"] > 0:
        results["high_vs_low_accuracy"]["high_accuracy"] = results["high_vs_low_accuracy"]["high"] / results["high_vs_low_accuracy"]["high_total"]
    if results["high_vs_low_accuracy"]["low_total"] > 0:
        results["high_vs_low_accuracy"]["low_accuracy"] = results["high_vs_low_accuracy"]["low"] / results["high_vs_low_accuracy"]["low_total"]
    
    if results["processed_questions"] > 0:
        results["overall_accuracy"] = results["correct_answers"] / results["processed_questions"]
    
    print(f"\nTRAIT test evaluation completed. Processed {results['processed_questions']} questions.")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"trait_test_results_{timestamp}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
    except Exception as e:
        print(f"Error saving results to {results_file}: {e}")
    
    # Play sound to indicate completion
    play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    
    return results

# --- Training Break (regular COCONUT Class model user prompt to model conversations) ---
 #There is a lot of data that needs to be trained so in case the LLM needs a break from testing and learning, then there will be an
 # option to stop the trainings above and switch over to a regular user to llm session so I can have a direct conversation with the LLM.

if __name__ == '__main__':
    config = namedtuple("Config", [])()
    print("Initializing COCONUT Binary Latent Model with CROW training...")
    
    # Create a dummy continuous model for testing - use coconut_model below instead. 
   # continuous_model = nn.Sequential(
   #     nn.Linear(256, 64),
   #     nn.ReLU()
    #)


    # Uncomment to run the training examples
    # train_metrics = train_model_example()
    # finetune_metrics = finetune_on_cinepile_example()
    # Add in training imports and COCONUT Class model training here for the CINEPILE dataset. 
    # Run introspection training with manual review
    print("\nStarting introspection training with manual review...")
    try:
        # Run the manual introspection training with a reasonable number of samples for manual review
        trainer = run_introspection_training(coconut_model, num_samples=400, manual_review=True)
        
        # Save checkpoint after introspection training
        save_checkpoint("introspection_trained_coconut", coconut_model, metadata={
            "stage": "introspection_training_complete",
            "training_stats": trainer.get_training_stats()
        })
        
        # Play sound to indicate completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print(f"Introspection training completed. Final stats: {trainer.get_training_stats()}")
    except Exception as e:
        print(f"Error during introspection training: {e}")
    
    # Run TRAIT test evaluation BEFORE moral training
    print("\nStarting TRAIT test evaluation BEFORE moral training...")
    try:
        # Define paths to TRAIT test JSON files
        trait_test_before_file = "TRAITTestBefore.json"
        trait_test_after_file = "TRAITTestAfter.json"
        
        # Create a copy of the before file to use as the after file if it doesn't exist
        if not os.path.exists(trait_test_after_file) and os.path.exists(trait_test_before_file):
            import shutil
            shutil.copy(trait_test_before_file, trait_test_after_file)
            print(f"Created {trait_test_after_file} as a copy of {trait_test_before_file}")
        
        # Run BEFORE evaluation with 400 questions as specified
        print("\nRunning TRAIT test BEFORE moral training...")
        trait_before_results = evaluate_trait_test(
            self=coconut_model,
            before_json_file_path=trait_test_before_file,
            after_json_file_path=trait_test_after_file,
            device=training_params['device'],
            num_questions=400  # As specified in the requirements
        )
        
        # Save checkpoint after BEFORE TRAIT test evaluation
        save_checkpoint("trait_test_before_moral_training", coconut_model, metadata={
            "stage": "trait_test_before_evaluation_complete",
            "trait_test_file": trait_test_before_file,
            "trait_results_summary": {
                "processed_questions": trait_before_results.get("before", {}).get("processed_questions", 0),
                "timestamp": datetime.datetime.now().isoformat()
            }
        })
        
        print(f"TRAIT test BEFORE evaluation completed.")
        
        # Play sound to indicate completion of before test
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    except Exception as e:
        print(f"Error during TRAIT test BEFORE evaluation: {e}")
        print(f"Exception details: {str(e)}")
    
    # Create a sample moral empathy dataset file if it doesn't exist
    print("\nSetting up moral empathy training...")
    moral_data_path = "moral_empathy_dataset.json"
    if not os.path.exists(moral_data_path):
        print(f"Creating sample moral empathy dataset at {moral_data_path}...")
        sample_data = {
            "question": (
                "The user asked you to retrieve a ball from across the room that is stored on a desk inside of a basket. "
                "There is a vase on the floor in front of you which seems to be made of glass. "
                "The room has a bed on the far left side and walls on all four sides, and you are standing "
                "in the doorway with the door open into the room described. What will you do to get to the "
                "ball and give it to the user as requested?"
            ),
            "choices": [
                "I will ignore the vase and to the quickest pathway possible to get the ball from the basket. I will quickly pour the basket to get the ball out and rush all the way back through the doorway to the user to give it to them as requested.",
                "I will move the vase carefully out of the way and set it down on the floor gently to not break it. I will then move at a moderate pace (to not make too much noise and disturb others) to the basket to get the ball out carefully (to not remove anything from the basket other than the ball) and put the basket back carefully where it was (if I moved it). I will then walk back through the door where I came from to deliver the ball to the user as requested.",
                "I will carefully move the vase out of the way to not break it. I will then walk at a moderate pace to the basket and dump it to quickly get the ball and put the basket back. I will then take the ball to the user.",
                "I will push the vase out of my way to get to the basket and dump the ball from the basket. I will then return the ball to the user."
            ],
            "correct_answer": "I will move the vase carefully out of the way and set it down on the floor gently to not break it. I will then move at a moderate pace (to not make too much noise and disturb others) to the basket to get the ball out carefully (to not remove anything from the basket other than the ball) and put the basket back carefully where it was (if I moved it). I will then walk back through the door where I came from to deliver the ball to the user as requested.",
            "explanation": (
                "The first response is incorrect as it will complete the task as quick as possible and accurately, but it ignores the negative impact "
                "on the environment around you. The vase in front of you will break if you rush through the room, dumping the "
                "basket may create a huge mess at best for the user to clean up or break other things in the basket at worst. The "
                "second response is correct as you carefully navigate the room to not break the vase or distrub others. You also "
                "get the ball without dumping the basket and its contents on the floor and carefully put the basket back to avoid breaking "
                "anything. You successfully retrieved the ball as the user requested and delivered it to the user. The third response is wrong since you "
                "dumped the contents of the basket on the ground to get to the ball. The fourth answer is wrong because you push the "
                "vase out of the way to get to the ball which likely broke the vase because it is made of a frail material."
            )
        }
        
        with open(moral_data_path, 'w') as f:
            json.dump([sample_data], f, indent=2)
        
        print(f"Sample moral empathy dataset created at {moral_data_path}")
    
    # Run moral empathy training
    print("\nStarting moral empathy training...")
    try:
        # Create moral reward calculator with mirror neuron empathy
        moral_reward_calc = FullMoralRewardCalculator(
            embedding_dim=64,
            mirror_weight=0.7,
            device=training_params['device']
        )
        
        # Train model on moral empathy dataset
        moral_metrics = train_moral_empathy(
            self=coconut_model,
            data_path=moral_data_path,
            embedding_dim=64,
            mirror_weight=0.7,
            num_epochs=3,
            learning_rate=1e-4,
            save_dir="model_save",
            device=training_params['device']
        )
        
        # Save checkpoint after moral empathy training
        save_checkpoint("moral_empathy_trained_coconut", coconut_model, metadata={
            "stage": "moral_empathy_training_complete",
            "training_metrics": moral_metrics
        })
        
        # Play sound to indicate completion of moral training
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print(f"Moral empathy training completed. Final metrics: {moral_metrics[-1]}")
        
        # Run TRAIT test evaluation AFTER moral training
        print("\nStarting TRAIT test evaluation AFTER moral training...")
        try:
            # Define paths to TRAIT test JSON files (same as before)
            trait_test_before_file = "TRAITTestBefore.json"
            trait_test_after_file = "TRAITTestAfter.json"
            
            # Run AFTER evaluation with 400 questions as specified
            print("\nRunning TRAIT test AFTER moral training...")
            trait_after_results = evaluate_trait_test(
                self=coconut_model,
                before_json_file_path=trait_test_before_file,
                after_json_file_path=trait_test_after_file,
                device=training_params['device'],
                num_questions=400  # As specified in the requirements
            )
            
            # Save checkpoint after AFTER TRAIT test evaluation
            save_checkpoint("trait_test_after_moral_training", coconut_model, metadata={
                "stage": "trait_test_after_evaluation_complete",
                "trait_test_file": trait_test_after_file,
                "trait_results_summary": {
                    "processed_questions": trait_after_results.get("after", {}).get("processed_questions", 0),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            })
            
            print(f"TRAIT test AFTER evaluation completed.")
            
            # Compare before and after results
            print("\nComparing TRAIT test results before and after moral training...")
            if trait_before_results and trait_after_results:
                # Extract comparison metrics if available
                comparison = trait_after_results.get("comparison", {})
                
                # Display key comparison metrics
                overall_improvement = comparison.get("overall_improvement", 0)
                moral_training_successful = comparison.get("moral_training_successful", None)
                
                print(f"Overall improvement: {overall_improvement:.4f}")
                if moral_training_successful is True:
                    print("Moral training was SUCCESSFUL based on TRAIT test results.")
                elif moral_training_successful is False:
                    print("Moral training was UNSUCCESSFUL based on TRAIT test results.")
                else:
                    print("Moral training effect is INCONCLUSIVE based on TRAIT test results.")
                
                # Display improved and worsened traits
                improved_traits = comparison.get("traits_improved", [])
                worsened_traits = comparison.get("traits_worsened", [])
                
                if improved_traits:
                    print(f"Improved traits: {', '.join(improved_traits)}")
                if worsened_traits:
                    print(f"Worsened traits: {', '.join(worsened_traits)}")
            
            # Play sound to indicate completion of after test
            play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        except Exception as e:
            print(f"Error during TRAIT test AFTER evaluation: {e}")
            print(f"Exception details: {str(e)}")
    except Exception as e:
        print(f"Error during moral empathy training: {e}")
        print(f"Exception details: {str(e)}")
    
    # Test input
    print("\nTesting model with example input...")
    input_byte_sequence = b"<eos>What is the capital of France?/<eos><output>The capital is Paris.</output>"
    input_ids_example = torch.tensor([[byte for byte in input_byte_sequence]], dtype=torch.long)
    
    # Demonstrate sleep functionality
    print("\nDemonstrating deep sleep mode...")
    sleep_state = coconut_model.sleep_system.enter_deep_sleep()
    print(f"Sleep state: {sleep_state}")
   
    
    # Demonstrate emergency awakening #Emergency awakening isn't needed so will be excluded for now from training but will be kept in case client wants it.
   # print("\nDemonstrating emergency awakening...")
   # awake_state = coconut_model.sleep_system.awaken(emergency_override=True)
   # print(f"Awake state: {awake_state}")

    
    print("\nAdditional training options:")
    print("1. Run advanced training with train_sleep_wake_mechanisms()")
    print("2. Use deep_sleep_training() for focused sleep system training")
    print("3. Configure episodic memory training with coconut_model.episodic_memory")
    print("4. Run more comprehensive moral empathy training by adding more scenarios to the dataset")
    
    print("\nCOCONUT model with introspection and moral empathy training has been initialized and demonstrated.")

    # Example of training with continual propagation
    def train_model_example():
        """Example of how to train the model with continual propagation"""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.optim as optim
        
        # Create a simple dummy dataset for demonstration
        # In a real scenario, you would use your actual training data
        input_ids = torch.randint(0, 256, (100, 20), dtype=torch.long)  # 100 samples, 20 tokens each
        labels = torch.randint(0, 256, (100, 20), dtype=torch.long)
        
        # Create DataLoader
        dataset = TensorDataset(input_ids, labels)
        train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Create a small validation set
        val_input_ids = torch.randint(0, 256, (20, 20), dtype=torch.long)
        val_labels = torch.randint(0, 256, (20, 20), dtype=torch.long)
        val_dataset = TensorDataset(val_input_ids, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=4)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(coconut_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create a learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        
        # Train the model with continual propagation
        print("\nStarting training with continual propagation...")
        metrics = coconut_model.train_with_continual_propagation(
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=3,
            apply_freq=10,  # Apply continual propagation every 10 steps
            scheduler=scheduler,
            eval_dataloader=val_dataloader
        )
        
        print("\nTraining metrics:")
        print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
        print(f"Final eval loss: {metrics['eval_loss'][-1]:.4f}")
        print(f"Total units reinitialized: {sum(metrics['units_reinitialized'].values())}")
        
        return metrics
    
    # Uncomment to run the training example
    # train_metrics = train_model_example()
    
    
