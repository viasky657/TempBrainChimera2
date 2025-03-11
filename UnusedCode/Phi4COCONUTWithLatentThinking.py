import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import time
import datetime
import os
import json
import platform
import subprocess
from contextlib import nullcontext
import torch.nn.functional as F
from TOVACompression import TOVACompression, tova_compress
import math
from tqdm import tqdm
import numpy as np
from scipy import integrate

# Define the Outputs namedtuple for consistency with COCONUT
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

def save_checkpoint(step_name, model=None, metadata=None):
    """
    Save a checkpoint of the model.
    
    Args:
        step_name: Name of the checkpoint step
        model: Model to save
        metadata: Additional metadata to include
    
    Returns:
        Tuple of (checkpoint_filename, config_filename)
    """
    import os, json, torch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = "model_save"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Save model checkpoint as a safetensor file if a model is provided; otherwise, save dummy data
    checkpoint_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}.safetensors")
    if model is not None:
        # In practice, one would use a dedicated safetensors library
        torch.save(model.state_dict(), checkpoint_filename)
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
        }
    }
    
    # Add any additional metadata if provided
    if metadata:
        config_data["metadata"] = metadata
    
    config_filename = os.path.join(checkpoint_dir, f"{step_name}_{timestamp}_config.json")
    with open(config_filename, "w") as cf:
        json.dump(config_data, cf, indent=4)
    print("Config file saved:", config_filename)
    
    return checkpoint_filename, config_filename

def play_sound(sound_file):
    """
    Play a sound file.
    
    Args:
        sound_file: Path to the sound file
    """
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
        diffusion_beta_schedule="linear"
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
            self.generation_mode = "autoregressive"  # or "diffusion"
        
        # Initialize continual propagation tracking if enabled
        if self.enable_continual_prop:
            self._init_continual_propagation()
            self._register_activation_hooks()
    
    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs): #Continue from Here. 
        """
        Forward pass through the model with continuous thinking and diffusion.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for loss calculation
            position_ids: Position IDs
            **kwargs: Additional arguments
            
        Returns:
            Outputs namedtuple with loss, inputs_embeds, and logits
        """
        # Check if we should use diffusion-based forward pass
        use_diffusion = kwargs.get("use_diffusion", False)
        
        # First apply continuous thinking to get enhanced embeddings
        continuous_outputs = self.forward_continuous_thinking(input_ids, attention_mask, labels, position_ids, **kwargs)
        
        # Pass the continuous thinking outputs to the diffusion model
        if kwargs.get("use_diffusion", False) and self.use_diffusion:
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
        
        # Return continuous thinking outputs directly if diffusion is not used. This needs to be here because the program will need to call this
        # function to start the model processing process. 
            return continuous_outputs 

    
    def forward_continuous_thinking(self, input_ids, attention_mask, labels, position_ids, **kwargs):
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


# Example usage
if __name__ == "__main__":
    print("Initializing Phi4COCONUTWithLatentThinking model...")
    
    # Create the model
    model = Phi4COCONUTWithLatentThinking()
    
    # Set to evaluation mode
    model.eval()
    
    # Example prompt with latent tokens for continuous thinking
    prompt = "What is the capital of France? <|latent|> Let me think about this. France is a country in Europe. Its capital city is <|latent|> Paris."
    
    # Replace <|latent|> with the actual latent token ID in the tokenized input
    # This is just a demonstration - in practice, you would use the tokenizer
    print("\nGenerating response with continuous thinking in latent space...")
    response = model.generate_text(
        prompt,
        max_new_tokens=100,
        use_diffusion=True,
        temperature=0.8
    )
    
    print(f"\nResponse: {response}")
    
    # Example of saving the model
    # model.save_pretrained("phi4_coconut_latent_thinking_model")
    
    # Example of loading the model
    # loaded_model = Phi4COCONUTWithLatentThinking.from_pretrained("phi4_coconut_latent_thinking_model")
    
    print("\nExample of training with continual propagation:")
    print("model.train_with_continual_propagation(train_dataloader, optimizer, criterion, num_epochs=3, apply_freq=100)")
    
    print("\nExample of applying continual propagation manually:")
    print("model.apply_continual_propagation()")
    
    print("\nExample of getting latent space statistics:")
    print("stats = model.get_latent_space_stats()")
    print("print(f'Latent spaces used: {stats[\"num_latent_spaces_used\"]}, Penalty applied: {stats[\"total_penalty\"]}')")

# The get_latent_space_stats method is already defined in the class (lines 1761-1784)

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
    checkpoint_filename, config_filename = save_checkpoint("model_checkpoint", model=self, metadata=metadata)
    
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
        model = cls(
            base_causallm=base_causallm,
            latent_token_id=config.get("latent_token_id"),
            start_latent_id=config.get("start_latent_id"),
            end_latent_id=config.get("end_latent_id"),
            eos_token_id=config.get("eos_token_id"),
            enable_continual_prop=config.get("enable_continual_prop", True),
            replacement_rate=config.get("replacement_rate", 1e-5),
            maturity_threshold=config.get("maturity_threshold", 1000),
            surprise_threshold=config.get("surprise_threshold", 0.5),
            use_tova=config.get("use_tova", True),
            tova_cache_size=config.get("tova_cache_size", 512),
            tova_layer_wise=config.get("tova_layer_wise", True),
            use_diffusion=config.get("use_diffusion", True),
            num_diffusion_steps=config.get("num_diffusion_steps", 20),
            diffusion_beta_schedule=config.get("diffusion_beta_schedule", "linear")
        )
        
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
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with base model only...")
        
        print(f"Model loaded from {model_path}")
        return model
