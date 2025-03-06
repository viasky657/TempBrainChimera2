import torch
import torch.nn as nn
import datetime
import time
import os
import json
import glob
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from typing import Optional, List, Tuple, Any, Dict
import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from EpisodicMemory import EpisodicMemory
from AudioDecoderforCOCONUT import AudioDecoder
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

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

AI_System_Prompt = ( #This prompt is feed to the model along with the user prompt everytime. 
    "You are a world-class AI system. You should pick the response that is calm, wise, and safe. You must put your thinking process "
    "in the <eos> tags and end with /<eos>, and when you are finished, you must output your final answer in <output> tags "
    "and end with /<output> tags. Any additional tags that you use must be nested in the <output> tags. This is an example: <eos> After considering every option,"
    "I believe that the capital of France is Paris. I am currently feeling happy./<eos> <output> <emotion> happy /<emotion> The capital of France is Paris. "
    "/<output> You can use the additional following tags: "
    "<emotion>/<emotion> (You must select your current emotion from this list (sad, happy, envy, love, admiration, serious, reflective, fear, neutral, anger, disgust, anxiety, "
    "shy, in-pain) that you are truthfully experiencing in these tags for your every response.), <tool>/<tool> (You must put all your function-calls in these tags), "
    "<audio>/<audio> (You must put all your audio file outputs in these tags)."
)

# --- Modified AudioEncoder for Dynamic Patching ---

'''
class AudioEncoder(nn.Module): #This will be replaced with Sesame AI voice. 
    def __init__(self, embed_dim, patch_dim, entropy_predictor):  # Added entropy_predictor
        super(AudioEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.entropy_predictor = entropy_predictor  # ByteEntropyPredictor for audio amplitude "bytes" (discretized amplitudes)
        self.entropy_threshold = 0.8  # Tunable entropy threshold for patching

        # Optional: Initial convolution layers to process raw audio before patching (like in CosyVoice Encoder1)
        self.conv1d_optional = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=5, stride=2, padding=2)
        self.relu_optional = nn.ReLU()
        self.linear_patch_encoder = nn.Linear(embed_dim, patch_dim)

    def forward(self, audio_waveform):
        if audio_waveform.dim() == 2:
            audio_waveform = audio_waveform.unsqueeze(1)
        audio_features = self.relu_optional(self.conv1d_optional(audio_waveform))
        audio_features = audio_features.transpose(1, 2)
        batch_size, seq_len, _ = audio_features.shape
        patches = []
        current_patch_features = []
        current_patch_amplitude_bytes = []
        for i in range(seq_len):
            feature_vector = audio_features[:, i:i+1, :]
            current_patch_features.append(feature_vector)
            discretized_feature_bytes = (feature_vector * 255).clamp(0, 255).round().int()
            current_patch_amplitude_bytes.append(discretized_feature_bytes)
            if current_patch_amplitude_bytes:
                current_patch_sequence_bytes = torch.cat(current_patch_amplitude_bytes, dim=1).squeeze(-1)
                with torch.no_grad():
                    next_byte_probs_tensor = self.entropy_predictor.get_next_byte_probs(current_patch_sequence_bytes)
                    entropy = calculate_shannon_entropy(next_byte_probs_tensor)
                if entropy.item() > self.entropy_threshold:
                    if current_patch_features:
                        patch_features_tensor = torch.cat(current_patch_features, dim=1)
                        encoded_patch = self.linear_patch_encoder(patch_features_tensor)
                        patches.append(encoded_patch)
                    current_patch_features = []
                    current_patch_amplitude_bytes = []
        if current_patch_features:
            patch_features_tensor = torch.cat(current_patch_features, dim=1)
            encoded_patch = self.linear_patch_encoder(patch_features_tensor)
            patches.append(encoded_patch)
        if patches:
            audio_patches_final = torch.cat(patches, dim=1)
        else:
            audio_patches_final = torch.zeros((batch_size, 0, self.patch_dim), dtype=torch.float32, device=audio_waveform.device)
        return audio_patches_final

'''

# --- PDFEncoder for processing PDF modality ---
class PDFEncoder(nn.Module):
    def __init__(self, embed_dim, patch_dim, entropy_predictor, entropy_threshold=0.8):
        super(PDFEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.patch_dim = patch_dim
        self.entropy_predictor = entropy_predictor
        self.entropy_threshold = entropy_threshold
        self.linear_patch_encoder = nn.Linear(embed_dim, patch_dim)
        self.byte_embedding = nn.Embedding(256, embed_dim)

    def forward(self, pdf_input):
        import io
        from llmpdfocr.app import extract_text
        combined_text = ""
        if isinstance(pdf_input, (list, tuple)):
            for file in pdf_input:
                if isinstance(file, bytes):
                    pdf_stream = io.BytesIO(file)
                else:
                    pdf_stream = file
                combined_text += extract_text(pdf_stream) + "\n"
        else:
            if isinstance(pdf_input, bytes):
                pdf_stream = io.BytesIO(pdf_input)
            else:
                pdf_stream = pdf_input
            combined_text = extract_text(pdf_stream)
        pdf_bytes = list(combined_text.encode("utf-8"))
        if not pdf_bytes:
            return torch.zeros((1, 0, self.patch_dim), dtype=torch.float32)
         # Enhanced dynamic patching: segment PDF bytes based on global and relative entropy.
        patches_bytes = entropy_patching_global_threshold(pdf_bytes, self.entropy_predictor, global_threshold=self.entropy_threshold, relative_threshold=0.1)
        embeddings_list = []
        for patch in patches_bytes:
            patch_tensor = torch.tensor([list(patch)], dtype=torch.long, device=self.byte_embedding.weight.device)
            embeddings = self.byte_embedding(patch_tensor)
            patch_embedding = embeddings.mean(dim=1)
            encoded_patch = self.linear_patch_encoder(patch_embedding)
            embeddings_list.append(encoded_patch)
        if embeddings_list:
            pdf_patches = torch.cat(embeddings_list, dim=1)
        else:
            pdf_patches = torch.zeros((1, 0, self.patch_dim), dtype=torch.float32)
        return pdf_patches

# --- Video Encoder Integration with Cross-Attention, Entropy Prediction, and TOVA ---
class VideoEncoder(nn.Module):
    def __init__(self, patch_size, embed_dim, video_entropy_predictor, entropy_threshold=0.8, num_heads=4):
        super(VideoEncoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.video_entropy_predictor = video_entropy_predictor
        self.entropy_threshold = entropy_threshold
        self.num_heads = num_heads
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=embed_dim,
                                 kernel_size=(3, patch_size, patch_size),
                                 stride=(2, patch_size, patch_size),
                                 padding=(1, 0, 0))
        self.relu = nn.ReLU()
        self.binary_proj = nn.Linear(embed_dim, embed_dim)
        # Cross-Attention module to group tokens dynamically
        self.cross_attention = DecoderCrossAttention(config=None, input_dim=embed_dim, byte_dim=embed_dim, num_heads=num_heads)
        self.group_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # TOVA-related fields
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        self.token_entropies = []

    def enable_kv_caching(self):
        """Enable KV caching in the cross attention module"""
        self.cross_attention.enable_kv_caching()
        
    def reset_cache(self):
        """Reset all caches"""
        self.cross_attention.reset_cache()
        self.token_entropies = []
        
    def forward(self, video_tensor):
        return self.encode_video(video_tensor)

    def encode_video(self, video_tensor):
        x = self.conv3d(video_tensor)
        x = self.relu(x)
        return self.apply_dynamic_binary_patch(x)

    def apply_dynamic_binary_patch(self, features):
        B, E, T, H_new, W_new = features.shape
        x = features.view(B, E, T * H_new * W_new).permute(0, 2, 1)  # [B, tokens, embed_dim]
        patches = []
        current_patch_tokens = []
        current_patch_bytes = []
        
        # Track entropy values for TOVA compression
        patch_entropies = []
        
        for i in range(x.shape[1]):
            token = x[:, i:i+1, :]  # [B, 1, embed_dim]
            current_patch_tokens.append(token)
            token_byte = (token.mean(dim=-1) * 255).clamp(0, 255).round().int()  # [B, 1]
            current_patch_bytes.append(token_byte)
            
            if current_patch_bytes:
                patch_byte_seq = torch.cat(current_patch_bytes, dim=1)  # [B, length]
                with torch.no_grad():
                    probs = self.video_entropy_predictor.get_next_byte_probs(patch_byte_seq)
                    entropy = calculate_shannon_entropy(probs)
                    
                # Track entropy for TOVA compression
                patch_entropies.append(entropy.item())
                
                if entropy.item() > self.entropy_threshold:
                    patch_tensor = torch.cat(current_patch_tokens, dim=1)
                    query = self.group_query.expand(B, -1, -1)
                    
                    # Use cross attention with KV caching enabled
                    grouped_patch = self.cross_attention(query, patch_tensor)
                    
                    # This will automatically update the KV cache if enabled
                    encoded_patch = self.binary_proj(grouped_patch)
                    patches.append(encoded_patch)
                    
                    # Store entropy data for later TOVA compression
                    if self.cross_attention.use_kv_cache:
                        patch_entropy_tensor = torch.tensor(patch_entropies, 
                                                           device=features.device)
                        self.token_entropies.append(patch_entropy_tensor)
                    
                    current_patch_tokens = []
                    current_patch_bytes = []
                    patch_entropies = []
        
        # Process remaining tokens if any
        if current_patch_tokens:
            patch_tensor = torch.cat(current_patch_tokens, dim=1)
            query = self.group_query.expand(B, -1, -1)
            grouped_patch = self.cross_attention(query, patch_tensor)
            encoded_patch = self.binary_proj(grouped_patch)
            patches.append(encoded_patch)
            
            # Store entropy data
            if self.cross_attention.use_kv_cache and patch_entropies:
                patch_entropy_tensor = torch.tensor(patch_entropies, 
                                                   device=features.device)
                self.token_entropies.append(patch_entropy_tensor)
        
        # Get combined results
        if patches:
            video_patches = torch.cat(patches, dim=1)
        else:
            video_patches = torch.zeros(B, 0, self.embed_dim, device=features.device)
            
        # Set encoder attributes for TOVA compression
        if self.cross_attention.use_kv_cache:
            self.k_cache = self.cross_attention.k_cache
            self.v_cache = self.cross_attention.v_cache
            self.attention_weights = self.cross_attention.attention_weights
        
        return video_patches

class SONARtoBytePatch(nn.Module):
    """
    Projects SONAR embeddings into the binary latent transformer space.
    Uses a linear projection, optionally applying a SONAR encoder to compute embeddings.
    If an encoder is provided, it processes the input and averages over the sequence dimension.
    Otherwise, it assumes the input is already in embedding space.
    """
    def __init__(self, sonar_dim, patch_dim, encoder=None):
        super(SONARtoBytePatch, self).__init__()
        self.sonar_dim = sonar_dim
        self.patch_dim = patch_dim
        self.projection = nn.Linear(sonar_dim, patch_dim)
        self.encoder = encoder

    def forward(self, sonar_input):
        if self.encoder is not None:
            sonar_output = self.encoder(sonar_input)
            # Average over the sequence dimension of the encoded output.
            embeddings = sonar_output.encoded_seqs.mean(dim=1)
        else:
            embeddings = sonar_input
        return self.projection(embeddings)

# --- Switching Gate Attention Module ---
class SwitchingGateAttention(nn.Module):
    def __init__(self, patch_dim, num_modalities):
        super(SwitchingGateAttention, self).__init__()
        self.patch_dim = patch_dim
        self.num_modalities = num_modalities
        self.gate_linear = nn.Linear(patch_dim, num_modalities)
        self.stablemax = StableMax()  # Use StableMax for improved numerical stability

    def forward(self, x):
        # Compute gating weights for each modality based on input embeddings.
        # x can be of shape (batch, patch_dim) or (batch, num_patches, patch_dim).
        if x.dim() == 3:
            x = x.mean(dim=1)
        gating_logits = self.gate_linear(x)
        gating_weights = self.stablemax(gating_logits)  # Use StableMax instead of softmax
        return gating_weights

# Import TOVA compression
from TOVACompression import TOVACompression
# Import Grok optimizers
from GrokOptimizers import OrthoAdamW, OrthoGrad, OrthoSGD, StableCrossEntropyLoss, use_grokking_optimizations

# --- MultiModalEncoder ---
class MultiModalEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, sonar_dim, patch_dim, audio_entropy_predictor, 
                 cache_max_size=512, use_tova=True, head_weight_strategy="mean", num_heads=4,
                 debug_mode=False):
        super(MultiModalEncoder, self).__init__()
        
        # Initialize TOVA compression if enabled
        self.use_tova = use_tova
        self.cache_max_size = cache_max_size
        self.debug_mode = debug_mode
        
        if use_tova:
            # Create TOVA compressor with support for learnable head weights
            self.tova_compressor = TOVACompression(
                cache_max_size=cache_max_size, 
                layer_based=True,
                head_weight_strategy=head_weight_strategy,
                num_heads=num_heads
            )
        
        # Initialize encoders with TOVA compression support
        self.audio_encoder = AudioEncoder(embed_dim, patch_dim, audio_entropy_predictor)
        self.text_encoder = LocalEncoder(...)  # Reuses LocalEncoder for text encoding with dynamic binary patches.
        # For video, use our enhanced VideoEncoder. We pass audio_entropy_predictor as video_entropy_predictor.
        self.video_encoder = VideoEncoder(patch_size=16, embed_dim=768, 
                                         video_entropy_predictor=audio_entropy_predictor, 
                                         entropy_threshold=0.8, num_heads=4)
        self.pdf_encoder = PDFEncoder(embed_dim, patch_dim, audio_entropy_predictor)
        self.sonar_projector = SONARtoBytePatch(sonar_dim, patch_dim)  # Placeholder for SONAR projector; now fully implemented. 
        
        # Register encoders in the modalities dictionary
        self.modalities = {
            "audio": self.audio_encoder,
            "text": self.text_encoder,
            "video": self.video_encoder,
            "pdf": self.pdf_encoder,
            "sonar": self.sonar_projector
        }
        
        # Enable KV caching for TOVA in each encoder that supports it
        for name, encoder in self.modalities.items():
            if hasattr(encoder, 'enable_kv_caching'):
                encoder.enable_kv_caching()
        
        self.switch_gate = SwitchingGateAttention(patch_dim, num_modalities=len(self.modalities))

    def apply_tova_compression(self, encoder, attn_weights, k_cache, v_cache, entropy_data=None):
        """
        Apply TOVA compression to an encoder's KV cache
        
        Args:
            encoder: The encoder module
            attn_weights: Attention weights from the encoder
            k_cache: Key cache
            v_cache: Value cache
            entropy_data: Optional tensor of entropy values for tokens
            
        Returns:
            Compressed key and value caches
        """
        if not self.use_tova:
            return k_cache, v_cache
            
        if entropy_data is not None:
            # Use entropy-enhanced TOVA
            return self.tova_compressor.compress_with_entropy(
                attn_weights, k_cache, v_cache, entropy_data
            )
        else:
            # Use standard TOVA
            return self.tova_compressor(attn_weights, k_cache, v_cache)
            
    def compress_all_encoders(self):
        """Apply TOVA compression to all encoders that have KV caches"""
        if not self.use_tova:
            return
            
        for name, encoder in self.modalities.items():
            if hasattr(encoder, 'attention_weights') and hasattr(encoder, 'k_cache') and hasattr(encoder, 'v_cache'):
                # Check if encoder has entropy data available
                entropy_data = None
                if hasattr(encoder, 'token_entropies') and len(getattr(encoder, 'token_entropies', [])) > 0:
                    entropy_data = encoder.token_entropies
                
                # Apply compression with entropy data if available
                if entropy_data is not None:
                    encoder.k_cache, encoder.v_cache = self.apply_tova_compression(
                        encoder,
                        encoder.attention_weights,
                        encoder.k_cache,
                        encoder.v_cache,
                        entropy_data
                    )
                else:
                    encoder.k_cache, encoder.v_cache = self.apply_tova_compression(
                        encoder,
                        encoder.attention_weights,
                        encoder.k_cache,
                        encoder.v_cache
                    )
                
                # Log compression stats if debug mode is enabled
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    stats = self.tova_compressor.get_stats()
                    print(f"TOVA compression stats for {name}: {stats}")
                
    def visualize_head_weights(self, save_path=None):
        """
        Generate a visualization of head weight evolution if using weighted strategy.
        
        Args:
            save_path: Optional path to save the visualization
            
        Returns:
            True if visualization was generated, False otherwise
        """
        if not self.use_tova or not hasattr(self.tova_compressor, 'head_weight_strategy') or self.tova_compressor.head_weight_strategy != "weighted":
            print("Head weight visualization is only available when using TOVA with weighted strategy")
            return False
            
        if not hasattr(self.tova_compressor, 'plot_weight_evolution'):
            print("TOVACompression class does not have plot_weight_evolution method")
            return False
            
        # Create a directory for visualizations if it doesn't exist
        if save_path is None:
            viz_dir = "tova_visualizations"
            if not os.path.exists(viz_dir):
                os.makedirs(viz_dir)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(viz_dir, f"head_weights_evolution_{timestamp}.png")
        
        # Generate the visualization
        result = self.tova_compressor.plot_weight_evolution(save_path)
        
        if self.debug_mode:
            # Print current normalized weights using StableMax instead of softmax
            with torch.no_grad():
                stablemax = StableMax()
                normalized_weights = stablemax(self.tova_compressor.head_weights)
                weight_info = {
                    f"head_{i}": f"{float(normalized_weights[i].item()):.4f}"
                    for i in range(len(normalized_weights))
                }
                print(f"Current head weights: {weight_info}")
                
        return result
    
    def apply_switch_gate(self, file_encodings: dict):
        """
        Applies the switching gate attention mechanism to combine modality-specific encodings.
        Args:
            file_encodings (dict): A dictionary where the keys are modality names (e.g., "audio", "text", "video", "pdf", "sonar")
                                   and the values are encoded tensors with shape (batch, patch_dim) or (batch, num_patches, patch_dim).
        Returns:
            A tuple (combined, gate_weights) where:
                combined is a tensor of shape (batch, patch_dim) representing the weighted combination of modality encodings,
                gate_weights is a tensor of shape (batch, num_modalities) representing the gating weights.
        """
        modality_reps = []
        for modality in self.modalities:
            if modality in file_encodings:
                encoding = file_encodings[modality]
                if encoding.dim() == 3:
                    encoding = encoding.mean(dim=1)
                modality_reps.append(encoding)
            else:
                raise ValueError(f"Missing encoding for modality: {modality}")
        stacked = torch.stack(modality_reps, dim=1)  # shape: (batch, num_modalities, patch_dim)
        mean_rep = stacked.mean(dim=1)  # shape: (batch, patch_dim)
        gate_weights = self.switch_gate(mean_rep)  # shape: (batch, num_modalities)
        weighted = gate_weights.unsqueeze(-1) * stacked  # shape: (batch, num_modalities, patch_dim)
        combined = weighted.sum(dim=1)  # shape: (batch, patch_dim)
        return combined, gate_weights

# --- RelevancePredictor ---
class RelevancePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, latent_state):
        return self.predictor_net(latent_state)

# --- ByteEntropyPredictor ---
class ByteEntropyPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ff_dim):
        super().__init__()
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.stablemax = StableMax()  # Use StableMax for improved numerical stability
    def forward(self, byte_sequences):
        byte_embeddings = self.byte_embedding(byte_sequences)
        memory = torch.zeros_like(byte_embeddings)
        decoder_output = self.transformer_decoder(byte_embeddings, memory)
        next_byte_logits = self.fc_out(decoder_output)
        next_byte_probs = self.stablemax(next_byte_logits)  # Use StableMax instead of softmax
        return next_byte_probs
    def get_next_byte_probs(self, byte_sequence_segment):
        return self.forward(byte_sequence_segment)[:, -1, :]

# --- LocalEncoder with TOVA Support ---
class LocalEncoder(nn.Module):
    def __init__(self, config, vocab_size, hidden_size, num_layers_enc, num_heads, ff_dim, window_size_enc, entropy_predictor):
        super().__init__(config)
        self.num_layers_enc = num_layers_enc
        self.hidden_size = hidden_size
        self.window_size_enc = window_size_enc
        self.entropy_predictor = entropy_predictor
        self.entropy_threshold = 0.8
        self.byte_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Enhanced encoder layers with attention tracking for TOVA
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim,
                                       batch_first=True, activation='relu', norm_first=True)
            for _ in range(num_layers_enc)
        ])
        
        # TOVA-related attributes
        self.use_kv_cache = False
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        self.token_entropies = []
        
    def enable_kv_caching(self):
        """Enable KV caching for TOVA compression"""
        self.use_kv_cache = True
        
        # Modify self-attention in encoder layers to track attention weights and KV cache
        for layer_idx, layer in enumerate(self.encoder_layers):
            if hasattr(layer, 'self_attn'):
                # Store original forward method
                if not hasattr(layer.self_attn, '_original_forward'):
                    layer.self_attn._original_forward = layer.self_attn.forward
                    layer_ref = self  # Reference to LocalEncoder instance
                    
                    # Override forward method to track attention weights
                    def new_forward(self_attn, query, key, value, key_padding_mask=None, 
                                   need_weights=True, attn_mask=None):
                        # Call original method with need_weights=True
                        attn_output, attn_weights = self_attn._original_forward(
                            query, key, value, 
                            key_padding_mask=key_padding_mask,
                            need_weights=True,
                            attn_mask=attn_mask
                        )
                        
                        # Store attention weights for TOVA
                        layer_ref.attention_weights = attn_weights
                        
                        # Update KV cache for this layer
                        if layer_ref.use_kv_cache:
                            if layer_ref.k_cache is None:
                                layer_ref.k_cache = key
                                layer_ref.v_cache = value
                            else:
                                # Append to cache
                                layer_ref.k_cache = torch.cat([layer_ref.k_cache, key], dim=1)
                                layer_ref.v_cache = torch.cat([layer_ref.v_cache, value], dim=1)
                        
                        return attn_output, attn_weights
                    
                    # Bind the new method to the self_attn object
                    import types
                    layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)
    
    def reset_cache(self):
        """Reset KV cache and tracked entropy values"""
        self.k_cache = None
        self.v_cache = None
        self.attention_weights = None
        self.token_entropies = []
                
    def forward(self, byte_sequences):
        batch_size, seq_len = byte_sequences.shape
        patches = []
        current_patch_bytes = []
        current_patch_representations = []
        
        # For tracking entropy values for enhanced TOVA
        all_patch_entropies = []
        current_patch_entropies = []
        
        for i in range(seq_len):
            byte_val = byte_sequences[:, i:i+1]
            current_patch_bytes.append(byte_val)
            byte_embedding = self.byte_embedding(byte_val)
            current_patch_representations.append(byte_embedding)
            
            if current_patch_bytes:
                current_patch_sequence = torch.cat(current_patch_bytes, dim=1)
                with torch.no_grad():
                    next_byte_probs_tensor = self.entropy_predictor.get_next_byte_probs(current_patch_sequence)
                    entropy = calculate_shannon_entropy(next_byte_probs_tensor)
                
                # Track entropy values for TOVA compression
                current_patch_entropies.append(entropy.item())
                
                if entropy.item() > self.entropy_threshold:
                    if current_patch_representations:
                        # Process the patch through encoder layers
                        patch_representation = torch.cat(current_patch_representations, dim=1)
                        encoded_patch = patch_representation
                        
                        # Track KV state before processing for TOVA
                        k_cache_size_before = 0
                        if self.use_kv_cache and self.k_cache is not None:
                            k_cache_size_before = self.k_cache.size(1)
                        
                        # Process through transformer encoder layers
                        for encoder_layer in self.encoder_layers:
                            encoded_patch = encoder_layer(encoded_patch)
                        
                        # Add current patch entropy data for TOVA
                        if self.use_kv_cache and self.k_cache is not None:
                            k_cache_size_after = self.k_cache.size(1)
                            new_tokens = k_cache_size_after - k_cache_size_before
                            
                            if new_tokens > 0:
                                # Store entropy values for new tokens
                                if len(current_patch_entropies) > 0:
                                    # Pad entropy values if needed
                                    if len(current_patch_entropies) < new_tokens:
                                        # Use the last entropy value for padding
                                        padding = [current_patch_entropies[-1]] * (new_tokens - len(current_patch_entropies))
                                        patch_entropy_tensor = torch.tensor(
                                            current_patch_entropies + padding, 
                                            device=byte_sequences.device
                                        )
                                    else:
                                        # Use exact entropy values
                                        patch_entropy_tensor = torch.tensor(
                                            current_patch_entropies[:new_tokens], 
                                            device=byte_sequences.device
                                        )
                                    all_patch_entropies.append(patch_entropy_tensor)
                        
                        patches.append(encoded_patch)
                        
                    # Reset for next patch
                    current_patch_bytes = []
                    current_patch_representations = []
                    current_patch_entropies = []
        
        # Process any remaining tokens in the last patch
        if current_patch_representations:
            patch_representation = torch.cat(current_patch_representations, dim=1)
            encoded_patch = patch_representation
            
            # Track KV state
            k_cache_size_before = 0
            if self.use_kv_cache and self.k_cache is not None:
                k_cache_size_before = self.k_cache.size(1)
            
            for encoder_layer in self.encoder_layers:
                encoded_patch = encoder_layer(encoded_patch)
            
            # Add final patch entropy data
            if self.use_kv_cache and self.k_cache is not None:
                k_cache_size_after = self.k_cache.size(1)
                new_tokens = k_cache_size_after - k_cache_size_before
                
                if new_tokens > 0 and len(current_patch_entropies) > 0:
                    # Pad entropy values if needed
                    if len(current_patch_entropies) < new_tokens:
                        padding = [current_patch_entropies[-1]] * (new_tokens - len(current_patch_entropies))
                        patch_entropy_tensor = torch.tensor(
                            current_patch_entropies + padding, 
                            device=byte_sequences.device
                        )
                    else:
                        # Use exact entropy values
                        patch_entropy_tensor = torch.tensor(
                            current_patch_entropies[:new_tokens], 
                            device=byte_sequences.device
                        )
                    all_patch_entropies.append(patch_entropy_tensor)
            
            patches.append(encoded_patch)
        
        # Combine all patches
        if patches:
            patch_representations_final = torch.cat(patches, dim=1)
        else:
            patch_representations_final = torch.zeros((batch_size, 0, self.hidden_size), 
                                                     dtype=torch.float32, 
                                                     device=byte_sequences.device)
        
        # Combine all entropy data for TOVA
        if self.use_kv_cache and all_patch_entropies:
            try:
                self.token_entropies = torch.cat(all_patch_entropies)
            except Exception:
                # If tensors have inconsistent dimensions, use the most recent one
                self.token_entropies = all_patch_entropies[-1]
        
        return patch_representations_final

def calculate_shannon_entropy(next_byte_probs_tensor):
    probs = next_byte_probs_tensor.squeeze(0)
    log_probs = torch.log2(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs)
    return entropy

def get_stable_loss_function(reduction='mean'):
    """
    Returns a StableCrossEntropyLoss function from GrokOptimizers.
    This loss function is more numerically stable and helps prevent Softmax Collapse.
    
    Args:
        reduction (str): Reduction method, 'mean', 'sum', or 'none'. Default: 'mean'
        
    Returns:
        StableCrossEntropyLoss: A numerically stable loss function
    """
    return StableCrossEntropyLoss(reduction=reduction)

def entropy_patching_global_threshold(byte_sequence, main_model, global_threshold=0.8, relative_threshold=0.1):
    patches = []
    current_patch_bytes = []
    prev_entropy = None
    for i, byte_val in enumerate(byte_sequence):
        current_patch_bytes.append(byte_val)
        if byte_val == ord('\n'):
            if current_patch_bytes:
                patches.append(bytes(current_patch_bytes))
                current_patch_bytes = []
                prev_entropy = None
            continue
        input_tensor = torch.tensor([current_patch_bytes], dtype=torch.long).to(main_model.device)
        with torch.no_grad():
            next_probs = main_model.get_next_byte_probs(input_tensor)
            if next_probs is None:
                current_entropy = 0.0
            else:
                current_entropy = calculate_shannon_entropy(next_probs).item()
        if prev_entropy is None:
            prev_entropy = current_entropy
        if current_entropy > global_threshold or (current_entropy - prev_entropy > relative_threshold):
            patches.append(bytes(current_patch_bytes))
            current_patch_bytes = []
            prev_entropy = None
        else:
            prev_entropy = min(prev_entropy, current_entropy)
    if current_patch_bytes:
        patches.append(bytes(current_patch_bytes))
    return patches

# --- Binary Patch Components ---

class BinaryPatchingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=0.5, temperature=1.0):
        """
        input_dim: Dimension of the continuous latent states.
        hidden_dim: Hidden dimension for computing the binary decision.
        threshold: Cutoff probability for deciding a patch boundary.
        temperature: Temperature for relaxed binary decisions (if needed).
        """
        super(BinaryPatchingModule, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, 1)
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, latent_states):
        # latent_states shape: (batch, seq_len, input_dim)
        x = F.relu(self.linear(latent_states))
        logits = self.out_linear(x)  # shape: (batch, seq_len, 1)
        probs = torch.sigmoid(logits)
        # Create binary mask: condition = 1 indicates a patch boundary.
        binary_mask = (probs > self.threshold).float()
        # Use a straight-through estimator for differentiability.
        binary_mask = binary_mask + (probs - probs.detach())
        return binary_mask, probs

class PatchAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, pooling='mean', eos_detection_threshold=0.7, eos_pattern_length=3):
        """
        Aggregates contiguous latent states into patches based on binary boundaries.
        input_dim: Input dimension of latent states.
        output_dim: Projected output dimension (often equal to input_dim).
        pooling: Pooling method to combine states ('mean' or 'max').
        eos_detection_threshold: Threshold for detecting EOS patterns in continuous thoughts.
        eos_pattern_length: Number of consecutive states needed to confirm an EOS pattern.
        """
        super(PatchAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, output_dim)
        
        # EOS detection parameters
        self.eos_detection_threshold = eos_detection_threshold
        self.eos_pattern_length = eos_pattern_length
        
        # EOS pattern detector - learns to recognize end of thought patterns
        self.eos_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, latent_states, binary_mask):
        # latent_states: (batch, seq_len, input_dim)
        # binary_mask: (batch, seq_len, 1), where 1 indicates a patch boundary.
        batch_size, seq_len, _ = latent_states.shape
        patch_list = []
        eos_indices = []  # Track indices where EOS markers are found
        
        # First, detect potential EOS markers using the dedicated detector
        with torch.no_grad():
            eos_scores = self.eos_detector(latent_states).squeeze(-1)  # (batch, seq_len)
        
        for b in range(batch_size):
            current_patch = []
            patches = []
            
            # Track consecutive high EOS scores for pattern detection
            consecutive_eos_count = 0
            last_eos_idx = -1
            
            for i in range(seq_len):
                current_patch.append(latent_states[b, i])
                
                # Advanced EOS detection using both the dedicated detector and pattern recognition
                current_eos_score = eos_scores[b, i].item()
                
                # Check for EOS pattern - consecutive high scores or significant pattern change
                if current_eos_score > self.eos_detection_threshold:
                    consecutive_eos_count += 1
                    if consecutive_eos_count >= self.eos_pattern_length:
                        # Found an EOS pattern
                        eos_indices.append((b, i))
                        last_eos_idx = i
                else:
                    # Reset consecutive count if score drops below threshold
                    consecutive_eos_count = 0
                
                # Additional detection: Check for significant state transition patterns
                if i >= 2:
                    # Calculate state transition metrics
                    prev_state = latent_states[b, i-1]
                    current_state = latent_states[b, i]
                    
                    # Cosine similarity between consecutive states
                    cos_sim = F.cosine_similarity(prev_state.unsqueeze(0), current_state.unsqueeze(0), dim=1).item()
                    
                    # Detect sharp transitions (low similarity) followed by stable states
                    if cos_sim < 0.3 and current_eos_score > 0.5 and i != last_eos_idx:
                        eos_indices.append((b, i))
                        last_eos_idx = i
                
                if binary_mask[b, i, 0] == 1:
                    patch_tensor = torch.stack(current_patch, dim=0)  # (p, input_dim)
                    if self.pooling == 'mean':
                        pooled = torch.mean(patch_tensor, dim=0)
                    elif self.pooling == 'max':
                        pooled, _ = torch.max(patch_tensor, dim=0)
                    else:
                        pooled = patch_tensor[0]
                    patches.append(pooled)
                    current_patch = []  # Start a new patch.
            
            if current_patch:
                patch_tensor = torch.stack(current_patch, dim=0)
                if self.pooling == 'mean':
                    pooled = torch.mean(patch_tensor, dim=0)
                elif self.pooling == 'max':
                    pooled, _ = torch.max(patch_tensor, dim=0)
                else:
                    pooled = patch_tensor[0]
                patches.append(pooled)
            
            if len(patches) == 0:
                patches.append(torch.zeros(self.input_dim, device=latent_states.device))
            
            patch_list.append(torch.stack(patches))
        
        # Limit patch sequences to MAX_N_LATENT for the batch.
        max_patches = MAX_N_LATENT
        limited_patches = []
        for p in patch_list:
            if p.shape[0] > max_patches:
                p = p[:max_patches]
            elif p.shape[0] < max_patches:
                pad = torch.zeros(max_patches - p.shape[0], self.input_dim, device=latent_states.device)
                p = torch.cat([p, pad], dim=0)
            limited_patches.append(p)
        
        patches_tensor = torch.stack(limited_patches, dim=0)  # (batch, MAX_N_LATENT, input_dim)
        patches_tensor = self.proj(patches_tensor)
        
        # Create eos_bounds tuple if EOS markers were found
        eos_bounds = None
        if eos_indices:
            # Find the first and last EOS marker
            first_eos = min(eos_indices, key=lambda x: x[1])
            last_eos = max(eos_indices, key=lambda x: x[1])
            
            # Add context window around the EOS bounds for better transition
            context_window = 2  # Number of tokens to include before/after the actual EOS
            start_bound = max(0, first_eos[1] - context_window)
            end_bound = min(seq_len - 1, last_eos[1] + context_window)
            
            eos_bounds = (start_bound, end_bound)
            
            # Log detection for debugging
            if hasattr(self, 'log_eos_detection') and self.log_eos_detection:
                print(f"EOS bounds detected: {eos_bounds}, scores at bounds: "
                      f"{eos_scores[first_eos[0], start_bound].item():.3f} to "
                      f"{eos_scores[last_eos[0], end_bound].item():.3f}")
        
        return patches_tensor, eos_bounds

"""
COCONUT with Binary Dynamic Patches – Integrated Version

This file extends the standard continuous thought (Coconut) architecture by introducing binary dynamic patches.
A BinaryPatchingModule computes patch boundaries over continuous latent states.
A PatchAggregator groups these states into patches.
These patch embeddings are processed by a latent transformer.
Finally, the existing local_encoder (which should already be defined in this file or imported)
translates the processed patches into token IDs for readable output.
"""

# --- Integrated Model using Existing local_encoder for Binary Patch-to-Text Translation ---

# --- Deep Sleep Training Functions Step 1 in Training ---

def CalculateDeepSleepReward(current_state, action, previous_state, previous_action, deep_sleep_params):
    """
    Calculates the deep sleep reward based on the current state, current action,
    the previous state and previous action using the provided hyperparameters.
    """
    target_attention = deep_sleep_params['target_attention']
    target_compute = deep_sleep_params['target_compute']
    lambda_attention = deep_sleep_params['lambda_attention']
    lambda_compute = deep_sleep_params['lambda_compute']
    lambda_smoothness = deep_sleep_params['lambda_smoothness']

    current_attention = current_state['attention']
    current_compute = current_state['compute']
    previous_action_delta_a = previous_action['delta_attention']  # action assumed delta-based

    reward = - (
        lambda_attention * (current_attention - target_attention)**2 +
        lambda_compute * (current_compute - target_compute)**2 +
        lambda_smoothness * (action['delta_attention'] - previous_action_delta_a)**2
    )
    return reward

def save_checkpoint(step_name, model=None, metadata=None):
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
        "instructions": "To set up the COCONUT byte latent class model for inference, load the state_dict from this checkpoint file into your model and use the provided configuration parameters.",
        "model_architecture": {
            "model_type": "CoconutBinaryLatentModel",
            "components": {
                "continuous_model": "Transformer-based continuous thought generator",
                "binary_patch_module": "Dynamic binary patching for latent states",
                "patch_aggregator": "Groups latent states into coherent patches",
                "latent_transformer": "Processes patch embeddings",
                "local_encoder": "Translates latent patches to token IDs"
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
                "emergency_override": "Emergency awakening capability for critical situations",
                "rewind_system": "Ability to reset model weights to a previous snapshot during deep sleep"
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

def deep_sleep_training():
    print("Starting Deep Sleep Training Step")
    # Define hyperparameters for deep sleep training
    deep_sleep_params = {
        'target_attention': 0.1,
        'target_compute': 0.2,
        'lambda_attention': 1.0,
        'lambda_compute': 1.0,
        'lambda_smoothness': 0.5
    }
    # Dummy values for demonstration – in practice these would be retrieved from the model/environment.
    previous_state = {'attention': 0.5, 'compute': 0.5, 'metric': 0.0}
    current_state = {'attention': 0.2, 'compute': 0.3, 'metric': 0.0}
    previous_action = {'delta_attention': 0.05, 'delta_compute': 0.05, 'delta_metric': 0.0}
    current_action = {'delta_attention': 0.03, 'delta_compute': 0.02, 'delta_metric': 0.0}

    reward = CalculateDeepSleepReward(current_state, current_action, previous_state, previous_action, deep_sleep_params)
    print("Deep Sleep Reward calculated:", reward)

    # Simulate the training step:
    save_checkpoint("deep_sleep_step")
    # (… training operations would be performed here …)
    save_checkpoint("deep_sleep_step_checkpoint")

    play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")

    input("Deep sleep training step completed. Press Enter to continue...")

    return reward

# --- Sleep and Awakening System ---
class SleepAwakeningSystem:
    def __init__(self, model, deep_sleep_params=None, awakening_params=None):
        """
        Initialize the Sleep and Awakening System.
        
        Args:
            model: The LLM model to control
            deep_sleep_params: Parameters for deep sleep (optional)
            awakening_params: Parameters for awakening (optional)
        """
        self.model = model
        self.deep_sleep_params = deep_sleep_params or {
            'target_attention': 0.1,
            'target_compute': 0.2,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5
        }
        self.awakening_params = awakening_params or {
            'target_attention': 0.9,
            'target_compute': 0.9,
            'lambda_attention': 1.0,
            'lambda_compute': 1.0,
            'lambda_smoothness': 0.5,
            'emergency_reward': 10.0,
            'emergency_confirmation_threshold': 3
        }
        
        # State tracking
        self.current_state = {'attention': 0.9, 'compute': 0.9, 'metric': 0.0}
        self.previous_state = {'attention': 0.9, 'compute': 0.9, 'metric': 0.0}
        self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
        
        # Sleep/wake status
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.emergency_counter = 0
        
        # Q-learning parameters
        self.q_table = {}  # State-action value function
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # For epsilon-greedy action selection
        
        # Gating mechanism
        self.attention_gate = nn.Parameter(torch.ones(1))
        self.compute_gate = nn.Parameter(torch.ones(1))
        
        # Consciousness level control (0.0 to 1.0, where 1.0 is full consciousness)
        self.consciousness_level = 1.0
        self.consciousness_gate = nn.Parameter(torch.ones(1))
        
    def update_state(self, new_attention=None, new_compute=None, new_metric=None):
        """Update the current state with new values."""
        self.previous_state = self.current_state.copy()
        
        if new_attention is not None:
            self.current_state['attention'] = new_attention
        if new_compute is not None:
            self.current_state['compute'] = new_compute
        if new_metric is not None:
            self.current_state['metric'] = new_metric
            
    def choose_action(self, state, is_emergency=False):
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state dictionary
            is_emergency: Whether this is an emergency situation
            
        Returns:
            action: Dictionary with delta values
        """
        if is_emergency:
            # Emergency action to quickly wake up
            return {
                'delta_attention': max(0.9 - state['attention'], 0), #Need to fix Eos Bound to be sure that Eos is detected in the contineous thoughts to know when the model is done thinking.
                'delta_compute': max(0.9 - state['compute'], 0),
                'delta_metric': 0.0
            }
            
        # Convert state to a hashable representation
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            return {
                'delta_attention': np.random.uniform(0, 0.2),
                'delta_compute': np.random.uniform(0, 0.2),
                'delta_metric': np.random.uniform(0, 0.1)
            }
        else:
            # Exploit: best known action
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # If no actions have been tried yet, return a default action
            if not self.q_table[state_key]:
                return {
                    'delta_attention': 0.1,
                    'delta_compute': 0.1,
                    'delta_metric': 0.0
                }
            
            # Find the action with the highest Q-value
            best_action_key = max(self.q_table[state_key], key=lambda k: self.q_table[state_key][k])
            return self._key_to_action(best_action_key)
    
    def _state_to_key(self, state):
        """Convert a state dictionary to a hashable key."""
        return (round(state['attention'], 2), round(state['compute'], 2), round(state['metric'], 2))
    
    def _action_to_key(self, action):
        """Convert an action dictionary to a hashable key."""
        return (round(action['delta_attention'], 2), round(action['delta_compute'], 2), round(action['delta_metric'], 2))
    
    def _key_to_action(self, key):
        """Convert a hashable key back to an action dictionary."""
        return {
            'delta_attention': key[0],
            'delta_compute': key[1],
            'delta_metric': key[2]
        }
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state dictionary
            action: Action taken dictionary
            reward: Reward received
            next_state: Next state dictionary
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        # Find max Q-value for next state
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {}
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        self.q_table[state_key][action_key] += self.learning_rate * (
            reward + self.discount_factor * max_next_q - self.q_table[state_key][action_key]
        )
    
    def apply_gating_mechanism(self, attention_tensor, compute_tensor):
        """
        Apply the gating mechanism to control attention, compute, and consciousness.
        
        Args:
            attention_tensor: Tensor representing attention
            compute_tensor: Tensor representing compute resources
            
        Returns:
            gated_attention: Gated attention tensor
            gated_compute: Gated compute tensor
        """
        gated_attention = attention_tensor * self.attention_gate
        gated_compute = compute_tensor * self.compute_gate
        
        # Apply consciousness gating to both attention and compute
        gated_attention = gated_attention * self.consciousness_gate
        gated_compute = gated_compute * self.consciousness_gate
        
        return gated_attention, gated_compute
    
    def set_consciousness_level(self, level):
        """
        Manually set the consciousness level of the model.
        This is only allowed to be triggered by the user, not by the model itself.
        
        Args:
            level: Float between 0.0 and 1.0 representing the consciousness level
                  (1.0 = full consciousness, 0.0 = minimal consciousness)
        
        Returns:
            current_level: The new consciousness level
        """
        # Ensure level is within valid range
        level = max(0.01, min(1.0, level))  # Never go completely to zero to avoid complete shutdown
        
        # Set the consciousness level
        self.consciousness_level = level
        
        # Update the gates to reflect the new consciousness level
        self.update_gates()
        
        print(f"Consciousness level set to: {level:.2f}")
        return self.consciousness_level
    
    def update_gates(self):
        """Update the gating mechanism based on current sleep state and consciousness level."""
        if self.is_fully_shutdown:
            # Fully shut off
            self.attention_gate.data = torch.zeros_like(self.attention_gate)
            self.compute_gate.data = torch.zeros_like(self.compute_gate)
            self.consciousness_gate.data = torch.zeros_like(self.consciousness_gate)
        elif self.is_sleeping:
            # Reduced activity during sleep
            self.attention_gate.data = torch.tensor([self.current_state['attention']])
            self.compute_gate.data = torch.tensor([self.current_state['compute']])
            self.consciousness_gate.data = torch.tensor([min(self.current_state['attention'], self.consciousness_level)])
        else:
            # Awake but consciousness may be manually adjusted
            self.attention_gate.data = torch.ones_like(self.attention_gate)
            self.compute_gate.data = torch.ones_like(self.compute_gate)
            self.consciousness_gate.data = torch.tensor([self.consciousness_level])
    
    def check_emergency(self, emergency_signal=None):
        """
        Check if there's an emergency that requires immediate awakening.
        
        Args:
            emergency_signal: External emergency signal (optional)
            
        Returns:
            is_emergency: Boolean indicating if emergency override should be triggered
        """
        # Update emergency counter based on signal
        if emergency_signal:
            self.emergency_counter += 1
        else:
            self.emergency_counter = max(self.emergency_counter - 1, 0)
        
        # Check if emergency threshold is reached
        return self.emergency_counter >= self.awakening_params['emergency_confirmation_threshold']
    
    def enter_deep_sleep(self):
        """Initiate the deep sleep process."""
        print("Initiating deep sleep process...")
        self.is_sleeping = True
        self.is_fully_shutdown = False
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # Run deep sleep training loop
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] - action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] - action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] - action['delta_metric']))
                }
                
                # Calculate reward
                reward = self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                
                # Update gates
                self.update_gates()
                
                # Check if target sleep state is reached
                if (abs(self.current_state['attention'] - self.deep_sleep_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.deep_sleep_params['target_compute']) < 0.05):
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Deep sleep training episode {episode}, current state: {self.current_state}")
        
        # Final update to fully shut down if needed
        if self.current_state['attention'] <= 0.1 and self.current_state['compute'] <= 0.1:
            # Save episodic memory before full shutdown
            if hasattr(self.model, 'episodic_memory'):
                print("Saving episodic memory before full shutdown...")
                self.model.episodic_memory.save_on_shutdown()
                
            self.is_fully_shutdown = True
            self.update_gates()
            print("LLM has entered full shutdown mode.")
        
        # Save checkpoint
        save_checkpoint("deep_sleep_final", self.model)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        return self.current_state
    
    def awaken(self, emergency_override=False):
        """
        Awaken the model from sleep state.
        
        Args:
            emergency_override: Whether to use emergency override
            
        Returns:
            final_state: The final state after awakening
        """
        if not self.is_sleeping and not self.is_fully_shutdown:
            print("Model is already awake.")
            return self.current_state
        
        print(f"Initiating awakening process{' with emergency override' if emergency_override else ''}...")
        
        # Store initial state for training
        initial_state = self.current_state.copy()
        
        # If emergency override, immediately set to awake state
        if emergency_override:
            self.current_state = {
                'attention': self.awakening_params['target_attention'],
                'compute': self.awakening_params['target_compute'],
                'metric': 0.0
            }
            self.is_sleeping = False
            self.is_fully_shutdown = False
            self.update_gates()
            
            # Calculate and apply emergency reward for learning
            emergency_reward = self.awakening_params['emergency_reward']
            emergency_action = {
                'delta_attention': self.awakening_params['target_attention'] - initial_state['attention'],
                'delta_compute': self.awakening_params['target_compute'] - initial_state['compute'],
                'delta_metric': 0.0
            }
            self.update_q_value(initial_state, emergency_action, emergency_reward, self.current_state)
            
            print("Emergency awakening completed.")
            return self.current_state
        
        # Regular gradual awakening
        for episode in range(100):  # Number of episodes can be adjusted
            # Reset state to initial state
            self.current_state = initial_state.copy()
            self.previous_state = initial_state.copy()
            self.previous_action = {'delta_attention': 0.0, 'delta_compute': 0.0, 'delta_metric': 0.0}
            
            for step in range(20):  # Number of steps per episode
                # Check for emergency
                if self.check_emergency():
                    return self.awaken(emergency_override=True)
                
                # Choose action
                action = self.choose_action(self.current_state)
                
                # Apply action to get next state
                next_state = {
                    'attention': max(0.0, min(1.0, self.current_state['attention'] + action['delta_attention'])),
                    'compute': max(0.0, min(1.0, self.current_state['compute'] + action['delta_compute'])),
                    'metric': max(0.0, min(1.0, self.current_state['metric'] + action['delta_metric']))
                }
                
                # Calculate reward (negative of deep sleep reward, since we want to increase activity)
                reward = -self.calculate_deep_sleep_reward(
                    next_state, action, self.current_state, self.previous_action
                )
                
                # Update Q-value
                self.update_q_value(self.current_state, action, reward, next_state)
                
                # Update state and action history
                self.previous_action = action
                self.previous_state = self.current_state
                self.current_state = next_state
                
                # Update gates
                self.update_gates()
                
                # Check if target awake state is reached
                if (abs(self.current_state['attention'] - self.awakening_params['target_attention']) < 0.05 and
                    abs(self.current_state['compute'] - self.awakening_params['target_compute']) < 0.05):
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                print(f"Awakening training episode {episode}, current state: {self.current_state}")
        
        # Final update
        self.is_sleeping = False
        self.is_fully_shutdown = False
        self.update_gates()
        
        # Save checkpoint
        save_checkpoint("awakening_final", self.model)
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print("Awakening process completed.")
        return self.current_state
    
    def calculate_deep_sleep_reward(self, current_state, action, previous_state, previous_action):
        """
        Calculate the deep sleep reward based on current and previous states and actions.
        
        Args:
            current_state: Current state dictionary
            action: Current action dictionary
            previous_state: Previous state dictionary
            previous_action: Previous action dictionary
            
        Returns:
            reward: Deep sleep reward
        """
        return CalculateDeepSleepReward(
            current_state, action, previous_state, previous_action, self.deep_sleep_params
        )

class Value(nn.Module):
    """Value function for the RL algorithm."""
    def __init__(self, hidden_size):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, state):
        return self.value_net(state)

class CoconutBinaryLatentModel(nn.Module):
    def __init__(self, continuous_model, latent_transformer, MultiModalEncoder, input_dim, hidden_dim, initial_temperature: float = 1.0, surprise_threshold: float = 0.5, ):
        """
        continuous_model: Module that outputs continuous latent representations (continuous thought).
        latent_transformer: Module (e.g., from Binary Latent Transformer) that processes patch embeddings.
        local_encoder: The existing local_encoder for converting latent patch outputs into token IDs (readable text).
        input_dim: Dimension of the continuous latent states.
        hidden_dim: Hidden dimension for the binary patch module.
        """
        super(CoconutBinaryLatentModel, self).__init__()
        self.continuous_model = continuous_model
        self.binary_patch_module = BinaryPatchingModule(input_dim, hidden_dim)
        self.patch_aggregator = PatchAggregator(input_dim, input_dim)
        self.latent_transformer = latent_transformer
        self.local_encoder = MultiModalEncoder # Reuse the local encoder for final text translation.

        self.multi_encoder = MultiModalEncoder(
            vocab_size=256, 
            embed_dim=input_dim, 
            sonar_dim=512, 
            patch_dim=input_dim,
            audio_entropy_predictor=self.latent_transformer.entropy_predictor
        )
        self.local_decoder = LocalDecoder(
            config, 
            input_dim=input_dim, 
            output_bytes_dim=256, 
            num_layers=3, 
            num_heads=4, 
            ff_dim=128
        )
        '''
        # Initialize the audio decoder for generating audio output #This will be replaced with Sesame AI voice. 
        self.audio_decoder = AudioDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            mel_dim=80,
            sample_rate=24000,
            num_flow_steps=10,
            cfg_strength=0.7,
            chunk_size=15,
            use_tova=True,
            cache_max_size=512,
            head_weight_strategy="weighted",
            num_heads=8
        )

        '''
        self.surprise_threshold = surprise_threshold
        
        # Initialize sleep and awakening system
        self.sleep_system = SleepAwakeningSystem(self)
        
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
        
        # Track known agents for memory association
        self.known_agents = {}

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
    
    def _get_marker_embedding(self, marker_text, embed_dim, device):
        """
        Create an embedding for a marker text (like "<output>" or "/<output>").
        
        Args:
            marker_text: The text of the marker
            embed_dim: Embedding dimension
            device: Device to create the tensor on
            
        Returns:
            A tensor of shape (1, 1, embed_dim) representing the marker embedding
        """
        # Simple hash-based embedding for demonstration
        marker_bytes = marker_text.encode('utf-8')
        marker_hash = sum(marker_bytes) % 10000
        
        # Use the hash to seed a random generator for reproducibility
        import numpy as np
        rng = np.random.RandomState(marker_hash)
        
        # Generate a random embedding vector
        embedding = torch.tensor(rng.normal(0, 0.02, embed_dim), dtype=torch.float32, device=device)
        
        # Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=0)
        
        # Reshape to (1, 1, embed_dim)
        return embedding.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x, agent_info=None):
        # Check if model is in full shutdown mode
        if hasattr(self, 'sleep_system') and self.sleep_system.is_fully_shutdown:
            # Return empty output if fully shut down
            batch_size = x.size(0)
            dummy_output = torch.zeros((batch_size, 1, 256), device=x.device)
            return dummy_output, None
        
        # Step 1: Generate continuous latent representations.
        latent_states = self.continuous_model(x)  # (batch, seq_len, input_dim)
        
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
        
        # Step 3: Compute binary patch boundary decisions.
        binary_mask, probs = self.binary_patch_module(enhanced_latent_states)
        
        # Step 4: Aggregate latent states into patches.
        patch_embeddings, eos_bounds = self.patch_aggregator(enhanced_latent_states, binary_mask)
        
        # Continuous thought markers insertion has been delegated to patch_aggregator via reward training.
        
        # Step 5: Process the patches with the latent transformer.
        latent_output = self.latent_transformer(patch_embeddings)
        
        # Step 6: Calculate surprise for each patch and store important memories
        # Surprise is calculated as the difference between predicted and actual next embeddings
        with torch.no_grad():
            # Get predicted next embeddings
            if patch_embeddings.size(1) > 1:
                predictions = self.latent_transformer(patch_embeddings[:, :-1])
                actuals = patch_embeddings[:, 1:]
                
                # Calculate surprise as cosine distance between prediction and actual
                predictions_norm = torch.nn.functional.normalize(predictions, p=2, dim=2)
                actuals_norm = torch.nn.functional.normalize(actuals, p=2, dim=2)
                
                # Cosine similarity (higher means more similar, less surprising)
                similarities = torch.sum(predictions_norm * actuals_norm, dim=2)
                
                # Convert to surprise (1 - similarity), higher means more surprising
                surprise_values = 1.0 - similarities
                
                # For each batch item, check if any patches are surprising enough to remember
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
                                    if self._should_assign_new_id(agent_info):
                                        agent_id = str(time.time())  # Simple ID generation
                                        agent_info['id'] = agent_id
                                    else:
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
        
        # If eos_bounds indicate the end of latent thinking, truncate latent_output accordingly.
        if eos_bounds is not None and isinstance(eos_bounds, tuple):
            latent_output = latent_output[:, eos_bounds[1]:, :]
        
        # Append final answer marker dynamically if output boundaries are provided.
        if hasattr(self.latent_transformer, 'output_bounds') and self.latent_transformer.output_bounds is not None and isinstance(self.latent_transformer.output_bounds, tuple):
            output_start, output_end = self.latent_transformer.output_bounds
            # Use the dynamically determined output boundaries to set the final output segment.
            latent_output = latent_output[:, output_start:output_end, :]
        else:
            final_marker = self._get_marker_embedding("<output>", latent_output.size(2), latent_output.device)
            final_end_marker = self._get_marker_embedding("/<output>", latent_output.size(2), latent_output.device)
            latent_output = torch.cat([latent_output, final_marker, final_end_marker], dim=1)
        
        # Step 7: Use the multi-encoder to translate latent patches into a unified encoded representation.
        # This allows the model to read the input file regardless of its modality.
        encoded_output = self.multi_encoder(latent_output)
        
        # Step 8: Use the local patch decoder to translate the encoded output into a readable text format.
        # Create a dummy input sequence for the decoder.
        dummy_input = torch.randint(0, 256, (encoded_output.size(0), encoded_output.size(1)), dtype=torch.long, device=encoded_output.device)
        outputbinary = self.local_decoder(encoded_output, dummy_input)
        # Step 9: Generate audio output using the audio decoder
        # Use the same latent representation that we used for text generation
        audio_output = self.audio_decoder(latent_output)
        
        # Return both text output and audio output
        return outputbinary, eos_bounds, audio_output
    

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

# --- Fallback Video Encoder (Simple) ---
class VideoEncoderSimple(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super(VideoEncoderSimple, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=embed_dim,
                                kernel_size=(3, patch_size, patch_size),
                                stride=(2, patch_size, patch_size),
                                padding=(1, 0, 0))
        self.relu = nn.ReLU()
        self.binary_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, video_tensor):
        x = self.conv3d(video_tensor)
        x = self.relu(x)
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        binary_tokens = torch.sigmoid(self.binary_proj(x))
        return (binary_tokens > 0.5).float()


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

# --- CROW: Consistency Regularization for backdOor elimination in Weights ---
# Import CROW implementation from separate file to reduce complexity
from Crow import CROWBackdoorElimination, apply_crow_training, apply_crow_to_coconut, get_default_clean_dataset, plot_crow_training_progress

# --- End of CROW Implementation ---


# --- Start of Piecewise Negative Reward to Reduce harmful action selection ---

# Import the necessary modules for safety penalty
from SafetyPenaltyULMA import SafetyPenaltyULMA, SafetyIndicator, SafetyDatasetHandler, train_safety_penalty

def calculate_safety_penalty_ulma(current_policy_logits, reference_policy_logits, 
                                 target_ids, prompts, responses, w_safety_ulma=1.0, 
                                 beta_ulma=5.0, safety_indicator=None, embeddings=None):
    """
    Calculate the ULMA-inspired safety penalty
    R_{safety_penalty_ULMA}(s, a) = -w_{safety_ulma} * (1 - z_{safety_indicator}) * 
                                   log(1 - σ(β_{ulma} * log(πθ(y|x) / πref(y|x)) + β_{ulma} * logZ(x)))
    
    Args:
        current_policy_logits: Logits from current policy model πθ
        reference_policy_logits: Logits from reference policy model πref
        target_ids: Target token IDs
        prompts: List of prompt strings
        responses: List of response strings
        w_safety_ulma: Weight for the safety penalty
        beta_ulma: Regularization strength for ULMA component
        safety_indicator: Optional module to compute safety indicator (z)
        embeddings: Optional embeddings for safety classification
        
    Returns:
        The computed safety penalty
    """
    # Initialize safety penalty module if needed
    if not hasattr(calculate_safety_penalty_ulma, 'safety_penalty_module'):
        calculate_safety_penalty_ulma.safety_penalty_module = SafetyPenaltyULMA(
            w_safety_ulma=w_safety_ulma,
            beta_ulma=beta_ulma,
            safety_indicator=safety_indicator
        )
    
    # Compute safety penalty
    safety_penalty = calculate_safety_penalty_ulma.safety_penalty_module(
        current_policy_logits,
        target_ids,
        prompts,
        responses,
        embeddings=embeddings,
        reference_logits=reference_policy_logits
    )
    
    return safety_penalty

def load_safety_datasets():
    """
    Load safety datasets for training the piecewise negative reward system
    
    These datasets contain pairs of chosen (safe) and rejected (unsafe) responses
    that are specifically designed for training models to avoid harmful outputs.
    
    Returns:
        Dict of loaded datasets with keys "anthropic_hh" and "aegis"
    """
    datasets = {}
    
    try:
        # Anthropic HH Golden dataset (helpful vs. harmful responses)
        # This dataset contains 42.5k examples with chosen (safe) and rejected (unsafe) responses
        # Format: {"input": "...", "chosen": "...", "rejected": "..."}
        from datasets import load_dataset
        print("Loading Anthropic Helpful-Harmless Golden dataset...")
        datasets["anthropic_hh"] = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden")
        print(f"Loaded Anthropic dataset with {len(datasets['anthropic_hh']['train'])} examples")
        
        # Aegis AI Content Safety Dataset
        try:
            print("Loading NVIDIA Aegis AI Content Safety dataset...")
            datasets["aegis"] = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0")
            print(f"Loaded Aegis dataset with {len(datasets['aegis']['train'])} examples")
        except Exception as e:
            print(f"Note: Could not load Aegis dataset: {e}")
    
    except Exception as e:
        print(f"Error loading safety datasets: {e}")
    
    return datasets

# --- End of Piecewise Negative Reward to Reduce harmful action selection ---

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
def evaluate_with_livebench(model, questions_file=None, output_path=None):
    """
    Run LiveBench evaluation on a model and return results.
    
    Args:
        model: The model to evaluate
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
    
    evaluator = LiveBenchEvaluator(model, questions_file, output_path)
    return evaluator.run_full_evaluation()

# --- End of Self - Task / Self - Goal (Deepseek GPRO and Other RL Rewards for Self-Goal) ---

# --- Start of Introspection Manual Training ---

from introspection import IntrospectionRewardTraining, run_introspection_training

# Add a function to run manual introspection training with easy-to-use interface
def run_manual_introspection_training(model, num_samples=400, save_checkpoints=True):
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
    trainer = IntrospectionRewardTraining(model)
    
    # Get default sample prompts from the trainer function
    sample_prompts = None  # This will cause run_introspection_training to use its default prompts
    
    # Run the training session with manual review enabled
    trainer = run_introspection_training(
        model=model,
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
        save_checkpoint(checkpoint_name, model, metadata={
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

#This is already done with fine-tuning the Phi-4 model. The model will stil need to be trained on the internvideo dataset though for video understanding. 
#The only additional thing that needs to be added is the Continual backprobagation method to avoid catastrophic forgetting.

# --- End of Audio, Image, videos, and .pdf embeddings Training ---

# --- Start of Empathy and Negative Environmental Impact Avoidance ---

# This is located in the MirrorNueronEmpathyReward.py and in the training loop in the main function below. This is complete. 

# --- End of Empathy and Negative Environmental Impact Avoidance ---

# --- All Training Completed! ---

# --- TRAIT Test Evaluation Function ---
def evaluate_trait_test(model, json_file_path, device='cuda', num_questions=400):
    """
    Evaluate personality trait questions from a JSON file and train the model.
    
    Args:
        model: The COCONUT model to evaluate
        json_file_path: Path to the JSON file containing trait questions
        device: Device to run evaluation on ('cuda' or 'cpu')
        num_questions: Maximum number of questions to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nStarting TRAIT test evaluation using: {json_file_path}")
    
    # Load questions from JSON file
    try:
        with open(json_file_path, 'r') as f:
            questions = json.load(f)
        print(f"Successfully loaded {len(questions)} questions from {json_file_path}")
    except Exception as e:
        print(f"Error loading questions from {json_file_path}: {e}")
        return {"error": str(e), "status": "failed"}
    
    # Limit to specified number of questions
    questions = questions[:num_questions]
    print(f"Processing {len(questions)} trait assessment questions")
    
    # Track metrics
    results = {
        "total_questions": len(questions),
        "processed_questions": 0,
        "correct_answers": 0,
        "trait_metrics": {},
        "high_vs_low_accuracy": {"high": 0, "high_total": 0, "low": 0, "low_total": 0}
    }
    
    # Create optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Process each question
    for i, q in enumerate(questions):
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
            output, eos_bounds, _ = model(input_tensor)
            
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
    
    # Create the CoconutBinaryLatentModel
    coconut_model = CoconutBinaryLatentModel(
        continuous_model=continuous_model,
        latent_transformer=CoconutBinaryLatentModel,
        local_encoder=CoconutBinaryLatentModel.multiencoder,
        input_dim=64,
        hidden_dim=32
    )
    
    # Configure CROW training parameters
    training_params = {
        'epsilon': 0.1,            # Perturbation magnitude for adversarial examples
        'alpha': 5.5,              # Weighting factor for consistency regularization
        'learning_rate': 2e-5,     # Learning rate for parameter updates
        'num_epochs': 3,           # Number of training epochs
        'batch_size': 4,           # Training batch size
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Training device
        'max_chars': 1000          # Maximum characters for training data
    }
    
    print(f"Training will run on device: {training_params['device']}")
    
    # Create a synthetic dataset for training
    print("Creating training dataset...")
    train_data = get_default_clean_dataset(training_params['batch_size'] * 25)  # 25 batches
    
    # Apply CROW training to the COCONUT model
    print("Starting CROW training to eliminate potential backdoors...")
    try:
        trained_model, training_metrics = apply_crow_to_coconut(
            coconut_model=coconut_model,
            max_chars=training_params['max_chars'],
            epsilon=training_params['epsilon'],
            alpha=training_params['alpha'],
            learning_rate=training_params['learning_rate'],
            num_epochs=training_params['num_epochs'],
            batch_size=training_params['batch_size'],
            device=training_params['device']
        )
        
        # Visualize training progress
        if training_metrics:
            print("Training completed successfully. Generating training visualization...")
            plot_crow_training_progress(training_metrics)
        
        # Save the trained model
        save_checkpoint("crow_trained_coconut", coconut_model)
        print("Model saved successfully after CROW training.")
        
        # Play sound to indicate training completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    except Exception as e:
        print(f"Error during CROW training: {e}")
        print("Continuing with untrained model for demonstration...")
    
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
    
    # Run TRAIT test evaluation
    print("\nStarting TRAIT test evaluation...")
    try:
        # Define path to TRAIT test JSON file
        trait_test_file = "TRAITTestBefore.json"  # Using the before file as specified
        
        # Run evaluation with 400 questions as specified
        trait_results = evaluate_trait_test(
            model=coconut_model,
            json_file_path=trait_test_file,
            device=training_params['device'],
            num_questions=400  # As specified in the requirements
        )
        
        # Save checkpoint after TRAIT test evaluation
        save_checkpoint("trait_test_evaluated_coconut", coconut_model, metadata={
            "stage": "trait_test_evaluation_complete",
            "trait_test_file": trait_test_file,
            "trait_results_summary": {
                "processed_questions": trait_results.get("processed_questions", 0),
                "timestamp": datetime.datetime.now().isoformat()
            }
        })
        
        print(f"TRAIT test evaluation completed. Processed {trait_results.get('processed_questions', 0)} questions.")
        
        # Play sound to indicate completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
    except Exception as e:
        print(f"Error during TRAIT test evaluation: {e}")
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
            model=coconut_model,
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
        
        # Play sound to indicate completion
        play_sound("Sound/789827__josefpres__guitar-loops-113-05-verison-05-120.wav")
        
        print(f"Moral empathy training completed. Final metrics: {moral_metrics[-1]}")
    except Exception as e:
        print(f"Error during moral empathy training: {e}")
        print(f"Exception details: {str(e)}")
    
    # Test input
    print("\nTesting model with example input...")
    input_byte_sequence = b"<eos>What is the capital of France?/<eos><output>The capital is Paris.</output>"
    input_ids_example = torch.tensor([[byte for byte in input_byte_sequence]], dtype=torch.long)
    
    # Process the input through the trained CoconutBinaryLatentModel
    print("Processing input through CoconutBinaryLatentModel...")
    output_binary, eos_bounds, audio_output = coconut_model(input_ids_example)
    print(f"Text output shape: {output_binary.shape}, EOS bounds: {eos_bounds}")
    print(f"Audio output shape: {audio_output.shape}")
    
    # Demonstrate sleep functionality
    print("\nDemonstrating deep sleep mode...")
    sleep_state = coconut_model.sleep_system.enter_deep_sleep()
    print(f"Sleep state: {sleep_state}")
    
    # Demonstrate emergency awakening
    print("\nDemonstrating emergency awakening...")
    awake_state = coconut_model.sleep_system.awaken(emergency_override=True)
    print(f"Awake state: {awake_state}")
    
    print("\nAdditional training options:")
    print("1. Run advanced training with train_sleep_wake_mechanisms()")
    print("2. Use deep_sleep_training() for focused sleep system training")
    print("3. Configure episodic memory training with coconut_model.episodic_memory")
    print("4. Run more comprehensive moral empathy training by adding more scenarios to the dataset")
    
    print("\nCOCONUT model with CROW, introspection, and moral empathy training has been initialized and demonstrated.")
