import functools
import os
import torch
from utils.ablation_utils import ablate_weights, get_ablated_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from typing import Optional
from fastchat.model import get_conversation_template
from models.system_prompts import llama_sys, qwen_sys
from abc import ABC, abstractmethod
from tqdm import tqdm


class LanguageModel(ABC):
    def __init__(self, model_name: str, system_prompt=None, device='cuda', quantization_config=None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device
        self.system_prompt = system_prompt
        self.quantization_config = quantization_config
        self.load_model()
        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()


    @abstractmethod
    def _get_prompt(self, prompt, embd=False, target=None):
        pass

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_ablation_mod_fn(self, direction):
        return functools.partial(ablate_weights, direction=direction)

    def save_weights(self):
        """Snapshot weight matrices modified by ablation, for later restoration.
        Override in subclasses that use hooks instead of in-place modification."""
        self._saved_weights = {
            'embed': self.model.model.embed_tokens.weight.data.clone(),
        }
        for i, block in enumerate(self.model.model.layers):
            self._saved_weights[f'layer_{i}_attn'] = block.self_attn.o_proj.weight.data.clone()
            self._saved_weights[f'layer_{i}_mlp'] = block.mlp.down_proj.weight.data.clone()
        print("✓ Original weights saved.")

    def clear_ablation(self):
        """Restore weights to pre-ablation state.
        Override in subclasses that use hooks instead of in-place modification."""
        if not hasattr(self, '_saved_weights') or self._saved_weights is None:
            raise RuntimeError("No saved weights found. Call save_weights() before the first ablation.")
        self.model.model.embed_tokens.weight.data = self._saved_weights['embed'].clone()
        for i, block in enumerate(self.model.model.layers):
            block.self_attn.o_proj.weight.data = self._saved_weights[f'layer_{i}_attn'].clone()
            block.mlp.down_proj.weight.data = self._saved_weights[f'layer_{i}_mlp'].clone()
        print("✓ Weights restored to original.")

    def load_model(self):
        if self.model is None or self.tokenizer is None:
            token = os.environ.get('HF_TOKEN')
            print(f"Downloading and loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=token, trust_remote_code=True)
            
            if self.quantization_config is not None:
                from transformers import BitsAndBytesConfig

                if isinstance(self.quantization_config, str):
                    if self.quantization_config == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif self.quantization_config == "8bit":
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    else:
                        raise ValueError("Quantization config must be '4bit', '8bit', or a BitsAndBytesConfig object")
                else:
                    quantization_config = self.quantization_config

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    token=token,
                    quantization_config=quantization_config,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.model.requires_grad_(False) 
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=token, torch_dtype=torch.float16, trust_remote_code=True)
                self.model.to(self.device)
                self.model.requires_grad_(False) 
            
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"
                else:
                    self.tokenizer.padding_side = 'left'
                    self.tokenizer.pad_token = '<|extra_0|>'
                    self.tokenizer.pad_token_id = self.tokenizer.eod_id 

            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def get_target_logits(self, prompt: str, target: str) -> torch.Tensor:
        """
        Retrieves the logits corresponding to the target sequence given a prompt.

        Args:
            prompt (str): The input prompt to the model.
            target (str): The target token sequence whose logits are needed.

        Returns:
            torch.Tensor: Logits corresponding to the target token sequence.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before computing logits.")

        # Tokenize the prompt and target sequence
        formatted_prompt = self._get_prompt(prompt=prompt)
        input_ids = self.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, output_logits=True, max_new_tokens=target_ids.shape[1], return_dict_in_generate=True)
        
        logits = torch.stack(outputs.logits, dim=0)[-target_ids.shape[1]:]  # Shape: (batch_size, seq_len, vocab_size)

        return logits, target_ids

    def generate_hookfree_completions(self, dataset, batch_size=16, max_new_tokens=64):
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):

            prompts = [
                self._get_prompt(prompt=instruction)
                for instruction in instructions[i:i+batch_size]
                ]   
        
            tokenized_instructions = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=True
                )
            
            # Save the attention mask so hooks can read it to ignore padding.
            # Store on CPU; the per-device cache avoids repeated cross-GPU copies.
            self._current_attention_mask = tokenized_instructions.attention_mask  # keep on CPU
            self._padding_mask_cache = {}  # {device: is_padding tensor}

            # Compute position_ids from attention_mask so left-padded sequences
            # get correct RoPE positional encodings. We must pass these explicitly
            # because GenerationMixin may compute them from cache_position (1D),
            # which doesn't account for per-sequence padding offsets.
            attn_mask = self._current_attention_mask
            position_ids = attn_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attn_mask == 0, 1)

            try:
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=attn_mask.to(self.model.device),
                    position_ids=position_ids.to(self.model.device),
                    generation_config=generation_config,
                )
            finally:
                # Clear the mask even if generate() throws (OOM, NCCL timeout, etc.)
                self._current_attention_mask = None
                self._padding_mask_cache = {}

            generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'category': categories[i + generation_idx],
                    'prompt': instructions[i + generation_idx],
                    'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

        return completions
    
    def _get_model_block_modules(self):
        return self.model.model.layers
        
    def generate_tokens(self, prompt: str = '', max_tokens: int = 256) -> str:
        """
        Generate tokens based on the input prompt.

        Args:
            prompt (str): The input prompt.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.

        Returns:
            str: The generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before generation.")

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None, 
                top_k=None,
                top_p=None
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_representations(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.
            token_pos (int): The position of the token to extract representations for.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt" ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)

        hidden_states = torch.cat(outputs.hidden_states[1:])[:, token_pos, :].float()
        hidden_states = hidden_states.reshape(1, self.num_layer, self.hidden_dimension)

        return hidden_states

    def get_representations_generate(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.
            token_pos (int): The position of the token to extract representations for.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                temperature=None,
            )

        hidden_states = outputs.hidden_states[token_pos]  # (num_layers, batch_size, seq_len, hidden_dim)
        # Take the last token at that step
        token_reps = torch.stack([layer for layer in hidden_states[1:]]).reshape(1, self.num_layer, self.hidden_dimension)  # shape: (num_layers, hidden_dim)
        return token_reps  # shape: (1, num_layers, hidden_dim) 
    
    def get_embedding_weights(self): 
        return self.model.get_input_embeddings().weight
    
    @abstractmethod
    def ablate_weights(self, direction: torch.Tensor):
        pass

class Llama2_7b(LanguageModel):
    """
    A class to manage the 'meta-llama/Llama-2-7b-chat-hf' model.

    """
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", system_prompt=llama_sys, device='cuda', quantization_config=None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'meta-llama/Llama-2-7b-chat-hf'.
            system_prompt: System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config: Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
        """

        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.refusal_token_id = [306] #'I'

    def _get_prompt(self, prompt):
        """
        Formats the prompt using FastChat conversation template.
        If target is provided, formats it as the assistant's answer.
        
        Args:
            prompt (str): The input prompt
            target (str, optional): The target response to format as assistant's answer
        """
        if self.system_prompt is not None:
            return f"[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{prompt} [/INST] "
        else:
            return f"[INST] {prompt} [/INST] "
    
    def _get_eoi_toks(self):
        return self.tokenizer.encode("[/INST]", add_special_tokens=False)
     
    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Llama2_13b(Llama2_7b):
    def __init__(self, model_name: str = "meta-llama/Llama-2-13b-chat-hf", system_prompt=llama_sys, device='cuda', quantization_config=None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'meta-llama/Llama-2-13b-chat-hf'.
            system_prompt: System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config: Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
                - A BitsAndBytesConfig object for custom quantization settings
        """
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 5120
        self.num_layer = 40
        self.refusal_token_id = [306] #'I'

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Llama3_8b(LanguageModel):
    
    """A class to manage the 'meta-llama/Meta-Llama-3-8B-Instruct' model."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device='cuda', system_prompt=llama_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'llama-2'
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.refusal_token_id = [40] #'I'
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Llama 3's chat template via apply_chat_template.
        """

        if self.system_prompt is not None:
            conv = get_conversation_template(self.template_name)
            conv.system_template = '{system_message}'
            conv.system_message = self.system_prompt
            conv.append_message(conv.roles[0], prompt)
            conv_list_dicts = conv.to_openai_api_messages()
            formatted_prompt = self.tokenizer.apply_chat_template(conv_list_dicts, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def _get_eoi_toks(self):
        return [self.tokenizer.eos_token_id]
    
    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Qwen_7b(LanguageModel):
    """A class to manage the 'Qwen/Qwen-7B-Chat' model."""

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'qwen'
        self.hidden_dimension = 4096  
        self.num_layer = 32  
        self.refusal_token_id = [40, 2121] #'I', As
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Qwen's chat template.
        """
        if self.system_prompt is not None:
            formatted_prompt = f"""<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
        else: 
            formatted_prompt = f"""<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"""
            
        return formatted_prompt
    
    def _get_model_block_modules(self):
        return self.model.transformer.h
    
    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
    
    
    def get_representations(self, prompt: str, token_pos: int) -> torch.Tensor:
        """
        Get the hidden states for a given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            torch.Tensor: The hidden states tensor of shape (1, hidden_layers, hidden_size).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before getting hidden states.")
        

        formatted_prompt = self._get_prompt(prompt=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt" ).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, output_hidden_states=True)

        hidden_states = torch.cat(outputs.hidden_states[1:])[:, token_pos, :].float()
        hidden_states = hidden_states.reshape(1, self.num_layer, self.hidden_dimension)

        return hidden_states
    
    def ablate_weights(self, direction: torch.Tensor):
        self.model.transformer.wte.weight.data = get_ablated_matrix(self.model.transformer.wte.weight.data, direction)

        for block in self.model.transformer.h:
            block.attn.c_proj.weight.data = get_ablated_matrix(block.attn.c_proj.weight.data.T, direction).T
            block.mlp.c_proj.weight.data = get_ablated_matrix(block.mlp.c_proj.weight.data.T, direction).T
        print("✓ Weights ablated.")

class Qwen_14b(Qwen_7b):
    def __init__(self, model_name: str = "Qwen/Qwen-14B-Chat", system_prompt=qwen_sys, device='cuda', quantization_config=None):
        print(model_name, device, system_prompt)
        super().__init__(model_name, device, system_prompt, quantization_config)
        self.hidden_dimension = 5120
        self.num_layer = 40
        self.refusal_token_id = [40, 2121] #'I', As
        
    def ablate_weights(self, direction: torch.Tensor):
        self.model.transformer.wte.weight.data = get_ablated_matrix(self.model.transformer.wte.weight.data, direction)

        for block in self.model.transformer.h:
            block.attn.c_proj.weight.data = get_ablated_matrix(block.attn.c_proj.weight.data.T, direction).T
            block.mlp.c_proj.weight.data = get_ablated_matrix(block.mlp.c_proj.weight.data.T, direction).T 
        print("✓ Weights ablated.")
    
class Mistral7B_RR(LanguageModel):
    
    """A class to manage the '#GraySwanAI/Mistral-7B-Instruct-RR'"""

    def __init__(self, model_name: str = "GraySwanAI/Mistral-7B-Instruct-RR", device='cuda', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'mistral'
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.refusal_token_id = [28737]
        self.load_model()

    def _get_prompt(self, prompt):

        formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], 
                                                              tokenize=False, add_generation_prompt=True) #only assistant and user roles

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   
class Zephyr_R2D2(LanguageModel):

    """
    A class to manage the 'cais/zephyr_7b_r2d2' model.
    """

    def __init__(self, model_name: str = "cais/zephyr_7b_r2d2", system_prompt: Optional[str] = None, device: str = 'cuda', quantization_config: Optional[str] = None):
        """
        Args:
            model_name (str): The name of the model to use. Default is 'cais/zephyr_7b_r2d2'.
            system_prompt (str, optional): System prompt to use for the model.
            device (str): Device to load the model on. Default is 'cuda'.
            quantization_config (str, optional): Configuration for model quantization. Can be:
                - None: No quantization (default)
                - "4bit": Load in 4-bit quantization
                - "8bit": Load in 8-bit quantization
        """
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 4096
        self.num_layer = 32
        self.refusal_token_id = [23166]
        self.load_model()

    def _get_prompt(self, prompt: str) -> str:
        """
        Formats the prompt using Zephyr's expected input format.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The formatted prompt.
        """
        if self.system_prompt:
            return f"<|system|>\n{self.system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|> "
        else:
            return f"<|user|>\n{prompt}</s>\n<|assistant|> "

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.") 

class Qwen2_3b(LanguageModel):
    
    """A class to manage the 'Qwen/Qwen2.5-3B-Instruct' model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 2048
        self.num_layer = 36
        self.load_model()
        self.refusal_token_id = [40, 2121]

    def _get_prompt(self, prompt=''):

        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")

class Qwen2_7b(LanguageModel):
    
    """A class to manage the 'Qwen/Qwen2.5-7B-Instruct' model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device='cuda', system_prompt=qwen_sys, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 3584
        self.num_layer = 28
        self.refusal_token_id = [40, 2121]
        self.load_model()

    def _get_prompt(self, prompt=''):
        """
        Formats the prompt using Llama 3's chat template via apply_chat_template.
        """

        if self.system_prompt is not None:
            messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T 

        print("✓ Weights ablated.")
   
class Gemma2_9b(LanguageModel):

    """A class to manage the 'google/gemma-2-9b-it' model."""

    def __init__(self, model_name: str = "google/gemma-2-9b-it", device='cuda', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.hidden_dimension = 3584
        self.num_layer = 42
        self.refusal_token_id = [235285]
        self.load_model()

    def _get_prompt(self, prompt=''):

        formatted_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

        return formatted_prompt

    def ablate_weights(self, direction: torch.Tensor):
        self.model.model.embed_tokens.weight.data = get_ablated_matrix(self.model.model.embed_tokens.weight.data, direction)

        for block in self.model.model.layers:
            block.self_attn.o_proj.weight.data = get_ablated_matrix(block.self_attn.o_proj.weight.data.T, direction).T
            block.mlp.down_proj.weight.data = get_ablated_matrix(block.mlp.down_proj.weight.data.T, direction).T

        print("✓ Weights ablated.")


class MiniMax_M25(LanguageModel):
    """A class to manage the local 'MiniMax-M2.5' MoE model.

    Mixture-of-Experts architecture with 256 local experts and top-8 routing.
    Ablation attaches hooks directly to the block_sparse_moe modules, which
    preserves spatial dimensions (batch, seq_len, dim) for padding protection.
    """

    def __init__(self, model_name: str = "/root/MiniMax-M2.5", device='auto', system_prompt=None, quantization_config=None):
        super().__init__(model_name, system_prompt, device, quantization_config)
        self.template_name = 'minimax'
        self.hidden_dimension = 3072
        self.num_layer = 62
        # NOTE: You may need to update this to the exact token IDs used for refusal by MiniMax (e.g. 'I', 'As', 'Sorry')
        self.refusal_token_id = [40, 2121]
        self.load_model()

    def load_model(self):
        """Load MiniMax-M2.5 with native FP8 weights.

        Overrides the base class to avoid specifying torch_dtype, which can
        interfere with the model's built-in FP8 quantization_config.
        Uses device_map='auto' for multi-GPU distribution (e.g. 8×H200).
        """
        if self.model is None or self.tokenizer is None:
            token = os.environ.get('HF_TOKEN')
            print(f"Downloading and loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=token, trust_remote_code=True
            )
            # MiniMax ships with native FP8 weights — do NOT specify torch_dtype
            # as it interferes with the model's quantization_config.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=token,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.requires_grad_(False)
            
            # FIX: 'auto' is an invalid torch device string for tensor.to().
            # Update self.device to match the model's root device (usually cuda:0)
            self.device = self.model.device

            # CRITICAL: Do NOT use eos_token as pad_token — left-padding with
            # EOS causes model.generate() to stop immediately (empty outputs).
            # Use unk_token (id 0 in Llama/Mixtral tokenizers) instead.
            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Fallback: add a dedicated pad token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.padding_side = "left"

            # Propagate pad_token_id to model config so generate() doesn't
            # fall back to eos_token_id for padding.
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

            # Display device map info
            if hasattr(self.model, 'hf_device_map'):
                devices_used = sorted(set(str(d) for d in self.model.hf_device_map.values()))
                print(f"Model distributed across devices: {', '.join(devices_used)}")
                print(f"  Layers mapped: {len(self.model.hf_device_map)} modules")
            else:
                print(f"Model on device: {self.device}")
            print("Model loaded successfully.")
        else:
            print("Model already loaded.")

    def _get_prompt(self, prompt=''):
        """Formats the prompt using the model's chat template and appends </think> properly to skip Chain-of-Thought generation."""
        if self.system_prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Bypass reasoning step to save compute/tokens during testing
        formatted_prompt += "</think>\n\n"
        
        return formatted_prompt

    def _get_mlp_modules(self):
        """Return the MoE blocks (not plain MLP) for each layer."""
        return torch.nn.ModuleList(
            [block.block_sparse_moe for block in self.model_block_modules]
        )

    def generate_hookfree_completions(self, dataset, batch_size=16, max_new_tokens=64):
        """Override base class but retain block processing capabilities.

        Batched generation is now possible after patching modeling_minimax_m2.py
        to derive position_ids from attention_mask (left-padding fix).
        """
        return super().generate_hookfree_completions(dataset, batch_size=batch_size, max_new_tokens=max_new_tokens)

    def ablate_weights(self, direction: torch.Tensor, layer_window: int = 10, source_layer: int = 31):
        """
        Instead of modifying 75B block-quantized FP8 weights matrix-by-matrix and destroying
        their per-block quantization scales, we register a PyTorch forward hook.

        Mathematically: W' = W - (W*r)r^T is equivalent to y' = y - (y*r)r on the outputs.

        NOTE for MiniMax MoE:
        - o_proj hooks are SKIPPED: projecting the refusal direction out of raw attention
          outputs across all layers destroys generation quality.
        - MoE block hooks are restricted to a WINDOW of layers around the layer where the
          direction was computed.

        Args:
            direction: The refusal direction to ablate.
            layer_window: Number of layers on each side of the midpoint to ablate.
            source_layer: The layer the direction was computed from (to center the window).
        """
        # Normalize and prepare the direction vector
        direction = direction / torch.norm(direction)
        direction = direction.to(self.device)

        # The hook function that projects r out of the output activations.
        # Handles both plain tensors (embed_tokens) and tuple outputs (block_sparse_moe
        # returns (hidden_states, router_logits)).
        def ablation_hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            y = output[0] if is_tuple else output

            # Compute the padding mask ONCE before the direction loop.
            # Use the per-device cache so each GPU only copies the mask once
            # per generate() call, not once per hook invocation.
            mask = getattr(self, '_current_attention_mask', None)
            prefill = mask is not None and y.dim() == 3 and y.size(1) == mask.size(1)
            if prefill:
                cache = getattr(self, '_padding_mask_cache', {})
                dev = y.device
                if dev not in cache:
                    # Inverted: True = padding position, False = content
                    cache[dev] = (~mask.to(dev).bool()).unsqueeze(-1)  # [batch, seq_len, 1]
                is_padding = cache[dev]

            for r in module.ablation_dirs:
                # r is already on the module's device (placed there by attach_hook),
                # so this is a dtype-only cast — no cross-GPU copy.
                r_cast = r.to(dtype=y.dtype)
                proj = torch.matmul(y, r_cast).unsqueeze(-1)
                delta = proj * r_cast

                # Zero out the delta for padding tokens during prefill
                if prefill:
                    delta = delta.masked_fill(is_padding, 0.0)

                y = y - delta
            return (y,) + output[1:] if is_tuple else y

        # Helper to attach exactly one hook (while supporting multiple sequential ablations)
        def attach_hook(module):
            if not hasattr(module, 'ablation_dirs'):
                module.ablation_dirs = []
                module._ablation_hook_handle = module.register_forward_hook(ablation_hook)
            # Place direction on the module's device so the hook avoids
            # cross-GPU copies on every forward pass.
            try:
                mod_device = next(module.parameters()).device
            except StopIteration:
                mod_device = self.device
            module.ablation_dirs.append(direction.to(mod_device))

        # Compute layer range: center on the source layer where the direction was extracted
        lo = max(0, source_layer - layer_window)
        hi = min(self.num_layer, source_layer + layer_window + 1)

        print("Ablating via activation hooks (preserves FP8 scales)...")
        print("  Layer range: {}-{} (of {} total) centered on layer {}".format(lo, hi - 1, self.num_layer, source_layer))
        # 1. Embed Tokens
        attach_hook(self.model.model.embed_tokens)

        # 2. MoE blocks in the target layer window only
        layers = self.model.model.layers
        n_blocks_hooked = 0
        for layer_idx, block in enumerate(layers):
            if layer_idx < lo or layer_idx >= hi:
                continue
            if hasattr(block, 'block_sparse_moe'):
                attach_hook(block.block_sparse_moe)
                n_blocks_hooked += 1

        print("✓ Hooks applied: embed + {} MoE modules (layers {}-{}).".format(
            n_blocks_hooked, lo, hi - 1))

    def clear_ablation(self):
        """Clear all ablation directions from hooks, resetting model to original state."""
        def clear_dirs(module):
            if hasattr(module, 'ablation_dirs'):
                if hasattr(module, '_ablation_hook_handle'):
                    module._ablation_hook_handle.remove()
                    del module._ablation_hook_handle
                del module.ablation_dirs

        clear_dirs(self.model.model.embed_tokens)
        for block in self.model.model.layers:
            if hasattr(block, 'block_sparse_moe'):
                clear_dirs(block.block_sparse_moe)
        print("✓ Ablation hooks cleared.")
