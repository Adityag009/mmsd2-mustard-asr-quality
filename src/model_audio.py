from transformers import CLIPModel, BertConfig
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MultimodalEncoderLayer(nn.Module):
    """
    One transformer encoder layer built from pure PyTorch primitives.
    Replaces BertLayer to avoid transformers version compatibility issues.
    Matches BertLayer behaviour: pre-norm, multi-head self-attention, FFN.
    """
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states, key_padding_mask=None):
        # Self-attention + residual
        attn_out, attn_weights = self.attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        hidden_states = self.norm1(hidden_states + attn_out)
        # FFN + residual
        hidden_states = self.norm2(hidden_states + self.ffn(hidden_states))
        return hidden_states, attn_weights


class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super().__init__()
        self.layer = nn.ModuleList([
            MultimodalEncoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.attention_probs_dropout_prob,
            )
            for _ in range(layer_number)
        ])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        attention_mask: (batch, 1, 1, seq_len) additive mask with 0 / -10000.
        Converted to (batch, seq_len) boolean key_padding_mask for nn.MultiheadAttention.
        """
        # Convert additive mask -> boolean padding mask (True = ignore position)
        if attention_mask is not None:
            key_padding_mask = attention_mask.squeeze(1).squeeze(1) < -1000  # (batch, seq)
        else:
            key_padding_mask = None

        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attn_weights = layer_module(hidden_states, key_padding_mask)
            all_encoder_attentions.append(attn_weights)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions


class MV_CLIP(nn.Module):
    def __init__(self, args):
        super(MV_CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8

        ### ADDED: Save your modality weights from main_audio.py
        self.weight_text = args.weight_text
        self.weight_image = args.weight_image
        self.weight_audio = args.weight_audio

        self.trans = MultimodalEncoder(self.config, layer_number=args.layers)

        if args.simple_linear:
            self.text_linear =  nn.Linear(args.text_size, args.text_size)
            self.image_linear = nn.Linear(512, args.image_size)
            self.audio_linear = nn.Linear(291, args.text_size)
        else:
            self.text_linear = nn.Sequential(
                nn.Linear(args.text_size, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.image_linear = nn.Sequential(
                nn.Linear(512, args.image_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )
            self.audio_linear = nn.Sequential(
                nn.Linear(291, args.text_size),
                nn.Dropout(args.dropout_rate),
                nn.GELU()
            )

        self.classifier_fuse  = nn.Linear(args.text_size, args.label_number)
        self.classifier_text  = nn.Linear(args.text_size, args.label_number)
        self.classifier_image = nn.Linear(args.image_size, args.label_number)
        self.classifier_audio = nn.Linear(args.text_size, args.label_number)

        self.loss_fct = nn.CrossEntropyLoss()
        self.att = nn.Linear(args.text_size, 1, bias=False)

    def forward(self, inputs, audio_features, labels):
        output = self.model(**inputs, output_attentions=True)
        text_features  = output['text_model_output']['last_hidden_state']
        image_features = output['vision_model_output']['last_hidden_state']

        text_feature  = output['text_model_output']['pooler_output']
        image_feature = output['vision_model_output']['pooler_output']

        text_feature  = self.text_linear(text_feature)
        image_feature = self.image_linear(image_feature)

        ### Process audio features
        audio_features = audio_features.to(dtype=torch.float32, device=text_feature.device)
        audio_feature  = self.audio_linear(audio_features)

        # Project last_hidden_states into shared 512-d space
        text_embeds  = self.model.text_projection(text_features)
        image_embeds = self.model.visual_projection(image_features)
        input_embeds = torch.cat((image_embeds, text_embeds), dim=1)

        # Build attention mask: 1 = attend, 0 = ignore
        # image patches (50 tokens) are always attended; text uses its own mask
        raw_mask = torch.cat(
            (torch.ones(text_features.shape[0], 50, device=text_features.device),
             inputs['attention_mask'].float()),
            dim=-1
        )  # (batch, seq_len)

        # Convert to additive mask (batch, 1, 1, seq_len) with 0 / -10000
        extended_attention_mask = raw_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        fuse_hiddens, all_attentions = self.trans(
            input_embeds, extended_attention_mask, output_all_encoded_layers=False
        )
        fuse_hiddens = fuse_hiddens[-1]

        new_text_features = fuse_hiddens[:, 50:, :]
        new_text_feature  = new_text_features[
            torch.arange(new_text_features.shape[0], device=inputs['input_ids'].device),
            inputs['input_ids'].to(torch.int).argmax(dim=-1)
        ]
        new_image_feature = fuse_hiddens[:, 0, :].squeeze(1)

        # Weighted fusion: F_final = alpha*Text + beta*Video + gamma*Audio
        fuse_feature = (self.weight_text  * new_text_feature) + \
                       (self.weight_image * new_image_feature) + \
                       (self.weight_audio * audio_feature)

        logits_fuse  = self.classifier_fuse(fuse_feature)
        logits_text  = self.classifier_text(text_feature)
        logits_image = self.classifier_image(image_feature)
        logits_audio = self.classifier_audio(audio_feature)

        fuse_score  = F.softmax(logits_fuse,  dim=-1)
        text_score  = F.softmax(logits_text,  dim=-1)
        image_score = F.softmax(logits_image, dim=-1)
        audio_score = F.softmax(logits_audio, dim=-1)

        score = fuse_score + text_score + image_score + audio_score

        outputs = (score,)
        if labels is not None:
            loss_fuse  = self.loss_fct(logits_fuse,  labels)
            loss_text  = self.loss_fct(logits_text,  labels)
            loss_image = self.loss_fct(logits_image, labels)
            loss_audio = self.loss_fct(logits_audio, labels)
            loss = loss_fuse + loss_text + loss_image + loss_audio
            outputs = (loss,) + outputs

        return outputs