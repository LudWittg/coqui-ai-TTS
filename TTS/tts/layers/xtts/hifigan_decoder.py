import logging

import torch
from trainer.io import load_fsspec

from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from TTS.vocoder.models.hifigan_generator import HifiganGenerator

logger = logging.getLogger(__name__)


class HifiDecoder(torch.nn.Module):
    def __init__(
        self,
        input_sample_rate=22050,
        output_sample_rate=24000,
        output_hop_length=256,
        ar_mel_length_compression=1024,
        decoder_input_dim=1024,
        resblock_type_decoder="1",
        resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes_decoder=[3, 7, 11],
        upsample_rates_decoder=[8, 8, 2, 2],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        d_vector_dim=512,
        cond_d_vector_in_each_upsampling_layer=True,
        speaker_encoder_audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64,
        },
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.speaker_encoder_audio_config = speaker_encoder_audio_config
        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            1,
            resblock_type_decoder,
            resblock_dilation_sizes_decoder,
            resblock_kernel_sizes_decoder,
            upsample_kernel_sizes_decoder,
            upsample_initial_channel_decoder,
            upsample_rates_decoder,
            inference_padding=0,
            cond_channels=d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
            cond_in_each_up_layer=cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def base_forward(self, latents, g=None):
        """
        Args:
            latents (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform (B,1,T).
        """
        # Ensure tensors are contiguous for MPS performance
        latents = latents.contiguous()
        if g is not None:
            g = g.contiguous()

        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        ) #.squeeze(1)
        # print(f"After first interpolate shape: {z.shape}")
        # z = z.squeeze(1)  # This was causing the issue with batches
        # print(f"After first squeeze shape: {z.shape}")
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            ) #.squeeze(0)
            # print(f"After second interpolate shape: {z.shape}")
            # z = z.squeeze(0)  # This was also problematic for batches
            # print(f"After second squeeze shape: {z.shape}")
        o = self.waveform_decoder(z, g=g)
        return o

    @torch.inference_mode()
    def forward(self, latents, g=None):
        # Process in chunks for memory efficiency
        # FIX: increase chunk_size to keep GPU busier
        chunk_size = 1600000  # was 32000; now bigger to reduce overhead
        if latents.shape[1] > chunk_size:
            chunks = []
            for i in range(0, latents.shape[1], chunk_size):
                chunk = latents[:, i : i + chunk_size]
                out_chunk = self.base_forward(chunk, g)
                chunks.append(out_chunk)
            return torch.cat(chunks, dim=2)
        return self.base_forward(latents, g)

    @torch.inference_mode()
    def inference(self, c, g):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.

        Returns:
            Tensor: output waveform.

        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        return self.forward(c, g=g)

    def load_checkpoint(self, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        # remove unused keys
        state = state["model"]
        states_keys = list(state.keys())
        for key in states_keys:
            if "waveform_decoder." not in key and "speaker_encoder." not in key:
                del state[key]

        self.load_state_dict(state)
        if eval:
            self.eval()
            assert not self.training
            self.waveform_decoder.remove_weight_norm()
