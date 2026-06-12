# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import logging

import torch
import torch._dynamo.config
import torch._inductor.config
from fvcore.nn import FlopCountAnalysis


from tokenizer.tokenizer_image.compression.real.profiling import (
    _env_choice,
    _env_flag,
    _profile_add,
    _profile_tic,
    _profile_toc,
)
from tokenizer.tokenizer_image.compression.real.sampling import sample
from tokenizer.tokenizer_image.compression.real.validation import _compare_generation_results, _pack_result
from tokenizer.tokenizer_image.compression.real.streaming import (
    _append_streams,
    _current_valid_mask,
    _encode_streams,
    _fill_invalid_tokens,
    _should_transmit,
    verify_restore_from_encoder_packets,
)


logger = logging.getLogger(__name__)

# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future



class ARWrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, kwargs):
        return self.net.ar_one_step(**kwargs)




def prefill(
    model,
    cond_idx: torch.Tensor,
    input_pos: torch.Tensor,
    gt_indices=None,
    codebook=None,
    should_transmit=True,
    valid_token_mask=None,
    padding_token=0,
    profile=None,
    **sampling_kwargs,
):
    # First AR slice has no previous token history, so it is conditioned only
    # on the dummy class/condition token. When should_transmit=True, ar_one_step
    # also entropy-codes the ground-truth targets for this slice.
    if sampling_kwargs['return_flops']:
        with torch.no_grad():
            flops = FlopCountAnalysis(
                ARWrapper(model),
                {"idx": None, "cond_idx": cond_idx, "input_pos": input_pos, "targets": gt_indices if should_transmit else None, "codebook": codebook, "entropy_coding": False},
            )
        global total_flops
        total_flops += flops.total()
        torch.cuda.empty_cache()

    if should_transmit:
        logits, entropy, bitstreams, ind = model.ar_one_step(
            None,
            cond_idx,
            input_pos,
            targets=gt_indices,
            codebook=codebook,
            entropy_coding=True,
            valid_token_mask=valid_token_mask,
            padding_token=padding_token,
            profile=profile,
        )
    else:
        logits, entropy = model.ar_one_step(None, cond_idx, input_pos, targets=None, codebook=codebook, entropy_coding=False, profile=profile)
        bitstreams, ind = None, None

    t = _profile_tic(profile, logits)
    sampled, probs = sample(logits, **sampling_kwargs)
    _profile_toc(profile, "generate.sample", t, logits)
    return sampled, probs, entropy, bitstreams, ind


def decode_one_token(
    model,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    all_tokens=None,
    gt_indices=None,
    codebook=None,
    should_transmit=True,
    valid_token_mask=None,
    padding_token=0,
    profile=None,
    **sampling_kwargs,
):
    # Later AR slices consume the previous slice token plus the accumulated
    # all_tokens history. The same function is used for transmitted slices
    # and for sampled suffix slices controlled by transfer_slices.
    if input_pos.shape[-1] != 1:
        raise ValueError(f"decode_one_token expects a single input position, got shape {tuple(input_pos.shape)}")
    if sampling_kwargs['return_flops']:
        with torch.no_grad():
            flops = FlopCountAnalysis(
                ARWrapper(model),
                {"idx": x, "cond_idx": None, "input_pos": input_pos, "all_tokens": all_tokens, "targets": gt_indices if should_transmit else None, "codebook": codebook, "entropy_coding": False},
            )
        global total_flops
        total_flops += flops.total()
        torch.cuda.empty_cache()

    if should_transmit:
        logits, entropy, bitstreams, ind = model.ar_one_step(
            x,
            cond_idx=None,
            input_pos=input_pos,
            all_tokens=all_tokens,
            targets=gt_indices,
            codebook=codebook,
            entropy_coding=True,
            valid_token_mask=valid_token_mask,
            padding_token=padding_token,
            profile=profile,
        )
    else:
        logits, entropy = model.ar_one_step(
            x,
            cond_idx=None,
            input_pos=input_pos,
            all_tokens=all_tokens,
            targets=None,
            codebook=codebook,
            entropy_coding=False,
            profile=profile,
        )
        bitstreams, ind = None, None

    t = _profile_tic(profile, logits)
    sampled, probs = sample(logits, **sampling_kwargs)
    _profile_toc(profile, "generate.sample", t, logits)
    return sampled, probs, entropy, bitstreams, ind


def decode_n_tokens(
    model,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    num_slices,
    gt_indices=None,
    transfer_slices=1,
    codebook=None,
    entropy_all=None,
    valid_token_mask=None,
    padding_token=0,
    profile=None,
    **sampling_kwargs,
):
    # all_tokens is the running decoder/encoder history in flattened latent
    # order. mask_all maps each position back to its AR slice id.
    new_tokens, new_probs, streams_all = [], [], []
    all_tokens = torch.full(
        (cur_token.shape[0], num_new_tokens),
        int(padding_token),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )

    mask_all = model.transformer.mask_all
    pos_start = mask_all == 0
    all_tokens[:, pos_start] = cur_token.long()

    for _ in range(num_slices):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            t_empty_cache = _profile_tic(profile)
            torch.cuda.empty_cache()
            _profile_toc(profile, "generate.empty_cache", t_empty_cache)
            should_transmit = _should_transmit(input_pos, transfer_slices)
            # Transmitted prefix slices use ground-truth targets and emit rANS
            # packets; suffix slices are generated from the AR sampler only.
            next_token, next_prob, entropy, bitstreams, ind = decode_one_token(
                model,
                cur_token,
                input_pos,
                all_tokens=all_tokens,
                gt_indices=gt_indices,
                codebook=codebook,
                should_transmit=should_transmit,
                valid_token_mask=valid_token_mask,
                padding_token=padding_token,
                profile=profile,
                **sampling_kwargs,
            )
            pos_insert = mask_all == input_pos
            valid_current = _current_valid_mask(valid_token_mask, mask_all, input_pos)

            _append_streams(streams_all, bitstreams)

            if should_transmit:
                entropy_all[:, pos_insert] = entropy
                next_token = ind
            next_token = _fill_invalid_tokens(next_token, valid_current, padding_token)

            cur_token = next_token.view(-1, 1)
            all_tokens[:, pos_insert] = next_token
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())

    return new_tokens, new_probs, entropy_all, all_tokens, streams_all



_FULL_FORWARD_VALIDATION_CACHE = {}





def _full_forward_enabled():
    return _env_choice("RDVQ_ENCODER_FULL_FORWARD", "0") in {"1", "true", "yes", "on", "auto"}


def _full_forward_auto_mode():
    return _env_choice("RDVQ_ENCODER_FULL_FORWARD", "0") == "auto"


def _full_forward_supported(gt_indices, codebook, transfer_slices, num_slices, return_flops):
    # Full teacher-forced encoding only covers lossless/full-transfer coding.
    # Partial-transfer generation still needs the sampling path after transmitted slices.
    return (
        gt_indices is not None
        and codebook is not None
        and not return_flops
        and int(transfer_slices) >= int(num_slices)
    )


def _transmit_mask(mask_all, transfer_slices):
    return mask_all < int(transfer_slices)


def _full_forward_validation_key(model, shape_list, gt_indices):
    dtype = str(next(model.parameters()).dtype)
    device = str(next(model.parameters()).device)
    return (tuple(tuple(int(v) for v in shape) for shape in shape_list), tuple(gt_indices.shape), dtype, device)




def _build_entropy_packets_from_logits(model, logits_all, targets, transfer_slices, valid_token_mask, padding_token, profile=None):
    """Build per-slice entropy packets from teacher-forced logits.

    This keeps the exact old packet order: one packet per transmitted AR slice.
    The first validation run compares the resulting merged stream with the
    original sequential AR path before the fast path is trusted for a shape.
    """
    device = logits_all.device
    # full_forward_logits returns logits for all flattened token positions at
    # once. This loop repacks them into the same per-slice packet order used by
    # the sequential AR path so validation and downstream bit accounting match.
    mask_all = model.transformer.mask_all.to(device)
    targets = targets.to(device)
    valid_token_mask = None if valid_token_mask is None else valid_token_mask.to(device=device, dtype=torch.bool)
    entropy_all = torch.zeros((targets.shape[0], targets.shape[1]), dtype=torch.float32, device=device)
    streams_all = []

    for slice_idx in range(int(mask_all.max().item()) + 1):
        if slice_idx >= int(transfer_slices):
            continue
        curr_pos = mask_all == slice_idx
        if not bool(curr_pos.any().item()):
            continue
        logits = logits_all[:, curr_pos, :]
        targets_current = targets[:, curr_pos]
        current_valid_mask = None
        if valid_token_mask is not None:
            current_valid_mask = valid_token_mask[:, curr_pos]
            fill = torch.full_like(targets_current, int(padding_token))
            targets_current = torch.where(current_valid_mask, targets_current, fill)

        t = _profile_tic(profile, device)
        entropy = model.cross_entropy_log2(logits.reshape(-1, logits.size(-1)), targets_current.reshape(-1), reduction="none")
        entropy = entropy.view(targets_current.shape[0], -1)
        if current_valid_mask is not None:
            entropy = entropy * current_valid_mask.to(entropy.dtype)
        entropy_all[:, curr_pos] = entropy
        _profile_toc(profile, "ar.full_forward_entropy_loss", t, device)

        t = _profile_tic(profile, device)
        bitstreams, _ = model.entropy_code_ans(
            logits,
            targets_current,
            coding_mask=current_valid_mask,
            fill_value=padding_token,
            profile=profile,
            packet_position=slice_idx,
        )
        _profile_toc(profile, "ar.full_forward_entropy_coding_call", t, device)
        _append_streams(streams_all, bitstreams)

    return entropy_all, streams_all


@torch.no_grad()
def _generate_encoder_full_forward(
    model,
    shape_list,
    k=4,
    gt_indices=None,
    transfer_slices=1,
    Bs=1,
    mask_padded=None,
    codebook=None,
    padding_token=0,
    return_flops=False,
    return_stats=False,
    profile=None,
    **sampling_kwargs,
):
    # Full-forward mode is an encoder-side acceleration for full-transfer
    # cases: codebook vectors from gt_indices are fed to the AR model in one
    # teacher-forced pass, then entropy packets are built from the logits.
    max_new_tokens = sum([pn[0] * pn[1] for pn in shape_list])
    num_slices = sum([k * 2 ** i for i in range(len(shape_list))])
    device = model.start_token.device
    valid_token_mask = None if mask_padded is None else (mask_padded.to(device) == 0)

    t_total = _profile_tic(profile, device)
    model.transformer.destroy_caches()
    model.transformer.build_mask_and_freq_cis(shape_list, k=k)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        quant = codebook[gt_indices.to(device)]
        t = _profile_tic(profile, device)
        logits_all = model.full_forward_logits(quant, updated_shape=shape_list)
        _profile_toc(profile, "ar.full_forward_transformer", t, device)
    _profile_add(profile, "ar.full_forward_calls", 1)

    entropy_all, streams_all = _build_entropy_packets_from_logits(
        model,
        logits_all,
        gt_indices,
        transfer_slices,
        valid_token_mask,
        padding_token,
        profile=profile,
    )
    if valid_token_mask is not None:
        entropy_all *= valid_token_mask.to(entropy_all.dtype)
    bits_all = entropy_all.sum().item()

    encoded_entropy_stream = _encode_streams(streams_all, profile=profile)
    payload_bits = encoded_entropy_stream.payload_bits

    if valid_token_mask is None:
        all_tokens = gt_indices.to(device).long().clone()
        valid_token_count = Bs * max_new_tokens
        padded_token_count = 0
        skipped_padded_token_count = 0
    else:
        fill = torch.full_like(gt_indices.to(device), int(padding_token))
        all_tokens = torch.where(valid_token_mask, gt_indices.to(device).long(), fill.long())
        valid_token_count = int(valid_token_mask.sum().item())
        padded_token_count = int((~valid_token_mask).sum().item())
        skipped_padded_token_count = int((~valid_token_mask[:, _transmit_mask(model.transformer.mask_all, transfer_slices)]).sum().item())

    transmitted_token_count = int(_transmit_mask(model.transformer.mask_all, transfer_slices).sum().item()) * Bs
    if valid_token_mask is not None:
        transmitted_token_count = int(valid_token_mask[:, _transmit_mask(model.transformer.mask_all, transfer_slices)].sum().item())

    all_tokens, decoded_slice_count, decoded_token_count = verify_restore_from_encoder_packets(
        all_tokens,
        encoded_entropy_stream,
        model.transformer.mask_all,
        profile=profile,
    )

    if _env_flag("RDVQ_DECODER_TIMING_PASS", False):
        _replay_decoder_entropy_model(
            model,
            shape_list,
            all_tokens,
            transfer_slices=transfer_slices,
            codebook=codebook,
            valid_token_mask=valid_token_mask,
            padding_token=padding_token,
            profile=profile,
        )

    _profile_toc(profile, "real.generate_encoder_full_forward", t_total, device)

    stats = {
        "estimated_bits": float(bits_all),
        "payload_bits": int(payload_bits),
        "entropy_packet_count": int(encoded_entropy_stream.packet_count),
        "entropy_symbol_count": int(encoded_entropy_stream.symbol_count),
        "entropy_stream_backend": encoded_entropy_stream.backend,
        "_encoded_entropy_stream": encoded_entropy_stream,
        "encoder_entropy_model_mode": "teacher_forced_full_forward",
        "decoder_token_source": "rans_bitstream_decode" if _env_flag("RDVQ_USE_BITSTREAM_DECODED_TOKENS", True) else "encoder_simulation",
        "bitstream_decoded_slice_count": int(decoded_slice_count),
        "bitstream_decoded_token_count": int(decoded_token_count),
        "valid_token_count": int(valid_token_count),
        "padded_token_count": int(padded_token_count),
        "skipped_padded_token_count": int(skipped_padded_token_count),
        "transmitted_token_count": int(transmitted_token_count),
    }
    return _pack_result(all_tokens, bits_all, streams_all, stats, return_stats)


@torch.no_grad()
def _replay_decoder_entropy_model(
    model,
    shape_list,
    decoded_tokens,
    transfer_slices=1,
    codebook=None,
    valid_token_mask=None,
    padding_token=0,
    profile=None,
):
    """Replay decoder-side causal logits using already decoded history.

    The replay does not consume the rANS stream yet; it measures the AR model
    work the decoder must perform between entropy decode steps.  Each slice is
    computed before that slice's tokens are written into the history buffer.
    """
    device = model.start_token.device
    decoded_tokens = decoded_tokens.to(device=device, dtype=torch.long)
    valid_token_mask = None if valid_token_mask is None else valid_token_mask.to(device=device, dtype=torch.bool)
    max_new_tokens = decoded_tokens.shape[1]
    k = int(getattr(model, "num_ar_per_scale", 4))
    num_slices = sum([k * 2 ** i for i in range(len(shape_list))])
    cond = torch.zeros(decoded_tokens.shape[0], dtype=torch.int, device=device).view(decoded_tokens.shape[0], 1)
    all_tokens = torch.full_like(decoded_tokens, int(padding_token))

    t_total = _profile_tic(profile, device)
    with torch.device(device):
        model.transformer.setup_caches(max_batch_size=decoded_tokens.shape[0], shape_list=shape_list, dtype=model.start_token.dtype)

    input_pos = torch.arange(0, 1, device=device)
    t = _profile_tic(profile, device)
    model.ar_one_step(None, cond, input_pos, targets=None, codebook=codebook, entropy_coding=False, profile=profile)
    _profile_toc(profile, "real.entropy_model_decode_logits", t, device)
    pos = model.transformer.mask_all == 0
    current = decoded_tokens[:, pos]
    if valid_token_mask is not None:
        current = _fill_invalid_tokens(current, valid_token_mask[:, pos], padding_token)
    all_tokens[:, pos] = current
    cur_token = current.view(-1, 1)

    input_pos = torch.tensor([1], device=device, dtype=torch.int)
    for _ in range(num_slices - 1):
        slice_idx = int(input_pos.item())
        t = _profile_tic(profile, device)
        model.ar_one_step(
            cur_token,
            cond_idx=None,
            input_pos=input_pos,
            all_tokens=all_tokens,
            targets=None,
            codebook=codebook,
            entropy_coding=False,
            profile=profile,
        )
        _profile_toc(profile, "real.entropy_model_decode_logits", t, device)

        pos = model.transformer.mask_all == input_pos
        current = decoded_tokens[:, pos]
        if valid_token_mask is not None:
            current = _fill_invalid_tokens(current, valid_token_mask[:, pos], padding_token)
        all_tokens[:, pos] = current
        cur_token = current.view(-1, 1)
        input_pos += 1
        if slice_idx + 1 >= int(transfer_slices):
            # Continue writing history only for transmitted full-transfer mode.
            # Partial-transfer generation is intentionally handled by the old path.
            pass

    t_destroy = _profile_tic(profile, device)
    model.transformer.destroy_caches()
    _profile_toc(profile, "generate.decoder_destroy_caches", t_destroy, device)
    _profile_toc(profile, "real.entropy_model_decode", t_total, device)
    _profile_add(profile, "ar.decoder_replay_calls", 1)
    return all_tokens


@torch.no_grad()
def _generate_sequential(
    model,
    shape_list,
    k=4,
    gt_indices=None,
    transfer_slices=1,
    Bs=1,
    mask_padded=None,
    codebook=None,
    padding_token=0,
    return_flops=False,
    return_stats=False,
    profile=None,
    **sampling_kwargs,
):
    # Sequential mode mirrors the original AR decoding order: prefill slice 0,
    # then iterate slice by slice. It is slower but supports partial transfer
    # because suffix slices can be sampled after the transmitted prefix.
    max_new_tokens = sum([pn[0] * pn[1] for pn in shape_list])
    num_slices = sum([k * 2 ** i for i in range(len(shape_list))])
    cond = torch.zeros(Bs, dtype=torch.int, device=model.start_token.device).view(Bs, 1)
    cond_combined = cond
    T = 1
    max_batch_size = cond.shape[0]
    streams_all = []

    sampling_kwargs['return_flops'] = return_flops

    if sampling_kwargs['return_flops']:
        global total_flops
        total_flops = 0
    else:
        total_flops = None

    device = cond.device
    valid_token_mask = None if mask_padded is None else (mask_padded.to(device) == 0)
    entropy_all = torch.zeros_like(model.mask_all, dtype=torch.float, device=model.mask_all.device).unsqueeze(0).repeat(max_batch_size, 1)

    t_setup = _profile_tic(profile, device)
    with torch.device(device):
        model.transformer.setup_caches(max_batch_size=max_batch_size, shape_list=shape_list, dtype=model.start_token.dtype)
    _profile_toc(profile, "generate.setup_caches", t_setup, device)

    input_pos = torch.arange(0, T, device=device)
    should_transmit = _should_transmit(input_pos, transfer_slices)
    next_token, _, entropy_start, bitstreams, ind = prefill(
        model,
        cond_combined,
        input_pos,
        gt_indices=gt_indices,
        codebook=codebook,
        should_transmit=should_transmit,
        valid_token_mask=valid_token_mask,
        padding_token=padding_token,
        profile=profile,
        **sampling_kwargs,
    )
    _append_streams(streams_all, bitstreams)

    pos_start = model.transformer.mask_all == 0
    valid_current = _current_valid_mask(valid_token_mask, model.transformer.mask_all, input_pos)
    if should_transmit:
        next_token = ind
        entropy_all[:, pos_start] = entropy_start
    next_token = _fill_invalid_tokens(next_token, valid_current, padding_token)

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _, entropy_all, all_tokens, streams = decode_n_tokens(
        model,
        next_token,
        input_pos,
        max_new_tokens,
        num_slices - 1,
        gt_indices=gt_indices,
        transfer_slices=transfer_slices,
        codebook=codebook,
        entropy_all=entropy_all,
        valid_token_mask=valid_token_mask,
        padding_token=padding_token,
        profile=profile,
        **sampling_kwargs,
    )

    streams_all.extend(streams)
    if valid_token_mask is not None:
        entropy_all *= valid_token_mask.to(entropy_all.dtype)

    bits_all = entropy_all.sum().item()
    # Pack all per-slice entropy packets into the selected stream backend. The
    # optional replay below can replace/sanity-check all_tokens from the actual
    # rANS payload rather than trusting encoder-side tensors.
    encoded_entropy_stream = _encode_streams(streams_all, profile=profile)
    payload_bits = encoded_entropy_stream.payload_bits

    all_tokens, decoded_slice_count, decoded_token_count = verify_restore_from_encoder_packets(
        all_tokens,
        encoded_entropy_stream,
        model.transformer.mask_all,
        profile=profile,
    )

    if valid_token_mask is None:
        valid_token_count = max_batch_size * max_new_tokens
        padded_token_count = 0
        skipped_padded_token_count = 0
    else:
        valid_token_count = int(valid_token_mask.sum().item())
        padded_token_count = int((~valid_token_mask).sum().item())
        transmit_positions = model.transformer.mask_all < transfer_slices
        skipped_padded_token_count = int((~valid_token_mask[:, transmit_positions]).sum().item())

    transmitted_token_count = int((model.transformer.mask_all < transfer_slices).sum().item()) * max_batch_size
    if valid_token_mask is not None:
        transmitted_token_count = int(valid_token_mask[:, model.transformer.mask_all < transfer_slices].sum().item())

    t_destroy = _profile_tic(profile, device)
    model.transformer.destroy_caches()
    _profile_toc(profile, "generate.destroy_caches", t_destroy, device)

    if _env_flag("RDVQ_DECODER_TIMING_PASS", False):
        _replay_decoder_entropy_model(
            model,
            shape_list,
            all_tokens,
            transfer_slices=transfer_slices,
            codebook=codebook,
            valid_token_mask=valid_token_mask,
            padding_token=padding_token,
            profile=profile,
        )

    stats = {
        "estimated_bits": float(bits_all),
        "payload_bits": int(payload_bits),
        "entropy_packet_count": int(encoded_entropy_stream.packet_count),
        "entropy_symbol_count": int(encoded_entropy_stream.symbol_count),
        "entropy_stream_backend": encoded_entropy_stream.backend,
        "_encoded_entropy_stream": encoded_entropy_stream,
        "decoder_token_source": "rans_bitstream_decode" if _env_flag("RDVQ_USE_BITSTREAM_DECODED_TOKENS", True) else "encoder_simulation",
        "bitstream_decoded_slice_count": int(decoded_slice_count),
        "bitstream_decoded_token_count": int(decoded_token_count),
        "valid_token_count": int(valid_token_count),
        "padded_token_count": int(padded_token_count),
        "skipped_padded_token_count": int(skipped_padded_token_count),
        "transmitted_token_count": int(transmitted_token_count),
    }

    if sampling_kwargs['return_flops']:
        if return_stats:
            return all_tokens, bits_all, streams_all, total_flops, stats
        return all_tokens, bits_all, streams_all, total_flops
    if return_stats:
        return all_tokens, bits_all, streams_all, stats
    return all_tokens, bits_all, streams_all


@torch.no_grad()
def generate(
    model,
    shape_list,
    k=4,
    gt_indices=None,
    transfer_slices=1,
    Bs=1,
    mask_padded=None,
    codebook=None,
    padding_token=0,
    return_flops=False,
    return_stats=False,
    profile=None,
    **sampling_kwargs,
):
    # Entry point used by compression_pipeline fast mode. Decide whether the
    # current image/patch can use the full-forward encoder shortcut; otherwise
    # fall back to the sequential AR path.
    num_slices = sum([k * 2 ** i for i in range(len(shape_list))])
    can_use_full = _full_forward_enabled() and _full_forward_supported(gt_indices, codebook, transfer_slices, num_slices, return_flops)
    if not can_use_full:
        return _generate_sequential(
            model,
            shape_list,
            k=k,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            Bs=Bs,
            mask_padded=mask_padded,
            codebook=codebook,
            padding_token=padding_token,
            return_flops=return_flops,
            return_stats=return_stats,
            profile=profile,
            **sampling_kwargs,
        )

    key = _full_forward_validation_key(model, shape_list, gt_indices)
    validate = _env_flag("RDVQ_ENCODER_FULL_FORWARD_VALIDATE", True)
    strict = _env_flag("RDVQ_ENCODER_FULL_FORWARD_STRICT", False)
    cached = _FULL_FORWARD_VALIDATION_CACHE.get(key)

    if validate and cached is None:
        # Before trusting the shortcut for a new shape, compare its tokens, bit
        # counts, and stream metadata against the sequential reference path.
        validation_profile = {} if profile is not None else None
        full_result = _generate_encoder_full_forward(
            model,
            shape_list,
            k=k,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            Bs=Bs,
            mask_padded=mask_padded,
            codebook=codebook,
            padding_token=padding_token,
            return_flops=return_flops,
            return_stats=return_stats,
            profile=validation_profile,
            **sampling_kwargs,
        )
        seq_result = _generate_sequential(
            model,
            shape_list,
            k=k,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            Bs=Bs,
            mask_padded=mask_padded,
            codebook=codebook,
            padding_token=padding_token,
            return_flops=return_flops,
            return_stats=return_stats,
            profile=None,
            **sampling_kwargs,
        )
        ok, reason = _compare_generation_results(full_result, seq_result, return_stats=return_stats)
        _FULL_FORWARD_VALIDATION_CACHE[key] = ok
        if profile is not None:
            _profile_add(profile, "ar.full_forward_validation_pass", int(ok))
            _profile_add(profile, "ar.full_forward_validation_fail", int(not ok))
        if ok:
            if profile is not None and validation_profile is not None:
                for key_profile, value_profile in validation_profile.items():
                    _profile_add(profile, key_profile, value_profile)
            logger.info("[RDVQ] encoder full-forward validation passed for shape=%s", shape_list)
            return full_result
        message = f"[RDVQ] encoder full-forward validation failed for shape={shape_list}: {reason}"
        if strict or not _full_forward_auto_mode():
            raise AssertionError(message)
        logger.warning("%s; falling back to sequential AR encoder", message)
        return _generate_sequential(
            model,
            shape_list,
            k=k,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            Bs=Bs,
            mask_padded=mask_padded,
            codebook=codebook,
            padding_token=padding_token,
            return_flops=return_flops,
            return_stats=return_stats,
            profile=profile,
            **sampling_kwargs,
        )

    if cached is False:
        return _generate_sequential(
            model,
            shape_list,
            k=k,
            gt_indices=gt_indices,
            transfer_slices=transfer_slices,
            Bs=Bs,
            mask_padded=mask_padded,
            codebook=codebook,
            padding_token=padding_token,
            return_flops=return_flops,
            return_stats=return_stats,
            profile=profile,
            **sampling_kwargs,
        )

    return _generate_encoder_full_forward(
        model,
        shape_list,
        k=k,
        gt_indices=gt_indices,
        transfer_slices=transfer_slices,
        Bs=Bs,
        mask_padded=mask_padded,
        codebook=codebook,
        padding_token=padding_token,
        return_flops=return_flops,
        return_stats=return_stats,
        profile=profile,
        **sampling_kwargs,
    )


# Semantic aliases for legacy/benchmark callers. The old underscored names stay
# available inside this module, while external code can import intent-revealing
# names that distinguish fast encoder shortcuts from the simplified causal codec.
legacy_fast_sequential_roundtrip = _generate_sequential
legacy_teacher_forced_encoder = _generate_encoder_full_forward
