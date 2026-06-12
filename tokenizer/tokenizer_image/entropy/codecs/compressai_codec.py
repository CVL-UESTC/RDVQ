"""CompressAI rANS backend wrapper."""

from __future__ import annotations

from compressai.ans import BufferedRansEncoder, RansDecoder

from tokenizer.tokenizer_image.entropy.utils.profiling import _profile_add, _profile_tic, _profile_toc


class CompressAIEntropyCodec:
    """Encode and verify list-based CompressAI entropy packets."""

    backend = "compressai"

    def encode_packets(self, packets, profile=None) -> tuple[bytes, int, int]:
        if not packets:
            return b"", 0, 0

        t = _profile_tic(profile)
        encoder = BufferedRansEncoder()
        for packet in packets:
            encoder.encode_with_indexes(packet.symbols, packet.indexes, packet.cdfs, packet.cdf_lengths, packet.offsets)
        merged_stream = encoder.flush()
        _profile_toc(profile, "stream_merge.rans_encode_flush", t)

        symbol_count = sum(len(packet.symbols) for packet in packets)
        bits = len(merged_stream) * 8
        _profile_add(profile, "stream_merge.packets", len(packets))
        _profile_add(profile, "stream_merge.symbols", symbol_count)
        _profile_add(profile, "stream_merge.payload_bits", bits)
        return merged_stream, bits, symbol_count

    def verify_packets(self, stream, packets, profile=None) -> None:
        if not packets:
            return

        t = _profile_tic(profile)
        decoder = RansDecoder()
        decoder.set_stream(stream)
        for packet in packets:
            decoded = decoder.decode_stream(packet.indexes, packet.cdfs, packet.cdf_lengths, packet.offsets)
            if decoded != packet.symbols:
                raise AssertionError("Merged stream decode mismatch!")
        _profile_toc(profile, "stream_merge.rans_decode", t)
