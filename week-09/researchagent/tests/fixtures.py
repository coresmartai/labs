"""Synthetic chunks, so the tests run with no corpus fetched and no network."""
from app.schemas import Chunk, SearchHit

CHUNKS = [
    Chunk(chunk_uid="rfc9989:0007", rfc="rfc9989", section="4.4 Alignment",
          page=14, text="Identifier alignment requires the RFC5322.From domain to match "
                        "the domain authenticated by SPF or DKIM."),
    Chunk(chunk_uid="rfc6376:0031", rfc="rfc6376", section="3.5 The DKIM-Signature Header",
          page=22, text="The h= tag lists the header fields that were signed, in order."),
    Chunk(chunk_uid="rfc7208:0012", rfc="rfc7208", section="2.4 Checking Host",
          page=9, text="The check_host() function fetches SPF records and evaluates them "
                       "against the MAIL FROM domain."),
    Chunk(chunk_uid="rfc8617:0004", rfc="rfc8617", section="4 ARC Header Fields",
          page=6, text="An ARC set comprises an ARC-Authentication-Results, an "
                       "ARC-Message-Signature and an ARC-Seal."),
]


def hits(*chunks, start=1):
    return [SearchHit(rank=i, score=10.0 - i, chunk=c)
            for i, c in enumerate(chunks, start=start)]
