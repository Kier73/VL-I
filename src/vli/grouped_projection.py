"""Project an activation through repeated BF16 coefficients, using binary64 accumulation."""
import math
import struct


def residual(matrix, activation, embedding):
    """r_i=e_i+Σ_b decode(b)·Σ_{j:bits(W_ij)=b} z_j; no new weight quantization."""
    rows,cols=len(embedding),len(activation)
    if not rows or not cols or len(matrix)!=2*rows*cols:
        raise ValueError("BF16 projection extent disagrees with geometry")
    if not all(map(math.isfinite,activation)) or not all(map(math.isfinite,embedding)):
        raise ValueError("Nonfinite projection input")
    decoded={}
    products=0
    result=[]
    for i,codes in enumerate(struct.iter_unpack(f"<{cols}H",matrix)):
        sums={}
        for code,x in zip(codes,activation):
            sums[code]=sums.get(code,0.)+x
        for code in sums.keys()-decoded.keys():
            value=struct.unpack("<f",struct.pack("<I",code<<16))[0]
            if not math.isfinite(value):
                raise ValueError("Nonfinite trained coefficient")
            decoded[code]=value
        products+=len(sums)
        result.append(embedding[i]+math.fsum(decoded[b]*x for b,x in sums.items()))
    return result,products
