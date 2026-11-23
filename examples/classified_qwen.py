"""Route Qwen first-block residuals by exact state identity inside a pinned session."""
import argparse
from contextlib import ExitStack
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
from time import perf_counter

from .run_qwen_mlp import advance


def library(path):
    """Bind F: BF16^(m×n) → (I,B), and its two evaluation maps ℝⁿ → ℝᵐ."""
    lib = ct.CDLL(str(path.resolve()))
    size, handle, doubles = ct.c_size_t, ct.c_void_p, ct.POINTER(ct.c_double)
    lib.form.argtypes = [ct.c_char_p, size, size, size]
    lib.form.restype = handle
    lib.release.argtypes = [handle]
    lib.classes.argtypes = [handle]
    lib.classes.restype = size
    lib.interpret.argtypes = [handle, doubles, doubles, doubles]
    lib.direct.argtypes = [ct.c_char_p, size, size, doubles, doubles]
    lib.release.restype = lib.interpret.restype = lib.direct.restype = None
    return lib


class Session:
    def __init__(self, stack, directory, lib, benchmark=False):
        """Fix W by its digests and bind F_W once for the lifetime of this session."""
        self.directory, self.lib = directory, lib
        self.manifest = json.loads((directory / "manifest.json").read_bytes())
        self.plans, self.original = {}, {}
        self.last = None
        self.plan_bytes = 0
        for key in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            entry = self.manifest["operands"][key]
            rows, cols = entry["shape"]
            data = (directory / key).read_bytes()
            if len(data) != 2*rows*cols or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Unbound source for {key}")
            handle = lib.form(data, len(data), rows, cols)
            if not handle:
                raise ValueError(f"Failed to form finite coefficient classes for {key}")
            stack.callback(lib.release, handle)
            count = lib.classes(handle)
            self.plans[key] = (handle, rows, cols, count)
            self.plan_bytes += len(data) + 2*count + ct.sizeof(ct.c_size_t)*(rows+1)
            if benchmark:
                self.original[key] = data

    def projection(self, directory, key, operand, vector, bias):
        """Apply Σ_k B_ik Σ_{j:I_ij=k} x_j to each bound row."""
        handle, rows, cols, count = self.plans[key]
        if (len(vector), len(bias)) != (cols, rows):
            raise ValueError(f"Wrong activation extent for {key}")
        x = (ct.c_double*cols)(*vector)
        y = (ct.c_double*rows)(*bias)
        self.lib.interpret(handle, x, y, (ct.c_double*min(cols, 65536))())
        return list(y), count

    def direct(self, directory, key, operand, vector, bias):
        """Comparator y=b+Wx over the identical BF16 source in resident memory."""
        _, rows, cols, _ = self.plans[key]
        if (len(vector), len(bias)) != (cols, rows):
            raise ValueError(f"Wrong activation extent for {key}")
        x = (ct.c_double*cols)(*vector)
        y = (ct.c_double*rows)(*bias)
        self.lib.direct(self.original[key], rows, cols, x, y)
        return list(y), rows*cols

    def dispatch(self, encoded):
        """If (W,r)=(W,r_last), return the stored F_W(r); otherwise evaluate F_W(r)."""
        encoded = bytes(encoded)
        if self.last is not None and self.last[0] == encoded:
            return "replay", self.last[1], 0
        state = struct.unpack(f"<{self.manifest['hidden']}d", encoded)
        if not all(map(math.isfinite, state)):
            raise ValueError("Residual outside finite domain")
        output, count = advance(self.directory, self.manifest, state, self.projection)
        if not all(map(math.isfinite, output)):
            raise ArithmeticError("Nonfinite block output")
        result = struct.pack(f"<{len(output)}d", *output)
        self.last = encoded, result
        return "interpret", result, count


def main():
    """Measure formation once and T(route,r) per request; compare ||y−y_dense||∞ on demand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--case", nargs=2, action="append", required=True, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        start = perf_counter()
        session = Session(stack, args.directory, library(args.library), args.benchmark)
        formation = perf_counter()-start
        records = []
        passed = True
        for source, destination in args.case:
            encoded = Path(source).read_bytes()
            start = perf_counter()
            route, result, count = session.dispatch(encoded)
            elapsed = perf_counter()-start
            Path(destination).write_bytes(result)
            record = dict(route=route, seconds=elapsed, products=count,
                          input_sha256=hashlib.sha256(encoded).hexdigest(),
                          output_sha256=hashlib.sha256(result).hexdigest())
            if args.benchmark:
                fmt = f"<{session.manifest['hidden']}d"
                state = struct.unpack(fmt, encoded)
                start = perf_counter()
                expected, _ = advance(args.directory, session.manifest, state, session.direct)
                record["native_dense_seconds"] = perf_counter()-start
                errors = [abs(a-b) for a,b in zip(struct.unpack(fmt,result),expected,strict=True)]
                error = max(errors) if all(map(math.isfinite, errors)) else math.inf
                record["native_dense_max_error"] = error if math.isfinite(error) else "nonfinite"
                passed &= error <= 1e-10
            records.append(record)
        manifest_bytes = (args.directory / "manifest.json").read_bytes()
        print(json.dumps(dict(page=session.manifest["page"], formation_seconds=formation,
                              manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                              native_library_bytes=args.library.stat().st_size,
                              native_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                              logical_plan_bytes=session.plan_bytes, cases=records), indent=2))
        return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
