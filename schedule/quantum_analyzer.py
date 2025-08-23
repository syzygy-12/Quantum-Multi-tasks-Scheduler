#!/usr/bin/env python3
"""
quantum_analyzer.py

Scan a directory (recursively) for .qpy/.qasm circuits, analyze with Qiskit,
and write ONE JSON file (array of objects).

Enhancements:
- --basis-csv allows specifying the implementable basis gates via CSV.
- The transpilation will be constrained to those gates (when feasible).
"""

from __future__ import annotations
import argparse, importlib, json, os, sys, csv, warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from qiskit import QuantumCircuit, transpile, qpy
except Exception as e:
    print("Qiskit is required. Try: pip install 'qiskit>=1.0'", file=sys.stderr)
    raise

SUPPORTED_EXTS = {".qpy", ".qasm"}
FAKE_PROVIDER_MODULES = [
    "qiskit.providers.fake_provider",
    "qiskit_ibm_runtime.fake_provider",
]

# --------------------------- basis gates from CSV -----------------------------

# Common alias normalization to keep inputs friendly
_GATE_NORMALIZE = {
    "cnot": "cx",
    "cz": "cz",
    "iswap": "iswap",
    "sqrtx": "sx",
    "u1": "rz",   # u1 ~= RZ (for many targets)
    "u2": "u",    # collapse to 'u'
    "u3": "u",
    "cu3": "cu",
    "id": "id",
    "i": "id",
}

def _normalize_gate_name(name: str) -> str:
    n = name.strip().lower()
    return _GATE_NORMALIZE.get(n, n)

def load_basis_from_csv(csv_path: str) -> List[str]:
    """
    Load allowed basis gates from a CSV file.
    - Accepts one or more columns. Column names do not matter.
    - Each cell can contain multiple gate names separated by comma and/or spaces.
    - Returns a de-duplicated, order-preserving, normalized list of gates.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Basis CSV not found: {csv_path}")

    basis: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            for cell in row:
                if not cell:
                    continue
                # Support comma/space mixed separators in a single cell
                parts = [p for chunk in cell.split(",") for p in chunk.split()]
                for p in parts:
                    if p:
                        basis.append(_normalize_gate_name(p))

    # de-duplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for g in basis:
        if g and g not in seen:
            seen.add(g)
            ordered.append(g)

    if not ordered:
        raise ValueError(f"No gate names found in CSV: {csv_path}")
    return ordered

# --------------------------- I/O utils for circuits --------------------------

def load_backend(name: str):
    last_err = None
    for mod_name in FAKE_PROVIDER_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, name)
            return cls()
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not import backend '{name}'. Tried: {', '.join(FAKE_PROVIDER_MODULES)}.\n"
        f"Original error: {last_err}"
    )

def list_supported_files_under_dir(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"Input is not a directory: {root_dir}")
    files: List[str] = []
    for r, _d, fns in os.walk(root_dir):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.join(r, fn))
    files.sort()
    return files

def load_circuits_from_path(path: str) -> List[Tuple[str, QuantumCircuit]]:
    ext = os.path.splitext(path)[1].lower()
    out: List[Tuple[str, QuantumCircuit]] = []
    if ext == ".qpy":
        with open(path, "rb") as f:
            loaded = list(qpy.load(f))
        if len(loaded) == 1:
            out.append((os.path.relpath(path), loaded[0]))
        else:
            for i, qc in enumerate(loaded):
                out.append((f"{os.path.relpath(path)}::#{i}", qc))
    elif ext == ".qasm":
        qc = QuantumCircuit.from_qasm_file(path)
        out.append((os.path.relpath(path), qc))
    else:
        raise ValueError(f"Unsupported file type (should not happen): {path}")
    return out

# ------------------------------ simple graph utils ---------------------------

def unique_edges(qc: QuantumCircuit) -> List[Tuple[int,int]]:
    qidx = {q: i for i, q in enumerate(qc.qubits)}
    edges = set()
    for _inst, qargs, _ in qc.data:
        if len(qargs) == 2:
            a, b = qidx[qargs[0]], qidx[qargs[1]]
            if a != b:
                x, y = (a, b) if a < b else (b, a)
                edges.add((x, y))
    return sorted(edges)

def degree_by_qubit(edges: List[Tuple[int,int]], n: int) -> List[int]:
    deg = [0]*n
    for a,b in edges:
        deg[a]+=1; deg[b]+=1
    return deg

# --------------------------------- metrics -----------------------------------

@dataclass
class CircuitMetrics:
    algo_id: str
    file: str
    num_qubits: int
    num_clbits: int
    depth: int
    size: int
    gate_counts: Dict[str,int]
    connectivity_topology: List[Tuple[int,int]]
    degree_by_qubit: List[int]
    transpiled_basis_gates: Optional[List[str]]
    coupling_map: Optional[List[Tuple[int,int]]]
    estimated_runtime_seconds: Optional[float]

# --------------------------------- analyze -----------------------------------

def analyze_circuit(
    algo_id: str,
    qc: QuantumCircuit,
    backend=None,
    opt_level: int = 1,
    seed: Optional[int] = None,
    allowed_basis_gates: Optional[List[str]] = None,
) -> CircuitMetrics:
    """
    Transpile the circuit (optionally to a backend, optionally constrained by basis_gates),
    and return various metrics. If allowed_basis_gates is provided, we pass it to Qiskit
    as basis_gates so that decomposition is restricted to that set (when feasible).
    """
    tkwargs = {"optimization_level": opt_level}
    if seed is not None:
        tkwargs["seed_transpiler"] = seed
    if allowed_basis_gates:
        tkwargs["basis_gates"] = allowed_basis_gates

    basis_gates_from_backend = None
    coupling_map = None

    if backend is not None:
        tqc = transpile(qc, backend=backend, **tkwargs)
        try:
            target = getattr(backend, "target", None)
            if target is not None:
                try:
                    basis_gates_from_backend = sorted({str(instr) for instr in target.instructions})
                except Exception:
                    basis_gates_from_backend = None
                cm = set()
                try:
                    for (_g, qtuple), _props in target._instruction_properties.items():  # type: ignore[attr-defined]
                        if len(qtuple) == 2:
                            a,b = qtuple
                            x,y = (a,b) if a<b else (b,a)
                            cm.add((x,y))
                except Exception:
                    pass
                coupling_map = sorted(cm) if cm else None
            else:
                conf = getattr(backend, "configuration", lambda: None)()
                if conf is not None and hasattr(conf, "basis_gates"):
                    basis_gates_from_backend = list(conf.basis_gates)
                if conf is not None and getattr(conf, "coupling_map", None):
                    cm=set()
                    for a,b in conf.coupling_map:
                        x,y=(a,b) if a<b else (b,a); cm.add((x,y))
                    coupling_map = sorted(cm)
        except Exception:
            basis_gates_from_backend = None
            coupling_map = None
    else:
        tqc = transpile(qc, **tkwargs)

    depth = int(tqc.depth() or 0)
    size  = int(tqc.size() or 0)
    gate_counts = {str(k): int(v) for k,v in tqc.count_ops().items()}
    edges = unique_edges(tqc)
    deg = degree_by_qubit(edges, tqc.num_qubits)

    est_seconds: Optional[float] = None
    if backend is not None:
        try:
            scheduled = transpile(
                tqc, backend=backend, scheduling_method="asap",
                **({} if seed is None else {"seed_transpiler": seed})
            )
            dt = getattr(getattr(backend, "target", None), "dt", None)
            if dt is None:
                conf = getattr(backend, "configuration", lambda: None)()
                if conf is not None and hasattr(conf, "dt"):
                    dt = conf.dt
            if hasattr(scheduled, "duration") and scheduled.duration is not None and dt:
                est_seconds = float(scheduled.duration) * float(dt)
        except Exception:
            est_seconds = None

    # Effective basis we record in metrics: prefer CSV if provided, otherwise backend-provided info
    effective_basis = list(allowed_basis_gates) if allowed_basis_gates else (
        list(basis_gates_from_backend) if basis_gates_from_backend else None
    )

    return CircuitMetrics(
        algo_id=algo_id,
        file=algo_id.split("::#")[0],
        num_qubits=tqc.num_qubits,
        num_clbits=tqc.num_clbits,
        depth=depth,
        size=size,
        gate_counts=gate_counts,
        connectivity_topology=edges,
        degree_by_qubit=deg,
        transpiled_basis_gates=effective_basis,
        coupling_map=coupling_map,
        estimated_runtime_seconds=est_seconds,
    )

# ----------------------------------- CLI -------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze all .qpy/.qasm under a directory and write ONE JSON array"
    )
    ap.add_argument("input_dir", help="Directory to recursively search for .qpy and .qasm")
    ap.add_argument("-o", "--output", default="results.json",
                    help="Output JSON file (default: results.json)")
    ap.add_argument("--backend", help="Fake backend class name (e.g., FakeLima, FakeNairobi)")
    ap.add_argument("--opt-level", type=int, default=1, choices=range(0,4),
                    help="Transpile optimization level (0-3)")
    ap.add_argument("--seed", type=int, help="Seed for transpiler")
    ap.add_argument("--basis-csv", help="CSV listing implementable basis gates (e.g., rz,sx,cx)")
    args = ap.parse_args(argv)

    backend = load_backend(args.backend) if args.backend else None

    allowed_basis: Optional[List[str]] = None
    if args.basis_csv:
        allowed_basis = load_basis_from_csv(args.basis_csv)

        # If a backend is also given, warn (non-fatal) when some CSV gates don't appear in target
        try:
            if backend is not None:
                tgt = getattr(backend, "target", None)
                if tgt is not None and hasattr(tgt, "instructions"):
                    backend_instrs = {str(instr).lower() for instr in tgt.instructions}
                    not_in_target = [g for g in allowed_basis if g not in backend_instrs]
                    if not_in_target:
                        warnings.warn(
                            "Some CSV gates may not be directly present in backend target: "
                            f"{not_in_target}. The transpiler will still try to decompose."
                        )
        except Exception:
            pass  # never block analysis just for the hint

    files = list_supported_files_under_dir(args.input_dir)
    if not files:
        ap.error("No .qpy or .qasm files found in the directory.")

    results: List[Dict] = []
    for path in files:
        try:
            for algo_id, qc in load_circuits_from_path(path):
                metrics = analyze_circuit(
                    algo_id,
                    qc,
                    backend=backend,
                    opt_level=args.opt_level,
                    seed=args.seed,
                    allowed_basis_gates=allowed_basis,
                )
                results.append(asdict(metrics))
        except Exception as e:
            results.append({
                "algo_id": os.path.relpath(path),
                "file": os.path.relpath(path),
                "error": f"{type(e).__name__}: {e}"
            })

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return results

if __name__ == "__main__":
    raise SystemExit(main())
