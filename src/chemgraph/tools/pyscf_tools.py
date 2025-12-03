import math
from pathlib import Path
from typing import Optional, Tuple, Union, List, Sequence
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chemgraph.models.atomsdata import AtomsData

from rdkit import Chem

HARTREE_TO_EV = 27.211386245988  # Hartree to eV conversion constant


def _atomic_number_to_symbol(z: int) -> str:
    """Convert atomic number to symbol for PySCF geometry."""
    from ase.data import chemical_symbols
    try:
        return chemical_symbols[int(z)]
    except Exception as exc:
        raise ValueError(f"Invalid atomic number for symbol lookup: {z}") from exc



def _valence_electrons(z: int) -> int:
    """Return a valence-electron count for atomic number ``z`` without needing a bond topology.
    RDKit's GetTotalValence/ExplicitValence are zero for isolated atoms (e.g., H2 built
    from atom numbers only). Use periodic-table data instead and fall back to a small
    default map to avoid crashes on simple systems.
    """
    z = int(z)
    pt = Chem.GetPeriodicTable()

    try:
        valence = pt.GetNOuterElecs(z)
    except Exception:
        valence = 0

    if not valence:
        try:
            valence = pt.GetDefaultValence(z)
        except Exception:
            valence = 0

    if not valence:
        fallback = {
            1: 1,   # H
            2: 2,   # He
            3: 1,   # Li
            4: 2,   # Be
            5: 3,
            6: 4,
            7: 5,
            8: 6,
            9: 7,
            10: 8,
        }
        valence = fallback.get(z, max(min(z, 8), 1))

    return int(valence)

def _total_valence_budget(numbers: Sequence[int], charge: int) -> int:
    total = sum(_valence_electrons(z) for z in numbers)
    return max(total - charge, 0)


def _is_transition_metal(z: int) -> bool:
    """Crude transition-metal check for heuristic branching."""
    return (21 <= z <= 30) or (39 <= z <= 48) or (57 <= z <= 80)


def _molecular_weight(numbers: Sequence[int]) -> float:
    """Rough molecular weight from atomic numbers only."""
    pt = Chem.GetPeriodicTable()
    return float(sum(pt.GetAtomicWeight(int(z)) for z in numbers))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))) ** 0.5


def _guess_aromaticity(numbers: Sequence[int], positions: Sequence[Sequence[float]]) -> Tuple[bool, Optional[str]]:
    """Crude aromaticity detection by inferring bonds from covalent radii and sanitizing with RDKit."""
    try:
        pt = Chem.GetPeriodicTable()
        mol = Chem.RWMol()
        for z in numbers:
            mol.AddAtom(Chem.Atom(int(z)))

        n = len(numbers)
        bonds_added = 0
        for i in range(n):
            for j in range(i + 1, n):
                r1 = pt.GetRcovalent(int(numbers[i])) or 0.7
                r2 = pt.GetRcovalent(int(numbers[j])) or 0.7
                cutoff = 1.25 * (r1 + r2)
                if _distance(positions[i], positions[j]) <= cutoff:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    bonds_added += 1

        warning = None
        sanitized = True
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            sanitized = False
            warning = "Sanitization failed; aromaticity guess may be unreliable."

        if bonds_added == 0:
            return False, "No bonds inferred; aromaticity unknown."

        aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms()) or any(
            bond.GetIsAromatic() for bond in mol.GetBonds()
        )
        if not sanitized and not warning:
            warning = "Sanitization skipped."
        return aromatic, warning
    except Exception as exc:
        return False, f"Aromaticity detection error: {exc}"


class PySCFCASSCFInput(BaseModel):
    atomsdata: AtomsData = Field(description="Molecular geometry to run PySCF on.")
    basis: str = Field(default="sto-3g", description="Basis set name (e.g., sto-3g, def2-svp).")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")
    n_active_electrons: Optional[Union[int, Tuple[int, int]]] = Field(
        default=None, description="Active electrons for CASSCF. Accepts total or (n_alpha, n_beta)."
    )
    n_active_orbitals: Optional[int] = Field(default=None, description="Active orbitals for CASSCF.")
    max_cycle: int = Field(default=50, description="Max SCF/CASSCF macro iterations.")
    active_orbital_indices: Optional[Sequence[int]] = Field(
        default=None, description="Optional list of HF MO indices to define the CAS (passed to sort_mo)."
    )


class PySCFCASSCFOutput(BaseModel):
    hf_energy: float
    casscf_energy: Optional[float] = None
    hf_converged: bool = True
    casscf_converged: Optional[bool] = None
    converged: bool = True
    n_electrons: Optional[int] = None
    n_orbitals: Optional[int] = None
    s2: Optional[float] = None
    multiplicity: Optional[float] = None
    message: str = ""


class XYZFileInput(BaseModel):
    path: str = Field(description="Filesystem path to the XYZ geometry file.")


class XYZFileOutput(BaseModel):
    atomsdata: AtomsData
    n_atoms: int
    comment: Optional[str] = None
    path: str = Field(description="Resolved path to the XYZ file.")


@tool
def load_xyz_geometry(params: XYZFileInput) -> XYZFileOutput:
    """Read a standard XYZ file and return an AtomsData geometry."""

    xyz_path = Path(params.path).expanduser().resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    with xyz_path.open("r", encoding="utf-8") as handle:
        raw_lines = handle.readlines()

    if len(raw_lines) < 2:
        raise ValueError("XYZ file must include atom count and comment lines.")

    try:
        n_atoms = int(raw_lines[0].strip().split()[0])
    except Exception as exc:
        raise ValueError("First line of XYZ must be the atom count.") from exc

    comment_line = raw_lines[1].rstrip("\n") if len(raw_lines) >= 2 else ""
    coord_lines = [line.strip() for line in raw_lines[2:] if line.strip()]

    if len(coord_lines) < n_atoms:
        raise ValueError(
            f"XYZ file lists {n_atoms} atoms but only {len(coord_lines)} coordinate lines were found."
        )

    pt = Chem.GetPeriodicTable()
    numbers: List[int] = []
    positions: List[List[float]] = []

    for idx in range(n_atoms):
        fields = coord_lines[idx].split()
        if len(fields) < 4:
            raise ValueError(f"XYZ line {idx + 3} must contain symbol and three coordinates.")
        symbol = fields[0]
        try:
            atomic_number = pt.GetAtomicNumber(symbol)
        except Exception as exc:
            raise ValueError(f"Unknown chemical symbol '{symbol}' in XYZ file.") from exc
        try:
            x, y, z = map(float, fields[1:4])
        except Exception as exc:
            raise ValueError(f"Invalid coordinates on line {idx + 3} in XYZ file.") from exc

        numbers.append(int(atomic_number))
        positions.append([x, y, z])

    atomsdata = AtomsData(numbers=numbers, positions=positions)

    return XYZFileOutput(
        atomsdata=atomsdata,
        n_atoms=n_atoms,
        comment=comment_line or None,
        path=str(xyz_path),
    )


class StretchBondInput(BaseModel):
    atomsdata: AtomsData = Field(description="Base geometry to be modified.")
    atom_a: int = Field(description="Zero-based index of the first atom defining the bond.")
    atom_b: int = Field(description="Zero-based index of the second atom defining the bond.")
    delta: float = Field(description="Change in Å to apply to the current bond length (positive stretches).")
    mode: str = Field(
        default="move_b",
        description="How to displace atoms: 'move_b' (default), 'move_a', or 'symmetric'.",
    )


class StretchBondOutput(BaseModel):
    atomsdata: AtomsData
    atom_a: int
    atom_b: int
    original_distance: float
    new_distance: float
    mode: str


@tool
def stretch_bond(params: StretchBondInput) -> StretchBondOutput:
    """Displace atoms along bond vector to change a bond length by a target increment."""

    numbers = list(params.atomsdata.numbers)
    positions = [list(map(float, pos)) for pos in params.atomsdata.positions]
    n_atoms = len(numbers)

    if not (0 <= params.atom_a < n_atoms) or not (0 <= params.atom_b < n_atoms):
        raise ValueError(f"atom indices must be within [0, {n_atoms - 1}]")
    if params.atom_a == params.atom_b:
        raise ValueError("atom_a and atom_b must be different atoms")

    a_pos = positions[params.atom_a]
    b_pos = positions[params.atom_b]
    vector = [b - a for a, b in zip(a_pos, b_pos)]
    distance = math.sqrt(sum(comp * comp for comp in vector))
    if distance == 0:
        raise ValueError("Selected atoms occupy the same coordinates; cannot define a bond direction")

    delta = float(params.delta)
    new_distance = distance + delta
    if new_distance <= 0:
        raise ValueError("Resulting bond length must stay positive; reduce |delta|.")

    unit_vec = [comp / distance for comp in vector]

    mode = params.mode.lower().strip()
    if mode not in {"move_b", "move_a", "symmetric"}:
        raise ValueError("mode must be 'move_b', 'move_a', or 'symmetric'")

    new_positions = [pos.copy() for pos in positions]
    if mode == "move_b":
        shift = [unit * delta for unit in unit_vec]
        new_positions[params.atom_b] = [b + s for b, s in zip(b_pos, shift)]
    elif mode == "move_a":
        shift = [unit * delta for unit in unit_vec]
        new_positions[params.atom_a] = [a - s for a, s in zip(a_pos, shift)]
    else:  # symmetric
        half_shift = [unit * (delta / 2.0) for unit in unit_vec]
        new_positions[params.atom_b] = [b + s for b, s in zip(b_pos, half_shift)]
        new_positions[params.atom_a] = [a - s for a, s in zip(a_pos, half_shift)]

    cell = getattr(params.atomsdata, "cell", None)
    pbc = getattr(params.atomsdata, "pbc", None)
    stretched = AtomsData(numbers=numbers, positions=new_positions, cell=cell, pbc=pbc)

    return StretchBondOutput(
        atomsdata=stretched,
        atom_a=params.atom_a,
        atom_b=params.atom_b,
        original_distance=distance,
        new_distance=new_distance,
        mode=mode,
    )


@tool
def run_pyscf_casscf(params: PySCFCASSCFInput) -> PySCFCASSCFOutput:
    """Run a quick RHF/ROHF followed by optional CASSCF using PySCF."""
    from pyscf import gto, scf, mcscf

    try:
        # Build geometry; AtomsData positions are in Angstrom by default.
        coords = [
            (_atomic_number_to_symbol(int(z)), tuple(pos))
            for z, pos in zip(params.atomsdata.numbers, params.atomsdata.positions)
        ]

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = params.basis
        mol.charge = params.charge
        mol.spin = params.spin
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        nelec = mol.nelectron
        if (nelec - mol.spin) % 2 != 0:
            raise ValueError(
                f"Inconsistent spin ({mol.spin}) and total electrons ({nelec}). nelec - spin must be even."
            )

        mf = scf.RHF(mol) if params.spin == 0 else scf.ROHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-8
        mf.max_cycle = params.max_cycle
        hf_e = mf.kernel()
        hf_conv = bool(mf.converged)
        casscf_e = None
        casscf_conv = None
        s2 = None
        multiplicity = None

        if params.n_active_electrons is not None and params.n_active_orbitals is not None:
            if params.n_active_orbitals <= 0:
                raise ValueError("n_active_orbitals must be > 0")
            if isinstance(params.n_active_electrons, int) and params.n_active_electrons <= 0:
                raise ValueError("n_active_electrons must be > 0")
            if isinstance(params.n_active_electrons, tuple):
                if any(ne <= 0 for ne in params.n_active_electrons):
                    raise ValueError("Each component of n_active_electrons tuple must be > 0")

            if not hf_conv:
                return PySCFCASSCFOutput(
                    hf_energy=float(hf_e),
                    casscf_energy=None,
                    hf_converged=False,
                    casscf_converged=None,
                    converged=False,
                    n_electrons=nelec,
                    n_orbitals=mol.nao,
                    s2=None,
                    multiplicity=None,
                    message="HF did not converge; CASSCF not attempted.",
                )

            mc = mcscf.CASSCF(mf, params.n_active_orbitals, params.n_active_electrons)
            mc.max_cycle_macro = params.max_cycle
            mc.verbose = 0
            mo = mf.mo_coeff
            nmo = mo.shape[1] if hasattr(mo, "shape") else len(getattr(mf, "mo_energy", []))
            if params.active_orbital_indices is not None:
                idx = [int(i) for i in params.active_orbital_indices]
                bad = [i for i in idx if i < 0 or i >= nmo]
                if bad:
                    raise ValueError(f"active_orbital_indices out of range for {nmo} MOs: {bad}")
                if len(idx) < params.n_active_orbitals:
                    raise ValueError(
                        f"Need at least {params.n_active_orbitals} active_orbital_indices, got {len(idx)}"
                    )
                mo = mc.sort_mo(idx, mo)

            mc.kernel(mo)
            casscf_e = mc.e_tot
            casscf_conv = bool(getattr(mc, "converged", True))
            if casscf_e is not None and hasattr(mc, "ci"):
                try:
                    s2, multiplicity = mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas)
                    s2 = float(s2)
                    multiplicity = float(multiplicity)
                except Exception:
                    s2 = None
                    multiplicity = None

        conv = hf_conv and (casscf_conv if casscf_conv is not None else True)

        return PySCFCASSCFOutput(
            hf_energy=float(hf_e),
            casscf_energy=float(casscf_e) if casscf_e is not None else None,
            hf_converged=hf_conv,
            casscf_converged=casscf_conv,
            converged=conv,
            n_electrons=nelec,
            n_orbitals=mol.nao,
            s2=s2,
            multiplicity=multiplicity,
            message="" if conv else "SCF or CASSCF did not fully converge",
        )
    except Exception as exc:
        return PySCFCASSCFOutput(
            hf_energy=float("nan"),
            casscf_energy=None,
            hf_converged=False,
            casscf_converged=None,
            converged=False,
            n_electrons=None,
            n_orbitals=None,
            message=str(exc),
        )


class TDExcitedState(BaseModel):
    root: int = Field(description="Zero-based excited-state index.")
    energy: float = Field(description="Total excited-state energy (Hartree).")
    excitation_energy: float = Field(description="Vertical excitation energy relative to HF ground state (Hartree).")
    excitation_energy_ev: float = Field(description="Vertical excitation energy (eV).")
    oscillator_strength: Optional[float] = Field(
        default=None,
        description="Isotropic oscillator strength if available (TDHF/TDA).",
    )


class TDExcitedStatesInput(BaseModel):
    atomsdata: AtomsData = Field(description="Molecular geometry to run PySCF on.")
    basis: str = Field(default="sto-3g", description="Basis set name (e.g., sto-3g, def2-svp).")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")
    method: str = Field(
        default="TDHF",
        description="Excited-state method: 'TDHF' (default) or 'TDA'.",
    )
    n_roots: int = Field(
        default=5,
        description="Number of excited states to compute (roots of the TD problem).",
    )
    max_cycle: int = Field(default=50, description="Max SCF iterations for the reference HF/UHF run.")


class TDExcitedStatesOutput(BaseModel):
    hf_energy: float
    hf_converged: bool = True
    states: List[TDExcitedState] = Field(default_factory=list)
    message: str = ""

##Dec 2 excited states
@tool
def run_pyscf_td_excited_states(params: TDExcitedStatesInput) -> TDExcitedStatesOutput:
    """Compute vertical excited states with TDHF/TDA on top of HF/UHF."""
    from pyscf import gto, scf

    try:
        coords = [
            (_atomic_number_to_symbol(int(z)), tuple(pos))
            for z, pos in zip(params.atomsdata.numbers, params.atomsdata.positions)
        ]

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = params.basis
        mol.charge = params.charge
        mol.spin = params.spin
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        if params.spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)

        mf.verbose = 0
        mf.conv_tol = 1e-8
        mf.max_cycle = params.max_cycle
        hf_e = mf.kernel()
        hf_conv = bool(mf.converged)

        if not hf_conv:
            return TDExcitedStatesOutput(
                hf_energy=float(hf_e),
                hf_converged=False,
                states=[],
                message="HF/UHF did not converge; TD calculation not attempted.",
            )

        method = (params.method or "TDHF").upper()
        if method == "TDHF":
            td = mf.TDHF()
        elif method == "TDA":
            td = mf.TDA()
        else:
            raise ValueError(f"Unknown excited-state method '{params.method}'. Use 'TDHF' or 'TDA'.")

        td.verbose = 0
        td.nroots = max(1, int(params.n_roots))

        td.kernel()
        exc_energies = [float(e) for e in getattr(td, "e", [])]

        osc_list: List[Optional[float]] = [None] * len(exc_energies)
        try:
            fos = td.oscillator_strength()
            raw = list(fos)
            for i in range(min(len(raw), len(exc_energies))):
                val = raw[i]
                if isinstance(val, (tuple, list)):
                    for v in val:
                        try:
                            osc_list[i] = float(v)
                            break
                        except Exception:
                            continue
                else:
                    try:
                        osc_list[i] = float(val)
                    except Exception:
                        osc_list[i] = None
        except Exception:
            pass

        states: List[TDExcitedState] = []
        for i, dE in enumerate(exc_energies):
            total_e = float(hf_e + dE)
            ev = float(dE * HARTREE_TO_EV)
            osc = osc_list[i] if i < len(osc_list) else None
            states.append(
                TDExcitedState(
                    root=i,
                    energy=total_e,
                    excitation_energy=float(dE),
                    excitation_energy_ev=ev,
                    oscillator_strength=osc,
                )
            )

        msg = "" if states else "TD calculation ran, but no excitation energies were returned."
        return TDExcitedStatesOutput(
            hf_energy=float(hf_e),
            hf_converged=hf_conv,
            states=states,
            message=msg,
        )

    except Exception as exc:
        return TDExcitedStatesOutput(
            hf_energy=float("nan"),
            hf_converged=False,
            states=[],
            message=str(exc),
        )


class CASSCFExcitedStatesInput(BaseModel):
    atomsdata: AtomsData = Field(description="Molecular geometry for CASSCF excited states.")
    basis: str = Field(default="sto-3g", description="Basis set name (e.g., sto-3g, def2-svp).")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")
    n_active_electrons: Union[int, Tuple[int, int]] = Field(
        description="Active electrons for CASSCF (total or (n_alpha, n_beta))."
    )
    n_active_orbitals: int = Field(description="Number of active orbitals for CASSCF.")
    n_roots: int = Field(
        default=2,
        description="Number of CASSCF roots in the state-averaged calculation (including ground).",
    )
    root_weights: Optional[List[float]] = Field(
        default=None,
        description="Optional state-averaging weights; if None, use equal weights for all roots.",
    )
    max_cycle: int = Field(default=50, description="Max SCF/CASSCF macro iterations.")
    active_orbital_indices: Optional[Sequence[int]] = Field(
        default=None,
        description="Optional list of HF MO indices to define the CAS (passed to sort_mo).",
    )


class CASSCFRootInfo(BaseModel):
    root: int = Field(description="Zero-based root index.")
    energy: float = Field(description="State-averaged CASSCF total energy for this root (Hartree).")
    excitation_energy: float = Field(
        description="Energy relative to the lowest root (Hartree)."
    )

##Dec 2 excited states
class CASSCFExcitedStatesOutput(BaseModel):
    hf_energy: float
    hf_converged: bool = True
    casscf_converged: bool = True
    roots: List[CASSCFRootInfo] = Field(default_factory=list)
    s2: Optional[List[float]] = None
    multiplicities: Optional[List[float]] = None
    message: str = ""


@tool
def run_pyscf_casscf_excited_states(params: CASSCFExcitedStatesInput) -> CASSCFExcitedStatesOutput:
    """Run a state-averaged CASSCF to obtain several roots (ground + excited states)."""
    from pyscf import gto, scf, mcscf

    try:
        coords = [
            (_atomic_number_to_symbol(int(z)), tuple(pos))
            for z, pos in zip(params.atomsdata.numbers, params.atomsdata.positions)
        ]

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = params.basis
        mol.charge = params.charge
        mol.spin = params.spin
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        nelec = mol.nelectron
        if (nelec - mol.spin) % 2 != 0:
            raise ValueError(
                f"Inconsistent spin ({mol.spin}) and total electrons ({nelec}). nelec - spin must be even."
            )

        if params.n_active_orbitals <= 0:
            raise ValueError("n_active_orbitals must be > 0")

        if isinstance(params.n_active_electrons, int):
            if params.n_active_electrons <= 0:
                raise ValueError("n_active_electrons must be > 0")
        else:
            if any(ne <= 0 for ne in params.n_active_electrons):
                raise ValueError("Each component of n_active_electrons tuple must be > 0")

        mf = scf.RHF(mol) if params.spin == 0 else scf.ROHF(mol)
        mf.verbose = 0
        mf.conv_tol = 1e-8
        mf.max_cycle = params.max_cycle
        hf_e = mf.kernel()
        hf_conv = bool(mf.converged)

        if not hf_conv:
            return CASSCFExcitedStatesOutput(
                hf_energy=float(hf_e),
                hf_converged=False,
                casscf_converged=False,
                roots=[],
                s2=None,
                multiplicities=None,
                message="HF did not converge; CASSCF not attempted.",
            )

        n_roots = max(1, int(params.n_roots))
        mc = mcscf.CASSCF(mf, params.n_active_orbitals, params.n_active_electrons)
        mc.max_cycle_macro = params.max_cycle

        if params.root_weights is not None:
            if len(params.root_weights) != n_roots:
                raise ValueError(
                    f"len(root_weights)={len(params.root_weights)} but n_roots={n_roots}"
                )
            weights = [float(w) for w in params.root_weights]
        else:
            weights = [1.0 / n_roots] * n_roots

        mc = mc.state_average_(weights)

        mo = mf.mo_coeff
        nmo = mo.shape[1] if hasattr(mo, "shape") else len(getattr(mf, "mo_energy", []))
        if params.active_orbital_indices is not None:
            idx = [int(i) for i in params.active_orbital_indices]
            bad = [i for i in idx if i < 0 or i >= nmo]
            if bad:
                raise ValueError(f"active_orbital_indices out of range for {nmo} MOs: {bad}")
            if len(idx) < params.n_active_orbitals:
                raise ValueError(
                    f"Need at least {params.n_active_orbitals} active_orbital_indices, got {len(idx)}"
                )
            mo = mc.sort_mo(idx, mo)

        mc.kernel(mo)
        casscf_conv = bool(getattr(mc, "converged", True))

        if hasattr(mc, "e_states") and mc.e_states is not None:
            casscf_energies = [float(e) for e in mc.e_states]
        elif isinstance(getattr(mc, "e_tot", None), (list, tuple)):
            casscf_energies = [float(e) for e in mc.e_tot]
        else:
            casscf_energies = [float(mc.e_tot)]

        if len(casscf_energies) != n_roots:
            n_roots = len(casscf_energies)

        e0 = min(casscf_energies)
        roots = [
            CASSCFRootInfo(root=i, energy=e, excitation_energy=e - e0)
            for i, e in enumerate(casscf_energies)
        ]

        s2_vals: List[float] = []
        mult_vals: List[float] = []
        try:
            if hasattr(mc, "ci") and mc.ci is not None:
                ci_list = mc.ci
                if not isinstance(ci_list, (list, tuple)):
                    ci_list = [ci_list]
                for ci in ci_list:
                    s2, mult = mc.fcisolver.spin_square(ci, mc.ncas, mc.nelecas)
                    s2_vals.append(float(s2))
                    mult_vals.append(float(mult))
        except Exception:
            s2_vals = []
            mult_vals = []

        msg = "" if casscf_conv else "CASSCF state-averaged optimization did not fully converge."

        return CASSCFExcitedStatesOutput(
            hf_energy=float(hf_e),
            hf_converged=hf_conv,
            casscf_converged=casscf_conv,
            roots=roots,
            s2=s2_vals or None,
            multiplicities=mult_vals or None,
            message=msg,
        )

    except Exception as exc:
        return CASSCFExcitedStatesOutput(
            hf_energy=float("nan"),
            hf_converged=False,
            casscf_converged=False,
            roots=[],
            s2=None,
            multiplicities=None,
            message=str(exc),
        )


class ActiveSpaceGuessInput(BaseModel):
    atomsdata: AtomsData = Field(description="Geometry used to seed the active space guess.")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")
    target_active_electrons: Optional[int] = Field(
        default=None, description="Force a specific number of active electrons if known."
    )
    target_active_orbitals: Optional[int] = Field(
        default=None, description="Force a specific number of active orbitals if known."
    )


class ActiveSpaceGuess(BaseModel):
    n_active_electrons: int
    n_active_orbitals: int
    charge: int
    spin: int
    rationale: str


@tool
def propose_active_space_guess(params: ActiveSpaceGuessInput) -> ActiveSpaceGuess:
    """Provide a lightweight active-space guess from molecular makeup."""
    numbers = [int(z) for z in params.atomsdata.numbers]
    total_electrons = max(sum(numbers) - params.charge, 0)
    valence_electrons = _total_valence_budget(numbers, params.charge)
    electron_budget = min(valence_electrons, total_electrons)
    has_tm = any(_is_transition_metal(z) for z in numbers)
    n_unpaired = max(params.spin, 0)

    # Start from provided targets when present.
    n_elec = params.target_active_electrons
    n_orb = params.target_active_orbitals

    if n_elec is None:
        # Heuristic: stay within valence, branch on TM vs main group, cap size.
        max_active_electrons = 12
        base_budget = min(electron_budget, max_active_electrons)

        if has_tm:
            # Cover d-shell with some pairing room.
            min_tm_elec = max(n_unpaired + 2, 6)
            n_elec = max(min_tm_elec, min(base_budget, 10))
        else:
            # For main-group, ensure unpaired + at least one pair.
            min_mg_elec = max(n_unpaired + 2, 2)
            n_elec = max(min_mg_elec, min(base_budget, 8))

    if n_orb is None:
        rough = (n_elec + 1) // 2  # ~1.5 electrons per orbital
        if has_tm:
            n_orb = max(5, min(rough + 1, 10))
        else:
            n_orb = max(2, min(rough, 8))

    # Enforce parity of active electrons to match total electron parity (avoid odd/even mismatches).
    if isinstance(n_elec, int) and n_elec % 2 != total_electrons % 2:
        if n_elec > 2:
            n_elec -= 1
        elif n_elec + 1 <= electron_budget:
            n_elec += 1
    rationale_parts = [
        f"Total electrons (after charge): {total_electrons}",
        f"Valence electrons (after charge): {valence_electrons}",
        f"Spin (2S): {params.spin}",
        f"Transition metal present: {has_tm}",
    ]
    rationale = "; ".join(rationale_parts)

    return ActiveSpaceGuess(
        n_active_electrons=int(n_elec),
        n_active_orbitals=int(n_orb),
        charge=params.charge,
        spin=params.spin,
        rationale=rationale,
    )


class CASSCFDiagnosticsInput(BaseModel):
    casscf_output: PySCFCASSCFOutput = Field(
        description="Result object from run_pyscf_casscf to evaluate."
    )
    energy_tolerance: float = Field(
        default=1e-4,
        description="If |CASSCF - HF| < tolerance, flag the active space as possibly too small.",
    )
    require_casscf_energy: bool = Field(
        default=False, description="Flag missing CASSCF energy as a failure even if SCF converged."
    )


class CASSCFDiagnostics(BaseModel):
    converged: bool
    needs_revision: bool
    delta_energy: Optional[float] = None
    summary: str
    recommendations: list[str] = Field(default_factory=list)


class OrbitalPrepInput(BaseModel):
    atomsdata: AtomsData = Field(description="Geometry for SCF orbital preparation.")
    basis: str = Field(default="sto-3g", description="Basis set name (e.g., sto-3g, def2-svp).")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")
    max_cycle: int = Field(default=50, description="Max SCF iterations.")
    n_active_orbitals: Optional[int] = Field(
        default=None, description="If provided, return this many frontier orbitals as a suggested CAS span."
    )
    max_orbitals_returned: int = Field(
        default=30, description="Trim returned orbital lists to at most this many entries to stay concise."
    )


class OrbitalPrepOutput(BaseModel):
    converged: bool
    homo_lumo_gap: Optional[float] = None
    mo_energies: List[float] = Field(default_factory=list)
    mo_occ: List[float] = Field(default_factory=list)
    recommended_active_orbitals: List[int] = Field(default_factory=list)
    message: str = ""


@tool
def evaluate_casscf_diagnostics(params: CASSCFDiagnosticsInput) -> CASSCFDiagnostics:
    """Summarize convergence and flag simple issues from a PySCF CASSCF run."""
    result = params.casscf_output
    recommendations = []
    needs_revision = False

    delta_energy = None
    if result.casscf_energy is not None:
        delta_energy = result.casscf_energy - result.hf_energy
        if abs(delta_energy) < params.energy_tolerance:
            recommendations.append("CASSCF energy change is small; consider expanding the active space.")
            needs_revision = True
    elif params.require_casscf_energy:
        recommendations.append("No CASSCF energy returned; ensure active orbitals/electrons are provided.")
        needs_revision = True

    if not result.converged:
        recommendations.append("SCF or CASSCF did not converge; adjust max_cycle or initial guess.")
        needs_revision = True

    summary_bits = [
        f"HF energy: {result.hf_energy}",
        f"CASSCF energy: {result.casscf_energy}" if result.casscf_energy is not None else "CASSCF energy: missing",
    ]
    summary = " | ".join(summary_bits)

    return CASSCFDiagnostics(
        converged=result.converged,
        needs_revision=needs_revision,
        delta_energy=delta_energy,
        summary=summary,
        recommendations=recommendations,
    )


@tool
def prepare_orbitals_and_rank(params: OrbitalPrepInput) -> OrbitalPrepOutput:
    """Run a quick SCF to generate frontier orbitals and rank them for CAS seeding."""
    from pyscf import gto, scf

    try:
        coords = [
            (_atomic_number_to_symbol(int(z)), tuple(pos))
            for z, pos in zip(params.atomsdata.numbers, params.atomsdata.positions)
        ]

        mol = gto.Mole()
        mol.atom = coords
        mol.basis = params.basis
        mol.charge = params.charge
        mol.spin = params.spin
        mol.unit = "Angstrom"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol) if params.spin == 0 else scf.ROHF(mol)
        mf.verbose = 0
        mf.max_cycle = params.max_cycle
        mf.conv_tol = 1e-8
        hf_e = mf.kernel()
        conv = bool(mf.converged)

        mo_energy = list(getattr(mf, "mo_energy", []))
        mo_occ = list(getattr(mf, "mo_occ", []))
        nao = len(mo_energy)

        homo_lumo_gap = None
        homo_idx = None
        lumo_idx = None
        if mo_occ:
            occ_pairs = [(i, e, o) for i, (e, o) in enumerate(zip(mo_energy, mo_occ))]
            occ_indices = [i for i, _, o in occ_pairs if o > 1e-3]
            virt_indices = [i for i, _, o in occ_pairs if o < 2 - 1e-3]
            if occ_indices and virt_indices:
                homo_idx = max(occ_indices)
                lumo_idx = min(virt_indices)
                homo_lumo_gap = mo_energy[lumo_idx] - mo_energy[homo_idx]

        # Rank orbitals by proximity to half-occupation and frontier placement.
        scores = []
        for i, (e, occ) in enumerate(zip(mo_energy, mo_occ)):
            frontier_penalty = abs(occ - 1.0)
            scores.append((frontier_penalty, abs(e), i))
        scores.sort()

        n_pick = params.n_active_orbitals or min(8, max(2, nao // 5))
        recommended = [idx for _, _, idx in scores[:n_pick]]
        recommended = sorted(set(recommended))

        # Trim outputs to keep payload small, centered around frontier if known.
        limit = max(1, min(params.max_orbitals_returned, nao))
        if homo_idx is not None and lumo_idx is not None:
            center = (homo_idx + lumo_idx) // 2
        else:
            center = nao // 2
        half = limit // 2
        start = max(0, center - half)
        end = min(nao, start + limit)
        mo_energies_trim = mo_energy[start:end]
        mo_occ_trim = mo_occ[start:end]

        return OrbitalPrepOutput(
            converged=conv,
            homo_lumo_gap=float(homo_lumo_gap) if homo_lumo_gap is not None else None,
            mo_energies=[float(x) for x in mo_energies_trim],
            mo_occ=[float(x) for x in mo_occ_trim],
            recommended_active_orbitals=recommended,
            message="" if conv else f"SCF not converged; hf_energy={hf_e}",
        )
    except Exception as exc:
        return OrbitalPrepOutput(
            converged=False,
            homo_lumo_gap=None,
            mo_energies=[],
            mo_occ=[],
            recommended_active_orbitals=[],
            message=str(exc),
        )


class ActiveSpacePlanInput(BaseModel):
    atomsdata: AtomsData = Field(description="Geometry used to build the active space plan.")
    charge: int = Field(default=0, description="Molecular charge.")
    spin: int = Field(default=0, description="2S value (0 for closed shell, 1 for doublet, etc.).")


class ActiveSpacePlan(BaseModel):
    n_active_electrons: int
    n_active_orbitals: int
    orbital_indices: List[int] = Field(
        default_factory=list, description="Sorted HF MO indices recommended for the active space."
    )
    homo_lumo_gap: Optional[float] = Field(
        default=None, description="Estimated HOMO-LUMO gap (Hartree) from the ranking SCF."
    )
    rationale: str
    warnings: List[str] = Field(default_factory=list)


@tool
def pick_active_space(params: ActiveSpacePlanInput) -> ActiveSpacePlan:
    """Let the agent select an active space based on aromaticity, elemental makeup, and molecular weight."""

    numbers = [int(z) for z in params.atomsdata.numbers]
    positions = params.atomsdata.positions
    total_electrons = max(sum(numbers) - params.charge, 0)
    weight = _molecular_weight(numbers)
    has_tm = any(_is_transition_metal(z) for z in numbers)

    aromatic, aromatic_warning = _guess_aromaticity(numbers, positions)
    warnings: List[str] = []
    if aromatic_warning:
        warnings.append(aromatic_warning)

    scale = "small" if weight < 50 else ("medium" if weight < 150 else "large")
    n_unpaired = max(params.spin, 0)

    if has_tm:
        # Favor d-shell coverage; scale with molecular size.
        n_orb = 8 if scale == "small" else (10 if scale == "medium" else 12)
        n_elec = n_orb
    elif aromatic:
        # Capture pi system; grow mildly with size.
        n_orb = 6 if scale in ("small", "medium") else 8
        n_elec = n_orb
    else:
        # Non-aromatic main-group: scale gently with molecular weight.
        if scale == "small":
            n_orb = 4
        elif scale == "medium":
            n_orb = 6
        else:
            n_orb = 8
        n_elec = n_orb

    # Respect spin/unpaired electrons and total electron parity.
    n_elec = max(n_elec, n_unpaired + (n_unpaired % 2))
    if n_elec % 2 != total_electrons % 2:
        n_elec = max(n_elec - 1, n_unpaired)
    n_elec = min(n_elec, total_electrons)

    rationale_parts = [
        f"Molecular weight ~ {weight:.1f} amu ({scale})",
        f"Aromatic: {aromatic}",
        f"Transition metal present: {has_tm}",
        f"Spin (2S): {params.spin}",
    ]
    rationale = "; ".join(rationale_parts)

    return ActiveSpacePlan(
        n_active_electrons=int(n_elec),
        n_active_orbitals=int(n_orb),
        orbital_indices=[],
        homo_lumo_gap=None,
        rationale=f"Frontier-free selection - {rationale}",
        warnings=warnings,
    )
