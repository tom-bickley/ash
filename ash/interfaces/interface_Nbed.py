import nbed
import pyscf
from typing import Optional
import ash

from ash.functions.functions_general import ashexit, print_line_with_mainheader

# NbedTheory object
# https://github.com/UCL-CCS/Nbed

class NbedTheory:
    def __init__(
        self,

        # nbed driver parameters:
        geometry: str,
        n_active_atoms: int,
        basis: str,
        xc_functional: str,
        projector: str,
        localization: Optional[str] = "spade",
        convergence: Optional[float] = 1e-6,
        charge: Optional[int] = 0,
        spin: Optional[int] = 0,
        symmetry: Optional[bool] = False,
        mu_level_shift: Optional[float] = 1e6,
        run_ccsd_emb: Optional[bool] = False,
        run_fci_emb: Optional[bool] = False,
        run_virtual_localization: Optional[bool] = True,
        run_dft_in_dft: Optional[bool] = False,
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        unit: Optional[str] = "angstrom",
        occupied_threshold: Optional[float] = 0.95,
        virtual_threshold: Optional[float] = 0.95,
        max_shells: Optional[int] = 4,
        init_huzinaga_rhf_with_mu: bool = False,
        max_hf_cycles: int = 50,
        max_dft_cycles: int = 50,
        force_unrestricted: Optional[bool] = False,
        run_qmmm: Optional[bool] = False,
        mm_coords: Optional[list] = None,
        mm_charges: Optional[list] = None,
        mm_radii: Optional[list] = None,

    ):
        print_line_with_mainheader("NbedTheory initialization")
        
        self.theorytype="QM"
        
        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis
        self.xc_functional = xc_functional
        self.projector = projector
        self.localization = localization
        self.convergence = convergence
        self.charge = charge
        self.spin = spin
        self.symmetry = symmetry
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.run_virtual_localization = run_virtual_localization
        self.run_dft_in_dft = run_dft_in_dft
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.unit = unit
        self.occupied_threshold = occupied_threshold
        self.virtual_threshold = virtual_threshold
        self.max_shells = max_shells
        self.init_huzinaga_rhf_with_mu = init_huzinaga_rhf_with_mu
        self.max_hf_cycles = max_hf_cycles
        self.max_dft_cycles = max_dft_cycles
        self.force_unrestricted = force_unrestricted
        self.run_qmmm = run_qmmm
        self.mm_coords = mm_coords
        self.mm_charges = mm_charges
        self.mm_radii = mm_radii

    def run(
            self,
            current_coords=None,
            current_MM_coords=None,
            MMcharges=None,
            qm_elems=None,
            mm_elems=None,
            elems=None,
            Grad=False,
            PC=False,
            numcores=None,
            pe=False,
            potfile=None,
            restart=False,
            label=None,
            charge=None,
            mult=None,
            Hessian=False
        ):

        driver = nbed.driver.NbedDriver(
            geometry=self.geometry,
            n_active_atoms=self.n_active_atoms,
            basis=self.basis,
            xc_functional=self.xc_functional,
            projector=self.projector,
            localization=self.localization,
            convergence=self.convergence,
            charge=self.charge,
            spin=self.spin,
            symmetry=self.symmetry,
            mu_level_shift=self.mu_level_shift,
            run_ccsd_emb=self.run_ccsd_emb,
            run_fci_emb=self.run_fci_emb,
            run_virtual_localization=self.run_virtual_localization,
            run_dft_in_dft=self.run_dft_in_dft,
            max_ram_memory=self.max_ram_memory,
            pyscf_print_level=self.pyscf_print_level,
            unit=self.unit,
            occupied_threshold=self.occupied_threshold,
            virtual_threshold=self.virtual_threshold,
            max_shells=self.max_shells,
            init_huzinaga_rhf_with_mu=self.init_huzinaga_rhf_with_mu,
            max_hf_cycles=self.max_hf_cycles,
            max_dft_cycles=self.max_dft_cycles,
            force_unrestricted=self.force_unrestricted,
            run_qmmm=self.run_qmmm,
        )

        if self.projector == 'mu':
            proj_data = driver._mu
        if self.projector == 'huzinaga':
            proj_data = driver._huzinaga
        
        emb_corr = (
            driver.e_env
            + driver.two_e_cross
            - proj_data["correction"]
            - proj_data["beta_correction"]
        )

        emb_ccsd, _ = driver._run_emb_CCSD(driver.embedded_scf)
        emb_ccsdt_energy = emb_ccsd.e_tot + emb_ccsd.ccsd_t()

        self.energy = emb_ccsdt_energy + emb_corr

        # approx gradient with DFT gradient 
        if Grad:
            g = driver._global_ks.nuc_grad_method()
            self.gradient = g.kernel()

            # approx point charge gradient with DFT 
            if PC:
                from .interface_pyscf import pyscf_pointcharge_gradient
                import numpy as np
                dm = driver._global_ks.make_rdm1()
                current_MM_coords_bohr = current_MM_coords*ash.constants.ang2bohr
                self.pcgrad = pyscf_pointcharge_gradient(
                    driver._global_ks.mol,
                    np.array(current_MM_coords_bohr),
                    np.array(MMcharges),
                    dm,
                    GPU=self.GPU_pcgrad
                )

                return self.energy, self.gradient, self.pcgrad
            
            else:
                return self.energy, self.gradient
        
        else:
            return self.energy
        
