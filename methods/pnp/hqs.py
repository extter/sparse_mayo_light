import torch
import torch.nn as nn

from methods.end_to_end_2.unet import SimpleUNet


class PnpHQS_Solver(nn.Module):
    def __init__(self, denoiser_path, device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.denoiser = SimpleUNet(in_ch=1, out_ch=1, base_ch=32).to(self.device)
        self.denoiser.load_state_dict(torch.load(denoiser_path, map_location=self.device))
        self.denoiser.eval()

        for param in self.denoiser.parameters():
            param.requires_grad = False

    def _matvec(self, v, projector, mu, y_scale):
        """
        Calcola il prodotto matrice-vettore M*v = (A^T A + mu*I) * v
        senza mai costruire M esplicitamente.

        E' il cuore del CG: basta saper applicare l'operatore forward
        e il suo aggiunto (backprojection).
        """
        return projector.T(projector(v)) / y_scale + mu * v

    def _conjugate_gradient(self, projector, b, mu, x_init, n_cg=5, y_scale=1):
        """
        Risolve il sistema lineare:

            (A^T A + mu * I) x = b

        con il metodo del Conjugate Gradient (CG).

        Questo e' il sottoproblema ESATTO in x dell'HQS:

            x^{k+1} = argmin_x  ||Ax - y||^2 + mu * ||x - z^k||^2

        Il termine noto b = A^T y + mu * z^k viene passato dall'esterno
        perche' dipende da z^k che cambia ad ogni iterazione esterna.

        Parametri
        ---------
        b      : termine noto  A^T y + mu * z^k  (cambia ad ogni iter HQS)
        mu     : parametro di accoppiamento (stesso usato nell'HQS esterno)
        x_init : punto di partenza (tipicamente x^k, la soluzione precedente)
        n_cg   : numero di iterazioni CG (3-5 bastano grazie alla convergenza
                 superlineare del CG su sistemi ben condizionati)

        Note
        ----
        Il CG non richiede step size ne' ATA_norm: gestisce autonomamente
        la direzione e l'ampiezza di ogni passo tramite i coefficienti
        alpha_cg e beta_cg calcolati dai prodotti scalari.
        """
        x = x_init.clone()

        # Residuo iniziale: r = b - M*x
        r = b - self._matvec(x, projector, mu, y_scale)
        p = r.clone()           # direzione di ricerca iniziale
        r_dot = (r * r).sum()   # <r, r>  (prodotto scalare)

        for _ in range(n_cg):
            if r_dot.item() < 1e-10:
                # Gia' converguto: residuo nullo, soluzione esatta trovata
                break

            Mp = self._matvec(p, projector, mu, y_scale)      # M * p

            # Passo ottimale lungo la direzione p:
            # alpha = <r,r> / <p, Mp>
            alpha_cg = r_dot / ((p * Mp).sum() + 1e-12)

            x = x + alpha_cg * p                     # aggiorna soluzione
            r = r - alpha_cg * Mp                    # aggiorna residuo

            r_dot_new = (r * r).sum()

            # Coefficiente di aggiornamento della direzione di ricerca:
            # beta = <r_new, r_new> / <r_old, r_old>
            beta_cg = r_dot_new / (r_dot + 1e-12)

            p = r + beta_cg * p                      # nuova direzione coniugata
            r_dot = r_dot_new

        return torch.clamp(x, 0.0, 1.0)

    def reconstruct(self, sinogram, projector, x_init=None,
                    num_iterations=20, mu=0.1, n_cg=5):
        """
        PnP-HQS con sottoproblema in x risolto via Conjugate Gradient.

        Ad ogni iterazione esterna k:

          FASE 1 (x-step, esatto):
            b^k = A^T y + mu * z^k
            x^{k+1} = CG( (A^T A + mu I) x = b^k )   <- n_cg iterazioni

          FASE 2 (z-step, Plug and Play):
            z^{k+1} = D_CNN( x^{k+1} )

        Parametri
        ---------
        num_iterations : iterazioni HQS esterne (tipico: 15-30;
                         meno che prima perche' ogni x-step e' esatto)
        mu             : accoppiamento x=z  (tipico: 0.05-0.3)
        n_cg           : iterazioni CG interne  (tipico: 3-7)
        """
        with torch.no_grad():
            y = sinogram.to(self.device)

            # --- Inizializzazione da FBP ---
            if x_init is not None:
                x = x_init.to(self.device)
            else:
                try:
                    x = projector.FBP(y).to(self.device)
                except AttributeError:
                    x = projector.T(y).to(self.device)

            x = torch.clamp(x, 0.0, 1.0)
            z = x.clone()

            # Precalcola A^T y: e' costante per tutte le iterazioni
            # (y non cambia mai, quindi questo prodotto si fa una volta sola)
            ATy = projector.T(y)

            y_scale = ATy.mean().item()
            ATy_norm = ATy / y_scale

            print(f"y: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
            print(f"ATy: min={ATy_norm.min():.4f}, max={ATy_norm.max():.4f}, mean={ATy_norm.mean():.4f}")

            print(f'   [HQS-CG] Avvio {num_iterations} iter esterne x {n_cg} iter CG '
                  f'(mu={mu})')

            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

            for k in range(num_iterations):

                # -----------------------------------------------------------
                # FASE 1: x-step  — risolve (A^T A + mu*I) x = A^T y + mu*z
                #
                # b^k cambia ad ogni iterazione perche' z^k cambia.
                # A^T y e' precalcolato fuori dal loop (costante).
                # -----------------------------------------------------------
                b = ATy_norm + mu * z
                x = self._conjugate_gradient(
                    projector=projector,
                    b=b,
                    mu=mu,
                    x_init=x,   # warm start: parte dalla soluzione precedente
                    n_cg=n_cg,
                    y_scale=y_scale
                )

                # -----------------------------------------------------------
                # FASE 2: z-step  — Plug and Play denoiser
                #   z^{k+1} = D_CNN(x^{k+1})
                # -----------------------------------------------------------
                with torch.amp.autocast(device_type):
                    residuo_pred = self.denoiser(x)
                    z = x - residuo_pred
                z = torch.clamp(z, 0.0, 1.0)

                if (k + 1) % 1 == 0:
                    residuo  = (projector(x) - y).norm().item()
                    delta_xz = (x - z).abs().mean().item()
                    print(f'      iter {k+1:2d} | ||Ax-y||={residuo:.4f} | mean|x-z|={delta_xz:.5f}')
            return x
                



