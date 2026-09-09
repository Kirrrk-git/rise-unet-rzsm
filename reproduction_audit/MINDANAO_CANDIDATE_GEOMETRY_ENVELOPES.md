<!-- markdownlint-disable -->
# Step 21A.4 Audit Dossier: Candidate 0.25° Spatial Envelopes & Model Tensor Dimensions

**Project**: Enhanced RISE-UNet for Subseasonal RZSM Drought Forecasting in Mindanao  
**Repository Branch**: `mindanao-adaptation`  
**Execution Date**: September 10, 2026  
**Audited Reference Geometry**: `mindanao_analysis_boundary.gpkg` (SHA-256: `9c7478ec02718153c124e8f528211951c6a855c0221dfd8e443b5c430bd623d2`)  
**Status**: **PASSED (Candidate Envelopes Defined & Divisibility Confirmed)**

---

## 1. Geographic Bounds & Geometric Constraints

From the certified PSA-NAMRIA unified Mindanao boundary (`mindanao_analysis_boundary.gpkg`), the exact outer geographic extrema are:

$$\begin{aligned}
\text{West Longitude } (\lambda_{\min}) &= 118.064236^\circ\text{E} \quad \text{(Turtle Islands / Sitangkai, Tawi-Tawi)} \\
\text{East Longitude } (\lambda_{\max}) &= 126.604966^\circ\text{E} \quad \text{(Cape San Agustin / Caraga Coast)} \\
\text{South Latitude } (\phi_{\min}) &= 4.587294^\circ\text{N} \quad \text{(Sitangkai / Frances Reef, Tawi-Tawi)} \\
\text{North Latitude } (\phi_{\max}) &= 10.471595^\circ\text{N} \quad \text{(Dinagat Islands / Surigao Strait)}
\end{aligned}$$

### Span Requirements at 0.25° Resolution ($\Delta = 0.25^\circ$):
* **Latitude Span**: $\Delta\phi = 10.471595^\circ - 4.587294^\circ = 5.884301^\circ \implies \lceil 5.884301 / 0.25 \rceil = \mathbf{24\text{ cells}}$ (minimum).
* **Longitude Span**: $\Delta\lambda = 126.604966^\circ - 118.064236^\circ = 8.540730^\circ \implies \lceil 8.540730 / 0.25 \rceil = \mathbf{35\text{ cells}}$ (minimum).

### Architectural Divisibility Constraint:
The frozen `UNET_RZSM` backbone implements 4 pooling stages ($2\times 2$ downsampling with stride 2), requiring that both height $H$ and width $W$ must be strictly divisible by $2^4 = \mathbf{16}$:
$$H \pmod{16} = 0, \quad W \pmod{16} = 0$$

Therefore:
* Minimum feasible height $H \ge 24$ satisfying $H \pmod{16} = 0 \implies \mathbf{H \in \{32, 48\}}$.
* Minimum feasible width $W \ge 35$ satisfying $W \pmod{16} = 0 \implies \mathbf{W \in \{48, 64\}}$.

---

## 2. Enumeration of Candidate Domains ($H \times W$)

All candidates are aligned with the official standard ERA5 / ERA5-Land $0.25^\circ$ coordinate convention ($0.25^\circ$ cell centers at multiples of $0.25^\circ$):

| Candidate ID | Grid Dimensions ($H \times W$) | Total Cells | Lat Extent ($\phi_{\min} \to \phi_{\max}$) | Lon Extent ($\lambda_{\min} \to \lambda_{\max}$) | Lat Buffer ($\text{S} / \text{N}$) | Lon Buffer ($\text{W} / \text{E}$) | Regional Coverage Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Candidate A** | **$32 \times 48$** | **1,536** | $4.00^\circ\text{N} \to 11.75^\circ\text{N}$ ($8.00^\circ$) | $116.00^\circ\text{E} \to 127.75^\circ\text{E}$ ($12.00^\circ$) | $+0.59^\circ / +1.28^\circ$ | $+2.06^\circ / +1.15^\circ$ | **Tight Focused Domain**: Fits complete Mindanao boundary with clean $0.5^\circ\text{--}2.0^\circ$ buffer on all four sides. Minimal compute footprint. |
| **Candidate B** | **$32 \times 64$** | **2,048** | $4.00^\circ\text{N} \to 11.75^\circ\text{N}$ ($8.00^\circ$) | $114.50^\circ\text{E} \to 130.25^\circ\text{E}$ ($16.00^\circ$) | $+0.59^\circ / +1.28^\circ$ | $+3.56^\circ / +3.65^\circ$ | **East-West Synoptic Buffer**: Adds wider maritime buffers covering Sulu Sea, western Celebes Sea, and Philippine Sea approaches. |
| **Candidate C** | **$48 \times 48$** | **2,304** | $2.00^\circ\text{N} \to 13.75^\circ\text{N}$ ($12.00^\circ$) | $116.00^\circ\text{E} \to 127.75^\circ\text{E}$ ($12.00^\circ$) | $+2.59^\circ / +3.28^\circ$ | $+2.06^\circ / +1.15^\circ$ | **Symmetric Square**: Extends into Visayas northward and northern Indonesia / Celebes Sea southward. |
| **Candidate D** | **$48 \times 64$** | **3,072** | $2.00^\circ\text{N} \to 13.75^\circ\text{N}$ ($12.00^\circ$) | $114.50^\circ\text{E} \to 130.25^\circ\text{E}$ ($16.00^\circ$) | $+2.59^\circ / +3.28^\circ$ | $+3.56^\circ / +3.65^\circ$ | **Broad Regional Domain**: Maximum regional context enclosing central Philippines and maritime seas. |

---

## 3. Boundary Containment & Zero-Truncation Audit

Every candidate domain strictly encloses the Mindanao multi-polygon boundary with **zero geometric truncation**:

1. **Southernmost Point** (Frances Reef / Sitangkai, $4.587294^\circ\text{N}$):
   * Candidate A & B south bound ($4.00^\circ\text{N}$): Margin $= 0.587^\circ \approx 2.35$ grid cells buffer.
   * Candidate C & D south bound ($2.00^\circ\text{N}$): Margin $= 2.587^\circ \approx 10.35$ grid cells buffer.
2. **Northernmost Point** (Dinagat Islands, $10.471595^\circ\text{N}$):
   * Candidate A & B north bound ($11.75^\circ\text{N}$): Margin $= 1.278^\circ \approx 5.11$ grid cells buffer.
   * Candidate C & D north bound ($13.75^\circ\text{N}$): Margin $= 3.278^\circ \approx 13.11$ grid cells buffer.
3. **Westernmost Point** (Turtle Islands, $118.064236^\circ\text{E}$):
   * Candidate A & C west bound ($116.00^\circ\text{E}$): Margin $= 2.064^\circ \approx 8.26$ grid cells buffer.
   * Candidate B & D west bound ($114.50^\circ\text{E}$): Margin $= 3.564^\circ \approx 14.26$ grid cells buffer.
4. **Easternmost Point** (Cape San Agustin / Caraga, $126.604966^\circ\text{E}$):
   * Candidate A & C east bound ($127.75^\circ\text{E}$): Margin $= 1.145^\circ \approx 4.58$ grid cells buffer.
   * Candidate B & D east bound ($130.25^\circ\text{E}$): Margin $= 3.645^\circ \approx 14.58$ grid cells buffer.

---

## 4. Multi-Stage Pooling & Upsampling Verification (Step 21A.5 Input)

Every candidate domain satisfies exact integer divisibility across all 4 downsampling ($2\times$) and upsampling ($2\times$) stages of the frozen `UNET_RZSM` architecture:

* **Level 1 (Input)**: $(H, W) \implies (32, 48), (32, 64), (48, 48), (48, 64)$
* **Level 2 (Pool 1)**: $(H/2, W/2) \implies (16, 24), (16, 32), (24, 24), (24, 32)$
* **Level 3 (Pool 2)**: $(H/4, W/4) \implies (8, 12), (8, 16), (12, 12), (12, 16)$
* **Level 4 (Pool 3)**: $(H/8, W/8) \implies (4, 6), (4, 8), (6, 6), (6, 8)$
* **Bottleneck (Pool 4)**: $(H/16, W/16) \implies (2, 3), (2, 4), (3, 3), (3, 4)$

All intermediate spatial feature map dimensions are strictly whole positive integers. Convolutions (`padding='same'`) and Transposed Convolutions (`strides=(2, 2)`, `padding='same'`) produce zero shape mismatch upon concatenation across all levels.

---

## 5. Certification & Next Step

* **Step 21A.4 Status**: **PASSED**.
* **Next Step (Step 21A.5)**: Construct and execute `notebooks/02_mindanao_geometry_and_cascade_gate.ipynb` executing synthetic 4-lead forward recursive cascade through frozen `UNET_RZSM` across these candidate dimensions.
