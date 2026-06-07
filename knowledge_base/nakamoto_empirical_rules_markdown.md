# Nakamoto Empirical Rules — Spectral-to-Structural Transition Reasoning Dataset

> **Source:** Infrared and Raman Spectra of Inorganic and Coordination Compounds, 6th Ed., Part A, Ch.2 — Kazuo Nakamoto  
> **Total rules:** 181 across 12 categories  
> **Format:** This file is optimized for both human reading and agentic retrieval (semantic chunking + structured frontmatter)

---

## Quick Index

| Category | Tag | #Rules | Purpose |
|---|---|---|---|
| Selection Rules | `[SEL]` | 10 | Determine IR vs Raman activity from symmetry |
| Frequency Trends | `[FRQ]` | 11 | Predict relative frequencies from atomic/molecular properties |
| Geometry Determination | `[GEO]` | 16 | Deduce molecular structure from spectral patterns |
| Group Frequencies | `[GFQ]` | 85 | Characteristic spectral ranges for functional groups |
| Charge & Oxidation Effects | `[CHG]` | 5 | How oxidation state/charge shifts frequencies |
| Substituent Effects | `[SUB]` | 8 | Effect of changing atoms/groups on spectra |
| Isotope Effects | `[ISO]` | 5 | Using isotopic substitution for band assignment |
| Environmental Effects | `[ENV]` | 10 | Matrix, H-bonding, pressure, temperature effects |
| Force Constant Correlations | `[FRC]` | 4 | Relating frequencies to bond strengths/distances |
| Metal-Ligand Vibrations | `[VIB]` | 6 | Coordination compound diagnostics |
| Cluster Vibrations | `[CRY]` | 6 | Cluster/framework structure-spectra relationships |
| Special Topics | `[SPC]` | 11 | Fermi resonance, J-T effect, RR, isotope patterns |

---

## [SEL] Selection Rules — IR vs Raman Activity

### SEL-001: Mutual Exclusion Rule
**Rule:** For centrosymmetric molecules, IR-active vibrations are Raman-inactive, and vice versa.
**Systems:** All centrosymmetric point groups (D∞h, D2h, D4h, Oh, Ih)
**Confidence:** DEFINITIVE
**Section:** 2.2, 2.4.1, 2.6.3, 2.8.1
**Note:** Violations can occur due to dynamic effects, electronic degeneracy, or symmetry lowering in condensed phases.

### SEL-002: Linear X₃ (D∞h)
**Rule:** ν₁(sym stretch) = Raman only; ν₂(bend) + ν₃(asym stretch) = IR only
**Systems:** CO₂, CS₂, linear triatomics
**Confidence:** DEFINITIVE

### SEL-003: Linear XYZ (C∞v)
**Rule:** All 3 modes (ν(XY), δ, ν(YZ)) are both IR- and Raman-active
**Systems:** HCN, NNO, OCS, HArF
**Confidence:** DEFINITIVE

### SEL-004: Bent XY₂ (C2v) / Bent XYZ (Cs)
**Rule:** All 3 normal modes are both IR- and Raman-active
**Systems:** H₂O, SO₂, NO₂, O₃
**Confidence:** DEFINITIVE

### SEL-005: Planar XY₃ (D3h)
**Rule:** ν₁(A₁′) = Raman only; ν₂(A₂″) = IR only; ν₃(E′) + ν₄(E′) = both IR+Raman
**Systems:** BF₃, CO₃²⁻, NO₃⁻, SO₃
**Confidence:** DEFINITIVE
**Contrast:** Pyramidal XY₃ (C3v) has all 4 modes both IR+Raman active

### SEL-006: Pyramidal XY₃ (C3v)
**Rule:** All 4 modes (ν₁-ν₄) are both IR- and Raman-active
**Systems:** NH₃, PH₃, AsH₃, NF₃, PCl₃
**Confidence:** DEFINITIVE

### SEL-007: Tetrahedral XY₄ (Td)
**Rule:** ν₁(A₁) + ν₂(E) = Raman only; ν₃(F₂) + ν₄(F₂) = both IR+Raman
**Systems:** CH₄, NH₄⁺, SO₄²⁻, ClO₄⁻, BF₄⁻
**Confidence:** DEFINITIVE

### SEL-008: Square-Planar XY₄ (D4h)
**Rule:** ν₁(A₁g)+ν₂(B₁g)+ν₄(B₂g) = Raman; ν₃(A₂u)+ν₆(Eu)+ν₇(Eu) = IR; ν₅(B₂u) = inactive
**Systems:** XeF₄, PdCl₄²⁻, PtCl₄²⁻, AuCl₄⁻
**Confidence:** DEFINITIVE
**Note:** ν₅ observable by inelastic neutron scattering (INS)

### SEL-009: Octahedral XY₆ (Oh)
**Rule:** ν₁(A₁g)+ν₂(Eg)+ν₅(F₂g) = Raman; ν₃(F₁u)+ν₄(F₁u) = IR; ν₆(F₂u) = inactive
**Systems:** SF₆, UF₆, MF₆³⁻/²⁻
**Confidence:** DEFINITIVE

### SEL-010: Icosahedral C₆₀ (Ih)
**Rule:** Only 4 F₁u IR-active; only 2 A_g + 8 H_g Raman-active
**Systems:** C₆₀, endohedral fullerenes
**Spectra:** IR at 1428, 1182, 577, 527 cm⁻¹; Raman at 1575, 1469, 1427, 1250, 1100, 774, 711, 496, 436, 273 cm⁻¹
**Confidence:** EXPERIMENTALLY_VERIFIED

---

## [FRQ] Frequency Trends

### FRQ-001: Bond Order vs Frequency
**Rule:** Higher bond order → higher vibrational frequency
**Example:** O₂⁺(1876) > O₂(1580) > O₂⁻(1094) > O₂²⁻(791/736 cm⁻¹)
**Confidence:** DEFINITIVE

### FRQ-002: Mass Effect Within Group
**Rule:** Heavier central atom → lower frequency
**Example:** Li₂(351) > Na₂(159) > K₂(93) > Rb₂(57) > Cs₂(42 cm⁻¹)
**Confidence:** DEFINITIVE

### FRQ-003: Across-Period Trend
**Rule:** Frequency increases left-to-right across period for hydrides
**Example:** HB(2366) < HC(2860) < HN(3300) < HO(3735) < HF(4139 cm⁻¹)
**Confidence:** DEFINITIVE

### FRQ-004: Halogen Frequency Trend
**Rule:** M-F > M-Cl > M-Br > M-I (universal)
**Ratios:** ν(MBr)/ν(MCl) ≈ 0.76 (ν₃), 0.61 (ν₁); ν(MI)/ν(MCl) ≈ 0.62 (ν₃), 0.42 (ν₁)
**Confidence:** DEFINITIVE

### FRQ-005: Anharmonicity
**Rule:** Largest for light molecules (H₂: xₑωₑ = 117.9; Cs₂: xₑωₑ = 0.08 cm⁻¹)
**Confidence:** DEFINITIVE

### FRQ-006: Tetrahedral XY₄ Trend
**Rule:** ν₃ > ν₁ and ν₄ > ν₂ for most compounds
**Confidence:** HIGH

### FRQ-007: Octahedral XY₆ Trend
**Rule:** Bending: ν₄ > ν₅ > ν₆ (most cases)
**Confidence:** HIGH

### FRQ-008: Trigonal-Bipyramidal Trend
**Rule:** ν₅ > ν₃ > ν₁ > ν₂ (equatorial > axial stretches)
**Reason:** Equatorial bonds are stronger than axial
**Confidence:** HIGH

### FRQ-009: Tetragonal-Pyramidal Trend
**Rule:** ν₁(axial) > equatorial stretches — opposite to TBP
**Confidence:** HIGH

### FRQ-010: Bonding/Antibonding Electron Removal
**Rule:** Losing bonding e⁻ → ν decreases; losing antibonding e⁻ → ν increases
**Examples:** N₂(2359) > N₂⁺(2207); CO⁺(2214) > CO(2170)
**Confidence:** DEFINITIVE

### FRQ-011: Carbon Cluster Frequencies
**Rule:** Neutral Cₙ linear clusters have characteristic νₐ frequencies
**Values:** C₃ 2039, C₄ 1543, C₅ 2164, C₆ 1952, C₇ 2128, C₈ 2072, C₉ 2014, C₁₁ 1943, C₁₃ 1809 cm⁻¹
**Confidence:** EXPERIMENTALLY_VERIFIED

---

## [GEO] Geometry Determination from Spectra

### GEO-001: Linear vs Bent XY₂
**Criterion:** ν₁ is IR-inactive for linear (D∞h), IR-active for bent (C2v)
**Warning:** Many bent XY₂ have angles 95-170°, matrix can change structure
**Confidence:** HIGH

### GEO-002: Pyramidal vs Planar XY₃
**Criterion:** Pyramidal → 2 stretching bands in IR (A₁+E); Planar → 1 stretching band (E′)
**Test for:** SO₃²⁻ (pyramidal) vs SO₃ (planar)
**Confidence:** DEFINITIVE

### GEO-003: TBP vs Tetragonal-Pyramidal XY₅
**Criterion:** TBP → 2 IR-active stretches; TP → 3 IR-active stretches
**Additional test:** TBP: equatorial > axial; TP: axial > equatorial
**Confidence:** HIGH

### GEO-004: Bond Angle from Isotope Shift
**Formula:** ν₃′/ν₃ ratio with isotopic substitution at X gives bond angle α
**Example:** NiF₂ angle = 154-167° from isotope data
**Confidence:** HIGH

### GEO-005: Trans vs Gauche X₂Y₄
**Criterion:** Trans(C2h) → 6 IR fundamentals; Gauche(C2) → 12 IR fundamentals
**Examples:** N₂H₄ = pure gauche; P₂H₄ = gauche gas, trans solid
**Confidence:** HIGH

### GEO-006: Planar vs Staggered B₂X₄
**Criterion:** D2h(planar) → mutual exclusion; D2d(staggered) → no mutual exclusion
**Confidence:** HIGH

### GEO-007: Pentagonal-Planar XY₅
**Criterion:** Only 3 IR + 3 Raman bands (D5h) vs many for TBP or TP
**Example:** XeF₅⁻ spectra confirm D5h
**Confidence:** EXPERIMENTALLY_VERIFIED

### GEO-008: Cis/Trans Octahedral XY₄Z₂
**Criterion:** Trans(D4h) → 3 XY stretches (Eu IR-only); Cis(C2v) → 4 XY stretches (all active)
**Confidence:** DEFINITIVE

### GEO-009: Cis/Trans/Nonplanar X₂Y₂
**Criterion:** ν₆ distinguishes cis-planar, trans-planar, and nonplanar (C2, C2v, C2h)
**Confidence:** HIGH

### GEO-010: Planar (D2h) vs Twisted (D2)
**Criterion:** D2h → mutual exclusion; D2 → no mutual exclusion
**Systems:** B₂H₄, B₂F₄, N₂O₄
**Confidence:** HIGH

### GEO-011: D3d vs D3h for X₂Y₆
**Criterion:** D3d → Eu(IR) + Eg(Raman) separate; D3h → E′ both IR+Raman
**Systems:** Si₂Cl₆, Si₂F₆
**Confidence:** HIGH

### GEO-012: Siₙ Cluster Geometry
**Criterion:** Count Raman-active fundamentals → identify geometry
**Examples:** Si₄(D2h): 3 Raman; Si₆(D4h): 5 Raman; Si₇(D5h): 5 Raman
**Confidence:** EXPERIMENTALLY_VERIFIED

### GEO-013: Bond Strength in XY₅
**Criterion:** TBP: equatorial > axial; TP: axial > equatorial (confirmed by force constants)
**Confidence:** HIGH

### GEO-014: T-Shaped XY₃ (C2v)
**Criterion:** 3 stretching modes (vs 1 for planar D3h), derived from TBP with 2 lone pairs
**Systems:** ClF₃, BrF₃, XeF₃⁺
**Confidence:** HIGH

### GEO-015: Endohedral Fullerene Symmetry
**Criterion:** #IR/Raman bands > predicted for Ih → symmetry lowered by metal-cage interaction
**Example:** Sc₃N@C₈₀: expected 8+17, observed 37+59 bands
**Confidence:** EXPERIMENTALLY_VERIFIED

### GEO-016: Square-Planar X₄²⁺
**Criterion:** D4h → A1g(R)+B1g(R)+B2g(R)+B2u(inactive)+Eu(IR)
**Systems:** S₄²⁺(ν₁584), Se₄²⁺(ν₁321), Te₄²⁺(ν₁219 cm⁻¹)
**Confidence:** EXPERIMENTALLY_VERIFIED

---

## [GFQ] Group Frequencies — Characteristic Spectral Ranges

### Comprehensive Reference Table (cm⁻¹)

| Bond | Type / Context | ν (cm⁻¹) | Note |
|------|---------------|----------|------|
| **O-H** | free | 3600-3700 | sharp, gas phase |
| **O-H** | H-bonded | 3200-3450 | broad, cond. phase |
| **O-H** | δ(HOH) | 1595-1650 | bend |
| **O-H** | libration (H₂O) | 1000-330 | liquid/ice |
| **O-H** | restricted transl. | 170, 60 | liquid/ice |
| **O-D** | stretch | 2490-2788 | isotopic shift ~√2 |
| **H₃O⁺** | ν₁/ν₂/ν₃/ν₄ | 3385/1095/3470/1700 | hydronium |
| **H₂O₂** | ν(OH) | 3607/3608 | nonplanar C₂ |
| **H₂O₂** | ν(OO) | 1394 | |
| **H₂O₂** | δ(HOO) | 864/1266 | |
| **N-H** | NH₃ stretch | 3336-3414 | ν₁/ν₃ |
| **N-H** | NH₃ δ | 931-1628 | ν₂/ν₄ inversion |
| **NH₄⁺** | ν₁/ν₂/ν₃/ν₄ | 3040/1680/3145/1400 | tetrahedral Td |
| **N-H** | amine/amide | 3300-3100 | |
| **N-H** | δ(NH₂) rocking | 700-400 | |
| **C-H** | CH₄ stretch | 2917/3019 | ν₁/ν₃ |
| **C-H** | δ(CH) | 1306-1534 | ν₂/ν₄ |
| **C-H** | diamond C-H | 2834 | H-doped surface |
| **Si-H** | SiH₄ stretch | 2180/2183 | |
| **Si-H** | δ(SiH) | 910-970 | |
| **Si-H** | ν(SiH) silanes | ~2100-2200 | |
| **Ge-H** | GeH₄ stretch | 2106/2114 | |
| **Ge-H** | δ(GeH) | 819-931 | |
| **Sn-H** | SnH₄ stretch | 1901 | |
| **P-H** | PH₃ stretch | 2327/2421 | ν₁/ν₃ |
| **P-H** | δ(PH) | 992-1121 | inversion |
| **PH₄⁺** | ν₃ | ~2366 | |
| **As-H** | AsH₃ stretch | 2122/2185 | |
| **AsH₄⁺** | ν₃ | ~2341 | |
| **Sb-H** | SbH₃ stretch | 1891/1894 | |
| **SbH₄⁺** | ν₃ | ~2061 | |
| **S-H** | H₂S stretch | 2615/2627 | |
| **S-H** | δ(SH) | 1183 | |
| **Se-H** | H₂Se | ~2345/2358 | |
| **B-H** | terminal | 2600-2500 | boranes |
| **B-H** | bridging | 2200-2100 | B-H-B |
| **B-H** | δ(BH) | ~1200-1100 | |
| **Ga-H** | terminal | 2005-1982 | |
| **Ga-H** | bridging | 1435-1315 | Ga-H-B |
| **H-Ar** | HArF | 1969.5 | noble gas |
| **H-Kr** | HKrF | 1950 | |
| **N≡N** | N₂ molecular | ~2400 | triple bond |
| **N≡N** | N₂⁺ | ~2207 | |
| **N-N** | polymeric N | ~840 | cubic gauche |
| **N=N** | N₂H₂ str | ~1522-1525 | cis/trans |
| **N≡C** | CN⁻ | ~2080 | |
| **N≡C** | HCN | 2097 | ν₃ |
| **N≡C** | FCN | 2290 | halogen effect |
| **N≡C** | ClCN | 2249 | |
| **N≡C** | BrCN | 2200 | |
| **N≡C** | ICN | 2188 | |
| **C≡P** | phosphaalkyne | 1543-1565 | |
| **C=O** | CO ground | 2170 | triplet lower |
| **C=O** | CO⁺ | 2214 | |
| **C=O** | metal carbonyl terminal | 2000-2130 | |
| **C=O** | metal carbonyl bridge | 1800-1860 | |
| **C=O** | normal ketones | ~1700 | |
| **C=O** | aromatic CₙOₙ²⁻ | <1600 | ring-stabilized |
| **C=O** | C₃O₂ | ~2290 | νₐ |
| **C=O** | δ(OCO) | 660-670 | CO₂ bend |
| **C=C** | C≡C stretch | 1700-2200 | C₂ clusters |
| **C=C** | C₆₀ A_g(2) | 1469 | pentagonal pinch |
| **C=C** | C₆₀Br₂₄ | 1688-1610 | IR |
| **C=C** | diamond | 1332 | Raman |
| **C=C** | graphite | 1575 | Raman E₂g |
| **C=C** | SWNT RBM | ~186 | d~1.4 nm |
| **C=N** | cyanate OCN⁻ | ~2155 | ν(C≡N) |
| **C=S** | thiocyanate SCN⁻ | ~2053 | ν(C≡N) |
| **N=O** | NO⁺ | ~2273 | linear |
| **N=O** | NO neutral | ~1880 | |
| **N=O** | NO⁻ | 1358-1374 | |
| **N=O** | (NO)₂ dimer cis | 1862/1768 | |
| **N=O** | (NO)₂ trans | 1740 | |
| **N=O** | NO₂ νₐ | 1610 | neutral |
| **N=O** | NO₂⁻ νₐ | 1286 | nitrite |
| **N=O** | NO₂⁺ ν₃ | 2360 | nitronium |
| **N=O** | N₂O ν₁/ν₂/ν₃ | 2277/597/1300 | |
| **N₃⁻** | νₐ/νₛ/δ | 2048/1343/638 | azide |
| **N₃** | νₐ organic | ~2107 | |
| **N₃** | νₛ organic | ~1060 | |
| **N₅⁺** | ν(N≡N) | ~2237.6 | IR |
| **C=O** | carbonate CO₃²⁻ ν₃ | 1429-1492 | |
| **C=O** | carbonate ν₁ Raman | 1084-1087 | |
| **N=O** | nitrate NO₃⁻ ν₃ | 1370-1405 | |
| **N=O** | nitrate ν₁ Raman | 1049-1068 | |
| **S=O** | SO₂ νₐ | 1351 | |
| **S=O** | SO₂ νₛ | 1147 | |
| **S=O** | SO₄²⁻ ν₃ | 1105 | tetrahedral |
| **S=O** | SO₄²⁻ ν₁ | 983 | Raman |
| **P=O** | PO₄³⁻ ν₃ | 1017 | tetrahedral |
| **P=O** | PO₄³⁻ ν₁ | 938 | |
| **P=O** | OPCl₃ | ~1290 | |
| **U=O** | uranyl ν₃ | ~950 | |
| **Pu=O** | ν₃ | 962 | |
| **Am=O** | ν₃ | 939 | |
| **Si=O** | silanone | ~1202 | matrix isolated |
| **Si=O** | SiO₄ ν₁ | 750-830 | silicates |
| **Si=O** | SiO₄ ν₃ | 800-1000 | |
| **Si=O** | SiO₄ ν₄ | 450-600 | |
| **Si=O** | SiO₄ ν₂ | 300-400 | |
| **Si=O** | 4 terminal O | ~850 | silica glass |
| **Si=O** | 3 terminal O | ~900 | |
| **Si=O** | 2 terminal O | 950-1000 | |
| **Si=O** | 1 terminal O | 1050-1100 | |
| **Al-O** | aluminosilicates | 750-650 | zeolites |
| **Mo=O** | terminal | 940-930 | Mo₆O₁₉²⁻ |
| **Si=C** | silaethene | ~985 | |
| **Si=Si** | disilene | ~589 | |
| **Te=C** | telluroketone | 1240 | |
| **O-O** | O₂⁺ | 1876 | |
| **O-O** | O₂ | 1580 | |
| **O-O** | O₂⁻ superoxo | 1094 | |
| **O-O** | O₂²⁻ peroxo | 791-736 | |
| **O-O** | H₂O₂ | 1394 | |
| **O-O** | side-on peroxo | 885, 550 | W-O₂ |
| **O₃** | ν₁/ν₂/ν₃ | 1104/700/1040 | ozone |
| **O₃⁻** | ν₁/ν₂/ν₃ | 1020/620/816 | ozonide |
| **NO** | N₂O₄ | ~1205(NO₂), ~1314(N₂O₂) | cis/trans |
| **S-S** | S₈ ring | ~470 | |
| **S-S** | chelate | 492-460 | |
| **Se-Se** | Se₂ | 513 | |
| **Se-Se** | Se₃ bent | 350 | |
| **Se-Se** | Se₄ trans | 345 | |
| **Se-Se** | Se₄ cis | 370 | |
| **Te-Te** | Te₇²⁺ chain | 396, 126, 65 | |
| **I-I** | I₂ | ~213 | |
| **I-I** | I₃⁻ ν₁ | 116-125 | linear |
| **I-I** | I₃⁻ ν₃ | ~125 | |
| **I-I** | I₅⁻ | 178-166 | |
| **Re-Re** | Re₂X₈²⁻ | 258-320 | depending on X |
| **Os-Os** | Os₂X₈²⁻ | 270-287 | |
| **Mo-Mo** | Mo₆X₈Y₆²⁻ | <300 | |
| **Nb-Nb** | Nb₆Cl₁₂ | ~140 | |
| **Sc-N** | Sc₃N@C₈₀ | 411/599/210 | νₛ/νₐ/δ |
| **Y-N** | Y₃N@C₈₀ | 429/707/194 | |
| **Sc-C** | Sc₂@C₈₄ | ~250 | |
| **B-F** | BF₃ ν₁/ν₂/ν₃/ν₄ | 888/719/1506/481 | planar |
| **B-Cl** | BCl₃ | 473/950/951/254 | |
| **B-F** | BF₄⁻ ν₁/ν₂/ν₃/ν₄ | 777/360/1070/533 | tetrahedral |
| **B-Cl** | BCl₄⁻ | 406/192/722/278 | |
| **C-F** | CF₄ | 908/435/1283/631 | |
| **C-Cl** | CCl₄ | 460/214/792/314 | |
| **C-Br** | CBr₄ | 267/123/672/183 | |
| **Si-F** | SiF₄ | 801/264/1030/389 | |
| **Si-Cl** | SiCl₄ | 423/145/617/220 | |
| **Si-Br** | SiBr₄ | 247/85/494/134 | |
| **Si-F** | SiFClBrI | 910 | specific |
| **Si-Cl** | SiFClBrI | 587 | |
| **Si-Br** | SiFClBrI | 486 | |
| **Si-I** | SiFClBrI | 333 | |
| **Ge-Cl** | GeCl₄ | 397/125/459/171 | |
| **Sn-Cl** | SnCl₄ | 369/95/408/126 | |
| **Sn-Br** | SnBr₄ | 222/59/284/86 | |
| **Sn-I** | SnI₄ | 148/42/210/63 | |
| **Ti-Cl** | TiCl₄ | 389/114/498/136 | |
| **V-Cl** | VCl₄ | 383/128/475/150 | |
| **P-F** | PF₃ pyramidal | 893/487/858/346 | |
| **P-Cl** | PCl₃ | 515/258/504/186 | |
| **As-F** | AsF₃ | 739/337/699/262 | |
| **As-Cl** | AsCl₃ | 417/193/391/150 | |
| **Sb-Cl** | SbCl₃ | 381/359/122 | |
| **M-Cl** | SF₆ ν₁/ν₃ | 775/939 | octahedral |
| **M-Cl** | UF₆ ν₁/ν₃ | 667/628 | |
| **M-F** | AlF₆³⁻ ν₁/ν₃ | 541/568 | |
| **M-F** | SiF₆²⁻ ν₁/ν₃ | 663/741 | |
| **M-F** | PF₆⁻ ν₁/ν₃ | 745/840 | |
| **P-F** | PF₅ eq/ax | 944/575 | TBP |
| **Sn-I** | SnCl₃⁻ ν₁/ν₂/ν₃/ν₄ | 297/130/256/104 | pyramidal |
| **I-F** | IF₅ ν₁(ax) | ~709 | TP |
| **Xe-F** | XeF₄ ν₁/ν₂/ν₄ Raman | 505/524/245 | square planar |
| **Xe-F** | XeF₄ ν₃/ν₆/ν₇ IR | 714/372/170 | |
| **F-H-F** | [FHF]⁻ ν₁/ν₂/ν₃ | 603/1233/1450 | linear |
| **F-H-Cl** | [FHCl]⁻ type I | ~2500 | asymmetric |
| **F-H-Cl** | [FHCl]⁻ type II | ~933 | symmetric |
| **Cl-O** | ClO₂ | 945/449/1108 | |
| **Br-O** | BrO₂ | 796/317/845 | |
| **Cr-Cr** | Cr-O-Cr νₛ | ~527 | bridge |
| **Cr-Cr** | Cr-O-Cr δ | ~230 | |
| **P-O-P** | νₐ | ~915 | pyrophosphate |
| **P-O-P** | νₛ | ~730 | |
| **As-O-As** | νₐ | ~735 | |
| **As-O-As** | νₛ | ~550 | |
| **S-O-S** | νₐ | ~825 | disulfate |
| **C₃Cl₃⁺** | ν(C-C) | 1790/1315 | cyclopropenium |
| **C₃Cl₃⁺** | ν(C-Cl) | 733/458 | |
| **CF₃** | ν(C-F) | 1277-1024 | |
| **CF₃** | δₛ | 750-739 | |
| **CF₃** | δₐ | 546 | |
| **ClF₃** | ν(XY′) | 754/672 | T-shaped |
| **ClF₃** | ν(XY) B₁ | 683/597 | |
| **BrF₃** | ν(XY′) | 672/547 | T-shaped |
| **C₆₀** | IR | 1428, 1182, 577, 527 | 4 bands |
| **C₆₀** | Raman A_g | 1469, 496 | |
| **C₆₀** | Raman H_g | 1575, 1427, 1250, 1100, 774, 711, 436, 273 | 8 bands |
| **C₆₀** | doping shift | ~-7 cm⁻¹/e⁻ | A_g(2) |
| **C₇₀** | IR | ~31 predicted | D5h |
| **C₇₀** | Raman | ~53 predicted | D5h |
| **N_4 H_4 S_4** | (SNH)₄ ν(SN) | ~800-700 | C4v crown |
| **N_4 H_4 S_4** | ν(NH) | ~3250 | |

---

## [CHG] Charge and Oxidation State Effects

1. **Higher oxidation → higher frequency** (e.g., VF₆⁻ > VF₆²⁻ > VF₆³⁻)
2. **FeX₄⁻/FeX₄²⁻ ratio ≈ 1.32** for ν₃
3. **ν₁/ν₃ ratio in MO₄ⁿ⁻** increases with negative charge and central atom mass
4. **Bonding MO removal → ν decreases; Antibonding MO removal → ν increases**
5. **NH₄⁺ in halides:** ν decreases, δ increases as H-bond weakens (F > Cl > Br > I)

---

## [SUB] Substituent Effects

1. **Group IIIB hydride anomaly:** BH₄⁻(2270) > GaH₄⁻(1807) > AlH₄⁻(1757 cm⁻¹)
2. **Ratios for mixed halogens:** ν(MBr)/ν(MCl) ≈ 0.76 (ν₃), 0.61 (ν₁); ν(MI)/ν(MCl) ≈ 0.62 (ν₃), 0.42 (ν₁)
3. **Characteristic mixed-halogen frequencies:** e.g., SiF 910, SiCl 587, SiBr 486, SiI 333 cm⁻¹
4. **Terminal vs bridging:** terminal 1.3-1.6× higher than bridging; ratio 0.61-0.74
5. **M-O-M bridge angle correlation:** νₐ-νₛ separation increases with angle; Δ = -38.79 + 0.39α for V₂O₇
6. **M₂X₉ clusters:** terminal ~340-430, bridging ~200-320 cm⁻¹
7. **M₂X₁₀ clusters:** terminal ~320-370, bridging ~250-280 cm⁻¹
8. **XH₄⁺ series:** ν₃: NH₄⁺ 3145 > PH₄⁺ 2366 > AsH₄⁺ 2341 > SbH₄⁺ 2061 cm⁻¹

---

## [ISO] Isotope Effects

1. **H/D shift:** ν(HX)/ν(DX) ≈ √2 ≈ 1.414
2. **³⁵Cl/³⁷Cl pattern:** 5-peak pattern in MCl₄/MCl₆; shift ≈ 3.8 cm⁻¹ per mass unit for Cl
3. **¹⁰B/¹¹B (19.8%/80.2%):** characteristic splitting in boron compounds
4. **¹²C/¹³C scrambling:** number of isotopomer peaks determines cluster geometry
5. **²³⁵U/²³⁸U:** only 0.604 cm⁻¹ shift in UF₆ — requires cold-jet technique

---

## [ENV] Environmental Effects

1. **H-bonding:** ν(XH) decreases, δ(XH) increases
2. **Matrix effects:** Ne/Ar/Kr/N₂ matrices give different ν and can change structure
3. **High pressure:** increases ν via bond shortening; can induce phase transitions
4. **Solution vs solid:** bands may merge or split; ν₂/ν₄ often overlap in Raman
5. **Crystal site symmetry:** degenerate modes split in crystalline state
6. **Temperature-dependent isomer equilibria:** observable via frequency changes
7. **Adduct formation:** H-bonded vs anti-H-bonded HF complexes distinguished by ν(HF)
8. **Cation effect on anion structure:** small cations → symmetric; large → asymmetric
9. **Raman intensity anomaly in d⁶ octahedral:** I(ν₂) > I(ν₁) for Pt/Pd hexahalides
10. **Free internal rotation:** low barrier → dynamic group symmetry G₃₆ → selection rule violations

---

## [FRC] Force Constant Correlations

1. **U=O distance formula:** R(pm) = 9141(ν̃₃)⁻²/³ + 80.4
2. **Pt(II)/Pd(II) M-Cl:** [νₐ(M-Cl)]² = P/(R-1.6)³ where P=38988(Pt), 41440(Pd)
3. **X₄⁴⁻/Y₄ ratio:** ν(X₄⁴⁻)/ν(Y₄) ≈ 0.77 for isoelectronic pairs
4. **M₆O₁₉⁸⁻ force constants:** terminal:bridging:central ≈ 6:3:1

---

## [VIB] Metal-Ligand Vibrations

1. **M-M stretch in clusters:** ν(M-M) 100-400 cm⁻¹; strong Raman; overtones under resonance
2. **M-M excited state:** can be higher or lower than ground state depending on MO involved
3. **Metal carbonyls:** terminal ν(CO) ~2000-2130; bridging ~1800-1860 cm⁻¹
4. **Dioxygen ligands:** ν(O-O) identifies charge: O₂⁺(1876) > O₂(1580) > O₂⁻(1094) > O₂²⁻(791-736)
5. **Polymeric N₂ at high P:** ν(N-N) 840 cm⁻¹ vs molecular 2400 cm⁻¹
6. **KrF₂ upon coordination:** ν shifts indicate covalent bonding

---

## [CRY] Cluster Vibrations

1. **B₆X₆²⁻ (Oh):** B-B modes insensitive, B-X modes sensitive to halogen
2. **B₁₂X₁₂²⁻ (Ih):** only 3 IR + 6 Raman out of 66 modes (highest symmetry)
3. **Cₙ clusters:** neutral = linear (n=3-13, except C₆); anions = markedly lower ν
4. **Sulfur clusters Sₙ:** neutral = puckered rings; Sₙ²⁻ = open chains
5. **[M₆X₈Y₆]²⁻ (Oh):** ν(M-M) ~200-300 cm⁻¹, force constant 1.2-1.5 mdyn/Å
6. **Nb₆Cl₁₂:** ν(Nb-Nb) ~140 cm⁻¹, force constant < 0.3 mdyn/Å

---

## [SPC] Special Topics

1. **Inversion doubling in NH₃:** each level splits (+/-), IR between different signs, Raman between same
2. **Fermi resonance:** fundamental + overtone mix when similar frequency/symmetry
3. **Resonance Raman:** overtones (nν₁ to n=9) when excitation matches electronic absorption
4. **Jahn-Teller in d¹/d² hexafluorides:** combination bands broaden anomalously
5. **Anomalous polarization in d⁵ (IrCl₆²⁻):** all bands show anomalous depolarization
6. **Electronic structure effect in MCl₆³⁻:** t₂g³e_g⁰ > t₂g³e_g¹ (J-T) > t₂g³e_g² (low ν₃)
7. **Group frequency charts:** see Nakamoto Appendix VIII
8. **Diamond:** 1332 cm⁻¹ Raman (pressure sensor); IR defects ~2000, ~1200 cm⁻¹
9. **Graphite:** Raman 1575 cm⁻¹; no IR bands
10. **¹³C isotopomers in C₆₀:** A_g(2) splits into 3 peaks (1471, 1470, 1469 cm⁻¹)
11. **Simultaneous transitions:** combination bands from different molecules → complex formation

---

## Usage Guide for LLM Agents

1. **Search by tag:** `[SEL]` for selection rules, `[GEO]` for structure determination, `[GFQ]` for group frequencies
2. **Forward inference (spectrum → structure):** Start with `[SEL]` to determine symmetry, then `[GEO]` for specific geometry, then `[GFQ]` for functional groups
3. **Reverse inference (structure → spectrum):** Start with `[GFQ]` for expected frequencies, then `[SEL]` for activity, then `[FRQ]`/`[CHG]` for trends
4. **Band assignment:** Use `[ISO]` for isotope shifts, `[SUB]` for substituent comparisons, `[ENV]` for environmental corrections
5. **Advanced interpretation:** Use `[SPC]` for Fermi resonance, J-T effect, RR phenomena
6. **For coordination compounds:** Prioritize `[VIB]` and `[SUB]` (terminal vs bridging, ligand type)
7. **For solid-state materials:** Prioritize `[CRY]` and `[FRC]`
