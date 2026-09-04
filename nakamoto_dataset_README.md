# Nakamoto Empirical Rules Dataset

A structured knowledge base of 181 empirical rules extracted from **Kazuo Nakamoto**, *Infrared and Raman Spectra of Inorganic and Coordination Compounds*, 6th Edition (2009), Part A, Chapter 2.

## Files

| File | Format | Size | Purpose |
|------|--------|------|---------|
| `nakamoto_empirical_rules_dataset.json` | JSON | ~28KB | Machine-readable, structured for function calling / API retrieval |
| `nakamoto_empirical_rules_markdown.md` | Markdown | ~20KB | Human-readable, chunked for semantic search / RAG |

## How to Use for Agentic Retrieval

### JSON Mode (Function Calling)
Each rule is a JSON object with these keys:
- `id` — Unique tag (e.g., `SEL-001`, `GEO-002`)
- `category` — One of 12 categories
- `rule` — The empirical rule text
- `molecular_systems` — Which molecules/compounds it applies to
- `spectral_details` — Numerical frequencies, ratios, or spectral patterns
- `confidence` — `definitive` | `high` | `experimentally_verified`
- `chapter_section` — Nakamoto section reference
- `notes` (optional) — Caveats, exceptions, or context

### Markdown Mode (RAG / Semantic Search)
Rules are organized under category headings with consistent `**Rule:**`, `**Systems:**`, `**Confidence:**` fields. Search by:
- `[TAG]` prefixes for category filtering
- Rule text for similarity search
- Molecular formula/formula type for specific compound queries

## Category Overview

| Tag | Category | Count | Purpose |
|-----|----------|-------|---------|
| SEL | Selection Rules | 10 | IR vs Raman activity by symmetry |
| FRQ | Frequency Trends | 15 | Predict relative frequencies |
| GEO | Geometry Determination | 16 | Deduce structure from spectra |
| GFQ | Group Frequencies | 85 | **Comprehensive** spectral ranges for all bond types |
| CHG | Charge & Oxidation Effects | 5 | Oxidation state → frequency shifts |
| SUB | Substituent Effects | 8 | Atom/group substitution effects |
| ISO | Isotope Effects | 5 | Isotopic substitution for assignment |
| ENV | Environmental Effects | 10 | Matrix, H-bonding, pressure effects |
| FRC | Force Constant Correlations | 4 | Frequency ↔ bond strength relations |
| VIB | Metal-Ligand Vibrations | 6 | Coordination compound diagnostics |
| CRY | Cluster Vibrations | 6 | Framework structure-spectra |
| SPC | Special Topics | 11 | Fermi res., J-T, RR, diamond/graphite |

## Reasoning Patterns

**Forward (spectrum → structure):** SEL → selection rules → symmetry → GEO → geometry → GFQ → functional groups

**Reverse (structure → spectrum):** GFQ → expected frequencies → SEL → activity → FRQ/CHG → trends

**Band assignment:** ISO → isotope shifts → SUB → substituent → ENV → environmental corrections

**Advanced:** SPC for Fermi resonance, Jahn-Teller, resonance Raman effects

## Source Text

Extracted from pages 149-298 of Nakamoto Part A (Chapter 2: "Applications in Inorganic Chemistry"). The source PDF OCR text is saved at `nakamoto_extracted_text.txt` (11,612 lines, 151 pages).
