# Arsenic K-edge XANES sample data

A subset of the `example/arsenic` data from [MrFitty](https://github.com/jklynch/mr-fitty),
MIT License, Copyright (c) 2015 Joshua Lynch, Sarah Nicholas.

- `unknown/OTT3_55_spot0.e`: one measured spectrum.
- `reference/`: ten reference spectra. They include the three in MrFitty's reported best fit of
  `OTT3_55_spot0` (Arsenopyrite_Julcani, orpiment, arsenate sorbed on diopside) and the pool's
  most nearly identical pair (As2O3 and sodium arsenite 13BM).

Files are Athena exports: `#` header lines, then columns `energy norm bkg_norm der_norm`.
The tutorial uses `energy` (eV) and `norm` (normalized absorption).
