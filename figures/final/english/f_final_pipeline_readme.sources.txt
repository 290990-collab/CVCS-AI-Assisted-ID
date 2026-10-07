# f_final_pipeline_readme — figura del report
data:     2026-10-07 13:17
comando:  python -m src.figures.pipeline --setup reset --layout readme --name f_final_pipeline_readme --svg --out-dir figures/final/english

numeri della figura:
  vision 768-d + graph 128-d → fusione 896-d
  α = 0.4 (da results/final_pipeline/fusion/s42/select_valid.json)
  gallery condivisa: 67405 piante

file letti:
  results/queryvec/valid/vision_pespatial_gem_whiten-train_partial-nowalls-random-f0.5_valid.npz
  results/final_pipeline/queryvec/valid/s42/graph_gat_rg_comb_s42_partial-random-f0.5_valid.npz
  results/shared_gallery.json
  results/final_pipeline/fusion/s42/select_valid.json
