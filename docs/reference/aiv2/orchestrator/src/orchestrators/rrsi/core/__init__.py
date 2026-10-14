"""Method core of RRSI, vendored from https://github.com/google-research/rrsi
(Apache License 2.0, see ../LICENSE).

schedule, components, history, selection, calibrate and evaluate are unmodified;
config drops the per-role model names and adds ``RRSIConfig.from_mapping``.
"""
