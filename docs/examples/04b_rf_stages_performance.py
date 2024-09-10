# # Modelling RF Photonic-Electronic Networks
#
# The goal of this notebook is to:
# - Demonstrate functionality of modelling performance parameters of a RF electronic network in the time and frequency domain.
# - Demonstrate some functionality of how the `piel` structure can streamline this analysis, alongside some default measurement.
#
# This enables us to determine how to test a corresponding device in terms of the required test equipment specification and also provide a reference range when comparing measurement results. This can then be understood in the context of an RF-photonic system of what is the expected optical performance of a photonic network accordingly.
#
# ## System to Model
#
# TODO make diagram of what we're simulating here.
#
# One important aspect of photonic-electronic codesign is synchronizing the time in between an electronic signal being generated from a corresponding photonic signal.s Let's say we have a system, where we can split a laser signal in two classically. One of the laser pulses is photo-detected and converted into an electronic pulse, whilst the other laser pulse continues to travel alongside a fibre. By the time the photo-detected electonic signal reaches a given logic level, how far away is the laser pulse that generated it? We would like a nice simulation of this type of circumstances.
#
# ## Constructing our Components Models
#
# First, we will understand how to create, or extract realistic electronic measurement for components that we might use in practice in our system. Let's get some reference files to use:
#
# **Relevant Data References for this notebook:**
# 1. Siew, Shawn Yohanes, et al. "Review of silicon photonics technology and platform development." Journal of Lightwave Technology 39.13 (2021): 4374-4389.
#
# Note that the raw files extracted from the papers can be accessed in the `docs/examples/reference_data` directory for each corresponding paper - as the files is publicly available online.

import piel
import pandas as pd

# #### Understanding Silicon Photodetector Performance

# Let's start by quantifying the performance of a silicon photodetector in RF terms. Let's extract the published files we will analyze:

foundry_file = piel.return_path(
    "./reference_data/review_silicon_photonics_technology/table_1.json"
)
pd.read_json(foundry_file)

mzm_file = piel.return_path(
    "./reference_data/review_silicon_photonics_technology/table_7.json"
)
pd.read_json(mzm_file)

detector_file = piel.return_path(
    "./reference_data/review_silicon_photonics_technology/table_7.json"
)
pd.read_json(detector_file)
