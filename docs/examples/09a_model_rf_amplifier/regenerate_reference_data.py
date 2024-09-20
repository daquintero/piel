# + endofcell="--"
import piel
from piel.types.electronic.lna import LNAMetrics
from piel.types.electronic.lna import LowNoiseTwoPortAmplifier as LNA
from piel.types import Reference, ScalarMetrics
from piel.types.units import Hz, dB, mW, V, nm, mm2
from typing import Optional

# Define Reference Instances
ref_chen2007 = Reference(
    text="Chen et al. (2007)",
    bibtex_id="chen2007ultra",
    bibtex="""
    @article{chen2007ultra,
       title={An Ultra-Wide-Band 0.4--10-GHz LNA in 0.18-um CMOS},
       author={Chen, Ke-Hou and Lu, Jian-Hao and Chen, Bo-Jiun and Liu, Shen-Iuan},
       journal={IEEE Transactions on Circuits and Systems II: Express Briefs},
       volume={54},
       number={3},
       pages={217--221},
       year={2007},
       publisher={IEEE}
    }
    """,
)

ref_liu20030 = Reference(
    text="Liu et al. (2003)",
    bibtex_id="liu20030",
    bibtex="""
    @inproceedings{liu20030,
       title={A 0.5-14-GHz 10.6-dB CMOS cascode distributed amplifier},
       author={Liu, Ren-Chieh and Lin, Chin-Shen and Deng, Kuo-Liang and Wang, Huei},
       booktitle={2003 Symposium on VLSI Circuits. Digest of Technical Papers (IEEE Cat. No. 03CH37408)},
       pages={139--140},
       year={2003},
       organization={IEEE}
    }
    """,
)

ref_parvizi2014 = Reference(
    text="Parvizi et al. (2014)",
    bibtex_id="parvizi2014sub",
    bibtex="""
    @article{parvizi2014sub,
       title={A sub-mW, ultra-low-voltage, wideband low-noise amplifier design technique},
       author={Parvizi, Mahdi and Allidina, Karim and El-Gamal, Mourad N},
       journal={IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
       volume={23},
       number={6},
       pages={1111--1122},
       year={2014},
       publisher={IEEE}
    }
    """,
)

ref_asgaran2006 = Reference(
    text="Asgaran et al. (2006)",
    bibtex_id="asgaran20064",
    bibtex="""
    @article{asgaran20064,
       title={A 4-mW monolithic CMOS LNA at 5.7 GHz with the gate resistance used for input matching},
       author={Asgaran, Saman and Deen, M Jamal and Chen, Chih-Hung},
       journal={IEEE Microwave and Wireless Components Letters},
       volume={16},
       number={4},
       pages={188--190},
       year={2006},
       publisher={IEEE}
    }
    """,
)

# Create ScalarMetrics Instances

# Chen et al. (2007) Metrics
chen_bandwidth = ScalarMetrics(
    min=0.4 * 1e9,  # 0.4 GHz
    max=10 * 1e9,  # 10 GHz
    unit=Hz,
)

chen_noise_figure = ScalarMetrics(min=4.4, max=6.5, unit=dB)

chen_power_gain = ScalarMetrics(min=11.2, max=12.4, unit=dB)

chen_power_consumption = ScalarMetrics(value=12, min=12, max=12, unit=mW)

chen_supply_voltage = ScalarMetrics(value=1.8, min=1.8, max=1.8, unit=V)

chen_technology = ScalarMetrics(value=180, min=180, max=180, unit=nm)

chen_footprint = ScalarMetrics(value=0.42, min=0.42, max=0.42, unit=mm2)

# Liu et al. (2003) Metrics
liu_bandwidth = ScalarMetrics(
    min=0.5 * 1e9,  # 0.5 GHz
    max=14 * 1e9,  # 14 GHz
    unit=Hz,
)

liu_noise_figure = ScalarMetrics(min=3.2, max=5.4, unit=dB)

liu_power_gain = ScalarMetrics(value=10.6, min=10.6, max=10.6, unit=dB)

liu_power_consumption = ScalarMetrics(value=52, min=52, max=52, unit=mW)

liu_supply_voltage = ScalarMetrics(value=1.3, min=1.3, max=1.3, unit=V)

liu_technology = ScalarMetrics(value=180, min=180, max=180, unit=nm)

# Footprint "1.0 x 1.6" mm² is 1.6 mm²
liu_footprint = ScalarMetrics(value=1.0 * 1.6, min=1.6, max=1.6, unit=mm2)

# Parvizi et al. (2014) Metrics
parvizi_bandwidth = ScalarMetrics(
    min=0.1 * 1e9,  # 0.1 GHz
    max=7 * 1e9,  # 7 GHz
    unit=Hz,
)

parvizi_noise_figure = ScalarMetrics(value=5.5, min=5.5, max=5.5, unit=dB)

parvizi_power_gain = ScalarMetrics(value=12.6, min=12.6, max=12.6, unit=dB)

parvizi_power_consumption = ScalarMetrics(value=0.75, min=0.75, max=0.75, unit=mW)

parvizi_supply_voltage = ScalarMetrics(value=0.5, min=0.5, max=0.5, unit=V)

parvizi_technology = ScalarMetrics(value=90, min=90, max=90, unit=nm)

parvizi_footprint = ScalarMetrics(value=0.23, min=0.23, max=0.23, unit=mm2)

# Asgaran et al. (2006) Metrics
asgaran_bandwidth = ScalarMetrics(
    value=5.7 * 1e9,  # 5.7 GHz
    min=5.7 * 1e9,
    max=5.7 * 1e9,
    unit=Hz,
)

asgaran_noise_figure = ScalarMetrics(value=3.4, min=3.4, max=3.4, unit=dB)

asgaran_power_gain = ScalarMetrics(value=11.45, min=11.45, max=11.45, unit=dB)

asgaran_power_consumption = ScalarMetrics(value=4, min=4, max=4, unit=mW)

asgaran_supply_voltage = ScalarMetrics(value=0.5, min=0.5, max=0.5, unit=V)

asgaran_technology = ScalarMetrics(value=180, min=180, max=180, unit=nm)

# Footprint "0.950 x 0.900" mm² is 0.855 mm²
asgaran_footprint = ScalarMetrics(value=0.950 * 0.900, min=0.855, max=0.855, unit=mm2)

# Instantiate LNAMetrics Objects

# Chen et al. (2007) LNAMetrics
lna_chen2007 = LNA(
    metrics=[
        LNAMetrics(
            name="Chen2007Ultra",
            reference=ref_chen2007,
            bandwidth_Hz=chen_bandwidth,
            noise_figure=chen_noise_figure,
            power_consumption_mW=chen_power_consumption,
            power_gain_dB=chen_power_gain,
            supply_voltage_V=chen_supply_voltage,
            technology_nm=chen_technology,
            footprint_mm2=chen_footprint,
        )
    ]
)

# Liu et al. (2003) LNAMetrics
lna_liu20030 = LNA(
    metrics=[
        LNAMetrics(
            name="Liu20030",
            reference=ref_liu20030,
            bandwidth_Hz=liu_bandwidth,
            noise_figure=liu_noise_figure,
            power_consumption_mW=liu_power_consumption,
            power_gain_dB=liu_power_gain,
            supply_voltage_V=liu_supply_voltage,
            technology_nm=liu_technology,
            footprint_mm2=liu_footprint,
        )
    ]
)

# Parvizi et al. (2014) LNAMetrics
lna_parvizi2014 = LNA(
    metrics=[
        LNAMetrics(
            name="Parvizi2014Sub",
            reference=ref_parvizi2014,
            bandwidth_Hz=parvizi_bandwidth,
            noise_figure=parvizi_noise_figure,
            power_consumption_mW=parvizi_power_consumption,
            power_gain_dB=parvizi_power_gain,
            supply_voltage_V=parvizi_supply_voltage,
            technology_nm=parvizi_technology,
            footprint_mm2=parvizi_footprint,
        )
    ]
)

# Asgaran et al. (2006) LNAMetrics
lna_asgaran2006 = LNA(
    metrics=[
        LNAMetrics(
            name="Asgaran20064",
            reference=ref_asgaran2006,
            bandwidth_Hz=asgaran_bandwidth,
            noise_figure=asgaran_noise_figure,
            power_consumption_mW=asgaran_power_consumption,
            power_gain_dB=asgaran_power_gain,
            supply_voltage_V=asgaran_supply_voltage,
            technology_nm=asgaran_technology,
            footprint_mm2=asgaran_footprint,
        )
    ]
)

# Aggregate all LNAMetrics instances into a list
lna_metrics_list = [lna_chen2007, lna_liu20030, lna_parvizi2014, lna_asgaran2006]


# -

piel.write_model_to_json(
    piel.types.RFAmplifierCollection(components=lna_metrics_list),
    file_path="data/lna_180nm_metrics.json",
)
# --
