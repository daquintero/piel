import numpy as np
import piel

test_data = {"t": np.array([3000, 4000, 5000, 6000]), "x": np.array([2, 3, 4, 5])}

print(test_data)

piel.visual.append_row_to_dict(data=test_data, copy_index=1, set_value={"t": 10})
print(test_data)

"""
Valid output
{'t': array([3000, 4000, 5000, 6000]), 'x': array([2, 3, 4, 5])}
{'t': array([3000, 4000, 5000, 6000,   10]), 'x': array([2, 3, 4, 5, 3])}
"""

"""
Now we test that it creates a copy of the data with corresponding changed data points.
"""

test_data = {"t": np.array([3000, 4000, 5000, 6000]), "x": np.array([2, 3, 4, 5])}

out_data = piel.visual.points_to_lines_fixed_transient(
    data=test_data,
    time_index_name="t",
    fixed_transient_time=1,
)

print(test_data)
print(out_data)

"""
Valid output
{'t': array([3000, 4000, 5000, 6000]), 'x': array([2, 3, 4, 5])}
{'t': array([3000, 4000, 5000, 6000, 3999, 4999, 5999, 3998]), 'x': array([2, 3, 4, 5, 2, 3, 4, 5])}
"""

"""
Let's assume a different data format:
"""

test_data = {
    "Unnamed: 0": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    "a": {
        0: 101,
        1: 1001,
        2: 0,
        3: 100,
        4: 101,
        5: 11,
        6: 101,
        7: 1000,
        8: 1101,
        9: 1001,
        10: 1011,
    },
    "b": {
        0: 1010,
        1: 1001,
        2: 1011,
        3: 101,
        4: 0,
        5: 0,
        6: 1011,
        7: 101,
        8: 100,
        9: 11,
        10: 1111,
    },
    "x": {
        0: 1111,
        1: 10010,
        2: 1011,
        3: 1001,
        4: 101,
        5: 11,
        6: 10000,
        7: 1101,
        8: 10001,
        9: 1100,
        10: 11010,
    },
    "t": {
        0: 2001,
        1: 4001,
        2: 6001,
        3: 8001,
        4: 10001,
        5: 12001,
        6: 14001,
        7: 16001,
        8: 18001,
        9: 20001,
        10: 22001,
    },
    "phase": {
        0: 1.5201104775434482,
        1: 1.8241325730521378,
        2: 1.1147476835318622,
        3: 0.9120662865260689,
        4: 0.5067034925144828,
        5: 0.30402209550868964,
        6: 1.6214511760463448,
        7: 1.3174290805376552,
        8: 1.7227918745492414,
        9: 1.2160883820347586,
        10: 2.6348581610753103,
    },
    "unitary": {
        0: (
            np.array(
                [
                    [0.33489325 - 0.83300986j, -0.16426986 + 0.4086031j],
                    [0.16426986 - 0.4086031j, 0.33489325 - 0.83300986j],
                ]
            ),
            ("o2", "o1"),
        ),
        1: (
            np.array(
                [
                    [0.41794202 - 0.70638908j, -0.29089065 + 0.49165187j],
                    [0.29089065 - 0.49165187j, 0.41794202 - 0.70638908j],
                ]
            ),
            ("o2", "o1"),
        ),
        2: (
            np.array(
                [
                    [0.17290701 - 0.95251202j, -0.04476771 + 0.24661686j],
                    [0.04476771 - 0.24661686j, 0.17290701 - 0.95251202j],
                ]
            ),
            ("o2", "o1"),
        ),
        3: (
            np.array(
                [
                    [0.07725035 - 0.98544577j, -0.01183396 + 0.1509602j],
                    [0.01183396 - 0.1509602j, 0.07725035 - 0.98544577j],
                ]
            ),
            ("o2", "o1"),
        ),
        4: (
            np.array(
                [
                    [-0.12396978 - 0.99099238j, -0.00628735 - 0.05025993j],
                    [0.00628735 + 0.05025993j, -0.12396978 - 0.99099238j],
                ]
            ),
            ("o2", "o1"),
        ),
        5: (
            np.array(
                [
                    [-0.22129543 - 0.96337818j, -0.03390155 - 0.14758559j],
                    [0.03390155 + 0.14758559j, -0.22129543 - 0.96337818j],
                ]
            ),
            ("o2", "o1"),
        ),
        6: (
            np.array(
                [
                    [0.36681329 - 0.79368558j, -0.20359414 + 0.44052313j],
                    [0.20359414 - 0.44052313j, 0.36681329 - 0.79368558j],
                ]
            ),
            ("o2", "o1"),
        ),
        7: (
            np.array(
                [
                    [0.25997616 - 0.90099705j, -0.09628268 + 0.33368601j],
                    [0.09628268 - 0.33368601j, 0.25997616 - 0.90099705j],
                ]
            ),
            ("o2", "o1"),
        ),
        8: (
            np.array(
                [
                    [0.39459122 - 0.7513338j, -0.24594593 + 0.46830107j],
                    [0.24594593 - 0.46830107j, 0.39459122 - 0.7513338j],
                ]
            ),
            ("o2", "o1"),
        ),
        9: (
            np.array(
                [
                    [0.21774784 - 0.92896234j, -0.06831739 + 0.29145769j],
                    [0.06831739 - 0.29145769j, 0.21774784 - 0.92896234j],
                ]
            ),
            ("o2", "o1"),
        ),
        10: (
            np.array(
                [
                    [0.42706175 - 0.31214236j, -0.68513737 + 0.5007716j],
                    [0.68513737 - 0.5007716j, 0.42706175 - 0.31214236j],
                ]
            ),
            ("o2", "o1"),
        ),
    },
    "output_amplitude_array_0": {
        0: (0.3348932484400226 - 0.8330098644113894j),
        1: (0.41794202495830884 - 0.7063890782969602j),
        2: (0.172907011379935 - 0.9525120170487468j),
        3: (0.07725035327753166 - 0.985445766256692j),
        4: (-0.12396977624117292 - 0.9909923830926474j),
        5: (-0.22129543393308604 - 0.963378176041775j),
        6: (0.36681328554358295 - 0.7936855849972456j),
        7: (0.2599761601249619 - 0.9009970540474179j),
        8: (0.3945912222496168 - 0.7513337969298994j),
        9: (0.21774784446017936 - 0.9289623374584851j),
        10: (0.42706175251351547 - 0.3121423638257475j),
    },
    "output_amplitude_array_1": {
        0: (0.16426986489554396 - 0.40860309522788557j),
        1: (0.2908906510099731 - 0.4916518717461718j),
        2: (0.04476771225818654 - 0.24661685816779796j),
        3: (0.01183396305024137 - 0.15096020006539462j),
        4: (0.006287346214285949 + 0.050259929453309954j),
        5: (0.03390155326515837 + 0.14758558714522307j),
        6: (0.2035941443096877 - 0.4405231323314459j),
        7: (0.0962826752595154 - 0.33368600691282485j),
        8: (0.24594593237703394 - 0.46830106903747976j),
        9: (0.0683173918484482 - 0.2914576912480423j),
        10: (0.6851373654811859 - 0.5007715993013784j),
    },
    "output_amplitude_array_0_abs": {
        0: 0.8978078425016078,
        1: 0.8207685825879792,
        2: 0.9680784974404777,
        3: 0.9884690057430947,
        4: 0.9987163805450128,
        5: 0.9884680971853417,
        6: 0.8743504985323095,
        7: 0.937754389611398,
        8: 0.8486487536581585,
        9: 0.9541410525616151,
        10: 0.5289750426576291,
    },
    "output_amplitude_array_0_phase_rad": {
        0: -1.1885429400423677,
        1: -1.036531892288082,
        2: -1.391224337048158,
        3: -1.4925650355510531,
        4: -1.6952464325568433,
        5: -1.7965871310597383,
        6: -1.1378725907909772,
        7: -1.289883638545263,
        8: -1.0872022415394729,
        9: -1.3405539877967674,
        10: -0.6311690982765017,
    },
    "output_amplitude_array_0_phase_deg": {
        0: -68.09849423449812,
        1: -59.38890275881594,
        2: -79.71128286874539,
        3: -85.51767718586902,
        4: -97.13046582011627,
        5: -102.9368601372399,
        6: -65.19529707593958,
        7: -73.90488855162177,
        8: -62.29209991737451,
        9: -76.80808571018683,
        10: -36.16332549032143,
    },
    "output_amplitude_array_1_abs": {
        0: 0.4403874180112424,
        1: 0.5712608282006741,
        2: 0.25064720783082256,
        3: 0.15142332939563324,
        4: 0.050651665629769665,
        5: 0.15142926020683917,
        6: 0.48529496774261166,
        7: 0.34729915744866857,
        8: 0.5289567968317025,
        9: 0.2993573981324223,
        10: 0.8486373808908964,
    },
    "output_amplitude_array_1_phase_rad": {
        0: -1.1885429400423682,
        1: -1.0365318922880824,
        2: -1.3912243370481585,
        3: -1.4925650355510538,
        4: 1.4463462210329538,
        5: 1.3450055225300555,
        6: -1.1378725907909775,
        7: -1.2898836385452632,
        8: -1.087202241539473,
        9: -1.3405539877967674,
        10: -0.6311690982765019,
    },
    "output_amplitude_array_1_phase_deg": {
        0: -68.09849423449815,
        1: -59.388902758815966,
        2: -79.71128286874541,
        3: -85.51767718586906,
        4: 82.86953417988396,
        5: 77.06313986276014,
        6: -65.19529707593959,
        7: -73.90488855162178,
        8: -62.29209991737452,
        9: -76.80808571018683,
        10: -36.16332549032144,
    },
}


print(test_data["t"])

piel.visual.points_to_lines_fixed_transient(
    data=test_data,
    time_index_name="t",
    fixed_transient_time=1,
)


print(test_data["t"])
