import piel

test_s_parameters = sp = {
    ("in_o_0", "in_o_0"): 0j,
    ("out_o_0", "out_o_0"): 0j,
    ("out_o_0", "out_o_1"): 0j,
    ("out_o_1", "out_o_0"): 0j,
    ("out_o_1", "out_o_1"): 0j,
    ("out_o_2", "out_o_2"): 0j,
    ("out_o_2", "out_o_0"): 0j,
    ("out_o_2", "out_o_1"): 0j,
    ("out_o_0", "out_o_2"): 0j,
    ("out_o_1", "out_o_2"): 0j,
    ("out_o_3", "out_o_3"): 0j,
    ("out_o_3", "out_o_2"): 0j,
    ("out_o_3", "out_o_0"): 0j,
    ("out_o_3", "out_o_1"): 0j,
    ("out_o_2", "out_o_3"): 0j,
    ("out_o_0", "out_o_3"): 0j,
    ("out_o_1", "out_o_3"): 0j,
    ("in_o_1", "in_o_1"): 0j,
    ("in_o_1", "in_o_0"): 0j,
    ("in_o_0", "in_o_1"): 0j,
    ("in_o_2", "in_o_2"): 0j,
    ("in_o_2", "in_o_3"): 0j,
    ("in_o_3", "in_o_2"): 0j,
    ("in_o_3", "in_o_3"): 0j,
    ("out_o_3", "in_o_1"): (0.2716484597569184 + 0.02196099222952891j),
    ("out_o_3", "in_o_0"): (-0.0794665269569963 - 0.8649883105768688j),
    ("out_o_2", "in_o_1"): (0.007822711346586048 + 0.08514973701412346j),
    ("out_o_2", "in_o_0"): (0.2631000097841309 - 0.0710883655713215j),
    ("out_o_0", "in_o_1"): (-0.27779553196759077 + 0.08816013242362411j),
    ("out_o_0", "in_o_0"): (0.24680707373523947 - 0.15921314209229298j),
    ("in_o_1", "out_o_3"): (0.2716484597569184 + 0.02196099222952891j),
    ("in_o_1", "out_o_2"): (0.007822711346586048 + 0.08514973701412346j),
    ("in_o_1", "out_o_0"): (-0.27779553196759077 + 0.08816013242362411j),
    ("in_o_1", "out_o_1"): (-0.17334637253982665 + 0.8963378967793234j),
    ("in_o_0", "out_o_3"): (-0.0794665269569963 - 0.8649883105768688j),
    ("in_o_0", "out_o_2"): (0.2631000097841309 - 0.0710883655713215j),
    ("in_o_0", "out_o_0"): (0.24680707373523947 - 0.15921314209229298j),
    ("in_o_0", "out_o_1"): (-0.2777955319675908 + 0.08816013242362407j),
    ("out_o_1", "in_o_1"): (-0.17334637253982665 + 0.8963378967793234j),
    ("out_o_1", "in_o_0"): (-0.2777955319675908 + 0.08816013242362407j),
    ("out_o_3", "in_o_3"): (0.31510861955012587 - 0.05641407084995648j),
    ("out_o_3", "in_o_2"): (-0.26202477498796306 - 0.008494149670075501j),
    ("in_o_3", "out_o_3"): (0.31510861955012587 - 0.05641407084995648j),
    ("in_o_3", "out_o_2"): (-0.2620247749879632 - 0.008494149670075501j),
    ("in_o_3", "out_o_0"): (-0.07946652695699663 - 0.8649883105768683j),
    ("in_o_3", "in_o_1"): 0j,
    ("in_o_3", "in_o_0"): 0j,
    ("in_o_3", "out_o_1"): (0.27164845975691837 + 0.021960992229528853j),
    ("out_o_2", "in_o_3"): (-0.2620247749879632 - 0.008494149670075501j),
    ("out_o_2", "in_o_2"): (-0.17250486028103565 + 0.9054977065122333j),
    ("out_o_0", "in_o_3"): (-0.07946652695699663 - 0.8649883105768683j),
    ("out_o_0", "in_o_2"): (0.2631000097841307 - 0.07108836557132153j),
    ("in_o_1", "in_o_3"): 0j,
    ("in_o_1", "in_o_2"): 0j,
    ("in_o_0", "in_o_3"): 0j,
    ("in_o_0", "in_o_2"): 0j,
    ("in_o_2", "out_o_3"): (-0.26202477498796306 - 0.008494149670075501j),
    ("in_o_2", "out_o_2"): (-0.17250486028103565 + 0.9054977065122333j),
    ("in_o_2", "out_o_0"): (0.2631000097841307 - 0.07108836557132153j),
    ("in_o_2", "in_o_1"): 0j,
    ("in_o_2", "in_o_0"): 0j,
    ("in_o_2", "out_o_1"): (0.007822711346586061 + 0.08514973701412341j),
    ("out_o_1", "in_o_3"): (0.27164845975691837 + 0.021960992229528853j),
    ("out_o_1", "in_o_2"): (0.007822711346586061 + 0.08514973701412341j),
}


def test_import_functions():
    sax_s_parameters_to_matrix_exists = "sax_s_parameters_to_matrix" in dir(piel.sax)
    return sax_s_parameters_to_matrix_exists


def test_sax_s_parameters_to_matrix():
    matrix = piel.sax_to_s_parameters_standard_matrix(sax_input=test_s_parameters)
    return matrix


if __name__ == "__main__":
    test_import_functions()
    test_sax_s_parameters_to_matrix()