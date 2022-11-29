import bioviz

model_path = "wu_converted_definitif.bioMod"
b = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=True)
b.exec()
