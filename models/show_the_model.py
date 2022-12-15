import bioviz

model_path = "wu_converted_definitif_without_floating_base.bioMod"
b = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=True)
b.exec()
