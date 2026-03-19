import onnx
lgmodel = onnx.load("model/superpoint_lightglue_fixed_overwrite.onnx")
print("LightGlue model input info")
for input in lgmodel.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])


spmodel = onnx.load("model/superpoint_fixed_480x640.onnx")

print("\nLightGlue model output info")
for output in lgmodel.graph.output:
    print(output.name, [dim.dim_value for dim in output.type.tensor_type.shape.dim])



print("\nSuperPoint model input info")
for input in spmodel.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])

# --- Add this part ---
print("\nSuperPoint model output info")
for output in spmodel.graph.output:
    print(output.name, [dim.dim_value for dim in output.type.tensor_type.shape.dim])
# --- ---