import onnx
lgmodel = onnx.load("model/superpoint_lightglue_fixed_overwrite.onnx")
print("LightGlue 模型的输入信息:")
for input in lgmodel.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])


spmodel = onnx.load("model/superpoint_fixed_480x640.onnx")

print("\nLightGlue 模型的输出信息:")
for output in lgmodel.graph.output:
    print(output.name, [dim.dim_value for dim in output.type.tensor_type.shape.dim])



print("\nSuperPoint 模型的输入信息:")
for input in spmodel.graph.input:
    print(input.name, [dim.dim_value for dim in input.type.tensor_type.shape.dim])

# --- 添加这部分 ---
print("\nSuperPoint 模型的输出信息:")
for output in spmodel.graph.output:
    print(output.name, [dim.dim_value for dim in output.type.tensor_type.shape.dim])
# --- ---