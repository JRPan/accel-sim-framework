import accel_sim
import onnx

# Load the ONNX model
model = onnx.load("/home/pan251/accel-sim-framework-pim/resnet50.onnx")
# model = onnx.load("model.onnx")
accel_sim_instance = accel_sim.accel_sim_framework('../gpgpusim.config', '../empty.g')

accel_sim_instance.bind_onnx_model(model.graph.SerializeToString())

accel_sim_instance.simulation_loop()

print("GPGPU-Sim: *** simulation thread exiting ***")
print("GPGPU-Sim: *** exit detected ***")
