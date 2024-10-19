// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include "accel-sim.h"

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary
 * format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

int main(int argc, const char **argv) {
  accel_sim_framework accel_sim(argc, argv);

  // std::ifstream file("/home/pan251/accel-sim-framework-pim/resnet50.onnx", std::ios::binary);
  std::ifstream file("/home/pan251/accel-sim-framework-pim/bert-base-uncased.onnx", std::ios::binary);

  if (!file) {
    std::cerr << "Failed to open file for reading." << std::endl;
    return 1;
  }

  onnx::ModelProto msg;

  std::string serialized((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());


  accel_sim.bind_onnx_model(serialized);

  accel_sim.simulation_loop();

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}