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

pim_layer *parse_pim_layer_info(const std::string &pimlayer_desc);

// int main(int argc, const char **argv) {
//   std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
//   gpgpu_context *m_gpgpu_context = new gpgpu_context();
//   trace_config tconfig;

//   gpgpu_sim *m_gpgpu_sim =
//       gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
//   m_gpgpu_sim->init();

//   trace_parser tracer(tconfig.get_traces_filename());

//   tconfig.parse_config();

//   // for each kernel
//   // load file
//   // parse and create kernel info
//   // launch
//   // while loop till the end of the end kernel execution
//   // prints stats
//   bool concurrent_kernel_sm =
//       m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
//   unsigned window_size =
//       concurrent_kernel_sm
//           ? m_gpgpu_sim->get_config().get_max_concurrent_kernel()
//           : 1;
//   assert(window_size > 0);
//   std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
//   std::vector<unsigned long> busy_streams;
//   std::vector<trace_kernel_info_t *> kernels_info;
//   std::vector<pim_layer *> pim_layers;
//   kernels_info.reserve(window_size);

//   unsigned i = 0;
//   while (i < commandlist.size() || !kernels_info.empty()) {
//     // gulp up as many commands as possible - either cpu_gpu_mem_copy
//     // or kernel_launch - until the vector "kernels_info" has reached
//     // the window_size or we have read every command from commandlist
//     while (kernels_info.size() < window_size && i < commandlist.size()) {
//       trace_kernel_info_t *kernel_info = NULL;
//       if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
//         size_t addre, Bcount;
//         tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
//         std::cout << "launching memcpy command : "
//                   << commandlist[i].command_string << std::endl;
//         m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
//         i++;
//       } else if (commandlist[i].m_type == command_type::kernel_launch) {
//         // Read trace header info for window_size number of kernels
//         kernel_trace_t *kernel_trace_info =
//             tracer.parse_kernel_info(commandlist[i].command_string);
//         kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
//                                          &tconfig, &tracer);
//         kernels_info.push_back(kernel_info);
//         std::cout << "Header info loaded for kernel command : "
//                   << commandlist[i].command_string << std::endl;
//         i++;
//       } else if (commandlist[i].m_type == command_type::pim_layer_launch) {
//         pim_layer *layer = parse_pim_layer_info(commandlist[i].command_string);
//         if (layer != NULL) {
//           pim_layers.push_back(layer);
//         }
//         i++;
//       }
//       else {
//         // unsupported commands will fail the simulation
//         assert(0 && "Undefined Command");
//       }
//     }

//     m_gpgpu_sim->launch_pim(pim_layers);

//     // Launch all kernels within window that are on a stream that isn't already
//     // running
//     for (auto k : kernels_info) {
//       bool stream_busy = false;
//       for (auto s : busy_streams) {
//         if (s == k->get_cuda_stream_id()) stream_busy = true;
//       }
//       if (!stream_busy && m_gpgpu_sim->can_start_kernel() &&
//           !k->was_launched()) {
//         std::cout << "launching kernel name: " << k->get_name()
//                   << " uid: " << k->get_uid() << std::endl;
//         m_gpgpu_sim->launch(k);
//         k->set_launched();
//         busy_streams.push_back(k->get_cuda_stream_id());
//       }
//     }

//     bool active = false;
//     bool sim_cycles = false;
//     unsigned finished_kernel_uid = 0;

//     do {
//       if (!m_gpgpu_sim->active()) break;

//       // performance simulation
//       if (m_gpgpu_sim->active()) {
//         m_gpgpu_sim->cycle();
//         sim_cycles = true;
//         m_gpgpu_sim->deadlock_check();
//       } else {
//         if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
//           m_gpgpu_context->the_gpgpusim->g_stream_manager
//               ->stop_all_running_kernels();
//           break;
//         }
//       }

//       active = m_gpgpu_sim->active();
//       finished_kernel_uid = m_gpgpu_sim->finished_kernel();
//     } while (active && !finished_kernel_uid);

//     // cleanup finished kernel
//     if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit() ||
//         !m_gpgpu_sim->active()) {
//       trace_kernel_info_t *k = NULL;
//       for (unsigned j = 0; j < kernels_info.size(); j++) {
//         k = kernels_info.at(j);
//         if (k->get_uid() == finished_kernel_uid ||
//             m_gpgpu_sim->cycle_insn_cta_max_hit() || !m_gpgpu_sim->active()) {
//           for (unsigned int l = 0; l < busy_streams.size(); l++) {
//             if (busy_streams.at(l) == k->get_cuda_stream_id()) {
//               busy_streams.erase(busy_streams.begin() + l);
//               break;
//             }
//           }
//           tracer.kernel_finalizer(k->get_trace_info());
//           delete k->entry();
//           delete k;
//           kernels_info.erase(kernels_info.begin() + j);
//           if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
//             break;
//         }
//       }
//       if (pim_layers.empty()) {
//         assert(k);
//       }
//       m_gpgpu_sim->print_stats();
//     }

//     if (sim_cycles) {
//       m_gpgpu_sim->update_stats();
//       m_gpgpu_context->print_simulation_time();
//     }

//     if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
//       printf(
//           "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
//           "instructions) **\n");
//       fflush(stdout);
//       break;
//     }
//   }
// }

int main(int argc, const char **argv) {
  accel_sim_framework accel_sim(argc, argv);
  accel_sim.simulation_loop();

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}