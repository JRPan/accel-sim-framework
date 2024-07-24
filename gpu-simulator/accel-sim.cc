#include "accel-sim.h"
#include "accelsim_version.h"

accel_sim_framework::accel_sim_framework(std::string config_file,
                                          std::string trace_file) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  // mimic argv
  unsigned argc = 5;
  const char *argv[] = {"accel-sim.out", "-config", config_file.c_str(),
                        "-trace", trace_file.c_str()};

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

accel_sim_framework::accel_sim_framework(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  m_gpgpu_context = new gpgpu_context();

  m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig);
  m_gpgpu_sim->init();

  tracer = trace_parser(tconfig.get_traces_filename());

  tconfig.parse_config();

  init();
}

void accel_sim_framework::simulation_loop() {
  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
    m_gpgpu_sim->pim_active = true;
    m_gpgpu_sim->launch_pim(pim_layers);

  while (commandlist_index < commandlist.size() || !kernels_info.empty() || m_gpgpu_sim->pim_active) {
    parse_commandlist();

    // Launch all kernels within window that are on a stream that isn't already
    // running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s : busy_streams) {
        if (s == k->get_cuda_stream_id()) stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() &&
          !k->was_launched()) {
        std::cout << "launching kernel name: " << k->get_name()
                  << " uid: " << k->get_uid() << std::endl;
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }

    unsigned finished_kernel_uid = simulate();
    // cleanup finished kernel
    if (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit() ||
        !m_gpgpu_sim->active()) {
      cleanup(finished_kernel_uid);
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }

    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf(
          "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
          "instructions) **\n");
      fflush(stdout);
      break;
    }
  }
}

void accel_sim_framework::parse_commandlist() {
  // gulp up as many commands as possible - either cpu_gpu_mem_copy
  // or kernel_launch - until the vector "kernels_info" has reached
  // the window_size or we have read every command from commandlist
  while (kernels_info.size() < window_size && commandlist_index < commandlist.size()) {
    trace_kernel_info_t *kernel_info = NULL;
    if (commandlist[commandlist_index].m_type == command_type::cpu_gpu_mem_copy) {
      size_t addre, Bcount;
      tracer.parse_memcpy_info(commandlist[commandlist_index].command_string, addre, Bcount);
      std::cout << "launching memcpy command : "
                << commandlist[commandlist_index].command_string << std::endl;
      m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
      commandlist_index++;
    } else if (commandlist[commandlist_index].m_type == command_type::kernel_launch) {
      // Read trace header info for window_size number of kernels
      kernel_trace_t *kernel_trace_info =
          tracer.parse_kernel_info(commandlist[commandlist_index].command_string);
      kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context,
                                       &tconfig, &tracer);
      kernels_info.push_back(kernel_info);
      std::cout << "Header info loaded for kernel command : "
                << commandlist[commandlist_index].command_string << std::endl;
      commandlist_index++;
    } else {
      // unsupported commands will fail the simulation
      assert(0 && "Undefined Command");
    }
  }
}

void accel_sim_framework::cleanup(unsigned finished_kernel) {
  trace_kernel_info_t *k = NULL;
  for (unsigned j = 0; j < kernels_info.size(); j++) {
    k = kernels_info.at(j);
    if (k->get_uid() == finished_kernel ||
        m_gpgpu_sim->cycle_insn_cta_max_hit() || !m_gpgpu_sim->active()) {
      for (unsigned int l = 0; l < busy_streams.size(); l++) {
        if (busy_streams.at(l) == k->get_cuda_stream_id()) {
          busy_streams.erase(busy_streams.begin() + l);
          break;
        }
      }
      tracer.kernel_finalizer(k->get_trace_info());
      delete k->entry();
      delete k;
      kernels_info.erase(kernels_info.begin() + j);
      if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
        break;
    }
  }
  if (pim_layers.empty()) {
    assert(k);
  }
  m_gpgpu_sim->print_stats();
}

unsigned accel_sim_framework::simulate() {
  unsigned finished_kernel_uid = 0;
  do {
    if (!m_gpgpu_sim->active()) break;

    // performance simulation
    if (m_gpgpu_sim->active()) {
      m_gpgpu_sim->cycle();
      sim_cycles = true;
      m_gpgpu_sim->deadlock_check();
    } else {
      if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
        m_gpgpu_context->the_gpgpusim->g_stream_manager
            ->stop_all_running_kernels();
        break;
      }
    }

    active = m_gpgpu_sim->active();
    finished_kernel_uid = m_gpgpu_sim->finished_kernel();
  } while (active && !finished_kernel_uid);
  return finished_kernel_uid;
}

trace_kernel_info_t *accel_sim_framework::create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        gpgpu_context *m_gpgpu_context,
                                        trace_config *config,
                                        trace_parser *parser) {
  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y,
               kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y,
                kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info = new trace_kernel_info_t(
      gridDim, blockDim, function_info, parser, config, kernel_trace_info);

  return kernel_info;
}

gpgpu_sim *accel_sim_framework::gpgpu_trace_sim_init_perf_model(
    int argc, const char *argv[], gpgpu_context *m_gpgpu_context,
    trace_config *m_config) {
  srand(1);
  print_splash();

  option_parser_t opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);
  pim_icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options
  m_config->reg_options(opp);

  option_parser_cmdline(opp, argc, argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}

void accel_sim_framework::bind_onnx_model(const std::string node_proto_string) {
  onnx::GraphProto graph;
  graph.ParseFromString(node_proto_string);

  // inputs
  for (auto input : graph.input()) {
    std::string name = input.name();
    std::vector<unsigned> shape;
    for (auto dim : input.type().tensor_type().shape().dim()) {
      shape.push_back(dim.dim_value());
    }
    bind_onnx_input(name, shape);
    shape_info.insert(std::make_pair(name, shape));
  }

  // initializers
  for (auto initailizer : graph.initializer()) {
    std::string name = initailizer.name();
    std::vector<unsigned> shape;
    for (auto dim : initailizer.dims()) {
      shape.push_back(dim);
    }
    shape_info.insert(std::make_pair(name, shape));
  }

  // nodes (Layers)
  for (auto node : graph.node()) {
    pim_layer *layer = new pim_layer();
    layer->N = 1;
    layer->name = node.name();

    std::string layer_type = node.op_type();
    if (layer_type == "Conv") {
      assert(node.input_size() == 3);

      layer->type = CONV;
      parse_attributes(node, layer);

      std::string x = node.input(0);
      pim_layer *input_layer = output_to_node.at(x);
      // input: N * C * H * W
      // layer->prev_layers.push_back(input_layer);
      solve_dependencies(input_layer, layer);
      layer->H = input_layer->P;
      layer->W = input_layer->Q;
      layer->C = input_layer->K;

      // calculate output size
      layer->P =
          (layer->H + 2 * layer->pad_h - layer->R) / layer->stride_h + 1;
      layer->Q =
          (layer->W + 2 * layer->pad_w - layer->S) / layer->stride_w + 1;
      
      std::vector<unsigned> filter_shape = shape_info.at(node.input(1));
      // filter: K * C * R * S
      layer->K = filter_shape[0];
      layer->C = filter_shape[1];
      assert(filter_shape[2] == layer->R);
      assert(filter_shape[3] == layer->S);

    } else if (layer_type == "Relu") {
      std::string input = node.input(0);

      layer->type = RELU;

      assert(output_to_node.find(input) != output_to_node.end());
      pim_layer *input_layer = output_to_node.at(input);
      // layer->prev_layers.push_back(input_layer);
      solve_dependencies(input_layer, layer);
      layer->K = input_layer->K;
      layer->P = input_layer->P;
      layer->Q = input_layer->Q;
    } else if (layer_type == "MaxPool") {
      layer->type = MAXPOOL;
      parse_attributes(node, layer);

      std::string input = node.input(0);
      assert(output_to_node.find(input) != output_to_node.end());
      pim_layer *input_layer = output_to_node.at(input);
      // layer->prev_layers.push_back(input_layer);
      solve_dependencies(input_layer, layer);

      layer->K = input_layer->K;
      layer->P = (input_layer->P - layer->R + 2 * layer->pad_h) / layer->stride_h + 1;
      layer->Q = (input_layer->Q - layer->S + 2 * layer->pad_w) / layer->stride_w + 1;
    } else if (layer_type == "Add") {
      layer->type = ADD;

      for (auto input : node.input()) {
        assert(output_to_node.find(input) != output_to_node.end());
        // layer->prev_layers.push_back(output_to_node.at(input));
        solve_dependencies(output_to_node.at(input), layer);
      }

      std::string input = node.input(0);
      pim_layer *input_layer = output_to_node.at(input);
      // // layer->prev_layers.push_back(input_layer);
      // solve_dependencies(input_layer, layer);

      layer->K = input_layer->K;
      layer->P = input_layer->P;
      layer->Q = input_layer->Q;
    } else if (layer_type == "GlobalAveragePool") {
      assert(node.input_size() == 1);
      assert(node.output_size() == 1);
      layer->type = GLOBAL_AVG_POOL;

      std::string input = node.input(0);
      pim_layer *input_layer = output_to_node.at(input);
      // layer->prev_layers.push_back(input_layer);
      solve_dependencies(input_layer, layer);

      layer->K = input_layer->K;
      layer->P = 1;
      layer->Q = 1;
    }

    else {
      assert(0 && "unknown node");
    }
    assert(node.output_size() == 1);
    std::string output = node.output(0);
    assert(output_to_node.find(output) == output_to_node.end());
    output_to_node.insert(std::make_pair(output, layer));

    pim_layers.push_back(layer);
  }

  for (auto output : graph.output()) {
    std::string name = output.name();

    // output should be seen already
    assert(output_to_node.find(name) != output_to_node.end());
    pim_layer *layer = new pim_layer();
    layer->name = name;
    layer->type = OUTPUT;

    pim_layer *input_layer = output_to_node.at(name);
    // input_layer->next_layers.push_back(layer);
    solve_dependencies(input_layer, layer);
    pim_layers.push_back(layer);
  }
}

void accel_sim_framework::parse_attributes(onnx::NodeProto node, pim_layer *layer) {
  for (auto attr : node.attribute()) {
    std::string name = attr.name();
    if (name == "kernel_shape") {
      assert(attr.ints_size() == 2);
      layer->R = attr.ints(0);
      layer->S = attr.ints(1);
    } else if (name == "strides") {
      assert(attr.ints_size() == 2);
      layer->stride_h = attr.ints(0);
      layer->stride_w = attr.ints(1);
    } else if (name == "pads") {
      assert(attr.ints_size() == 4);
      layer->pad_h = attr.ints(0);
      layer->pad_w = attr.ints(1);
    } else if (name == "dilations") {
      assert(attr.ints_size() == 2);
      layer->dilation_h = attr.ints(0);
      layer->dilation_w = attr.ints(1);
    } else if (name == "group") {
      layer->group = attr.i();
    } else if (name == "ceil_mode"){

    }
    
    else {
      assert(0 && "unknown attribute");
    }
  }
}

void accel_sim_framework::bind_onnx_input(std::string input_name,
                                          std::vector<unsigned> input_shape) {
  assert(input_shape.size() >= 4);

  pim_layer *layer = new pim_layer();
  layer->name = input_name;
  layer->type = INPUT;
  layer->N = 1;
  layer->C = 3;
  layer->H = input_shape[2];
  layer->W = input_shape[3];

  // treat as passthrough, set output as input
  layer->P = layer->H;
  layer->Q = layer->W;


  assert(output_to_node.find(input_name) == output_to_node.end());
  output_to_node.insert(std::make_pair(input_name, layer));

  pim_layers.push_back(layer);
}