# convert_to_onnx_and_check_for_pytorch.py
# environment suggesstion:
# 1. need in drive docker and nuplan environment(python version: 3.9)
# 2. need reinstall tensorrt for python version: 3.9
# 3. need install pycuda, onnx, torch, onnxruntime, omegaconf, tqdm, polygraphy, matplotlib, onnx_graphsurgeon


import os
import sys

import numpy as np
import onnx
import onnxruntime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import torch
from omegaconf import DictConfig, OmegaConf

import glob

# set stdout and stderr to line-buffered
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Optional, Callable
from utils_for_test import *

import onnx_graphsurgeon
from polygraphy.backend.onnx import fold_constants

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, list):
        tensor = tensor[0]
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )

def generate_decoder_inputs(cfg, dataset):

    history_input, prediction_input, _, _ = dataset[0:1]

    history_mask = torch.ones(cfg.batch_size, cfg.history_length * 2 - 1)
    prediction_mask = torch.ones(cfg.batch_size, cfg.prediction_length)

    dummy_input = {
        "history_input": history_input,
        "prediction_input": prediction_input,
        "history_mask": history_mask,
        "prediction_mask": prediction_mask,
    }

    dummy_input_new = (
        history_input.to(cfg.device), prediction_input.to(cfg.device), history_mask.to(cfg.device), prediction_mask.to(cfg.device)
    )

    pth_inputs = (
        dummy_input["history_input"].to(cfg.device),
        dummy_input["prediction_input"].to(cfg.device),
    )

    onnx_inputs = {
        "history_input": to_numpy(history_input),
        "prediction_input": to_numpy(prediction_input),
        "history_mask": to_numpy(history_mask),
        "prediction_mask": to_numpy(prediction_mask),
    }

    return dummy_input, pth_inputs, onnx_inputs,dummy_input_new

def get_decoder_inputs_outputs_names(
):
    # prepare names
    input_names = [
        "history_input",
        "prediction_input",
        "history_mask",
        "prediction_mask",
    ]
    output_names = [
        "output"
    ]

    return input_names, output_names

def convert_decoder_to_onnx(
    pth_decoder,
    cfg,
    onnx_model_path,
    trt_engine_path,
    dataset,
):
    print(">>>> Convert decoder from pytorch to onnx ......")

    dummy_input, pth_inputs, onnx_inputs, dummy_input_new = generate_decoder_inputs(
        cfg, dataset
    )

    input_names, output_names = get_decoder_inputs_outputs_names()

    # export the model
    torch.onnx.export(
        pth_decoder,
        dummy_input_new,   #dummy_input,
        onnx_model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        do_constant_folding=True,
        opset_version=cfg.opset_version,
        verbose=cfg.export_verbose,
    )
    print(f"ONNX decoder model saved to {onnx_model_path}")

    try:
        decoder_onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(decoder_onnx_model)
    except onnx.checker.ValidationError as e:
        raise RuntimeError(f"Invalid ONNX decoder model: {e}")

    # fold constants
    decoder_onnx_model = fold_constants(decoder_onnx_model)
    onnx.save(decoder_onnx_model, onnx_model_path)

    onnx_decoder = onnxruntime.InferenceSession(decoder_onnx_model.SerializeToString())

    print("==== To test decoder outputs again with exported inputs")
    test_decoder_outputs(
        pth_decoder, onnx_decoder, pth_inputs, onnx_inputs, rtol=cfg.rtol, atol=cfg.atol
    )

    trt_engine = convert_decoder_to_trt(cfg, onnx_model_path, trt_engine_path)

    print(
        f"==== To test tensorrt decoder outputs"
    )
    _, pth_inputs, onnx_inputs,_ = generate_decoder_inputs(
        cfg, dataset
    )
    test_decoder_trt_outputs(
        pth_decoder,
        trt_engine,
        pth_inputs,
        onnx_inputs,
        output_names,
        rtol=cfg.rtol,
        atol=cfg.atol,
    )


def test_decoder_outputs(
    pth_decoder, onnx_decoder, pth_inputs, onnx_inputs, rtol, atol
):
    print("Test outputs between pytorch and onnx decoder model ......")

    pth_outputs = pth_decoder(*pth_inputs)
    onnx_outputs = onnx_decoder.run(None, onnx_inputs)
    output_names = [node.name for node in onnx_decoder.get_outputs()]

    # input_mean = torch.tensor(pth_outputs['input_mean'], dtype=torch.float32)
    # input_std = torch.tensor(pth_outputs['input_std'], dtype=torch.float32)
    # compare outputs
    # NOTE there is a rare chance below assert could fail with minor difference
    #      if fail, run again, do not try to enlarge the tolerance
    assert_error_num = 0
    for i in range(len(pth_outputs)):
        try:
            if i >= len(output_names):
                print(f" - Output {i} is not in output_names list")
                continue
            name = output_names[i]

            np.testing.assert_allclose(
                onnx_outputs[i].squeeze(axis=0), to_numpy(pth_outputs[i]), rtol=rtol, atol=atol
            )
            print(f" o Output {name} is valid")
        except AssertionError as e:
            assert_error_num += 1
            print(f" x Output {name} has differences:\n {e}")

    if assert_error_num == 0:
        print("[[PASS]] ONNX decoder is valid")
    else:
        print(
            f"[[FAIL]] ONNX decoder has differences in {assert_error_num} / {len(onnx_outputs)} outputs."
        )


def convert_onnx_to_trt(
    onnx_model_path: str,
    trt_engine_path=None,
    verbose: bool = True,
    profile_list: list[dict[str, dict[str, tuple]]] = [],
    mark_output: list = [str],
    builder_optimization_level=None,
):
    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_model_path, "rb") as model:
        if not parser.parse(model.read()):
            error_msg = "\n".join(
                [str(parser.get_error(i)) for i in range(parser.num_errors)]
            )
            raise RuntimeError(f"Failed to parse the ONNX file: \n{error_msg}")

    print(f"ONNX file {onnx_model_path} parsed by trt.OnnxParser successfully.")

    print("Going to build TensorRT engine for ONNX model. May take a while ......")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    for profile_cfg in profile_list:
        profile = builder.create_optimization_profile()
        for key, value in profile_cfg.items():
            profile.set_shape(key, min=value["min"], opt=value["opt"], max=value["max"])
        config.add_optimization_profile(profile)

    # nodes in mark_output list are marked as model output if not yet, others are unmarked
    if len(mark_output) > 0:
        marked_output_list = []
        for layer in network:
            for i in range(layer.num_outputs):
                tensor = layer.get_output(i)
                if tensor.name in mark_output:
                    if not tensor.is_network_output:
                        network.mark_output(tensor)
                    marked_output_list.append(tensor.name)
                elif tensor.is_network_output:
                    network.unmark_output(tensor)
        print(f"Marked outputs: {marked_output_list}")
        for name in mark_output:
            if name not in marked_output_list:
                print(f"Failed to mark output: {name}")

    # set lowest optimization level to have best precision result
    if builder_optimization_level is not None:
        config.builder_optimization_level = builder_optimization_level

    try:
        serialized_network = builder.build_serialized_network(network, config)
    except Exception as e:
        print(f"Error building serialized network: {e}")
        print("Check the network structure for unsupported layers.")

    if serialized_network is None:
        raise RuntimeError("Failed to serialize the network.")

    # Optionally save the serialized engine to a file
    if trt_engine_path is not None:
        with open(trt_engine_path, "wb") as f:
            f.write(serialized_network)

    # Deserialize the serialized network to create the engine
    runtime = trt.Runtime(trt_logger)
    if trt_engine_path is None:
        print("deserialize serialized network as engine")
        engine = runtime.deserialize_cuda_engine(serialized_network)
    else:
        print(f"deserialize engine from file: {trt_engine_path}")
        with open(trt_engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine for {onnx_model_path}.")

    print(f"TensorRT engine for {onnx_model_path} created successfully.")
    return engine


def convert_decoder_to_trt(cfg, onnx_model_path, trt_engine_path):
    print(">>>> Convert decoder from onnx to tensorrt ......")

    return convert_onnx_to_trt(
        onnx_model_path=onnx_model_path,
        trt_engine_path=trt_engine_path,
        verbose=cfg.export_verbose,
        mark_output=cfg.decoder_mark_output if cfg.enable_decoder_mark_output else [],
        builder_optimization_level=cfg.builder_optimization_level,
    )

def infer_trt(engine, inputs, encoder_type=None):
    """Run inference using a TensorRT engine.

    Args:
        engine: TensorRT engine
        inputs: Input tensors
        encoder_type: "hivt", "dtpp", or None (for decoder)
    """
    cuda.init() 
    device = cuda.Device(0)
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # Set the active optimization profile index to 0, so that tensorrt infer dynamic shapes correctly
    context.set_optimization_profile_async(0, stream.handle)

    trt_inputs = {}
    trt_outputs = {}
    bindings = [None] * engine.num_bindings

    input_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_bindings)
        if engine.get_tensor_mode(engine[i]) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_bindings)
        if engine.get_tensor_mode(engine[i]) == trt.TensorIOMode.OUTPUT
    ]

    onnx_inputs = inputs

    onnx_inputs = {k: to_numpy(v) for k, v in onnx_inputs.items()}

    for binding_name in input_names:
        input_array = onnx_inputs[binding_name]
        context.set_input_shape(binding_name, input_array.shape)

        binding_index = engine[binding_name]
        binding_shape = context.get_tensor_shape(binding_name)
        binding_dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        size = trt.volume(binding_shape)

        assert (
            tuple(binding_shape) == input_array.shape
        ), f"Binding name {binding_name}, expected shape {binding_shape}, got {input_array.shape}"
        assert (
            binding_dtype == input_array.dtype
        ), f"Binding name {binding_name}, expected dtype {binding_dtype}, got {input_array.dtype}"

        print(
            f"Binding name {binding_name}, shape {binding_shape}, dtype {binding_dtype}, size {size}"
        )

        host_mem = cuda.pagelocked_empty(size, binding_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        np.copyto(host_mem, input_array.ravel())
        cuda.memcpy_htod_async(device_mem, host_mem, stream)

        trt_inputs[binding_name] = {"host": host_mem, "device": device_mem}
        bindings[binding_index] = int(device_mem)

    for binding_name in output_names:
        binding_index = engine[binding_name]
        binding_shape = context.get_tensor_shape(binding_name)
        binding_dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        size = context.get_max_output_size(binding_name)

        assert (
            size > 0
        ), f"Binding name {binding_name}, max output size is none positive"

        print(
            f"Binding name {binding_name}, shape {binding_shape}, dtype {binding_dtype}, size {size}"
        )

        host_mem = cuda.pagelocked_empty(size, binding_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        trt_outputs[binding_name] = {"host": host_mem, "device": device_mem}
        bindings[binding_index] = int(device_mem)

    # Execute inference asynchronously
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer outputs from device to host
    for binding_name, output in trt_outputs.items():
        cuda.memcpy_dtoh_async(
            output["host"],
            output["device"],
            stream,
        )

    # Synchronize the stream to ensure all operations are completed
    stream.synchronize()

    # Construct output results
    outputs = {}
    for binding_name, output in trt_outputs.items():
        shape = context.get_tensor_shape(binding_name)
        host_output = output["host"]
        output_size = trt.volume(shape)
        outputs[binding_name] = host_output[:output_size].reshape(shape)

    # Clean up memory
    for mem in trt_inputs.values():
        mem["device"].free()
    for mem in trt_outputs.values():
        mem["device"].free()

    return outputs

def test_decoder_trt_outputs(
    pth_model, trt_engine, pth_inputs, onnx_inputs, output_names, rtol, atol
):
    print("Test outputs between pytorch and tensorrt decoder model ......")

    pth_outputs = pth_model(*pth_inputs)
    trt_outputs = infer_trt(trt_engine, onnx_inputs)

    assert_error_num = 0
    for i in range(len(pth_outputs)):
        try:
            if i >= len(output_names):
                print(f" - Output {i} is not in output_names list")
                continue
            name = output_names[i]
            if name not in trt_outputs:
                print(f" - Output {name} is not in tensorrt outputs")
                continue
            np.testing.assert_allclose(
                trt_outputs[name].squeeze(axis=0), to_numpy(pth_outputs[i]), rtol=rtol, atol=atol
            )
            print(f" o Output {name} is valid")
        except AssertionError as e:
            assert_error_num += 1
            print(f" x Output {name} has differences:\n {e}")

    if assert_error_num == 0:
        print("[[PASS]] TensorRT decoder is valid")
    else:
        print(
            f"[[FAIL]] TensorRT decoder has differences in {assert_error_num} / {len(trt_outputs)} outputs."
        )

def load_pth_models(cfg):
    def load_pth_model(model_state_dict, init_func):
        """Helper function to load and initialize a PyTorch model.

        Args:
            model_state_dict: Model state dictionary
            init_func: Function to initialize the model
        """
        model = init_func(cfg)
        model.load_state_dict(model_state_dict)
        model.to(cfg.device)
        model.eval()
        return model
    
    def get_decoder(cfg):
        return TorchTransformerDecoder(
            cfg.state_dim,
            cfg.action_dim,
            cfg.state_dim,
            cfg.latent_dim,
            cfg.num_heads,
            cfg.num_layers,
            cfg.device,
            cfg.dropout,
            cfg.history_length,
            cfg.prediction_length
        )

    if cfg.pth_model_path is None:
        return None,

    try:
        model_dict = torch.load(cfg.pth_model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load pth model: {e}")

    pth_decoder = load_pth_model(
        model_dict["model_state_dict"],
        get_decoder
    )

    return pth_decoder

def convert_to_onnx(cfg: DictConfig):
    torch.set_default_dtype(torch.float32)

    # load data from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_files = glob.glob(os.path.join(current_dir, '*.pkl')) # get all *.pkl file in this path
    dataset = MujocoDataset(dataset_files, cfg.history_length, cfg.prediction_length, use_zero_point=True)

    OmegaConf.register_new_resolver("eval", eval)

    pth_decoder = load_pth_models(cfg)

    convert_decoder_to_onnx(
        pth_decoder,
        cfg,
        onnx_model_path=cfg.decoder_onnx_model_path,
        trt_engine_path=cfg.decoder_trt_engine_path,
        dataset = dataset
    )


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file_path = os.path.join(current_dir, 'tool_config.yaml')
    config = OmegaConf.load(yaml_file_path)
    convert_to_onnx(config)