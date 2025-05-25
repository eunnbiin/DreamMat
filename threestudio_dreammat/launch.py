import argparse
import contextlib
import logging
import os
import sys


class ColoredFilter(logging.Filter):
    """
    A logging filter to add color to certain log levels.
    """

    RESET = "\\033[0m"
    RED = "\\033[31m"
    GREEN = "\\033[32m"
    YELLOW = "\\033[33m"
    BLUE = "\\033[34m"
    MAGENTA = "\\033[35m"
    CYAN = "\\033[36m"

    COLORS = {
        "WARNING": YELLOW,
        "INFO": GREEN,
        "DEBUG": BLUE,
        "CRITICAL": MAGENTA,
        "ERROR": RED,
    }

    RESET = "\\x1b[0m"

    def __init__(self):
        super().__init__()

    def filter(self, record):
        if record.levelname in self.COLORS:
            color_start = self.COLORS[record.levelname]
            record.levelname = f"{color_start}[{record.levelname}]"
            record.msg = f"{record.msg}{self.RESET}" 
        return True


def main(args, extras) -> None:
    # =====================================================================================
    # START: Submesh Processing Block 
    # =====================================================================================
    if args.enable_submesh_processing:
        import subprocess 

        try:
            from threestudio.utils.mesh_utils import split_mesh_by_segmentation
        except ImportError:
            print("Error: Could not import 'split_mesh_by_segmentation' from 'threestudio.utils.mesh_utils'.")
            print("Ensure 'threestudio_dreammat/threestudio/utils/mesh_utils.py' exists and is in the Python path.")
            sys.exit(1)

        if not all([args.original_mesh_path, args.segmentation_npy_path, args.target_segment_ids]):
            print("Error: For submesh processing, --original_mesh_path, --segmentation_npy_path, and --target_segment_ids must be provided.")
            sys.exit(1)

        current_script_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.abspath(os.path.join(current_script_dir, ".."))

        def resolve_path(user_path):
            if not user_path: return None
            if os.path.isabs(user_path):
                return user_path
            return os.path.normpath(os.path.join(project_root, user_path))

        original_mesh_abs_path = resolve_path(args.original_mesh_path)
        segmentation_npy_abs_path = resolve_path(args.segmentation_npy_path)
        
        if not original_mesh_abs_path or not os.path.exists(original_mesh_abs_path):
            print(f"Error: Original mesh file not found at resolved path: {original_mesh_abs_path}")
            sys.exit(1)
        if not segmentation_npy_abs_path or not os.path.exists(segmentation_npy_abs_path):
            print(f"Error: Segmentation NPY file not found at resolved path: {segmentation_npy_abs_path}")
            sys.exit(1)

        mesh_basename = os.path.splitext(os.path.basename(original_mesh_abs_path))[0]
        submesh_base_output_abs_dir = resolve_path(args.submesh_output_base_dir)
        submesh_parts_temp_dir = os.path.join(submesh_base_output_abs_dir, f"{mesh_basename}_parts_files")
        os.makedirs(submesh_parts_temp_dir, exist_ok=True)
        
        print(f"--- Submesh Processing Enabled ---")
        print(f"Project Root: {project_root}")
        print(f"Original Mesh: {original_mesh_abs_path}")
        print(f"Segmentation NPY: {segmentation_npy_abs_path}")
        print(f"Target IDs: {args.target_segment_ids}")
        print(f"Submesh .obj save dir: {submesh_parts_temp_dir}")
        print(f"Subprocess output base: {submesh_base_output_abs_dir}")

        target_sm_path, remaining_sm_path = split_mesh_by_segmentation(
            original_mesh_path=original_mesh_abs_path,
            segmentation_npy_path=segmentation_npy_abs_path,
            target_segment_ids=args.target_segment_ids,
            output_dir=submesh_parts_temp_dir, 
            base_output_name=mesh_basename
        )

        sub_processes_to_run = []
        target_ids_str = '_'.join(map(str, sorted(list(set(args.target_segment_ids)))))
        if target_sm_path and args.target_prompt:
            sub_processes_to_run.append({
                "name": f"{mesh_basename}_target_{target_ids_str}",
                "mesh_path": target_sm_path,
                "prompt": args.target_prompt,
            })
        if remaining_sm_path and args.remaining_prompt:
            sub_processes_to_run.append({
                "name": f"{mesh_basename}_remaining_after_{target_ids_str}",
                "mesh_path": remaining_sm_path, 
                "prompt": args.remaining_prompt,
            })

        if not sub_processes_to_run:
            print("No submeshes were created or no prompts provided for them. Exiting submesh processing.")
            sys.exit(0)

        base_subprocess_command_parts = [sys.executable, sys.argv[0]] # ['python', 'launch.py']

        if args.config:
            abs_config_path = resolve_path(args.config)
            if not abs_config_path or not os.path.exists(abs_config_path):
                print(f"Error: Config file not found at resolved path: {abs_config_path}")
                sys.exit(1)
            base_subprocess_command_parts.extend(["--config", abs_config_path])
        else: 
            print("Error: --config argument is missing for subprocess construction.")
            sys.exit(1)

        if args.train: base_subprocess_command_parts.append("--train")
        elif args.validate: base_subprocess_command_parts.append("--validate")
        elif args.test: base_subprocess_command_parts.append("--test")
        elif args.export: base_subprocess_command_parts.append("--export")
        
        if args.gpu: base_subprocess_command_parts.extend(["--gpu", args.gpu])
        if args.gradio: base_subprocess_command_parts.append("--gradio")
        if args.verbose: base_subprocess_command_parts.append("--verbose")
        if args.typecheck: base_subprocess_command_parts.append("--typecheck")

        original_cli_extras = extras # main(args, extras)에서 받은 extras

        for process_info in sub_processes_to_run:
            print(f"\n--- Preparing to run DreamMat for: {process_info['name']} ---")
            print(f"Mesh: {process_info['mesh_path']}")
            print(f"Prompt: \"{process_info['prompt']}\"")

            current_subprocess_extras = original_cli_extras.copy()
            current_subprocess_extras.extend([
                f"system.geometry.shape_init=mesh:{process_info['mesh_path']}",
                f"system.prompt_processor.prompt=\"{process_info['prompt']}\"",
                "data.blender_generate=true", # Blender pre-render 활성화
                f"name={process_info['name']}", 
                f"exp_root_dir={submesh_base_output_abs_dir}", 
                "tag=submesh_run" 
            ])
            
            final_cmd_for_subprocess = base_subprocess_command_parts + current_subprocess_extras
            
            print(f"Executing command: {' '.join(final_cmd_for_subprocess)}")
            
            try:
                # 동일한 환경 변수를 사용하며 실행
                subprocess.run(final_cmd_for_subprocess, check=True, env=os.environ.copy())
                print(f"Successfully processed: {process_info['name']}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {process_info['name']}: {e}")
            except FileNotFoundError:
                print(f"Error: Python executable or launch script not found for command: {' '.join(final_cmd_for_subprocess)}")
            except Exception as e:
                print(f"An unexpected error occurred while running subprocess for {process_info['name']}: {e}")


        print("\nAll submesh processing finished.")
        sys.exit(0) # 서브메쉬 처리 완료 후 메인 프로세스 종료
    # =====================================================================================
    # END: Submesh Processing Block
    # =====================================================================================

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0] 

    devices = -1 # 
    if len(env_gpus) > 0:
        n_gpus = len(env_gpus)
    else:
        if args.gpu:
            selected_gpus = list(args.gpu.split(","))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            n_gpus = len(selected_gpus)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
            n_gpus = 1


    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if args.typecheck:
        from jaxtyping import install_import_hook
        install_import_hook("threestudio", "typeguard.typechecked")

    import threestudio 
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank


    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        if handler.stream == sys.stderr: 
            if not args.gradio: 

                if not any(isinstance(f, ColoredFilter) for f in handler.filters):
                    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s")) 
                    handler.addFilter(ColoredFilter())
            else: 
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))


    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)

    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None
    )
    system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
   
    if hasattr(cfg, 'data_type') and cfg.data_type == 'random-camera-datamodule':
        pre_render_dir = os.path.join(cfg.trial_dir, "pre_render")
        os.makedirs(pre_render_dir, exist_ok=True) 

        dm = threestudio.find(cfg.data_type)(system.geometry.isosurface(), pre_render_dir, cfg.data)
    else:
        dm = threestudio.find(cfg.data_type)(cfg.data)


    if args.gradio:
        fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
        fh.setLevel(logging.INFO)
        if args.verbose: fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
            ),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(
                os.path.join(cfg.trial_dir, "code"), use_version=False
            ),
            ConfigSnapshotCallback(
                args.config,
                cfg,        
                os.path.join(cfg.trial_dir, "configs"),
                use_version=False,
            ),
        ]
        if args.gradio:
            callbacks += [ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))]
        else:
            callbacks += [CustomProgressBar(refresh_rate=1)] 

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\\n")

    loggers_list = [] 
    if args.train:
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers_list += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
            CSVLogger(cfg.trial_dir, name="csv_logs"),
        ]
        if hasattr(system, 'get_loggers') and callable(system.get_loggers):
            loggers_list.extend(system.get_loggers())

        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)], 
            )
        )()
    
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers_list,
        inference_mode=False,
        accelerator="gpu",
        devices=devices, 
        **cfg.trainer,
    )

    def set_system_status(system_instance: BaseSystem, ckpt_path: typing.Optional[str]): 
        if ckpt_path is None or not os.path.exists(ckpt_path): 
            if ckpt_path: print(f"Warning: Checkpoint path {ckpt_path} not found.")
            return
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system_instance.set_resume_status(ckpt["epoch"], ckpt["global_step"])


    if args.train:
        trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)
        if hasattr(cfg.trainer, 'test_after_train') and cfg.trainer.test_after_train:
            trainer.test(system, datamodule=dm) 
        if args.gradio: 
            trainer.predict(system, datamodule=dm) 

    elif args.validate:
        set_system_status(system, cfg.resume)
        trainer.validate(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.test:
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
    elif args.export:
        set_system_status(system, cfg.resume)
        trainer.predict(system, datamodule=dm, ckpt_path=cfg.resume) # export는 predict 사용


if __name__ == "__main__":
    import typing

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0", 
        help="GPU(s) to be used. e.g., 0 or 0,1 or 1,2. If CUDA_VISIBLE_DEVICES is set, this is ignored.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--export", action="store_true")

    parser.add_argument(
        "--gradio", action="store_true", help="if true, run in gradio mode"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )
    parser.add_argument(
        "--typecheck", action="store_true", help="whether to enable dynamic type checking"
    )

    # --- New arguments for mesh segmentation and sub-processing ---
    parser.add_argument(
        "--enable_submesh_processing",
        action="store_true",
        help="Enable processing of segmented submeshes. Requires submesh-related args."
    )
    parser.add_argument(
        "--original_mesh_path",
        type=str,
        default=None,
        help="Path to the original mesh file (relative to project root, e.g., 'threestudio_dreammat/load/shapes/objs/knight.obj')."
    )
    parser.add_argument(
        "--segmentation_npy_path",
        type=str,
        default=None,
        help="Path to the .npy file containing segmentation IDs (relative to project root, e.g., 'threestudio_dreammat/load/shapes/sep/knight_seg.npy')."
    )
    parser.add_argument(
        "--target_segment_ids",
        type=int,
        nargs='+', 
        default=None,
        help="List of segment IDs for the 'target' submesh (e.g., 0 1 2)."
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default="A high-quality PBR material.",
        help="Text prompt for the target submesh."
    )
    parser.add_argument(
        "--remaining_prompt",
        type=str,
        default="A high-quality PBR material for the rest.", 
        help="Text prompt for the remaining submesh."
    )
    parser.add_argument(
        "--submesh_output_base_dir", 
        type=str,
        default="outputs_submesh_runs", 
        help="Base directory (relative to project root) to save outputs for each submesh process."
    )
    # --- End of new arguments ---

    args, extras = parser.parse_known_args()

    if args.gradio:
        with contextlib.redirect_stdout(sys.stderr): # sys.stderr로 리다이렉션
            main(args, extras)
    else:
        main(args, extras)

