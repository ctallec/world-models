# import subprocess




def gen_dataset(logdir, rollouts=1000, display=False, max_workers=32, dataset_dir='datasets/carracing', random=False):
    if not random:
        args_str = f"--logdir {logdir} --rollouts {rollouts} --max-workers {max_workers} --dataset_dir {dataset_dir}"
        if display:
            args_str += " --display"

        return f'xvfb-run -a -s "-screen 0 1400x900x24" python eval_controller.py {args_str}'
        # subprocess.run(["python", "eval_controller.py", args_str], shell=True, capture_output=True)
    else:
        args_str = f"--logdir {logdir} --rollouts {rollouts} --threads {max_workers} --rootdir {dataset_dir}"
        # subprocess.run(["python", "data/generation_script.py", args_str], shell=True, capture_output=True)
        return f"python data/generation_script.py {args_str}"


def trainvae(logdir, batch_size=32, epochs=1000, noreload=False, 
             nosamples=False, dataset_dir='datasets/carracing', output_command=False):
    args_str = f"--logdir {logdir} --batch-size {batch_size} --epochs {epochs} --dataset_dir {dataset_dir}"
    if noreload:
        args_str += " --noreload"
    if nosamples:
        args_str += " --nosamples"

    # if output_command:
    return f"python trainvae.py {args_str}"
    # else:
    # subprocess.run(["python", "trainvae.py", args_str], shell=True, capture_output=True)


def trainmdrnn(logdir, noreload=False, include_reward=False, dataset_dir='datasets/carracing'):
    args_str = f"--logdir {logdir} --dataset_dir {dataset_dir}"
    if noreload:
        args_str += " --noreload"
    if include_reward:
        args_str += " --include_reward"
    

    # if output_command:
    return f"python trainmdrnn.py {args_str}"
    # else:
    # subprocess.run(["python", "trainmdrnn.py", args_str], shell=True, capture_output=True)



def traincontroller(logdir, n_samples=16, pop_size=64, target_return=950, display=False, max_workers=32, noreload=False, random_rnn=False, random_vae=False, max_epoch=1000000):
    args_str = f"--logdir {logdir} --n-samples {n_samples} --pop-size {pop_size} --target-return {target_return} --max-workers {max_workers} --max_epoch {max_epoch}"
    if noreload:
        args_str += " --noreload"
    if random_rnn:
        args_str += " --random-rnn"
    if display:
        args_str += " --display"
    if random_vae:
        args_str += " --random-vae"
    # subprocess.run(["python", "traincontroller.py", args_str], shell=True, capture_output=True)

    # if output_command:
    return f'xvfb-run -a -s "-screen 0 1400x900x24"  python traincontroller.py {args_str}'
    # else:
        # subprocess.run(["python", "traincontroller.py", args_str], shell=True, capture_output=True)







def main(logdir, total_epochs=200, period_restart=5, max_workers=32):


    script = ""
    for step in range(total_epochs // period_restart + 1):
        dataset_dir = f'~/data/carracing_step_{step}'
        ## First, generate the dataset
        first_step = (step == 0)

        # dataset_dir = '~/data/carracing'
        # if step != 0:
        script += f"######################\n"
        script += f"echo STEP {step}\n"
        script += "echo GEN DATASET\n"
        script += gen_dataset(logdir, max_workers=max_workers, 
                              dataset_dir=dataset_dir, random=first_step)
        script += "\n"
        ## Then train the Vae
        script += "echo TRAIN VAE\n"
        script += trainvae(logdir, dataset_dir=dataset_dir, noreload=first_step, nosamples=True)
        script += "\n"
        # Then the Mdrnn
        script += "echo TRAIN MDRNN\n"
        script += trainmdrnn(logdir, dataset_dir=dataset_dir, noreload=first_step)
        script += "\n"
        # Finally the controler 
        script += "echo TRAIN CONTROLLER\n"
        script += traincontroller(logdir, max_workers=max_workers, 
                                  max_epoch=(step+1) * period_restart,
                                  noreload=first_step)

        script += "\n"
        script += f"cp -r {logdir} {logdir}_step{step}\n"


    print(script)

if __name__ == "__main__":
    main('/private/home/leonardb/code/world-models/exp_dir/pipeline')