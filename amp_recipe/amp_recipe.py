# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

from typing import List, Tuple

import numpy as np
import torch, time, gc


import pytorch_onevar_model as onevar


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

# Timing utilities
start_time = None

#FIXME: need a better name
class MyDataset(onevar.OnesDataset):
    def __init__(self, stage_lens: List[int], stages: List[str]) -> None:
        assert len(stage_lens) == len(stages)
        self.stages = stages
        self.total_len = sum(stage_lens)
        stage_start_idx = [sum(stage_lens[:i]) for i in range(len(stage_lens))]
        assert stage_start_idx[-1] + stage_lens[-1] == self.total_len
        self.stage_stop_idx = stage_start_idx[1:]
        self.stage_stop_idx.append(self.total_len)  # This is so np.searchsorted works
        assert self.stage_stop_idx[-1] == self.total_len
        print(f"Created {self.__class__.__name__} with stage stop indices {self.stage_stop_idx}")

    def __len__(self) -> int:
        return self.total_len

    def _get_stage(self, index: int) -> str:
        stage_idx = np.searchsorted(self.stage_stop_idx, index, side="left")
        #print(f"{stage_idx=} for {index=}")
        stage = self.stages[stage_idx]
        return stage

    def __getitem__(self, index: int) -> Tuple:
        stage = self._get_stage(index)
        return self._get_stage_item(stage)

    def _get_stage_item(self, stage) -> Tuple:
        if stage.startswith("one"):
            x = 1
        elif stage.startswith("zero"):
            x = 0
        elif stage == "small":
            x = 2e-14
        elif stage == "large":
            x = 2e4
        else:
            raise ValueError(f"Unrecognized {self.__class__.__name__} stage {stage}")
        y = x
        return (
            torch.Tensor([float(x)]).cuda(),
            torch.Tensor([float(y)]).cuda()
        )


def get_data_loader() -> torch.utils.data.DataLoader:
    dataset = MyDataset([5, 1, 4, 1, 4, 1, 4], ["one", "large", "one", "small", "one", "zero", "one"])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    return data_loader


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def start_timer_and_print(local_msg):
    print("\n" + local_msg)
    start_timer()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print(local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print(
        "Max memory used by tensors = {} bytes".format(
            torch.cuda.max_memory_allocated()
        )
    )


def make_model(in_size, out_size, num_layers):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(torch.nn.Linear(in_size, in_size))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(in_size, out_size))
    return torch.nn.Sequential(*tuple(layers)).cuda()

def _train(net, loss_fn, input_, target, amp_enabled=False):
    output = net(input_)
    loss = loss_fn(output, target)
    if amp_enabled:
        # output is float16 because linear layers autocast to float16.
        assert output.dtype is torch.float16
        # loss is float32 because mse_loss layers autocast to float32.
        assert loss.dtype is torch.float32
    else:
        assert output.dtype is torch.float32
        assert loss.dtype is torch.float32
    return output, loss

def time_training(
    in_size,
    out_size,
    num_layers,
    epochs,
    loss_fn,
    data_loader,
    amp_enabled=True,
    checkpoint=None,
):
    start_timer_and_print(f"{'='*80}\n" + ("** Mixed precision start **" if amp_enabled else "** Default precision start **"))
    net = torch.nn.Linear(1, 1, bias=False).cuda()
    # Manually initialize the weight to 0.
    net.weight.data.fill_(0)
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(
            init_scale=INIT_SCALE,
            growth_interval=GROWTH_INTERVAL,
            enabled=amp_enabled,
    )
    if amp_enabled:
        print(f"Created and enabled GradScaler with {INIT_SCALE=} and {GROWTH_INTERVAL=}")

    if checkpoint is not None:
        print("\nReloading checkpoint...")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        if amp_enabled and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # So we can check for changes in the scale factor
    scale = scaler.get_scale()

    for epoch in range(1, 1+epochs):
        print("="*80)
        print(f"Epoch {epoch}")
        for b, ((input_, target),) in enumerate(zip(data_loader), 1):
            batch_prefix = "    "
            print(batch_prefix + "-"*(80-len(batch_prefix)))
            stage = data_loader.dataset._get_stage(b-1)
            print(f"{batch_prefix}Batch {b} ({stage.upper()})")
            step_prefix = 2*batch_prefix
            print(f"{step_prefix}input={input_.item()}, target={target.item()}")

            # From determined.harness.tests.experiment.fixtures.pytorch_onevar_model
            w_before = net.weight.data.item()
            loss_exp = (target.item() - input_.item() * w_before) ** 2
            w_exp = w_before + 2 * LR * input_.item() * (target.item() - (input_.item() * w_before))

            if amp_enabled:
                with torch.cuda.amp.autocast():
                    output, loss = _train(net, loss_fn, input_, target, amp_enabled=True)
            else:
                output, loss = _train(net, loss_fn, input_, target, amp_enabled=False)
            print(f"{step_prefix}{loss_exp=}")
            print(f"{step_prefix}loss    ={loss.item()}")

            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            if (new_scale := scaler.get_scale()) != scale:
                print(f"{step_prefix}scale changed from {scale} to {new_scale}")
                scale = new_scale

            # Inspect/modify gradients (e.g., clipping) may be done here

            # From PyTorch documentation
            # :meth:`step` carries out the following two operations:
            # 1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            #     earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
            # 2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            #     gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.
            scaler.unscale_(opt)
            if scaler.is_enabled():
                if found_inf := int(sum(scaler._found_inf_per_device(opt).values()).item()):
                    print(f"{step_prefix}{found_inf=}")
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(
                set_to_none=True  # can modestly improve performance
            )

            # From determined.harness.tests.experiment.fixtures.pytorch_onevar_model
            print(f"{step_prefix}Weight was {w_before}")
            w_after = net.weight.data.item()
            if w_after == w_exp:
                if w_after == w_before:
                    print(f"{step_prefix}Weight did not change as expected")
                else:
                    print(f"{step_prefix}Weight changed to {w_after} as expected")
            else:
                if w_after == w_before:
                    print(f"{step_prefix}Weight was expected to change to {w_exp} but did not change")
                else:
                    Δw_abserr = w_after - w_exp
                    print(f"{step_prefix}Weight was expected to change to {w_exp} but actually changed to {w_after}, {Δw_abserr=:.0e}")

        # end batch loop
    # end epoch loop

    end_timer_and_print("** Mixed precision end **" if amp_enabled else "** Default precision end**")
    checkpoint = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
    }
    if amp_enabled:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


LR = 0.001
BATCH_SIZE = 512//4
SIZE = 4096//1000
NUM_LAYERS = 3
NUM_BATCHES = 10
EPOCHS = 1

# One-indexed
BIG_BATCH_NB = 5
TINY_BATCH_NB = 10

SMALLEST_POS_SUBNORM = 2e-14 / 1024
LARGEST_NORM = 65504
INIT_SCALE = 8.0  # 65536.0
GROWTH_INTERVAL = 2000

def main(
    batch_size=BATCH_SIZE,
    in_size=SIZE,
    out_size=SIZE,
    num_layers=NUM_LAYERS,
    num_batches=NUM_BATCHES,
    epochs=EPOCHS,
    checkpoint=False,
):
    data_loader = get_data_loader()

    loss_fn = torch.nn.MSELoss().cuda()

    default_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data_loader, False)
    if checkpoint:
        default_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data_loader, False, default_cp)
    mixed_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data_loader, True)
    if checkpoint:
        mixed_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data_loader, True, mixed_cp)


if __name__ == "__main__":
    main()

