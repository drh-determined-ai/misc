from typing import Dict, Optional, Union

import numpy as np

import apex
import torch

import pytorch_onevar_model as onevar

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

LR = 0.01
BATCH_SIZE = 1
EPOCHS = 1
OPT = "O2"  # O1 patches torch functions, which can screw up successive tests
# With Apex, 32760 is the minimum scaled_loss value that causes a scale reduction
INIT_SCALE = 32760

def train(
    epochs,
    loss_fn,
    data_loader,
    checkpoint: Optional[Union[Dict, str]] = None,
    loss_prev: Optional[float] = None,
):
    net = torch.nn.Linear(1, 1, bias=False).cuda()
    net.weight.data.fill_(0)
    opt = torch.optim.SGD(net.parameters(), lr=LR)

    net, opt = apex.amp.initialize(net, opt, opt_level=OPT)


    try:
        checkpoint = torch.load(checkpoint)
        print("Loaded checkpoint from file")
    except:
        pass

    if checkpoint is not None:
        print("\nReloading checkpoint...")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        apex.amp.load_state_dict(checkpoint["amp"])
    apex.amp.frontend._amp_state.loss_scalers[0]._loss_scale = INIT_SCALE

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        batch_prefix = "    "  # Indent
        print(batch_prefix + "-"*(80-len(batch_prefix)))
        for b, ((input_, target),) in enumerate(zip(data_loader)):
            input_ = input_.cuda()
            target = target.cuda()
            if BATCH_SIZE == 1:
                # From determined.harness.tests.experiment.fixtures.pytorch_onevar_model
                w_before = net.weight.data.item()
                loss_exp = (target.item() - input_.item() * w_before) ** 2
                w_exp = w_before + 2 * LR * input_.item() * (target.item() - (input_.item() * w_before))
                #print(f"Expect loss = {loss_exp}")

            output = net(input_)
            loss = loss_fn(output, target)

            with apex.amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
                print(f"{batch_prefix}Batch {b:>2d}, loss = {loss.item()}, scaled_loss = {scaled_loss.item()}")
            opt.step()
            opt.zero_grad()

            if loss_prev is not None:
                assert loss <= loss_prev
                loss_prev = loss

    checkpoint = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "amp": apex.amp.state_dict(),
    }
    return checkpoint


def main(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    num_checkpoints=1,
):
    dataset = onevar.OnesDataset()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    loss_fn = torch.nn.MSELoss().cuda()

    checkpoint, loss_prev = None, None
    for _ in range(num_checkpoints):
        checkpoint = train(epochs, loss_fn, data_loader, checkpoint, loss_prev)
    torch.save(checkpoint, "amp_checkpoint.pt")
    return checkpoint


if __name__ == "__main__":
    main()

