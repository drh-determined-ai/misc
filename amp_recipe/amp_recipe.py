# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html

import torch, time, gc


# Timing utilities
start_time = None


def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()


def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
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



def time_training(
    in_size,
    out_size,
    num_layers,
    epochs,
    loss_fn,
    data,
    targets,
    amp_enabled=True,
    checkpoint=None,
):
    net = make_model(in_size, out_size, num_layers)
    opt = torch.optim.SGD(net.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if checkpoint is not None:
        print("\nReloading checkpoint...")
        net.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        if amp_enabled and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # So we can check for changes in the scale factor
    scale = scaler.get_scale()

    start_timer()
    for epoch in range(epochs):
        for i, (input, target) in enumerate(zip(data, targets)):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                output = net(input)
                loss = loss_fn(output, target)
                if amp_enabled:
                    # output is float16 because linear layers autocast to float16.
                    assert output.dtype is torch.float16
                    # loss is float32 because mse_loss layers autocast to float32.
                    assert loss.dtype is torch.float32
                else:
                    assert output.dtype is torch.float32
                    assert loss.dtype is torch.float32

            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            if new_scale := scaler.get_scale() != scale:
                print(f"{i=} : scale changed from {scale} to {new_scale}")
                scale = new_scale

            # Inspect/modify gradients (e.g., clipping) may be done here

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(
                set_to_none=True  # can modestly improve performance
            )
    end_timer_and_print("Mixed precision:" if amp_enabled else "Default precision:")
    checkpoint = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
    }
    if amp_enabled:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


BATCH_SIZE = 512
SIZE = 4096
NUM_LAYERS = 3
NUM_BATCHES = 50
EPOCHS = 3


def main(
    batch_size=BATCH_SIZE,
    in_size=SIZE,
    out_size=SIZE,
    num_layers=NUM_LAYERS,
    num_batches=NUM_BATCHES,
    epochs=EPOCHS,
    checkpoint=False,
):
    # Creates data in default precision.
    # The same data is used for both default and mixed precision trials below.
    # You don't need to manually change inputs' dtype when enabling mixed precision.
    data = [torch.randn(batch_size, in_size, device="cuda") for _ in range(num_batches)]
    targets = [
        torch.randn(batch_size, out_size, device="cuda") for _ in range(num_batches)
    ]

    loss_fn = torch.nn.MSELoss().cuda()

    default_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data, targets, False)
    if checkpoint:
        default_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data, targets, False, default_cp)
    mixed_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data, targets, True)
    if checkpoint:
        mixed_cp = time_training(in_size, out_size, num_layers, epochs, loss_fn, data, targets, True, mixed_cp)


if __name__ == "__main__":
    main()

