from argparse import ArgumentParser


if __name__ == "__main__":
    aparser = ArgumentParser() 
    # project arguments    
    aparser.add_argument("save_dir", type=str, default="./restults/decathlon/Task01_BrainTumour")
    #model arguments
    aparser.add_argument("--model_type", type=str, default="unet", choices=["unet", "vit"])
    # vit arguments
    aparser.add_argument("--patch_size", type=int, default=16)
    aparser.add_argument("--embeddings_dim", type=int, default=768)
    # unet arguments
    aparser.add_argument("--num_features", type=int, default=16)
    aparser.add_argument("--num_classes", type=int, default=3)
    aparser.add_argument("--in_channels", type=int, default=4)
    aparser.add_argument("--features_sizes", type=list, default=[16, 32, 64, 128])
    aparser.add_argument("--kernel_size", type=int, default=3)
    aparser.add_argument("--activation", type=str, default="relu")
    # data arguments
    aparser.add_argument("--dataset", type=str, default="BraTS", choices=["BraTS", "WMH"])
    aparser.add_argument("--data_dir", type=str, default="./data/decathlon/Task01_BrainTumour")
    aparser.add_argument("--img_size", type=tuple, default=(128, 128, 128))
    aparser.add_argument("--num_workers", type=int, default=4)
    # training arguments
    aparser.add_argument("--max_epochs", type=int, default=300)
    aparser.add_argument("--batch_size", type=int, default=4)
    aparser.add_argument("--lr", type=float, default=1e-4)
    aparser.add_argument("--weight_decay", type=float, default=1e-5)
    # validation arguments
    aparser.add_argument("--val_mode", type=str, default="max", choices=["min", "max"])
    aparser.add_argument("--save_interval", type=int, default=1)
    args = aparser.parse_args()

    # Define data loaders
    train_transform = MyTrainTransform()
    val_transform = MyValTransform()

    if dataset == "BraTS":
        train_ds = DecathlonDataset(root_dir=data_dir, task="Task01_BrainTumour", transform=train_transform, section="training")
        val_ds = DecathlonDataset(root_dir=data_dir, task="Task01_BrainTumour", transform=val_transform, section="validation")
    elif dataset == "WMH":
        train_ds = DecathlonDataset(root_dir=data_dir, task="Task02_WMH", transform=train_transform, section="training")
        val_ds = DecathlonDataset(root_dir=data_dir, task="Task02_WMH", transform=val_transform, section="validation")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Define my model
    if model_type == "unet":
        model = MyUNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=3)
    elif model_type == "vit":
        model = ViT(patch_size=16, embeddings_dim=768, in_channels=4, out_channels=3)

    # Define loss function, optimizer and learning rate scheduler
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Traning
    if dataset == "BraTS":
        val_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    elif dataset == "WMH":
        val_metric = HausdorffMetric()
    batch_metrics = {'train':[], 'val': []}
    epoch_metrics = {'train':[], 'val': []}
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        # training
        for batch_data in train_loader:
            inputs, labels = batch_data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            epoch_loss += loss.item()
        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validation
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_outputs = model(val_inputs)
                dice = dice_metric(y_pred=val_outputs, y=val_labels)
                # validation metrics
                batch_metrics['val'].append(dice.item())
        epoch_metrics['val'].append(np.mean(batch_metrics['val']))

        if (val_mode == "min" and epoch_metrics['val'][-1] > best_metric) \
            or (val_mode == "max" and epoch_metrics['val'][-1] < best_metric) \
            and epoch % save_interval == 0:
            best_metric = epoch_metrics['val'][-1]
            torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"), weights_only=True)
            print("saved new best metric model")

    plt.plot(epoch_metrics['train'], label='train')
    plt.plot(epoch_metrics['val'], label='val')
    plt.legend()
    plt.show()

    # Evaluation

    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), weights_only=True))
    model.eval()
    with torch.no_grad():
        # select one image to evaluate and visualize the model output
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        val_output = inference(val_input)
        val_output = post_trans(val_output[0])
        plt.figure("image", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
        plt.show()
        # visualize the 3 channels label corresponding to this image
        plt.figure("label", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
        plt.show()
        # visualize the 3 channels model output corresponding to this image
        plt.figure("output", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
        plt.show()


