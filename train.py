#%%
import torch
import torchvision
from dataset import FaceDataset, get_transform
from architecture import auto_encoder, small_auto_encoder
from torchvision.models.segmentation.fcn import FCNHead


user_option = input("Choose the architecure, type 'small', 'normal', 'torch_fcn'")

if user_option== 'normal':
    model=auto_encoder()  # one of the custom architecures
elif user_option=='small':
    model=small_auto_encoder() # another one of the custom architecures use if you have less data but ideally you should keep editing the architecure until you achieve the best performance on the dataset
elif: user_option=='torch_fcn':
    # next two lines are to implement the PyTorch base FCN as an autoencoder, the classifier is repurposed to ouput an RGB image
    model=torchvision.models.segmentation.fcn_resnet50(pretrained=True, progress=True)
    model.classifier=FCNHead(2048, 3)




device=torch.device('cuda')

trainset=FaceDataset('./train/', get_transform(True))
valset=FaceDataset('./val/', get_transform(True))

trainloader=train_loader = torch.utils.data.DataLoader(trainset, batch_size=2,num_workers=0, shuffle=True)
valloader=train_loader = torch.utils.data.DataLoader(valset, batch_size=2,num_workers=0, shuffle=True)



model.to(device)
params = [p for p in model.parameters() if p.requires_grad]

optimizer=torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

 # optimizer.load_state_dict(torch.load('./best_ADAM_state.pth'))

MSE=torch.nn.MSELoss()
BCE=torch.nn.BCELoss(size_average=None, reduce=None, reduction= 'mean')



num_epochs=10

best_val_loss=1

for epoch in range(num_epochs):
    print('Epoch: ',str(epoch))
    model.train()
    for images, targets in trainloader:
        images=images.to(device)
        targets=targets.to(device)
        optimizer.zero_grad()
        if user_option=='torch_fcn':
            outputs=model(images)['out']
            loss=MSE(outputs, targets)
        else:
            outputs=model(images)
            loss=MSE(outputs, targets)+BCE(outputs, targets)
        loss.backward()
        optimizer.step()
        #print('Train Loss: ', str(loss.detach().cpu().numpy()))

    if epoch==0:
        torch.save(model.state_dict(), './experiment_FCN/best_model.pth')
        torch.save(optimizer.state_dict(), './experiment_FCN/best_ADAM_state.pth')
    current_val_loss=0
    model.eval()
    with torch.no_grad():
        for images, targets in valloader:
            images=images.to(device)
            targets=targets.to(device)
            if user_option=='torch_fcn':
                outputs=model(images)['out']
                iter_val_loss=MSE(outputs, targets)
            else:
                outputs=model(images)
                iter_val_loss=MSE(outputs, targets)+BCE(outputs, targets)
            iter_val_loss=iter_val_loss.detach().cpu().numpy()
            current_val_loss= iter_val_loss + current_val_loss
            #print('Iterative Val Loss: ', str(iter_val_loss))

    if current_val_loss<best_val_loss or epoch==0:
        best_val_loss=current_val_loss
        torch.save(model.state_dict(), './experiment_FCN/best_model.pth')
        torch.save(optimizer.state_dict(), './experiment_FCN/best_ADAM_state.pth')
        print('Best Model Saved! Val loss is ',str(best_val_loss))
    else:
        print('Model is bad!', 'Current Loss: ', str(current_val_loss), 'Best Loss: ', str(best_val_loss))
        # print('Current Val loss is ', str(current_val_loss, 'while the best is ', str(best_val_loss)))

        

# %%
