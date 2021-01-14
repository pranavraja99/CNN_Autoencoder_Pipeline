# CNN_Autoencoder_Pipeline
I've always been interested in unsupervised deep learning so here's an autoencoder pipeline I want to share for those of you who want to try it out with your own data. You need alot of data for it to be effective since it's unsupervised.

## Usage
- To train, use the train.py code and just make sure your train and val set are in a folder called 'train' and 'val'
- If you actually want it to work with optimal performance, I suggest you read up on autoencoders and their literature in addition to how to use the albumentations library and basic torch code.
- In this project, I was training a simple inpainting autoencoder. If you want to change the application, you can change what is being done to the input image in 'dataset.py'. For this just check out alubmenations or use another augmentation library. The standard torch augmentations should work for simple application such as deblurring
- Just for simplicity, I only used horizontal flip augmentation. stronger and more augmentation should definetely help but will differ based on the use case. So I suggest you play around with that.
- Remember, you shouldnt just use one of my two custom arhitecures or the torch FCN and expect to work well with your data. Be sure to learn some basic torch code if don't know already and play around with the architecture until you find the opitimal one for your data.
- After training, the best validation checkpoint will be saved to a file called 'best_model.pth'. You can then use the 'evaluate.py' and 'inference.py' on the test and validate sets.
- In order to use this for compression/decompression purposes, I put the encoder and decoder into separate nn.Sequential blocks. Making it very easy to decouple the encoder and decoder post training. The encoder can be references using 'model.encoder' and the decoer using 'model.decoder'

## Credit
- If you use this implementation or any parts of it, all I ask is that you give me credit (Pranav Raja, researcher at UC Davis) and include a link to this repo if you post it on Github.
