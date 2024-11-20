import BatchMaker_16
import LearnTools
import ImageTools
import argparse
import torch
import numpy as np
from tifffile import imsave, imread, imwrite
import torch.nn as nn

import MinkowskiEngine as ME
from modules_ME_2_16 import *

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_dims = args.n_dims
squash, down_sample = args.squash_phases, args.down_sample
size_to_evaluate = args.volume_size_to_evaluate
g_file_name, super_sample = args.g_image_path, args.super_sampling
phases_to_low, g_epoch_id = args.phases_low_res_idx, args.g_epoch_id

progress_main_dir = 'progress/' + progress_dir
path_to_g_weights = progress_main_dir + '/g_weights' + g_epoch_id + '.pth'
G_image_path = 'data/' + g_file_name

rand_id = str(np.random.randint(10000))

file_name = 'generated_tif' + rand_id + '.tif'
crop_to_cube = False
input_with_noise = True
all_pore_input = False

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# the material indices to low-res:
to_low_idx = torch.LongTensor(phases_to_low).to(device)

scale_f = 16

###################################################################################################################################
class G_0(torch.nn.Module):
    def __init__(self, chin_out, channels, factors, steps):
        super(G_0, self).__init__()
        
        self.chin_out = chin_out
        self.channels = channels
        self.factors = factors
        self.steps = steps

        self.block_standard = nn.Sequential(
            nn.Conv3d(self.chin_out, self.channels[4], kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm3d(self.channels[4]),
            nn.InstanceNorm3d(self.channels[4], eps=1e-04),
            nn.ReLU(),
            nn.Conv3d(self.channels[4], self.channels[4], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(self.channels[4]),
            nn.InstanceNorm3d(self.channels[4], eps=1e-04),
            nn.ReLU(),)

        # This ME convolutional layer is to get used to the transition between the formats
        self.block_new_format = nn.Sequential(
            ME.MinkowskiConvolution(self.channels[4], self.channels[4], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[4], eps=1e-04),
            ME.MinkowskiReLU(),)
        
        self.classification_0 = ME.MinkowskiConvolution(self.channels[4], self.chin_out, kernel_size=1, bias=True, dimension=3)
        
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, stage):

        # Yake the parameters of the input tensor
        bs, ch, X, Y, Z = x.size()
        
        x = self.block_standard(x)
        
        # Transfer into the ME sparse tensor
        x = separate_coordinates_and_features_dense(x, factor=self.factors[0])

        # Put through 1 additional ME convolutional layer to get used for the transferring
        x = self.block_new_format(x)
        
        # Propagate mixed, memorize the rest
        sftmx = self.softmax(self.classification_0(x))

        mask = torch.argmax(sftmx.F, dim=1) == (sftmx.F.size(1) - 1)
        x = self.pruning(x, mask)
        memorized_0 = self.pruning(sftmx, ~mask)

        if stage == 0:
            sftmx_dense = back_to_dense_0(sftmx, self.factors[0], (bs, self.chin_out, X, Y, Z))
        else:
            sftmx_dense = None
        
        return x, memorized_0, sftmx_dense
###################################################################################################################################
class G_1(torch.nn.Module):
    def __init__(self, chin_out, channels, factors, steps):
        super(G_1, self).__init__()
        
        self.chin_out = chin_out
        self.channels = channels
        self.factors = factors
        self.steps = steps
        
        self.block1 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[4], self.channels[4], kernel_size=2, stride=2, dimension=3, bias=True), 
            ME.MinkowskiBatchNorm(self.channels[4], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[4], self.channels[4], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[4], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[4], self.channels[4], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[4], eps=1e-04),
            ME.MinkowskiReLU(),)

        self.classification_1 = ME.MinkowskiConvolution(self.channels[4], self.chin_out, kernel_size=1, bias=True, dimension=3)
        
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, memorized_0, stage):
    
        # Upconvolution
        x = self.block1(x)
        
        # Propagate mixed, memorize the rest
        sftmx = self.softmax(self.classification_1(x))

        mask = torch.argmax(sftmx.F, dim=1) == (sftmx.F.size(1) - 1)
        x = self.pruning(x, mask)
        memorized_1 = self.pruning(sftmx, ~mask)
        
        if stage == 1:
            sftmx_dense = back_to_dense_1(sftmx, memorized_0, self.factors[1], self.steps[3], (1, self.chin_out, 64, 64, 64))
        else:
            sftmx_dense = None

        return x, memorized_1, sftmx_dense
###################################################################################################################################
class G_2(torch.nn.Module):
    def __init__(self, chin_out, channels, factors, steps):
        super(G_2, self).__init__()
        self.chin_out = chin_out
        self.channels = channels
        self.factors = factors
        self.steps = steps
        
        self.block2 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[4], self.channels[4], kernel_size=2, stride=2, dimension=3, bias=True), 
            ME.MinkowskiBatchNorm(self.channels[4], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[4], self.channels[3], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[3], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[3], self.channels[3], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[3], eps=1e-04),
            ME.MinkowskiReLU(),)

        self.classification_2 = ME.MinkowskiConvolution(self.channels[3], self.chin_out, kernel_size=1, bias=True, dimension=3)
        
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, memorized_0, memorized_1, stage):
    
        # Upconvolution
        x = self.block2(x)
        
        # Propagate mixed, memorize the rest
        sftmx = self.softmax(self.classification_2(x))

        mask = torch.argmax(sftmx.F, dim=1) == (sftmx.F.size(1) - 1)
        x = self.pruning(x, mask)
        memorized_2 = self.pruning(sftmx, ~mask)
        
        if stage == 2:
            sftmx_dense = back_to_dense_2(sftmx, memorized_1, memorized_0, self.factors[2], self.steps[3], (1, self.chin_out, 128, 128, 128))
        else:
            sftmx_dense = None

        return x, memorized_2, sftmx_dense
###################################################################################################################################
class G_3(torch.nn.Module):
    def __init__(self, chin_out, channels, factors, steps):
        super(G_3, self).__init__()
        self.chin_out = chin_out
        self.channels = channels
        self.factors = factors
        self.steps = steps
        
        self.block3 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[3], self.channels[3], kernel_size=2, stride=2, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[3], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[3], self.channels[2], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[2], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[2], self.channels[2], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[2], eps=1e-04),
            ME.MinkowskiReLU(),)
        
        self.classification_3 = ME.MinkowskiConvolution(self.channels[2], self.chin_out, kernel_size=1, bias=True, dimension=3)
        
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, memorized_0, memorized_1, memorized_2, stage):
        
        # Upconvolution
        x = self.block3(x)

        # Propagate mixed, memorize the rest
        sftmx = self.softmax(self.classification_3(x))

        mask = torch.argmax(sftmx.F, dim=1) == (sftmx.F.size(1) - 1)
        x = self.pruning(x, mask)
        memorized_3 = self.pruning(sftmx, ~mask)
        
        if stage == 3:
            sftmx_dense = back_to_dense_3(sftmx, memorized_2, memorized_1, memorized_0, self.factors[3], self.steps[3], (1, self.chin_out, 256, 256, 256))
        else:
            sftmx_dense = None

        return x, memorized_3, sftmx_dense
###################################################################################################################################
class G_4(torch.nn.Module):
    def __init__(self, chin_out, channels, factors, steps):
        super(G_4, self).__init__()
        self.chin_out = chin_out
        self.channels = channels
        self.factors = factors
        self.steps = steps

        self.block4 = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(self.channels[2], self.channels[2], kernel_size=2, stride=2, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[2], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[2], self.channels[1], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[1], eps=1e-04),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(self.channels[1], self.channels[1], kernel_size=3, dimension=3, bias=True),
            ME.MinkowskiBatchNorm(self.channels[1], eps=1e-04),
            ME.MinkowskiReLU(),)
        
        self.classification_4 = ME.MinkowskiConvolution(self.channels[1], self.chin_out, kernel_size=1, bias=True, dimension=3)
        
        self.softmax = ME.MinkowskiSoftmax(dim=1)
        self.pruning = ME.MinkowskiPruning()

    def forward(self, x, memorized_0, memorized_1, memorized_2, memorized_3):
        
        # Upconv
        x = self.block4(x)

        # Predict their values
        sftmx = self.softmax(self.classification_4(x))
        
        # Transfer to the final dense tensor
        sftmx_dense = back_to_dense_4(sftmx, memorized_3, memorized_2, memorized_1, memorized_0, self.steps[3], (1, self.chin_out, 512, 512, 512))

        return sftmx_dense
###################################################################################################################################
class ProgressiveGenerator(nn.Module):
    def __init__(self, stage0, stage1, stage2, stage3, stage4):
        super(ProgressiveGenerator, self).__init__()
        
        self.stage0 = stage0
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4

    def set_stage(self, stage):
        self.current_stage = stage
        if stage == 0:
            for param in self.stage0.parameters():
                param.requires_grad = True
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage3.parameters():
                param.requires_grad = False
            for param in self.stage4.parameters():
                param.requires_grad = False
        if stage == 1:
            for param in self.stage0.parameters():
                param.requires_grad = False
            for param in self.stage1.parameters():
                param.requires_grad = True
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage3.parameters():
                param.requires_grad = False
            for param in self.stage4.parameters():
                param.requires_grad = False
        if stage == 2:
            for param in self.stage0.parameters():
                param.requires_grad = False
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = True
            for param in self.stage3.parameters():
                param.requires_grad = False
            for param in self.stage4.parameters():
                param.requires_grad = False
        if stage == 3:
            for param in self.stage0.parameters():
                param.requires_grad = False
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage3.parameters():
                param.requires_grad = True
            for param in self.stage4.parameters():
                param.requires_grad = False 
        if stage == 4:
            for param in self.stage0.parameters():
                param.requires_grad = False
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage3.parameters():
                param.requires_grad = False
            for param in self.stage4.parameters():
                param.requires_grad = True 

    def forward(self, x):

        x, memorized_0, sftmx_dense = self.stage0(x, self.current_stage)

        if self.current_stage >= 1:
        
            x, memorized_1, sftmx_dense = self.stage1(x, memorized_0, self.current_stage)
        
        if self.current_stage >= 2:
            
            x, memorized_2, sftmx_dense = self.stage2(x, memorized_0, memorized_1, self.current_stage)
            
        if self.current_stage >= 3:
            
            x, memorized_3, sftmx_dense = self.stage3(x, memorized_0, memorized_1, memorized_2, self.current_stage)
            
        if self.current_stage == 4:
            
            sftmx_dense = self.stage4(x, memorized_0, memorized_1, memorized_2, memorized_3)
            
        return sftmx_dense

###################################################################################################################################
def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
###################################################################################################################################

# Load the state dict
state_dict = torch.load(path_to_g_weights, map_location='cuda:0')

# Remove the "module." prefix
state_dict = remove_module_prefix(state_dict)

# Create the generator
chin_out = 5  # N of minerals + 1 channel for the mixed. Here we have pore + quartz + feltspar + clay + mixed -> 5
factors = [16, 8, 4, 2]  # we have 6 because input is 32 and this is 2^5 - we go 5 steps to 1x1x1 + 1 for the initial size for the res conv
steps = [16, 8, 4, 2]  # these steps are the distance between the nodes for different levels of upsampling. For x8 we have 3 levels of upsampling
channels = [32, 64, 128, 256, 512]

# Setup G
G_stage0 = G_0(chin_out, channels, factors, steps).to(device)
G_stage1 = G_1(chin_out, channels, factors, steps).to(device)
G_stage2 = G_2(chin_out, channels, factors, steps).to(device)
G_stage3 = G_3(chin_out, channels, factors, steps).to(device)
G_stage4 = G_4(chin_out, channels, factors, steps).to(device)

model_G = ProgressiveGenerator(G_stage0, G_stage1, G_stage2, G_stage3, G_stage4).to(device)

stage = 4
model_G.set_stage(4)

G_net = model_G

# G_net.load_state_dict(torch.load(path_to_g_weights, map_location='cuda:0'))
G_net.load_state_dict(state_dict)

G_net.eval()

with torch.no_grad():  # save the images
    # 1. Start a new run
    # wandb.init(project='SuperRes', name='making large volume',
    #            entity='tldr-group')

#    step_len = int(np.round(128/scale_f, 5))
    step_len = 32
#     overlap = int(step_len/2)
#     high_overlap = int(np.round(overlap / 2 * scale_f, 5))
    high_overlap = 384
#    step = step_len - overlap
    step = 16

    BM_G = BatchMaker_16.\
        BatchMaker(stage, device=device, to_low_idx=to_low_idx, path=G_image_path, sf=scale_f, dims=n_dims, stack=False, down_sample=down_sample, 
                   low_res=not down_sample, rot_and_mir=False, squash=squash, super_sample=super_sample)
    im_3d = BM_G.all_image_batch()

    if all_pore_input:
        im_3d[:] = 0
        im_3d[:, 0] = 1

    if input_with_noise:
        input_size = im_3d.size()
        print(input_size)
        # make noise channel and concatenate it to input:
        noise = torch.randn(input_size[0], 1, *input_size[2:], device=device, dtype=im_3d.dtype)
        im_3d = torch.cat((im_3d, noise), dim=1)
        print(im_3d.size())

    nz1, nz2, nz3 = size_to_evaluate
    first_img_stack = []
    with torch.no_grad():
        last_ind1 = int(np.ceil((nz1-step_len)/step))
        for i in range(last_ind1 + 1):
            print('large step = ' + str(i))
            if i == last_ind1:
                first_lr_vec = im_3d[..., nz1-step_len:nz1, :, :]
            else:
                first_lr_vec = im_3d[..., i*step:i*step+step_len, :, :]
            second_img_stack = []
            last_ind2 = int(np.ceil((nz2-step_len)/step))
            for j in range(last_ind2 + 1):
                print('middle step = ' + str(j))
                if j == last_ind2:
                    second_lr_vec = first_lr_vec[..., :, nz2-step_len:nz2, :]
                else:
                    second_lr_vec = first_lr_vec[..., :, j * step:j * step + step_len, :]
                third_img_stack = []
                last_ind3 = int(np.ceil((nz3-step_len)/step))
                for k in range(last_ind3 + 1):
                    print('small step = ' + str(k))
                    if k == last_ind3:
                        third_lr_vec = second_lr_vec[..., :, :, nz3-step_len:nz3]
                    else:
                        third_lr_vec = second_lr_vec[..., :, :, k * step:k * step + step_len]
                    g_output = G_net(third_lr_vec)
                    g_output = g_output.detach().cpu()
                    g_output = ImageTools.fractions_to_ohe(g_output)
                    g_output_grey = ImageTools.one_hot_decoding(g_output).astype('int8').squeeze()
                    if k == 0:  # keep the beginning
                        g_output_grey = g_output_grey[:, :, :high_overlap]
                    elif k == last_ind3:  # keep the middle+end
                        g_output_grey = g_output_grey[:, :, -high_overlap:]
                    else:  # keep the middle
                        g_output_grey = g_output_grey[:, :, -high_overlap:high_overlap]
                    third_img_stack.append(np.int8(g_output_grey))
                res2 = np.concatenate(third_img_stack, axis=2)
                if j == 0:
                    res2 = res2[:, :high_overlap, :]
                elif j == last_ind2:
                    res2 = res2[:, -high_overlap:, :]
                else:
                    res2 = res2[:, -high_overlap:high_overlap, :]
                second_img_stack.append(res2)
            res1 = np.concatenate(second_img_stack, axis=1)
            if i == 0:
                res1 = res1[:high_overlap, :, :]
            elif i == last_ind1:
                res1 = res1[-high_overlap:, :, :]
            else:
                res1 = res1[-high_overlap:high_overlap, :, :]
            first_img_stack.append(res1)
    img = np.concatenate(first_img_stack, axis=0)
    low_res = np.squeeze(ImageTools.one_hot_decoding(im_3d.cpu()))
    if all_pore_input:
        imwrite(progress_main_dir + '/' + file_name + '_pore', img)
    else:
        imwrite(progress_main_dir + '/' + file_name, img)

    # also save the low-res input.
    imwrite(progress_main_dir + '/' + file_name.split('.')[0] + '_low_res.tif', low_res)
