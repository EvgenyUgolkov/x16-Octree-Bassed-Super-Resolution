import LearnTools
from BatchMaker_16 import *
import os
import time
import random
import math
import torch.optim as optim
import torch.utils.data
import argparse
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
modes = ['bilinear', 'trilinear']
import torch.nn.functional as F

import MinkowskiEngine as ME
from modules_ME_2_16 import *

from torch.cuda.amp import GradScaler, autocast

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# The command below help to process data with smaller chunks and prevents OOM error
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Change the directory to pick up training dataset
if os.getcwd().endswith('code'):
    os.chdir('..')  

#################################################################
# All variables initialization
#################################################################

# Parsing arguments:
parser = argparse.ArgumentParser()

args = LearnTools.return_args(parser)

progress_dir, wd, wg = args.directory, args.widthD, args.widthG
n_res_blocks, pix_distance = args.n_res_blocks, args.pixel_coefficient_distance
num_epochs, g_update, n_dims = args.num_epochs, args.g_update, args.n_dims
squash, phases_to_low = args.squash_phases, args.phases_low_res_idx
D_dimensions_to_check, scale_f = args.d_dimensions_to_check, args.scale_factor
rotation, anisotropic = args.with_rotation, args.anisotropic
rotations_bool, down_sample = args.rotations_bool, args.down_sample
separator, super_sampling = args.separator, args.super_sampling

if not os.path.exists(ImageTools.progress_dir + progress_dir):
    os.makedirs(ImageTools.progress_dir + progress_dir)

# Where to save the trained model
PATH_G = 'progress/' + progress_dir + '/g_weights.pth'
PATH_D = 'progress/' + progress_dir + '/d_weights.pth'
eta_file = 'eta.npy'

# Root directory for dataset
dataroot = "data/"

# The path to the training dataset
D_images = [dataroot + d_path for d_path in args.d_image_path]
G_image = dataroot + args.g_image_path

# G and D slices to choose from
g_batch_slices = [0]  # in 3D different views of the cube, better to keep it as

# adding 45 degree angle instead of z axis slices
forty_five_deg = False

# The number of GPUs
ngpu = 2

# Number of HR number of phases:
nc_d = 5

# Batch size for G, parameter determying the nunber of epochs, and batch soze for D
batch_size_G_for_D, batch_size_G, batch_size_D = 1, 32, 64

# number of iterations in each epoch
epoch_iterations = 10000 // batch_size_G

# Learning rate for optimizers
lr = 0.0001

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Learning parameter for gradient penalty
Lambda = 10

# When to save progress
saving_num = 50

def main():
    config = parser.parse_args()
    config.world_size = ngpu 
    mp.spawn(main_worker, nprocs=ngpu, args=(ngpu, config))

def main_worker(gpu, ngpus_per_node, args):

    ###################################################################################################################################
    def plot_intermediate(g_input_plot, output_plot):
        """
        Plots the input and generated images each epoch
        """
        g_input_plot = ImageTools.one_hot_decoding(np.array(g_input_plot[:, :-1].detach().cpu()))
        output_plot = ImageTools.one_hot_decoding(ImageTools.fractions_to_ohe(np.array(output_plot.detach().cpu())))
        images = [g_input_plot, output_plot]
    
        f, axarr = plt.subplots(2, 1)
        for i in range(2):
            for j in range(1):
                length_im = images[i].shape[1]
                middle = int(length_im/2)
                axarr[i].imshow(images[i][j, middle, :, :], cmap='gray', vmin=0, vmax=4)
                axarr[i].set_xticks([0, length_im-1])
                axarr[i].set_yticks([0, length_im-1])
        plt.suptitle('Stage 3 slides comparison')
        wandb.log({"running slices": plt})
        plt.close()
    ###################################################################################################################################
    def save_differences_and_metrics(input_to_g, output_of_g, down, save_dir, filename, hr_metrics, generator, with_deg=False):
        """
        Saves the image of the differences between the high-res real and the
        generated images that are supposed to be similar.
        """
        images = [input_to_g.clone().detach().cpu()]
        g_output = output_of_g.cpu()
        down = down.detach().cpu()
        # metrics_loss = ImageTools.log_metrics(g_output, hr_metrics)
        # if metrics_loss < 0.015:  # mean difference is smaller than 1.5%
        #     difference_str = str(np.round(metrics_loss, 4))
        #     torch.save(generator.state_dict(), PATH_G + difference_str)
        #     wandb.save(PATH_G + difference_str)
        images = images + [down, g_output]
        ImageTools.plot_fake_difference(images, save_dir, filename, with_deg)
    ###################################################################################################################################
    # Stage 1
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
    # Stage 2
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
    # Stage 3
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
    # Stage 4
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
    # Stage 5
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
    # The class to combine 5 Stages together
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
    # D for Stage 1
    class D_0(nn.Module):
        def __init__(self):
            super(D_0, self).__init__()
            self.from_0 = nn.Conv2d(5, 512, 1, 1, 0)
            self.conv1 = nn.Conv2d(512, 512, 3, 2, 1)
            self.conv2 = nn.Conv2d(512, 512, 3, 2, 1)
            self.conv3 = nn.Conv2d(512, 512, 3, 2, 1)
            self.conv_cls = nn.Conv2d(512, 512, 4, 1, 0)
            self.linear = nn.Linear(512, 1)
    
        def forward_layers(self, x):
            x = nn.ReLU()(self.conv1(x))
            x = nn.ReLU()(self.conv2(x))
            x = nn.ReLU()(self.conv3(x))
            x = nn.ReLU()(self.conv_cls(x))
            x = torch.flatten(x, start_dim=1)
            x = self.linear(x)
            return x  
    
        def forward(self, x):
            x = nn.ReLU()(self.from_0(x))
            x = self.forward_layers(x)
            return x
    ###################################################################################################################################
    # D for Stage 2
    class D_1(nn.Module):
        def __init__(self):
            super(D_1, self).__init__()
            self.d0 = D_0()
            self.d0.load_state_dict(torch.load('D_0_pretrained_95.pth'))
            self.from_1 = nn.Conv2d(5, 256, 1, 1, 0)
            self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
            self.alpha = 0
    
        def forward_layers(self, x):
            x = nn.ReLU()(self.conv4(x))
            x = self.d0.forward_layers(x)
            return x  
    
        def forward(self, x):
            if self.alpha < 1:
                # Previous stage
                new_x = int(x.size()[2]/2)
                new_y = int(x.size()[3]/2)
                x_pred = F.interpolate(x, size=(new_x, new_y), mode='nearest')
                x_pred = nn.ReLU()(self.d0.from_0(x_pred))
                # Current stage
                x = nn.ReLU()(self.from_1(x))
                x = nn.ReLU()(self.conv4(x))
                # Combining together
                x = self.alpha * x + (1 - self.alpha) * x_pred
                # Running the rest
                x = self.d0.forward_layers(x)
            else:
                x = nn.ReLU()(self.from_1(x))
                x = self.forward_layers(x)
            return x

        def set_alpha(self, epoch, transition_epochs=20):
            self.alpha = min(epoch / transition_epochs, 1.0) 
    ###################################################################################################################################
    # D for Stage 3
    class D_2(nn.Module):
        def __init__(self):
            super(D_2, self).__init__()
            self.d1 = D_1()
            self.d1.load_state_dict(torch.load('D_1_pretrained_95.pth'))
            self.from_2 = nn.Conv2d(5, 128, 1, 1, 0)
            self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
            self.alpha = 0
            
        def forward_layers(self, x):
            x = nn.ReLU()(self.conv5(x))
            x = self.d1.forward_layers(x)
            return x  
            
        def forward(self, x):
            if self.alpha < 1:
                # Previous stage
                new_x = int(x.size()[2]/2)
                new_y = int(x.size()[3]/2)
                x_pred = F.interpolate(x, size=(new_x, new_y), mode='nearest')
                x_pred = nn.ReLU()(self.d1.from_1(x_pred))
                # Current stage
                x = nn.ReLU()(self.from_2(x))
                x = nn.ReLU()(self.conv5(x))
                # Combining together
                x = self.alpha * x + (1 - self.alpha) * x_pred
                # Running the rest
                x = self.d1.forward_layers(x)
            else:
                x = nn.ReLU()(self.from_2(x))
                x = self.forward_layers(x)
            return x

        def set_alpha(self, epoch, transition_epochs=20):
            self.alpha = min(epoch / transition_epochs, 1.0) 
    ###################################################################################################################################
    # D for Stage 4
    class D_3(nn.Module):
        def __init__(self):
            super(D_3, self).__init__()
            self.d2 = D_2()
            self.d2.load_state_dict(torch.load('D_2_pretrained_95.pth'))
            self.from_3 = nn.Conv2d(5, 64, 1, 1, 0)
            self.conv6 = nn.Conv2d(64, 128, 3, 2, 1)
            self.alpha = 0
    
        def forward_layers(self, x):
            x = nn.ReLU()(self.conv6(x))
            x = self.d2.forward_layers(x)
            return x  
    
        def forward(self, x):
            if self.alpha < 1:
                # Previous stage
                new_x = int(x.size()[2]/2)
                new_y = int(x.size()[3]/2)
                x_pred = F.interpolate(x, size=(new_x, new_y), mode='nearest')
                x_pred = nn.ReLU()(self.d2.from_2(x_pred))
                # Current stage
                x = nn.ReLU()(self.from_3(x))
                x = nn.ReLU()(self.conv6(x))
                # Combining together
                x = self.alpha * x + (1 - self.alpha) * x_pred
                # Running the rest
                x = self.d2.forward_layers(x)
            else:
                x = nn.ReLU()(self.from_3(x))
                x = self.forward_layers(x)
            return x

        def set_alpha(self, epoch, transition_epochs=20):
            self.alpha = min(epoch / transition_epochs, 1.0) 
    ###################################################################################################################################
    # D for Stage 5
    class D_4(nn.Module):
        def __init__(self):
            super(D_4, self).__init__()
            self.d3 = D_3()
            self.d3.load_state_dict(torch.load('D_3_pretrained_95.pth'))
            # self.from_4 = nn.Conv2d(5, 32, 1, 1, 0)
            # self.conv7 = nn.Conv2d(32, 64, 3, 2, 1)
            self.conv7 = nn.Conv2d(5, 64, 3, 2, 1)
            self.alpha = 0
    
        def forward(self, x):
            if self.alpha < 1:
                # Previous stage
                new_x = int(x.size()[2]/2)
                new_y = int(x.size()[3]/2)
                x_pred = F.interpolate(x, size=(new_x, new_y), mode='nearest')
                x_pred = nn.ReLU()(self.d3.from_3(x_pred))
                # Current stage
                x = nn.ReLU()(self.conv7(x))
                # Combining together
                x = self.alpha * x + (1 - self.alpha) * x_pred
                # Running the rest
                x = self.d3.forward_layers(x)
            else:
                x = nn.ReLU()(self.conv7(x))
                x = self.d3.forward_layers(x)
            return x

        def set_alpha(self, epoch, transition_epochs=20):
            self.alpha = min(epoch / transition_epochs, 1.0) 
    ###################################################################################################################################
    def generate_fake_image(stage, model_G, detach_output=True, batch_size=batch_size_G_for_D):
        """
        :param detach_output: to detach the tensor output from gradient memory.
        :param same_seed: generate the same random seed.
        :param batch_size: the batch size of the fake images.
        :return: the generated image from G
        """
        # Generate batch of G's input:
        g_slice = random.choice(g_batch_slices)
        input_to_G = BM_G.random_batch_for_fake(batch_size, g_slice)
        input_size = input_to_G.size()
        # make noise channel and concatenate it to input:
        noise = torch.randn(input_size[0], 1, *input_size[2:], device=device)
        input_to_G = torch.cat((input_to_G, noise), dim=1)
    
        # Generate fake image batch with G
        with_edges = model_G(input_to_G)
        torch.cuda.empty_cache()
        if detach_output:
            return input_to_G, with_edges.detach()
        else:
            return input_to_G, with_edges
    ###################################################################################################################################
    def take_fake_slices(fake_image, perm_idx):
        """
        :param fake_image: The fake image to slice at all directions.
        :param perm_idx: The permutation index for permutation before slicing.
        :return: batch of slices from the 3d image (if 2d image,
        just returns the image)
        """
        perm = perms_3d[perm_idx]
        # permute the fake output of G to make it into a batch
        # of images to feed D (each time different axis)
        fake_slices_for_D = fake_image.permute(0, perm[0], 1, *perm[1:])
        # the new batch size feeding D:
        batch_size_new = batch_size_G_for_D * BM_D.high_l
        # reshaping for the correct size of D's input
        return fake_slices_for_D.reshape(batch_size_new, nc_d, BM_D.high_l, BM_D.high_l)
    ###################################################################################################################################
    
    # ########################################################################################
    # # Print raw gradients for each layer directly
    # print("Gradients before everything:")
    # for name, param in model_G.named_parameters():
    #     if param.grad is not None:
    #         print(f'{name}: {param.grad}')
    #     else:
    #         print(f'{name}: None')
    # ########################################################################################

    ###################################################################################################################################
    # WGAN-GP training algorithm 
    def train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage):
        
        scalerG = GradScaler(growth_interval=100, init_scale=8192)
        min_scale = 1
        # scalerD = GradScaler()
    
        for epoch in range(num_epochs):
    
            j = np.random.randint(steps)  # to see different slices
            for i in range(steps):
    
                #######################
                # (1) Update D network:
                #######################
                _, fake_for_d = generate_fake_image(stage, model_G, detach_output=True)
    
                for k in range(math.comb(n_dims, 2)):
                    # Train with all-real batch
                    netD.zero_grad()

                    with autocast():
    
                        # Batch of real high res for D
                        high_res = BM_D.random_batch_for_real(batch_size_D)
        
                        # Forward pass real batch through D
                        output_real = netD(high_res).view(-1).mean()
        
                        # obtain fake slices from the fake image
                        fake_slices = take_fake_slices(fake_for_d, k)
        
                        # Classify all fake batch with D
                        output_fake = netD(fake_slices).view(-1).mean()
        
                        min_batch = min(high_res.size()[0], fake_slices.size()[0])
                        
                        fake_slices = fake_slices.to(device)
                        # Calculate gradient penalty
                        gradient_penalty = LearnTools.calc_gradient_penalty_MP(netD, high_res[:min_batch], fake_slices[:min_batch], 
                                                                            batch_size_D, BM_D.high_l, device, Lambda, nc_d, scalerD)
        
                        # discriminator is trying to minimize:
                        d_cost = output_fake - output_real + gradient_penalty
                    
                    # Calculate gradients for D in backward pass
                    # d_cost.backward()
                    # optimizerD.step()
                    scalerD.scale(d_cost).backward()
                    # if stage == 4:
                    #     scalerD.unscale_(optimizerD)
                    #     torch.nn.utils.clip_grad_norm_(netD.parameters(), 2.5)
                    scalerD.step(optimizerD)
                    scalerD.update()
                    if scalerD._scale < min_scale:
                        scalerD._scale = torch.tensor(min_scale).to(scalerD._scale)
    
                    wass = abs(output_fake.item() - output_real.item())
    
                del _
                del fake_for_d
                del high_res
                del fake_slices
                torch.cuda.empty_cache()
    
                #######################
                # (2) Update G network:
                #######################
    
                if (i % g_update) == 0:
                    
                    model_G.zero_grad()
                    # generate fake again to update G:
                    low_res, fake_for_g = generate_fake_image(stage, model_G, detach_output=False)
                    # save the cost of g to add from each axis:
                    g_cost = 0

                    with autocast():
                        # go through each axis
                        for k in range(math.comb(n_dims, 2)):
                            fake_slices = take_fake_slices(fake_for_g, k)
                            # perform a forward pass of all-fake batch through D
                            fake_output = netD(fake_slices).view(-1).mean()
        
                            if k == 0:
                                wandb.log({'yz_slice': fake_output})
                            if k == 1:
                                wandb.log({'xz_slice': fake_output})
           
                            low_res_without_noise = low_res[:, :-1]  # without noise
                            pix_loss = down_sample_objects[stage].voxel_wise_distance(fake_for_g, low_res_without_noise)
                            if pix_loss.item() > 0.05:
                                g_cost += -fake_output + pix_distance * pix_loss
                            else:
                                g_cost += -fake_output
    
                    del low_res
                    del fake_for_g
                    del fake_slices
                    del low_res_without_noise
                    torch.cuda.empty_cache()
                    
                    # Calculate gradients for G
                    # g_cost.backward()
                    # # Update G
                    # optimizerG.step()
                    scalerG.scale(g_cost).backward()
                    # if stage == 4:
                    #     scalerG.unscale_(optimizerG)
                    #     torch.nn.utils.clip_grad_norm_(model_G.parameters(), 2.5)
                    scalerG.step(optimizerG)
                    scalerG.update()
                    if scalerG._scale < min_scale:
                        scalerG._scale = torch.tensor(min_scale).to(scalerG._scale)

                    if stage == 4:
                        # wandb.log({"pixel distance": pix_loss})
                        wandb.log({"wass": wass})
                        wandb.log({"real": output_real, "fake": output_fake})
                    torch.cuda.empty_cache()
    
                    # ########################################################################################
                    # # Print raw gradients for each layer directly
                    # print("Gradients after iteration:")
                    # for name, param in model_G.named_parameters():
                    #     if param.grad is not None:
                    #         print(f'{name}: {param.grad}')
                    #     else:
                    #         print(f'{name}: None')
                    # ########################################################################################
                
                if i == j:
                    with torch.no_grad():  # only for plotting
                        g_input_plot, output_plot = generate_fake_image(stage, model_G, detach_output=True, batch_size=1)
                        plot_intermediate(g_input_plot, output_plot)
                        del g_input_plot
                        del output_plot
                        torch.cuda.empty_cache()
    
                # # Output training stats
                # if i == j and stage == 4:
                #     ImageTools.calc_and_save_eta(steps, time.time(), start, i, epoch, num_epochs, eta_file)
    
                #     with torch.no_grad():  # only for plotting
                #         g_input_plot, for_down = generate_fake_image(model_G, detach_output=True, batch_size=2)
    
                #         downsampled = down_sample_object(for_down)     ##################################
    
                #         # plot input without the noise channel
                #         save_differences_and_metrics\
                #             (g_input_plot[:, :-1], for_down, downsampled, progress_dir, 'running slices', hr_slice_metrics, model_G, forty_five_deg)
                        
                # print(i, j)
                
                torch.cuda.empty_cache()                                                       ############################################################
        
            print('Stage:', stage, ', Epoch:', epoch)
            
            if (epoch % 3) == 0 and stage == 4:
                torch.save(model_G.state_dict(), PATH_G)
                wandb.save(PATH_G)
                if (epoch % 60) == 0:
                    PATH_G_wo_ext = PATH_G.split('.')[0]
                    torch.save(model_G.state_dict(), PATH_G_wo_ext + str(epoch) + '.pth')
                    wandb.save(PATH_G_wo_ext + str(epoch) + '.pth')
                
            if stage > 0:
                netD.module.set_alpha(epoch)                                                    ############################################################
        
        print('Finished this Stage')

    # Initiate the DDP training
    args.gpu = gpu
    args.rank = 0 * ngpus_per_node + gpu
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:23095",
        world_size=args.world_size,
        rank=args.rank)

    torch.cuda.set_device(args.gpu)

    seed = 333 + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Decide which device we want to run on
    device = torch.device('cuda')
    print('device is ' + str(device))
    
    # the material indices to low-res:
    to_low_idx = torch.LongTensor(phases_to_low).to(device)
    
    # 1. Start a new run
    print('Initiating the project')
    print(progress_dir)
    # wandb.init(project='SuperRes', config=args, name=progress_dir,entity='tldr-group')
    wandb.init(project='SuperRes', config=args, name=progress_dir)
    print('The project was initiated OK')
    
    # Create the down-sample object to compare between super-res and low-res for each Stage
    down_sample_object = LearnTools.DownSample(squash, n_dims, to_low_idx, scale_f, device, super_sampling, separator).to(device)
    scale_f_1 = 1
    scale_f_2 = 2
    scale_f_3 = 4
    scale_f_4 = 8
    scale_f_5 = 16
    to_low_idx_interm = torch.LongTensor([]).to(device)
    down_sample_object_1 = LearnTools.DownSample_2(squash, n_dims, to_low_idx_interm, scale_f_1, device, super_sampling, separator).to(device)
    down_sample_object_2 = LearnTools.DownSample_2(squash, n_dims, to_low_idx_interm, scale_f_2, device, super_sampling, separator).to(device)
    down_sample_object_3 = LearnTools.DownSample_2(squash, n_dims, to_low_idx_interm, scale_f_3, device, super_sampling, separator).to(device)
    down_sample_object_4 = LearnTools.DownSample_2(squash, n_dims, to_low_idx_interm, scale_f_4, device, super_sampling, separator).to(device)
    down_sample_object_5 = LearnTools.DownSample_2(squash, n_dims, to_low_idx_interm, scale_f_5, device, super_sampling, separator).to(device)
    down_sample_objects = [down_sample_object_1, down_sample_object_2, down_sample_object_3, down_sample_object_4, down_sample_object_5]  
    
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
    model_G = torch.nn.parallel.DistributedDataParallel(model_G, device_ids=[args.gpu], find_unused_parameters=True)
    model_G = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model_G)
    
    # optimizerG = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, 0.999))                       ### exactly this one was the same for all before ###
    
    # Track it in the weight and biases
    wandb.watch(model_G, log='all')
    
    # Batch Maker for the Generator
    stage=0
    BM_G = BatchMaker(stage, device=device, to_low_idx=to_low_idx, path=G_image, sf=scale_f, dims=n_dims, stack=False, down_sample=down_sample, 
                      low_res=not down_sample, rot_and_mir=False, squash=squash)

    # scalerG = GradScaler()
    # Scaler optimization for the Mixed Precision training. Tricky and smart optimization to prevent Mixed Precision artefacts 
    scalerD = GradScaler(growth_interval=100, init_scale=8192)
    min_scale = 1
    
    steps = epoch_iterations
    print("Starting Training...")
    start = time.time()

    # Stage 0
    print('Running Stage 0')
    stage = 0
    num_epochs = 100
    model_G.module.set_stage(0)
    optimizerG = optim.AdamW(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    netD = D_0().to(device)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    BM_D = BatchMaker(stage, device, path=D_images[0], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=True)
    torch.cuda.empty_cache()
    train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage)
    torch.save(netD.module.state_dict(), 'D_0_pretrained_95.pth')
    
    # Stage 1
    print('Running Stage 1')
    stage = 1
    num_epochs = 100
    model_G.module.set_stage(1)
    optimizerG = optim.AdamW(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)               
    netD = D_1().to(device)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    BM_D = BatchMaker(stage, device, path=D_images[1], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=True)
    torch.cuda.empty_cache()
    train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage)
    torch.save(netD.module.state_dict(), 'D_1_pretrained_95.pth')
    
    # Stage 2
    print('Running Stage 2')
    stage = 2
    num_epochs = 100
    model_G.module.set_stage(2)
    optimizerG = optim.AdamW(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    netD = D_2().to(device)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
    optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    BM_D = BatchMaker(stage, device, path=D_images[2], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=True)
    torch.cuda.empty_cache()
    train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage)
    torch.save(netD.module.state_dict(), 'D_2_pretrained_95.pth')
    
    # Stage 3
    lr2 = 0.00001
    print('Running Stage 3')
    stage = 3
    num_epochs = 100
    model_G.module.set_stage(3)
    optimizerG = optim.AdamW(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr2, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    netD = D_3().to(device)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
    optimizerD = optim.AdamW(netD.parameters(), lr=lr2, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    BM_D = BatchMaker(stage, device, path=D_images[3], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=True)
    torch.cuda.empty_cache()
    train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage)
    torch.save(netD.module.state_dict(), 'D_3_pretrained_95.pth')
    
    # Stage 4
    lr2 = 0.00001
    print('Running Stage 4')
    stage = 4
    num_epochs = 100
    model_G.module.set_stage(4)
    optimizerG = optim.AdamW(filter(lambda p: p.requires_grad, model_G.parameters()), lr=lr2, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)               
    netD = D_4().to(device)
    netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=[args.gpu], find_unused_parameters=True)
    optimizerD = optim.AdamW(netD.parameters(), lr=lr2, betas=(beta1, 0.999), eps=1e-04, weight_decay=0.05)
    BM_D = BatchMaker(stage, device, path=D_images[4], sf=scale_f, dims=n_dims, stack=True, low_res=False, rot_and_mir=True)
    # Volume Fraction and Surface Area high-res metrics - herewe need it because it is the final stage:
    hr_slice_metrics = BM_D.hr_metrics
    torch.cuda.empty_cache()
    print('Starting actually training Stage 4')
    train_progressive_stage(num_epochs, model_G, netD, optimizerD, BM_D, stage)
    
    print('All stages were done')

if __name__ == "__main__":
    main()

# print(dir(self.stage3.block2[0]))
# print('The device of the kernel is', self.stage3.block2[0].kernel.device)

# allocated = torch.cuda.memory_allocated()
# reserved = torch.cuda.memory_reserved()
# print('1')
# print(f"Allocated GPU memory: {allocated / (1024**3):.2f} GB")
# print(f"Reserved GPU memory: {reserved / (1024**3):.2f} GB")