-- Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'image'
require 'lapp'
require 'nn'
require 'os'
require 'qt'
require 'torch'

require 'nnhelpers'

opt = lapp[[
-p,--type (default cuda) float or cuda
-n,--network (string) Pretrained Model to be loaded
--weights (string) model weights to load
--networkDirectory (default '') directory in which network exists
]]

local network_type = opt.type
local network = opt.network
local weights_filename = opt.snapshot
local network_directiory = opt.networkDirectory

-- GoogleNet (ImageNet) : 174, GoogLeNet(Cifar10) : 220, AlexNet(Cifar): 22
local push_layer = 174
local push_unit  = 8
local num_channels = 3

-- PARAMS:
local reg_params = nnhelpers.bestGoogLeNet()

local max_iter = 200
local input_size = {1,num_channels,227,227}
local mean_image = image.lena()*0+0.7
if num_channels == 1 then
  mean_image = mean_image[1]
end

cutorch.setDevice(1)

-- Setup Inputs
inputs = torch.Tensor(input_size[1],input_size[2], input_size[3],input_size[4])
im = image.scale(mean_image,input_size[3],input_size[4])
inputs[1] = im

-- Load Network
local model = nnhelpers.loadModel(network_directiory,network,weights_filename,network_type)
-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Color', 'model', './AlexNet(Cifar)_Color/snapshot_13_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Grey', 'model', './AlexNet(Cifar)_Grey/snapshot_30_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./GoogLeNet(Cifar)', 'model', './GoogLeNet(Cifar)/old/snapshot_60_Model.t7', 'cuda')
-- model = torch.load('./GoogLeNet(Cifar)/snapshot_60_Model.t7'):cuda()

w = image.display(im)
g = nnhelpers.gaussianKernel(reg_params.blur_radius)

-- Run Optimization
local mm = nnhelpers.getNetworkUntilPushLayer(model, push_layer):cuda()
mm:get(1).gradInput = torch.Tensor():cuda()

xx = inputs:cuda()
diffs = nnhelpers.generatePointGradient(model,xx,push_unit,reg_params.push_spatial):cuda()

for ii=1,max_iter do

  -- Run Forward with new image:
  xx = torch.cmin(torch.cmax(xx+inputs:cuda(), 0),1)-inputs:cuda()
  output = mm:forward(xx)

  -- Run backward:
  gradOutput = diffs:clone()
  gradInput = mm:backward(xx,gradOutput):clone()

  -- Apply Regularizations:
  xx = nnhelpers.regularize(xx,gradInput,ii,g,reg_params)
  image.display{image=xx, win=w}
end
