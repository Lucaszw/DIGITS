-- Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'image'
require 'nn'
require 'os'
require 'pl'
require 'qt'
require 'torch'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end

require 'nnhelpers'

opt = lapp[[
-p,--type (default cuda) float or cuda
-n,--network (string) Pretrained Model to be loaded
--weights (string) model weights to load
--networkDirectory (default '') directory in which network exists
-s,--save (default .) save directory
--height (number)
--width (number)
--layer (number)
<units...> (number) a list of units to activate
]]

local network_type = opt.type
local network = opt.network
local weights_filename = opt.weights
local network_directiory = opt.networkDirectory
local filename = paths.concat(opt.save, 'vis.h5')

-- GoogleNet (ImageNet) : 174, GoogLeNet(Cifar10) : 220, AlexNet(Cifar): 22
local push_layer = opt.layer
local units = opt.units

-- PARAMS:
local reg_params = nnhelpers.bestAlexNet()
local max_iter = 400

cutorch.setDevice(1)

-- Load Network
local model = nnhelpers.loadModel(network_directiory,network,weights_filename,network_type)

-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Color', 'model', './AlexNet(Cifar)_Color/snapshot_13_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Grey', 'model', './AlexNet(Cifar)_Grey/snapshot_30_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./GoogLeNet(Cifar)', 'model', './GoogLeNet(Cifar)/old/snapshot_60_Model.t7', 'cuda')
-- model = torch.load('/home/lzeerwanklyn/Desktop/gradientAscentPlaygroundTorch/GoogLeNet(Cifar)/snapshot_60_Model.t7'):cuda()

-- Run Optimization
local mm = nnhelpers.getNetworkUntilPushLayer(model, push_layer):cuda()

local input_layer = mm:get(1)
input_layer.gradInput = torch.Tensor():cuda()

-- Adjust Input Image to match # channels, and image size:
local channels = input_layer.nInputPlane
local height  = opt.height
local width  = opt.width

local input_size = {1,channels,height,width}
local mean_image = torch.ones(image.lena():size())*0.5

if channels == 1 then
  mean_image = 0.333 * torch.add(torch.add(mean_image[1],mean_image[2]),mean_image[3])
end

inputs = torch.Tensor(torch.LongStorage(input_size))
im = image.scale(mean_image,input_size[3],input_size[4])
inputs[1] = im

local w = image.display{image=im, min=0, max=1}

-- If no units then , solve for all units in layer:
if units[1] == 0 then
  local outputs = nnhelpers.getNumOutputs(mm,channels,height,width)
  units = {}
  for i=1,outputs do
    units[i] = i
  end
end

-- Run Optimization:
g = nnhelpers.gaussianKernel(reg_params.blur_radius)

for _,push_unit in ipairs(units) do
  local mean = inputs:clone():cuda()
  local xx   = inputs:clone():cuda()
  local diffs = nnhelpers.generatePointGradient(mm,xx,push_unit,reg_params.push_spatial):cuda()

  for ii=1,max_iter do

    -- Run Forward with new image:
    xx = torch.cmin(torch.cmax(xx+mean, 0),1)-mean
    mm:forward(xx)

    -- Run backward:
    local gradInput = mm:backward(xx,diffs)

    -- Apply Regularizations:
    xx = nnhelpers.regularize(xx,gradInput,ii,g,reg_params)
    image.display{image=xx, win=w}

  end

end
