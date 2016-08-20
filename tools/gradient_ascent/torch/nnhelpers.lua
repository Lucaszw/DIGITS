-- Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.

require 'torch'
require 'cutorch'
require 'paths'
require 'nn'
require 'cudnn'
require 'cunn'
require 'qt'
require 'image'
require 'os'

nnhelpers = {}


function nnhelpers.bestGoogLeNet()
  return {push_spatial=0.8,lr=1,decay=0.001,blur_every=4,blur_radius=1}
end
function nnhelpers.bestAlexNet()
  return {push_spatial=1,lr=1000,decay=0.001,blur_every=4,blur_radius=1}
end

function nnhelpers.loadNetwork(dir,name, weightsFile, tensorType)
  package.path = paths.concat(dir, "?.lua") ..";".. package.path
  local parameters = {
      ngpus = (tensorType =='cuda') and 1 or 0
  }
  if nn.DataParallelTable then
      -- set number of GPUs to use when deserializing model
      nn.DataParallelTable.deserializeNGPUs = 1
  end

  local network = require (name)(parameters)

  local model

  -- the full model was saved
  model = torch.load(weightsFile)
  network.model = model

  model:cuda()

  return network
end

function nnhelpers.loadModel(dir,name, weightsFile, tensorType)
  network = nnhelpers.loadNetwork(dir,name, weightsFile, tensorType)
  return network.model
end

function nnhelpers.getNetworkUntilPushLayer(model,push_layer)
  -- Create a network starting from input layer to output layer:
  local chainedNetwork = nnhelpers._generateChainedNetwork(model)
  local croppedNetwork = nil
  local layers = chainedNetwork:listModules()
  local containers = nnhelpers._getContainers(chainedNetwork:listModules())

  local mm = nn.Sequential()
  local c = containers[push_layer]

  if nnhelpers._isSequential(push_layer,layers,containers) then
    croppedNetwork = nnhelpers._cropNetwork(push_layer, chainedNetwork, containers)
  else
    local chain = layers[c]
    local local_index = push_layer - c
    local seq = nn.Sequential()

    croppedNetwork = nnhelpers._cropNetwork(containers[c]-1, chainedNetwork, containers)

    for i=1,local_index do
      croppedNetwork:add(chain:get(i))
    end
  end

  return croppedNetwork
end

function nnhelpers.generatePointGradient(model,input,unit,amplitude)
  -- Generates gradient to push back through the network
  output = model:forward(input)
  diffs = torch.Tensor(output:size()) * 0

  if output:nDimension() == 4 and output:size()[3] > 1 then
    p = {output:size()[3]/2,output:size()[4]/2}
    diffs[1][unit][p[1]][p[2]] = amplitude
  elseif output:nDimension() == 4 then
    diffs[1][unit][1][1] = amplitude
  elseif output:nDimension() == 2 then
    diffs[1][unit] = amplitude
  else
    diffs[unit] = amplitude
  end

  return diffs
end

function nnhelpers.gaussianKernel(size,radius)
  -- Create a blur effect:
  return image.gaussian({size=3,sigma=radius,normalize=true})
end

function nnhelpers.regularize(img,gradient,iteration,g,reg_params)
  -- Regularize:
  -- TODO: Add more regularization techniques
  grad = torch.cmin(torch.cmax(reg_params.lr*gradient, - 1), 1)
  img = torch.add(img:clone(),grad) * (1-reg_params.decay)
  if iteration % reg_params.blur_every == 0 then
    img[1] = image.convolve(img[1]:double(),g,'same'):cuda()
  end
  return img
end


function nnhelpers._getContainers(layers)
  -- Get array of index values for parent containers
  local containers = {}
  local layer_id = 0
  function generateChain(layer,container_id)
    layer_id = layer_id + 1
    table.insert(containers, container_id)
    if layer.modules then
      local id = layer_id
      for i=1, #layer.modules do
        child = layer:get(i)
        generateChain(child,id)
      end
    end
  end

  generateChain(layers[1],0)
  return containers
end

function nnhelpers._generateChainedNetwork(container)
  -- Chains sibling sequential containers into a single container
  local chainedNetwork = nn.Sequential();
  function chainContents(container)
    local children = container.modules
    for i=1, #children do
      local child = children[i]
      if child.modules and string.find(torch.type(child), "Sequential") then
        chainContents(child)
      else
        chainedNetwork:add(child);
      end
    end
  end
  chainContents(container)
  return chainedNetwork
end

function nnhelpers._isSequential(push_layer,layers,containers)
  -- Checks to see if a layer is part of the main sequence
  local pContainer = containers[push_layer]
  local gpContainer = containers[pContainer]

  if not gpContainer then
    -- Entire network must be sequential
    return true
  end

  if gpContainer < 1 then
    return true
  end

  return string.find(torch.type(layers[gpContainer]), "Sequential")
end

function nnhelpers._cropNetwork(push_layer,model,containers)
  -- Crops network such that it starts with the container
  -- we are doing backprop on
  local modules = model.modules
  local layers = model:listModules()

  local mm = nn.Sequential()
  local i = push_layer

  while i >  1 do
    local pContainer = containers[i]
    local gpContainer = containers[pContainer]
    if gpContainer then
      if (gpContainer ~= 0) then
        i = gpContainer
      end
    end
    mm:insert(layers[i],1)
    i = i - 1
  end

  return mm
end
