require 'torch'
require 'image'

local W = torch.load('pretrained.t7').W
W = W:reshape(32*4,8,8)

local F = image.toDisplayTensor{input=W, padding=1, nrow=4, symmetric=true}
image.save('filters.png', F)
