require 'nn'
require 'cunn'
require 'cudnn'

-- currently support 12 views 
-- inputs will be Tensors of size: bz x 12 x C x H x W
encoder = nn.Sequential() 
encoder:add(nn.SpatialConvolution(1, 96, 11, 11, 4, 4))
encoder:add(nn.ReLU(true)) 

mv_share_net = nn.ParallelTable()
-- siamese style 
mv_share_net:add(encoder) 
p, pg = mv_share_net:getParameters()

for k = 1, 11 do 
    mv_share_net:add(encoder:clone('weight','bias', 'gradWeight','gradBias'))
end 

-- p1, pg1 = mv_share_net:getParameters()