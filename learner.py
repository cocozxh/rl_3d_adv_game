import torch
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc, TotalVariation_3d


class Learner():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Yolo model:
        self.dnet = Darknet(self.config.cfgfile)
        self.dnet.load_weights(self.config.weightfile)
        self.dnet = self.dnet.eval()
        self.dnet = self.dnet.to(self.device)

        # self.output = 
        self.optimizer = torch.optim.Adam(self.dnet.parameters(), lr=1e-3)

    def reset(self):
        self.dnet = Darknet(self.config.cfgfile)
        self.dnet.load_weights(self.config.weightfile)
        self.dnet = self.dnet.eval()
        self.dnet = self.dnet.to(self.device)

    def evaluate(self, images, with_grad = False):
        images_as_tensor = torch.Tensor(images[0].cpu()).to(self.device)
        for i in range(1,len(images)):
            images_as_tensor = torch.cat([images_as_tensor,images[i]],axis = 0)
        output = self.dnet(images_as_tensor)

        d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
        number_of_detections_failed = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)

            # tv = total_variation(self.patches[0])
            # tv_loss = tv * 2.5
            
            # loss = d_loss + tv_loss

        if with_grad:
            return d_loss, number_of_detections_failed
        else:
            with torch.no_grad():
                return d_loss, number_of_detections_failed
    
    def update(self,images):
        loss,acc = self.evaluate(images, with_grad=True)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

    def update_params(self, delta_named_params):
        # delta_params = {}
        # for name in delta_named_params:
        #     params = delta_named_params[name]
        #     delta_params[name] = params.clone()

        new_params = {}
        for name, params in self.dnet.named_parameters():
            new_params[name] = params.add(torch.tensor(delta_named_params[name]).cuda())

        # leaf variable with requires_grad = True cannot used inplace operation
        for name, params in self.dnet.named_parameters():
            params.data.copy_(new_params[name])


    # from test_patch in generator.py
    def test_images(self, images):
        # run detection model in detector
        output = self.dnet(images)
        return output


