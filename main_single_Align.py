import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
import linklink as link

class LabelSmoothCELoss(_Loss):
    def __init__(self, smooth_ratio, num_classes):
        super(LabelSmoothCELoss, self).__init__()
        self.smooth_ratio = smooth_ratio
        self.v = self.smooth_ratio / num_classes

    def forward(self, input, label):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.v)
        y = label.to(torch.long).view(-1, 1)
        one_hot.scatter_(1, y, 1-self.smooth_ratio+self.v)

        loss = - torch.sum(F.log_softmax(input, 1) *
                           (one_hot.detach())) / input.size(0)
        return loss


class ClipInfoCELoss(_Loss):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()

    def forward(self, logits_per_image, logits_per_text):
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i+loss_t)/2
        return loss, labels

def D(p, z):
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    return (p * z).sum(dim=1).mean()

def D_minimize(p, z):
    z = z.detach()
    p = p / p.norm(dim=-1, keepdim=True)
    z = (z / z.norm(dim=-1, keepdim=True)).permute(0, 2, 1)
    sim = torch.bmm(p, z)
    return sim.max(dim=-1)[0].mean(dim=-1).mean()


class SimsiamLoss(nn.Module):
    def __init__(self, symmetry=True):
        super(SimsiamLoss, self).__init__()
        self.symmetry = symmetry

    def forward(self, p1, z1, p2, z2, minimize_loss=False):
        if self.symmetry:
            if minimize_loss:
                D1 = D_minimize(p1, z2)
                D2 = D_minimize(p2, z1)
                return -0.5 * (D1.mean() + D2.mean())
            else:
                D1 = D(p1, z2)
                D2 = D(p2, z1)
                return -0.5 * (D(p1, z2)  + D(p2, z1) )


class AlignCLIP(CLIP):
    def __init__(self, image_encode, text_encode, use_allgather, nn_size=2**16, nn_topk=1, \
                 return_dense=False, return_caption=False, return_nn_bank=False, text_mask_type=None, \
                 EDA=True, feature_dim=1024, embed_dim=768, forward_type='split', dense_mapping_image=2048, \
                 dense_mapping_language=512, dense_embed_dim=256, mask_rate=0.75, patch_number=14, \
                 text_mae_feature=False, return_simsiam=False, two_view=False, sparse=False, select_topk=False):
        super(AlignCLIP, self).__init__(image_encode, text_encode, use_allgather)
        self.return_dense = return_dense
        self.return_caption = return_caption

        self.text_mask_type = text_mask_type
        self.select_topk = select_topk
        if self.return_dense:
            self.image_mapping = nn.Linear(dense_mapping_image, dense_embed_dim)
            self.text_mapping = nn.Linear(dense_mapping_language, dense_embed_dim)

        self.logit_scale_dense = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_dense, np.log(1/0.07))

        if self.encode_text.text_encode_type == 'Transformer':
            self.sos_index = self.encode_text.tokenizer.encoder["<|startoftext|>"]
            self.padding_idx = 0
        if self.return_caption:
            self.caption_module = TransformerDecoderTextualHead(visual_feature_size=2048, vocab_size=self.encode_text.vocab_size, padding_idx=self.padding_idx)
        else:
            self.caption_module = None
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)

    def encode_text_dense(self, texts, return_dense=True):
        text_features, word_features = self.encode_text(texts, return_dense=return_dense)
        word_features_d = self.text_mapping(word_features)
        return word_features_d

    def encode_image_dense(self, image):
        image_features, image_features_dense = self.visual(image.type(self.dtype), return_dense=True)
        image_features_dense = self.image_mapping(image_features_dense)
        return image_features_dense

    def encode_image(self, image, return_all=False):
        output = self.visual(image.type(self.dtype), return_dense=return_all)
        return output

    def get_weighted_dense_logits(self, dense_feat_1, dense_feat_2, top_k=16):
        dense_feat_1 = dense_feat_1 / dense_feat_1.norm(dim=-1, keepdim=True)
        dense_feat_2 = dense_feat_2 / dense_feat_2.norm(dim=-1, keepdim=True)

        logit_scale_dense = self.logit_scale_dense.exp()

        if self.select_topk:
            dense_feat_cross_logit = torch.matmul(dense_feat_1, dense_feat_2.permute(0, 2, 1))
            _, dense_id_1 = torch.topk(dense_feat_cross_logit.sum(dim=2), dim=1, k=top_k)
            _, dense_id_2 = torch.topk(dense_feat_cross_logit.sum(dim=1), dim=1, k=top_k)
            bs, n1 = dense_feat_1.shape[:2]
            dense_id_1 = dense_id_1 + (torch.arange(bs) * n1).to(dense_id_1.device)[:, None]
            selected_feat_1 = dense_feat_1.reshape(bs * n1, -1)[dense_id_1].reshape(bs, top_k, -1)
            bs, n2 = dense_feat_2.shape[:2]
            dense_id_2 = dense_id_2 + (torch.arange(bs) * n2).to(dense_id_2.device)[:, None]
            selected_feat_2 = dense_feat_2.reshape(bs * n2, -1)[dense_id_2].reshape(bs, top_k, -1)

        selected_feat_1 = self.all_gather(selected_feat_1)
        selected_feat_2 = self.all_gather(selected_feat_2)

        def get_logits(dense_feat_1, selected_feat_2):
            i, j, k = dense_feat_1.shape
            l, m, k = selected_feat_2.shape
            dense_feat_1 = dense_feat_1.reshape(-1, k)
            selected_feat_2 = selected_feat_2.reshape(-1, k)
            final_logits_1 = logit_scale_dense * dense_feat_1 @ selected_feat_2.t()
            final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0,2,1,3)
            return final_logits_1
        final_logits_1 = get_logits(dense_feat_1, selected_feat_2).max(dim=-1)[0].mean(dim=-1)
        final_logits_2 = get_logits(dense_feat_2, selected_feat_1).max(dim=-1)[0].mean(dim=-1)
        return final_logits_1, final_logits_2

    def forward(self, input, return_dict=False):
        images = input['images']
        images_1, _ = torch.split(images, [3,3], dim=1)
        texts = input['captions']
        texts = self.sample_captions(texts)
        text_features, word_features, text_labels = self.encode_text(texts, mask_type = self.text_mask_type)
        image_concat = images_1
        image_features_1, image_features_d = self.encode_image(image_concat, return_all=True)
        logit_scale = self.logit_scale.exp()
        image_features_1 = image_features_1 / (image_features_1.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)
        if self.training and self.use_allgather:
            link.barrier()
            gathered_image_features_1 = self.all_gather(image_features_1)
            gathered_text_features = self.all_gather(text_features)
            logits_per_image_1 = logit_scale * image_features_1 @ gathered_text_features.t()
            logits_per_text_1 = logit_scale * text_features @ gathered_image_features_1.t()
        if self.return_dense:
            image_features_d1 = image_features_d
            image_features_d1 = self.image_mapping(image_features_d1)
            word_features_d = self.text_mapping(word_features)
            logits_per_image_dense_1, logits_per_text_dense_1 = self.get_weighted_dense_logits(image_features_d1, word_features_d)
        if return_dict:
            ret_dict = {}
            ret_dict['logits'] = logits_per_image_1, logits_per_text_1
            if self.return_dense:
                ret_dict['dense_logits'] = logits_per_image_dense_1, logits_per_text_dense_1
            return ret_dict
        raise NotImplementedError()


def alignclip_res50(**kwargs):
    image_encode = modified_resnet_R50(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = AlignCLIP(image_encode,text_encode,**kwargs['clip'], dense_mapping_image=2048)
    return model


def alignclip_vitb32(**kwargs):
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = AlignCLIP(image_encode,text_encode,**kwargs['clip'], dense_mapping_image=768)
    return model