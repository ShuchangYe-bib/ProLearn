import json
import os
import pickle
from tqdm import tqdm
import torch
from einops import repeat
from monai.transforms import (Compose, NormalizeIntensityd, RandZoomd, MapTransform,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision.transforms.functional import rgb_to_grayscale

class MMSegDataset(Dataset):

    def __init__(self, 
        data=None, 
        ann_path=None, 
        root_path=None, 
        tokenizer=None, 
        mode='train', 
        aug=False,
        image_size=[224,224], 
        vision_encoder = None,
        text_encoder = None,
        lazy=True
    ):

        super(MMSegDataset, self).__init__()

        self.mode = mode
        self.aug = aug

        self.image_list = []
        self.mask_list = []
        self.caption_list = []
        self.pseudolabel_list = []
        with open(ann_path, 'r') as anno_file:
            self.annotations = json.loads(anno_file.read())[mode]
        for anno in self.annotations:
            self.image_list.append("images/{}.png".format(anno))
            self.mask_list.append("masks/{}.png".format(anno))
            # self.image_list.append("images/{}.jpg".format(anno))
            # self.mask_list.append("masks/{}.jpg".format(anno))
            self.caption_list.append("reports/{}.txt".format(anno))
            self.pseudolabel_list.append("pseudolabels/{}.json".format(anno))

        self.root_path = root_path
        self.image_size = image_size

        self.lazy = lazy
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    def __len__(self):
        # return 64
        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, self.image_list[idx])
        mask = os.path.join(self.root_path, self.mask_list[idx])
        text = os.path.join(self.root_path, self.caption_list[idx])
        pseudolabel = os.path.join(self.root_path, self.pseudolabel_list[idx])
        with open(text, 'r') as file:
            report = file.read()
        try:
            with open(pseudolabel, 'r') as file:
                pseudolabel = json.load(file)
                pseudolabel = {key: torch.tensor(value) for key, value in pseudolabel.items()}
        except:
            pseudolabel = None

        data = {'image':image, 'mask':mask}
        data = trans(data)

        image, mask = data['image'], rgb_to_grayscale(data['mask'])
        mask = torch.where(mask > 0, 1, 0)

        if self.lazy:
            try:
                tensor_path = os.path.join(self.root_path, "precompute", self.mode)
                with open(os.path.join(tensor_path, f"{self.annotations[idx]}_vision.pkl"), 'rb') as f:
                    image_emb = pickle.load(f)
                with open(os.path.join(tensor_path, f"{self.annotations[idx]}_text.pkl"), 'rb') as f:
                    report_emb = pickle.load(f)
                return {
                    "image": image,
                    "text": report,
                    "pseudolabel": pseudolabel,
                    "image_emb": image_emb,
                    "text_emb": report_emb,
                    "label": mask,
                    "logits": None,
                    "loss": None
                }
            except:
                if idx == 0:
                    print("Wait for precomputing")

        return {
            "image": image,
            "text": report,
            "pseudolabel": pseudolabel,
            "image_emb": None,
            "text_emb": None,
            "label": mask,
            "logits": None,
            "loss": None
        }

    def precompute(self, encoder=None, device="cpu"):
        tensor_path = os.path.join(self.root_path, "precompute", self.mode)
        print(tensor_path)
        if os.path.exists(tensor_path):
            return 
        else:
            os.makedirs(tensor_path, exist_ok=True)
        for anno in tqdm(self.annotations, desc="Precomputing", leave=False):
            if encoder:
                trans = self.transform(self.image_size, lazy=True)
                image_path = os.path.join(self.root_path, f"images/{anno}.png")
                # image_path = os.path.join(self.root_path, f"images/{anno}.jpg")
                image = trans({"image":image_path})["image"]
                image = image.unsqueeze(0).to(device)
                if image.shape[1] == 1:   
                    image = repeat(image,'b 1 h w -> b c h w',c=3)
                image_feature = encoder.encode_image(image)
                with open(os.path.join(tensor_path, f"{anno}_vision.pkl"), 'wb') as f:
                    pickle.dump(image_feature.squeeze().detach().cpu(), f)

                text_path = os.path.join(self.root_path, f"reports/{anno}.txt")
                with open(text_path, 'r') as file:
                    report = file.read()
                text_feature = encoder.encode_text(report)
                with open(os.path.join(tensor_path, f"{anno}_text.pkl"), 'wb') as f:
                    pickle.dump(text_feature.squeeze().detach().cpu(), f)


    def transform(self, image_size=[224,224], lazy=False):

        if lazy:
            return Compose([
                LoadImaged(["image"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image"]),
            ])

        if self.aug:  # for training mode
            trans = Compose([
                LoadImaged(["image","mask"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image","mask"]),
                RandZoomd(['image','mask'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic","nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["mask"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","mask"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image","mask"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["mask"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","mask"]),
            ])

        return trans
