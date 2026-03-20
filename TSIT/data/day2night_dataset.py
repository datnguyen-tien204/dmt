import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class Day2NightDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        content_paths = sorted(make_dataset(opt.croot))
        style_paths = sorted(make_dataset(opt.sroot))
        instance_paths = []

        expanded_content = []
        expanded_style = []
        for c in content_paths:
            for s in style_paths:
                expanded_content.append(c)
                expanded_style.append(s)

        return expanded_content, expanded_style, instance_paths

    def paths_match(self, path1, path2):
        return True

    def __getitem__(self, index):
        from data.base_dataset import get_params, get_transform
        from PIL import Image

        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('RGB')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params)
        label_tensor = transform_label(label)

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        input_dict = {
            'label': label_tensor,
            'instance': 0,
            'image': image_tensor,
            'path': image_path,
            'cpath': label_path
        }

        self.postprocess(input_dict)
        return input_dict
