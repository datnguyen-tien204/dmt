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
        label_paths = sorted(make_dataset(opt.croot))
        image_paths = sorted(make_dataset(opt.sroot))
        instance_paths = []
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        return True
