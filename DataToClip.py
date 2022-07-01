import torch
import clip
import math
import numpy as np
import time
from glob import glob
from PIL import Image
import hydra
from hydra.core.config_store import ConfigStore
from conf.config import Configuration


def split(x, n):
    """ Creates a list with size n trying to divide elements which sum is x the most equally it cans.

    :param x: Total number of elements to be split.
    :param n: How many part it should be split into.
    :return: Return a list size of n which elements sum is x.
    """
    output = []
    if x < n:
        return -1

    elif x % n == 0:
        for i in range(n):
            output.append(x // n)
    else:
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if i >= zp:
                output.append(pp + 1)
            else:
                output.append(pp)

    return output


def run(cfg: Configuration):
    """ This .py file has the objective to use the CLIP model to encode each movie trailer into a tensor with the shape
    [N, 512], where N is the number of frames the trailer.

    :param cfg: The .yaml configuration setted. The paths will be used to save the data.
    """
    with torch.no_grad():
        print("Defining model and device")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        path_names = glob(cfg.file_path_config.trailer_dir)
        true_paths = []
        print("Defining paths")
        for name in path_names:
            if 'tar' not in name:  # in case the .tar compressed files are in the same dir of the trailers
                true_paths.append(name)

        count = 1
        for folder in true_paths:
            print("Film start " + str(count) + " out of " + str(len(true_paths)))
            film_code = folder.split("\\")
            files = glob(folder + '/*.jpg')
            film_aux = np.array([]).reshape((0, 512))
            batch_size = cfg.clip_encoder_config.batch_size
            numb_parts = int(len(files) / batch_size)
            parts = split(len(files), numb_parts)
            last_frame = 0
            start = time.process_time()
            for i, size in enumerate(parts):
                film_frame_count = 0
                film = torch.zeros((size, 3, 224, 224), dtype=torch.float32, device=device)
                for j in range(last_frame, last_frame + size):
                    film[film_frame_count] = preprocess(Image.open(files[i])).unsqueeze(0).to(device)
                    film_frame_count += 1
                last_frame += film_frame_count
                film_aux = np.concatenate((film_aux, model.encode_image(film).cpu().numpy()))
                del film
            np.save(cfg.file_path_config.encoded_trailer_tensor_dir + film_code[1], film_aux)
            del film_aux
            print("Film " + str(count) + " out of " + str(len(true_paths)) + " done in " + str(
                time.process_time() - start) + "seconds")
            count += 1
        print("Films are done and saved")
