import random
from random import shuffle

import numpy as np
from glob import glob
import json
import pickle

import hydra
from hydra.core.config_store import ConfigStore
from conf.config import Configuration

genre_to_position = {"Drama": 0,
                     "Comedy": 1,
                     "Thriller": 2,
                     "Action": 3,
                     "Romance": 4,
                     "Horror": 5,
                     "Crime": 6,
                     "Documentary": 7,
                     "Adventure": 8,
                     "Sci-Fi": 9,
                     "Family": 10,
                     "Fantasy": 11,
                     "Mystery": 12,
                     "Biography": 13,
                     "Animation": 14,
                     "History": 15,
                     "Music": 16,
                     "War": 17,
                     "Sport": 18,
                     "Musical": 19,
                     "Western": 20}

position_to_genre = {0: "Drama",
                     1: "Comedy",
                     2: "Thriller",
                     3: "Action",
                     4: "Romance",
                     5: "Horror",
                     6: "Crime",
                     7: "Documentary",
                     8: "Adventure",
                     9: "Sci-Fi",
                     10: "Family",
                     11: "Fantasy",
                     12: "Mystery",
                     13: "Biography",
                     14: "Animation",
                     15: "History",
                     16: "Music",
                     17: "War",
                     18: "Sport",
                     19: "Musical",
                     20: "Western"}

lb = ["Drama", "Comedy", "Thriller", "Action", "Romance", "Horror", "Crime", "Documentary", "Adventure",
      "Sci-Fi", "Family", "Fantasy", "Mystery", "Biography", "Animation", "History", "Music", "War", "Sport",
      "Musical", "Western"]


def genreToPos(genre_list: list) -> list[int]:
    """ Transforms the list containing the genres of the movies as strings into a list with the one-hot encoding.

    :param genre_list: List containing the genres of the movies as strings.
    :return: Return a list with the one-hot encoding of given movie.
    """
    genres_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for genre in genre_list:
        genres_pos[genre_to_position[genre]] = 1
    return genres_pos


def getGenre(movie_id: str, meta_dir: str) -> list[int]:
    """ Gets the one-hot encoding for the given movie.

    :param movie_id: The id of the movie to be fetched.
    :param meta_dir: The directory with the .json containing the metadata of the given movie.
    :return: Return a list with the one-hot encoding of given movie.
    """
    json_file_name = meta_dir + movie_id + ".json"
    file = open(json_file_name)
    json_file = json.load(file)
    file.close()
    return genreToPos(json_file["genres"])


def run(cfg: Configuration):
    """ Runs the data processing, making the data have the same dimensions across all the dataset.
    The variables for this process are stored in the .yaml file.

    :param cfg: The .yaml configuration setted. The paths will be used to save the data.
    """
    random.seed(cfg.seed_config.SEED)

    movie_map = {}
    list_id = []

    files_names = glob(cfg.file_path_config.encoded_trailer_tensor_dir)
    for file_name in files_names:
        sample_size = cfg.data_processing_config.patch_size
        final = np.zeros((sample_size, 512))
        final_size = len(final)
        array = np.load(file_name)
        size = len(array)
        film_code = file_name.split("\\")

        # if there will be more than one patch per movie, the id will be needed to generate for each patch
        if cfg.data_processing_config.number_patches > 1:
            if final_size > size:
                movie_id = film_code[-1][:-4] + "_0"
                final[-size:] = array
                list_id.append(movie_id)
                genres = getGenre(film_code[-1][:-4], cfg.file_path_config.meta_data_dir)
                movie_map[movie_id] = genres
                np.save(cfg.file_path_config.processed_tensor_dir + movie_id, final)
            else:
                for i in range(cfg.data_processing_config.number_patches):
                    rand_sample = random.randint(0, size-sample_size)
                    final = array[rand_sample: rand_sample+sample_size]
                    movie_id = film_code[-1][:-4] + "_" + str(i)
                    list_id.append(movie_id)
                    genres = getGenre(film_code[-1][:-4], cfg.file_path_config.meta_data_dir)
                    movie_map[movie_id] = genres
                    np.save(cfg.file_path_config.processed_tensor_dir + movie_id, final)
        # else the id can keep being the original id
        else:
            if final_size > size:
                movie_id = film_code[-1][:-4]
                final[-size:] = array
                list_id.append(movie_id)
                genres = getGenre(film_code[-1][:-4], cfg.file_path_config.meta_data_dir)
                movie_map[movie_id] = genres
                np.save(cfg.file_path_config.processed_tensor_dir + movie_id, final)
            else:
                rand_sample = random.randint(0, size-sample_size)
                final = array[rand_sample: rand_sample+sample_size]
                movie_id = film_code[-1][:-4]
                list_id.append(movie_id)
                genres = getGenre(film_code[-1][:-4], cfg.file_path_config.meta_data_dir)
                movie_map[movie_id] = genres
                np.save(cfg.file_path_config.processed_tensor_dir + movie_id, final)

    # shuffling the ids to prevent any bias from the data being ordered or not
    shuffle(list_id)

    # saves the information into pickled list and dictionary to be easily consulted when the training is done.
    with open(cfg.file_path_config.list_id_path, 'wb') as f:
        pickle.dump(list_id, f)
    with open(cfg.file_path_config.movie_map_path, 'wb') as f:
        pickle.dump(movie_map, f)


    print("Finished")
