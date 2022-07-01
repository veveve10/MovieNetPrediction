import hydra
import json
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


def genreToPos(genre_list: list):
    genres_pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for genre in genre_list:
        genres_pos[genre_to_position[genre]] = 1
    return genres_pos


def getGenre(movie_id: str):
    json_file_name = r"D:/MovieDataset/meta/" + movie_id + ".json"
    file = open(json_file_name)
    json_file = json.load(file)
    file.close()
    return genreToPos(json_file["genres"])
