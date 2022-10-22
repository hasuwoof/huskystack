from huskygen.filter.datatypes import SortedList

filters = [
    (SortedList(100000), "rating:explicit"),
    (SortedList(100000), "rating:safe"),
    (list, "species:wyvern || species:dragon || species:rare_fuck"),
    (list, "artist:angiewolf"),
    ([], "artist:zackary911"),
    (SortedList, "species:cat")
]