create table entries
(
    id              INTEGER not null
        constraint entries_pk
            primary key autoincrement,
    filename        TEXT    not null,
    original_width  INTEGER not null,
    original_height INTEGER not null,
    metadata        TEXT,
    width           INTEGER not null,
    height          INTEGER not null
);

create table tags
(
    id   INTEGER not null
        constraint tags_pk
            primary key autoincrement,
    tag  TEXT    not null,
    type INTEGER not null
);

create table entries_tags
(
    entry_id INTEGER not null
        constraint entries_tags_entries_fk
            references entries,
    tag_id   INTEGER not null
        constraint entries_tags_tags_id_fk
            references tags,
    constraint entries_tags_pk
        primary key (tag_id, entry_id)
);

create index entries_tags_entry_id_index
    on entries_tags (entry_id);



