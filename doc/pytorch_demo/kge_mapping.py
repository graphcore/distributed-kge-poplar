# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

"""Provide search and autocomplete for entity IDs."""

import dataclasses
import gzip
import json
import re
import sqlite3
from pathlib import Path
from typing import Callable, List

import numpy as np
import ogb.linkproppred


@dataclasses.dataclass
class Entry:
    """A database entry, corresponding to an entity or relation."""

    type: str  # "entity" | "relation"
    idx: int
    wikidata_id: str
    wikidata_label: str


@dataclasses.dataclass
class RawData:
    """Utility class for creating a mapping `Database` from a mapping dump and OGB dataset."""

    mapping: List[Entry]
    relation_count: np.ndarray
    head_entity_count: np.ndarray

    @classmethod
    def load(cls, mapping_path: Path, data_path: Path) -> "RawData":
        """Load raw data from raw inputs.

        E.g. `load("ogbl_wikikg2_mapping.jsonl.gz", "data/")`
        """
        with gzip.open(mapping_path, "rt") as f:
            mapping = [Entry(**json.loads(line)) for line in f]
        ogb_data = ogb.linkproppred.LinkPropPredDataset("ogbl-wikikg2", root=data_path)
        return cls(
            mapping=mapping,
            relation_count=np.bincount(ogb_data.graph["edge_reltype"].flatten()),
            head_entity_count=np.bincount(
                ogb_data.graph["edge_index"][0], minlength=ogb_data.graph["num_nodes"]
            ),
        )


class Database:
    """Uses SQLite FTS3 to provide text search on entity & relation names."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(str(path))

    @staticmethod
    def build(path: Path, data: RawData) -> None:
        connection = sqlite3.connect(str(path))
        with connection:
            connection.execute("BEGIN TRANSACTION")

            # Drop any previous contents
            connection.execute("DROP TABLE IF EXISTS mapping")
            connection.execute("DROP TABLE IF EXISTS entity_label")
            connection.execute("DROP TABLE IF EXISTS relation_label")

            # Create main table & FTS index
            # Note that we create separate FTS tables for entity & relation for performance reasons
            connection.execute(
                "CREATE TABLE mapping ("
                "type TEXT NOT NULL"
                ", idx INTEGER NOT NULL"
                ", wikidata_id TEXT NOT NULL"
                ", wikidata_label TEXT NOT NULL"
                ", score REAL NOT NULL"
                ")"
            )
            for type in ["entity", "relation"]:
                connection.execute(
                    f"CREATE VIRTUAL TABLE {type}_label USING fts3 (label, id)"
                )
                connection.execute(
                    f"CREATE TRIGGER mapping_insert_{type} AFTER INSERT ON mapping"
                    f" WHEN new.type = '{type}'"
                    " BEGIN"
                    f" INSERT INTO {type}_label"
                    " (docid, label, id) VALUES (new.rowid, new.wikidata_label, new.wikidata_id);"
                    " END;"
                )
            connection.executemany(
                "INSERT INTO mapping VALUES (:type, :idx, :wikidata_id, :wikidata_label, :score)",
                [
                    dict(
                        **entry.__dict__,
                        score=float(
                            dict(
                                relation=data.relation_count,
                                entity=data.head_entity_count,
                            )[entry.type][entry.idx]
                        ),
                    )
                    for entry in data.mapping
                ],
            )
            connection.execute(
                "CREATE UNIQUE INDEX mapping_index_typeidx ON mapping(type, idx)"
            )
        connection.close()

    @staticmethod
    def _entries(cursor: sqlite3.Cursor) -> List[Entry]:
        columns = [c[0] for c in cursor.description]
        return [Entry(**dict(zip(columns, x))) for x in cursor.fetchall()]

    def search(self, query: str, type: str, limit: int) -> List[Entry]:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT m.type, m.idx, m.wikidata_id, m.wikidata_label"
            " FROM mapping as m"
            f" INNER JOIN {type}_label as w"
            " WHERE m.rowid = w.docid"
            " AND m.type = :type"
            + (f" AND {type}_label MATCH :query" if query else "")
            + " ORDER BY m.score DESC"
            " LIMIT :limit",
            dict(type=type, query=query, limit=limit),
        )
        return self._entries(cursor)

    def get_entry(self, idx: int, type: str) -> Entry:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT m.type, m.idx, m.wikidata_id, m.wikidata_label"
            " FROM mapping as m"
            " WHERE m.type = ? AND m.idx = ?",
            [type, int(idx)],
        )
        (result,) = self._entries(cursor)
        return result

    @property
    def all_relations(self) -> List[Entry]:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT m.type, m.idx, m.wikidata_id, m.wikidata_label"
            " FROM mapping as m"
            " WHERE m.type = 'relation'"
            " ORDER BY m.score DESC"
        )
        return self._entries(cursor)


@dataclasses.dataclass
class Prediction(Entry):
    """Additional metadata for a tail entity prediction."""

    in_training: bool


@dataclasses.dataclass
class Predictions:
    """Render head & relation from a search query, and tail predictions from a model."""

    heads: List[Entry]
    relations: List[Entry]
    tail_predictions: List[Prediction]

    STYLE = """
<style>
.kge-q-part {
    display: inline-block;
    vertical-align: top;
}
.kge-q-part h1 {
    font-size: large;
}
.kge-q-part ol {
    list-style-type: none;
}
.kge-q-part li {
    padding: 0.8em;
    margin: 0.2em;
    background-color: #eee;
    color: #888;
}
.kge-q-part-head li:nth-of-type(1),.kge-q-part-relation li:nth-of-type(1) {
    background-color: #aaf;
    color: #000;
}
.kge-q-part-tail_predictions li {
    color: #000;
}
.kge-q-part a {
    color: #008 !important;
}
.kge-tail-present {
    outline: solid #4a48 3px;
}
</style>
"""

    @staticmethod
    def _shorten_text(text: str, limit: int) -> str:
        if len(text) < limit:
            return text
        return f"{text[:limit]}&mldr;"

    @classmethod
    def _render_entry(cls, entry: Entry) -> str:
        url = dict(
            entity=f"https://www.wikidata.org/wiki/{entry.wikidata_id}",
            relation=f"https://www.wikidata.org/wiki/Property:{entry.wikidata_id}",
        )[entry.type]
        html = (
            f"<b>{cls._shorten_text(entry.wikidata_label, 40)}</b>"
            f' (<a href="{url}" target=”_blank”>{entry.wikidata_id}</a>)'
        )
        return html

    @classmethod
    def _render_part(cls, entries: List[Entry]) -> str:
        html = "<ol>"
        for entry in entries:
            class_ = ""
            if isinstance(entry, Prediction) and entry.in_training:
                class_ = "kge-tail-present"
            html += f'<li class="{class_}">{cls._render_entry(entry)}</li>'
        html += "</ol>"
        return html

    def _repr_html_(self) -> str:
        html = self.STYLE
        for kind, entries in dict(
            head=self.heads,
            relation=self.relations,
            tail_predictions=self.tail_predictions,
        ).items():
            html += (
                f'<div class="kge-q-part kge-q-part-{kind}">'
                f'<h1>{kind.capitalize().replace("_", " ")}</h1>'
                f"{self._render_part(entries)}</div>"
            )
        return html


class Predictor:
    """Provide a user-friendly query interface for use in a Jupyter notebook.

    Constructing a predictor takes non-negligible time, so we recommend constructing once
    rather than inline for each query.
    """

    def __init__(
        self,
        database: Database,
        predict: Callable[[int, int], List[int]],
        train_hrt: np.ndarray,
        n_entity: int,
        n_relation_type: int,
        n_suggestions: int = 10,
    ):
        self.database = database
        self.predict = predict
        self.train_hrt_hash = {
            int(h)
            for h in np.sum(
                [
                    n_entity * n_relation_type,
                    n_entity,
                    1,
                ]
                * train_hrt.astype(np.int64),
                -1,
            )
        }
        self.n_entity = n_entity
        self.n_relation_type = n_relation_type
        self.n_suggestions = n_suggestions

    def search(self, query: str, type: str) -> List[Entry]:
        if query and not query.endswith(" ") and not re.match(r"^(P|Q)\d+$", query):
            query = query + "*"
        return self.database.search(query, type=type, limit=self.n_suggestions)

    def __call__(self, head_query: str, relation_query: str) -> Predictions:
        """Run text search on {head, relation} and a model prediction query for {tail}.

        head_query -- str
                   -- e.g. "england richa" -> "Richard I of England"
                      e.g. "Q62378" -> "Tower of London"
                      (note that "*" is appended for word completion unless
                       the query ends with " " or looks like a WikiData ID)

        relation_query -- str -- same syntax (and ID e.g. "P770")

        returns -- Predictions -- designed for rendering in Jupyter via `display()`
        """
        heads = self.search(head_query, type="entity")
        relations = self.search(relation_query, type="relation")
        predictions = []
        if heads and relations:
            head = heads[0].idx
            relation = relations[0].idx
            predictions = [
                Prediction(
                    **self.database.get_entry(tail, type="entity").__dict__,
                    in_training=(
                        (
                            self.n_entity * self.n_relation_type * head
                            + self.n_entity * relation
                            + tail
                        )
                        in self.train_hrt_hash
                    ),
                )
                for tail in self.predict(heads[0].idx, relations[0].idx)
            ]
        return Predictions(heads, relations, predictions)
