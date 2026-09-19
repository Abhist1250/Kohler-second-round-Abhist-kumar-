
"""
KOHLER AI Bathroom Designer - Backend

This module contains the application logic extracted and cleaned from
the working prototype notebook.

Pipeline:
    requirements
        -> structured filtering
        -> semantic retrieval (ChromaDB)
        -> candidate generation
        -> deterministic constraint validation
        -> bundle scoring
        -> three recommendation modes
        -> 2D spatial layout
        -> optional Groq explanation
"""

from __future__ import annotations

import json
import os
from itertools import product as cartesian_product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Rectangle

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError:
    DefaultEmbeddingFunction = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "products.json"
# Fallbacks keep the standalone app connected to the prototype catalog even
# when products.json was copied from the earlier raw extraction.
FALLBACK_CATALOG_PATHS = [
    DATA_DIR / "kohler_ai_bathroom.products_prototype_budget_dimensions.json",
    BASE_DIR.parent / "KOHLER" / "DATA" /
        "kohler_ai_bathroom.products_prototype_budget_dimensions.json",
]
CHROMA_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "kohler_products_v2"

FT_TO_MM = 304.8
SPATIAL_CLEARANCE_MM = 200.0
DEFAULT_DOOR_WIDTH_MM = 900.0
DEFAULT_DOOR_CLEARANCE_MM = 600.0

# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "not specified"}:
        return None
    return value


def clean_list(value: Any) -> List[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            item = clean_text(item)
            if item and item not in result:
                result.append(item)
        return result
    item = clean_text(value)
    return [item] if item else []


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_product_price(product: Any) -> Optional[float]:
    if product is None:
        return None

    price = None
    try:
        price = product.get("price_inr")
    except AttributeError:
        pass

    if price is None:
        try:
            pricing = product.get("pricing")
        except AttributeError:
            pricing = None
        if isinstance(pricing, dict):
            price = pricing.get("price_inr")

    if price is None:
        return None

    try:
        value = float(price)
    except (TypeError, ValueError):
        return None

    if np.isnan(value) or value <= 0:
        return None

    return value


def feet_to_mm(feet: float) -> float:
    return float(feet) * FT_TO_MM


def normalize_category(category: Any) -> str:
    if category is None or is_missing(category):
        return "Other"

    value = str(category).strip().lower()

    if not value or value in {"none", "nan", "not specified"}:
        return "Other"
    if "toilet seat" in value:
        return "Toilet Seat"
    if "toilet" in value:
        return "Toilet"
    if "faucet trim" in value:
        return "Faucet Trim"
    if "faucet" in value:
        return "Faucet"
    if "shower" in value:
        return "Shower"
    if "vanity" in value:
        return "Vanity"
    if "bathtub" in value or "bath tub" in value:
        return "Bathtub"
    if "mirror" in value or "cabinet" in value:
        return "Mirror"
    if "sink" in value or "basin" in value or "vessel" in value:
        return "Sink"
    return "Other"


# ---------------------------------------------------------------------
# Catalog / retrieval initialization
# ---------------------------------------------------------------------

_products: Optional[List[Dict[str, Any]]] = None
_catalog_df: Optional[pd.DataFrame] = None
_chroma_client = None
_collection = None
_embedding_function = None


def _catalog_is_usable(path: Path) -> bool:
    """Return True when a catalog contains the fields needed by the optimizer."""
    if not path.exists():
        return False

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list) or not data:
            return False

        df = pd.json_normalize(data)
        required = {
            "product_id",
            "category",
            "pricing.price_inr",
            "dimensions.width_mm",
            "dimensions.depth_mm",
        }
        if not required.issubset(df.columns):
            return False

        prices = pd.to_numeric(df["pricing.price_inr"], errors="coerce")
        widths = pd.to_numeric(df["dimensions.width_mm"], errors="coerce")
        depths = pd.to_numeric(df["dimensions.depth_mm"], errors="coerce")

        # We need at least one usable product in each core category.
        usable = (prices > 0) & (widths > 0) & (depths > 0)
        categories = df.loc[usable, "category"].astype(str).map(normalize_category)
        return {"Toilet", "Sink", "Faucet"}.issubset(set(categories))
    except Exception:
        return False


def resolve_catalog_path() -> Path:
    """Choose the portable project catalog, then the known prototype fallback."""
    if _catalog_is_usable(DATA_PATH):
        return DATA_PATH

    for fallback in FALLBACK_CATALOG_PATHS:
        if _catalog_is_usable(fallback):
            return fallback

    checked = [DATA_PATH, *FALLBACK_CATALOG_PATHS]
    raise FileNotFoundError(
        "No usable KOHLER prototype catalog was found. Checked: "
        + "; ".join(str(path) for path in checked)
    )


def load_catalog() -> pd.DataFrame:
    global _products, _catalog_df

    if _catalog_df is not None:
        return _catalog_df

    catalog_path = resolve_catalog_path()

    with open(catalog_path, "r", encoding="utf-8") as file:
        _products = json.load(file)

    if not isinstance(_products, list) or not _products:
        raise ValueError("Product catalog must contain a non-empty list.")

    df = pd.json_normalize(_products)

    required_columns = [
        "product_id",
        "product_name",
        "category",
        "subcategory",
        "collection",
        "material",
        "color",
        "features",
        "pricing.price_inr",
        "pricing.price_estimated",
        "dimensions.width_mm",
        "dimensions.depth_mm",
        "dimensions.height_mm",
        "installation.type",
        "installation.rough_in_mm",
        "installation.rough_in_available",
        "electrical.required",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["category_normalized"] = df["category"].apply(normalize_category)

    for field in [
        "dimensions.width_mm",
        "dimensions.depth_mm",
        "dimensions.height_mm",
        "electrical.power_w",
        "installation.rough_in_mm",
    ]:
        df[field] = pd.to_numeric(df[field], errors="coerce")
        df.loc[df[field] < 0, field] = pd.NA

    df["price_inr"] = pd.to_numeric(
        df["pricing.price_inr"], errors="coerce"
    )
    df["price_estimated"] = (
        df["pricing.price_estimated"]
        .fillna(False)
        .astype(bool)
    )

    df["semantic_text"] = df.apply(create_semantic_text, axis=1)
    df["semantic_text"] = df["semantic_text"].apply(
        lambda x: x if clean_text(x) else "KOHLER bathroom product."
    )

    _catalog_df = df
    return _catalog_df


def create_semantic_text(product: pd.Series) -> str:
    parts: List[str] = []

    for field in [
        "product_name",
        "collection",
        "subcategory",
        "category_normalized",
        "material",
        "color",
        "features",
    ]:
        value = product.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif not is_missing(value):
            parts.append(str(value))

    installation = clean_text(product.get("installation.type"))
    if installation:
        parts.append(f"installation {installation}")

    return " ".join(parts)


def initialize_retrieval() -> None:
    global _chroma_client, _collection, _embedding_function

    load_catalog()

    if _collection is not None:
        return

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if DefaultEmbeddingFunction is None:
        raise ImportError("ChromaDB embedding utilities are unavailable.")

    _embedding_function = DefaultEmbeddingFunction()

    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function,
        metadata={
            "description": "KOHLER prototype bathroom product catalog"
        },
    )

    # Upsert current catalog so the app remains synchronized with products.json.
    catalog = load_catalog()

    ids = []
    documents = []
    metadatas = []

    for _, row in catalog.iterrows():
        product_id = clean_text(row.get("product_id"))
        if not product_id:
            continue

        ids.append(product_id)
        documents.append(row.get("semantic_text", "KOHLER bathroom product."))
        metadatas.append({
            "product_id": product_id,
            "product_name": clean_text(row.get("product_name")) or "Not specified",
            "category": clean_text(row.get("category_normalized")) or "Other",
            "collection": clean_text(row.get("collection")) or "Not specified",
            "material": clean_text(row.get("material")) or "Not specified",
            "price_inr": safe_float(row.get("price_inr"), -1),
            "price_estimated": bool(row.get("price_estimated", False)),
            "width_mm": safe_float(row.get("dimensions.width_mm"), -1),
            "depth_mm": safe_float(row.get("dimensions.depth_mm"), -1),
            "height_mm": safe_float(row.get("dimensions.height_mm"), -1),
        })

    if ids:
        _collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


# ---------------------------------------------------------------------
# Structured retrieval
# ---------------------------------------------------------------------

def apply_max_constraint(
    dataframe: pd.DataFrame,
    field: str,
    maximum: Optional[float],
) -> pd.DataFrame:
    if maximum is None or field not in dataframe.columns:
        return dataframe

    values = pd.to_numeric(dataframe[field], errors="coerce")
    return dataframe[values.isna() | (values <= float(maximum))]


def apply_budget_constraint(
    dataframe: pd.DataFrame,
    maximum: Optional[float],
) -> pd.DataFrame:
    return apply_max_constraint(dataframe, "price_inr", maximum)


def filter_products(
    dataframe: pd.DataFrame,
    category: Optional[str] = None,
    max_price_inr: Optional[float] = None,
    max_width_mm: Optional[float] = None,
    max_depth_mm: Optional[float] = None,
    max_height_mm: Optional[float] = None,
    installation_type: Optional[str] = None,
    required_rough_in_mm: Optional[float] = None,
    electrical_required: Optional[bool] = None,
) -> pd.DataFrame:
    result = dataframe.copy()

    if category:
        normalized = normalize_category(category)
        result = result[
            result["category_normalized"].astype(str).str.lower()
            == normalized.lower()
        ]

    result = apply_budget_constraint(result, max_price_inr)
    result = apply_max_constraint(
        result, "dimensions.width_mm", max_width_mm
    )
    result = apply_max_constraint(
        result, "dimensions.depth_mm", max_depth_mm
    )
    result = apply_max_constraint(
        result, "dimensions.height_mm", max_height_mm
    )

    if installation_type:
        target = str(installation_type).strip().lower()
        values = result["installation.type"].fillna("").astype(str).str.lower()
        result = result[values == target]

    if required_rough_in_mm is not None:
        values = pd.to_numeric(
            result["installation.rough_in_mm"], errors="coerce"
        )
        result = result[
            values.isna() | (values == float(required_rough_in_mm))
        ]

    if electrical_required is not None:
        values = result["electrical.required"]
        normalized_values = values.apply(
            lambda x: bool(x) if not is_missing(x) else None
        )
        result = result[
            normalized_values.isna()
            | (normalized_values == bool(electrical_required))
        ]

    return result.copy()


def product_fits_inside_room(
    product: pd.Series,
    room_length_mm: float,
    room_width_mm: float,
) -> bool:
    width = product.get("dimensions.width_mm")
    depth = product.get("dimensions.depth_mm")

    if is_missing(width) or is_missing(depth):
        return False

    width = safe_float(width, -1)
    depth = safe_float(depth, -1)

    return (
        (width <= room_width_mm and depth <= room_length_mm)
        or (depth <= room_width_mm and width <= room_length_mm)
    )


def prepare_physical_candidates(
    dataframe: pd.DataFrame,
    category: str,
    room_length_mm: float,
    room_width_mm: float,
) -> pd.DataFrame:
    category_df = dataframe[
        dataframe["category_normalized"].astype(str).str.lower()
        == normalize_category(category).lower()
    ].copy()

    if category_df.empty:
        return category_df

    mask = category_df.apply(
        lambda row: product_fits_inside_room(
            row, room_length_mm, room_width_mm
        ),
        axis=1,
    )
    return category_df[mask].copy()


def product_search_text(product: pd.Series) -> str:
    parts = []
    for field in [
        "product_name",
        "collection",
        "subcategory",
        "material",
        "features",
    ]:
        value = product.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif not is_missing(value):
            parts.append(str(value))
    return " ".join(parts).lower()


def semantic_search(
    query: str,
    n_results: int = 5,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    initialize_retrieval()

    if not clean_text(query):
        raise ValueError("Semantic query cannot be empty.")

    kwargs = {
        "query_texts": [query],
        "n_results": max(1, int(n_results)),
    }

    if category:
        kwargs["where"] = {"category": normalize_category(category)}

    return _collection.query(**kwargs)


def get_theme_candidates(
    dataframe: pd.DataFrame,
    category: str,
    theme: str,
    room_length_mm: float,
    room_width_mm: float,
    top_k: int = 8,
) -> pd.DataFrame:
    physical = prepare_physical_candidates(
        dataframe, category, room_length_mm, room_width_mm
    )

    if physical.empty:
        return physical

    query = (
        f"{theme} bathroom design {category}, "
        "clean elegant coordinated style"
    )

    # Semantic search is used for relevance; physical candidates remain
    # the authoritative allowed pool.
    try:
        semantic = semantic_search(
            query=query,
            n_results=min(max(top_k * 2, top_k), max(1, len(dataframe))),
            category=normalize_category(category),
        )
    except Exception:
        # Retrieval should improve ranking, never make the core optimizer fail.
        physical = physical.copy()
        physical["semantic_distance"] = 999.0
        return physical.head(max(1, int(top_k))).copy()

    ids = semantic.get("ids", [[]])[0] or []
    distances = semantic.get("distances", [[]])[0] or []

    distance_map = {
        str(pid): safe_float(dist, 999)
        for pid, dist in zip(ids, distances)
    }

    physical = physical.copy()
    physical["semantic_distance"] = physical["product_id"].astype(str).map(
        distance_map
    )

    # Products not returned semantically remain valid but are ranked after
    # semantic hits.
    physical["semantic_distance"] = physical["semantic_distance"].fillna(999)
    physical = physical.sort_values(
        ["semantic_distance", "price_inr"],
        ascending=[True, True],
    )

    return physical.head(max(1, int(top_k))).copy()


# ---------------------------------------------------------------------
# Spatial validation
# ---------------------------------------------------------------------

def get_fixture_dimensions(product: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    width = product.get("dimensions.width_mm")
    depth = product.get("dimensions.depth_mm")

    if is_missing(width) or is_missing(depth):
        return None

    width = safe_float(width, -1)
    depth = safe_float(depth, -1)

    if width <= 0 or depth <= 0:
        return None

    return width, depth


def expand_rectangle(
    rect: Tuple[float, float, float, float],
    margin: float,
) -> Tuple[float, float, float, float]:
    x, y, width, depth = rect
    return (
        x - margin,
        y - margin,
        width + 2 * margin,
        depth + 2 * margin,
    )


def rectangles_overlap(a, b) -> bool:
    ax, ay, aw, ad = a
    bx, by, bw, bd = b
    return not (
        ax + aw <= bx
        or bx + bw <= ax
        or ay + ad <= by
        or by + bd <= ay
    )


def rectangle_inside_room(
    rect,
    room_length_mm: float,
    room_width_mm: float,
) -> bool:
    x, y, width, depth = rect
    return (
        x >= 0
        and y >= 0
        and x + width <= room_width_mm
        and y + depth <= room_length_mm
    )


def build_layout_for_bundle(
    bundle: Dict[str, Dict[str, Any]],
    room_length_mm: float,
    room_width_mm: float,
    door_width_mm: float = DEFAULT_DOOR_WIDTH_MM,
    door_clearance_mm: float = DEFAULT_DOOR_CLEARANCE_MM,
    fixture_clearance_mm: float = SPATIAL_CLEARANCE_MM,
) -> Dict[str, Any]:
    """
    Build a deterministic top-down bathroom layout.

    Important layout rule:
    - Faucets are NOT treated as independent floor fixtures.
    - When a sink/basin is present, the faucet is mounted at the back edge
      of that sink and is drawn directly above/behind it in the 2D view.
    - Because the faucet is sink-mounted, its clearance rectangle is not
      used as a separate floor-space obstacle.
    """
    if not bundle:
        return {
            "status": "FAIL",
            "reason": "No products were supplied for layout generation.",
            "placements": [],
        }

    door_x = (room_width_mm - door_width_mm) / 2
    door_zone = (door_x, 0, door_width_mm, door_clearance_mm)

    # Deterministic anchors for floor-mounted fixtures.
    anchors = [
        ("top-left", 0.0, room_length_mm),
        ("top-right", room_width_mm, room_length_mm),
        ("bottom-left", 0.0, 0.0),
        ("bottom-right", room_width_mm, 0.0),
        ("center-left", 0.0, room_length_mm / 2),
        ("center-right", room_width_mm, room_length_mm / 2),
        ("top-center", room_width_mm / 2, room_length_mm),
    ]

    placements: List[Dict[str, Any]] = []
    occupied_clearance: List[Tuple[float, float, float, float]] = []

    def place_floor_fixture(category: str, product: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dims = get_fixture_dimensions(product)
        if dims is None:
            return None

        width, depth = dims

        orientations = [
            (width, depth, False),
            (depth, width, True),
        ]

        for candidate_width, candidate_depth, rotated in orientations:
            for anchor_name, anchor_x, anchor_y in anchors:
                if "right" in anchor_name:
                    x = anchor_x - candidate_width
                elif "center" in anchor_name:
                    x = anchor_x - candidate_width / 2
                else:
                    x = anchor_x

                if "top" in anchor_name:
                    y = anchor_y - candidate_depth
                elif "center" in anchor_name:
                    y = anchor_y - candidate_depth / 2
                else:
                    y = anchor_y

                rect = (x, y, candidate_width, candidate_depth)

                if not rectangle_inside_room(
                    rect, room_length_mm, room_width_mm
                ):
                    continue

                if rectangles_overlap(rect, door_zone):
                    continue

                clearance_rect = expand_rectangle(
                    rect, fixture_clearance_mm
                )

                if any(
                    rectangles_overlap(clearance_rect, other)
                    for other in occupied_clearance
                ):
                    continue

                return {
                    "category": category,
                    "product_id": product.get("product_id"),
                    "product_name": product.get("product_name"),
                    "x_mm": float(x),
                    "y_mm": float(y),
                    "width_mm": float(candidate_width),
                    "depth_mm": float(candidate_depth),
                    "rotated": rotated,
                    "anchor": anchor_name,
                    "clearance_mm": float(fixture_clearance_mm),
                    "mounted_fixture": False,
                }

        return None

    # 1) Place all floor fixtures first, except faucets.
    #    This guarantees the sink exists before its faucet is positioned.
    deferred_faucets: List[Tuple[str, Dict[str, Any]]] = []

    for category, product in bundle.items():
        if normalize_category(category) == "Faucet":
            deferred_faucets.append((category, product))
            continue

        placement = place_floor_fixture(category, product)

        if placement is None:
            dims = get_fixture_dimensions(product)
            if dims is None:
                reason = f"Missing usable dimensions for {category}."
            else:
                reason = (
                    f"Could not place {category} without overlap, "
                    "room-boundary violation, or door obstruction."
                )

            return {
                "status": "FAIL",
                "reason": reason,
                "placements": placements,
            }

        placements.append(placement)
        occupied_clearance.append(
            expand_rectangle(
                (
                    placement["x_mm"],
                    placement["y_mm"],
                    placement["width_mm"],
                    placement["depth_mm"],
                ),
                fixture_clearance_mm,
            )
        )

    # 2) Mount each faucet directly above the sink/basin.
    sink_placement = next(
        (
            placement
            for placement in placements
            if normalize_category(placement.get("category")) == "Sink"
        ),
        None,
    )

    for category, product in deferred_faucets:
        dims = get_fixture_dimensions(product)

        if dims is None:
            return {
                "status": "FAIL",
                "reason": f"Missing usable dimensions for {category}.",
                "placements": placements,
            }

        faucet_width, faucet_depth = dims

        if sink_placement is not None:
            # Faucet is centered on the sink's back/top edge.
            # It slightly overlaps the sink rectangle to visually represent
            # a deck-mounted/back-mounted faucet rather than a floor fixture.
            sink_x = sink_placement["x_mm"]
            sink_y = sink_placement["y_mm"]
            sink_width = sink_placement["width_mm"]
            sink_depth = sink_placement["depth_mm"]

            # Position the faucet on the sink's rear/top edge.  The faucet
            # is allowed to overlap the sink rectangle because it represents
            # a mounted fixture, not a separate floor fixture.
            #
            # Clamp both axes so a sink placed against the wall never causes
            # the faucet to extend outside the bathroom boundary.
            faucet_x = sink_x + (sink_width - faucet_width) / 2
            faucet_x = max(0.0, min(
                faucet_x,
                room_width_mm - faucet_width,
            ))

            faucet_y = sink_y + sink_depth - faucet_depth
            faucet_y = max(0.0, min(
                faucet_y,
                room_length_mm - faucet_depth,
            ))

            faucet_rect = (
                faucet_x,
                faucet_y,
                faucet_width,
                faucet_depth,
            )

            # Keep the mounted faucet within the room. The faucet may overlap
            # the sink because it is intentionally attached to the sink.
            if not rectangle_inside_room(
                faucet_rect, room_length_mm, room_width_mm
            ):
                return {
                    "status": "FAIL",
                    "reason": (
                        "Could not mount the faucet above the sink "
                        "within the room boundary."
                    ),
                    "placements": placements,
                }

            placements.append({
                "category": category,
                "product_id": product.get("product_id"),
                "product_name": product.get("product_name"),
                "x_mm": float(faucet_x),
                "y_mm": float(faucet_y),
                "width_mm": float(faucet_width),
                "depth_mm": float(faucet_depth),
                "rotated": False,
                "anchor": "sink-mounted",
                "clearance_mm": 0.0,
                "mounted_fixture": True,
                "attached_to": "Sink",
                "mount_position": "above_sink",
            })

        else:
            # If a user selects a faucet without a sink, retain the old
            # deterministic floor-placement fallback.
            placement = place_floor_fixture(category, product)

            if placement is None:
                return {
                    "status": "FAIL",
                    "reason": (
                        f"Could not place {category} without overlap, "
                        "room-boundary violation, or door obstruction."
                    ),
                    "placements": placements,
                }

            placements.append(placement)
            occupied_clearance.append(
                expand_rectangle(
                    (
                        placement["x_mm"],
                        placement["y_mm"],
                        placement["width_mm"],
                        placement["depth_mm"],
                    ),
                    fixture_clearance_mm,
                )
            )

    return {
        "status": "PASS",
        "reason": (
            "All selected fixtures were placed without detected spatial "
            "conflicts; faucets are mounted directly above the sink when "
            "a sink is present."
        ),
        "placements": placements,
        "door": {
            "x_mm": door_x,
            "width_mm": door_width_mm,
            "clearance_mm": door_clearance_mm,
        },
        "fixture_clearance_mm": fixture_clearance_mm,
    }


# ---------------------------------------------------------------------
# Compatibility / scoring
# ---------------------------------------------------------------------

def check_product_compatibility(bundle: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    explicit_conflicts = []

    for category, product in bundle.items():
        text = " ".join([
            str(product.get("product_name", "")),
            str(product.get("features", "")),
        ]).lower()

        if "not compatible" in text or "incompatible" in text:
            explicit_conflicts.append(category)

    if explicit_conflicts:
        return {
            "status": "FAIL",
            "reason": (
                "Explicit incompatibility text found for: "
                + ", ".join(explicit_conflicts)
            ),
        }

    return {
        "status": "UNKNOWN",
        "reason": (
            "No complete cross-product compatibility graph is "
            "available in the prototype catalog."
        ),
    }


def calculate_bundle_price(bundle: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    prices = {}
    unknown_categories = []
    total = 0.0

    for category, product in bundle.items():
        price = get_product_price(product)
        prices[category] = price

        if price is None:
            unknown_categories.append(category)
        else:
            total += price

    return {
        "total_price": total,
        "prices": prices,
        "price_known": len(unknown_categories) == 0,
        "unknown_categories": unknown_categories,
    }


def validate_bundle_budget(
    bundle: Dict[str, Dict[str, Any]],
    budget_inr: float,
) -> Dict[str, Any]:
    price_info = calculate_bundle_price(bundle)

    if not price_info["price_known"]:
        return {"status": "UNKNOWN", **price_info}

    return {
        "status": (
            "PASS"
            if price_info["total_price"] <= float(budget_inr)
            else "FAIL"
        ),
        **price_info,
    }


def validate_bundle_space(
    bundle: Dict[str, Dict[str, Any]],
    room_length_mm: float,
    room_width_mm: float,
    door_width_mm: float,
    door_clearance_mm: float,
) -> Dict[str, Any]:
    layout = build_layout_for_bundle(
        bundle=bundle,
        room_length_mm=room_length_mm,
        room_width_mm=room_width_mm,
        door_width_mm=door_width_mm,
        door_clearance_mm=door_clearance_mm,
        fixture_clearance_mm=SPATIAL_CLEARANCE_MM,
    )

    return {
        "status": layout["status"],
        "reason": layout["reason"],
        "layout": layout,
    }


def status_to_score(status: str) -> float:
    if status == "PASS":
        return 1.0
    if status == "UNKNOWN":
        return 0.5
    return 0.0


def calculate_feature_score(
    bundle: Dict[str, Dict[str, Any]],
    preferred_features: List[str],
) -> float:
    if not preferred_features:
        return 1.0

    text = " ".join(
        product_search_text(pd.Series(product))
        for product in bundle.values()
    )

    matched = 0
    requested = 0

    for feature in preferred_features:
        feature = str(feature).strip().lower()
        if not feature:
            continue
        requested += 1
        if feature in text:
            matched += 1

    return matched / requested if requested else 1.0


def calculate_theme_score(
    bundle: Dict[str, Dict[str, Any]],
    theme: str,
) -> float:
    if not theme:
        return 0.5

    theme_words = [
        word for word in str(theme).lower().split()
        if len(word) > 2
    ]

    if not theme_words:
        return 0.5

    scores = []
    for product in bundle.values():
        text = product_search_text(pd.Series(product))
        matches = sum(word in text for word in theme_words)
        scores.append(matches / len(theme_words))

    return float(np.mean(scores)) if scores else 0.5


def calculate_final_bundle_score(
    budget_score: float,
    space_score: float,
    compatibility_score: float,
    theme_score: float,
    feature_score: float,
) -> float:
    return (
        0.30 * space_score
        + 0.25 * budget_score
        + 0.20 * compatibility_score
        + 0.15 * theme_score
        + 0.10 * feature_score
    )


# ---------------------------------------------------------------------
# Bundle optimization
# ---------------------------------------------------------------------

def dataframe_product_to_dict(row: pd.Series) -> Dict[str, Any]:
    return row.to_dict()


def generate_bundle_candidates(
    dataframe: pd.DataFrame,
    requirements: Dict[str, Any],
    top_k_per_category: int = 8,
) -> List[Dict[str, Dict[str, Any]]]:
    room_length_mm = feet_to_mm(requirements["length_ft"])
    room_width_mm = feet_to_mm(requirements["width_ft"])
    theme = requirements.get("theme", "Minimalist Modern")

    category_pools: Dict[str, pd.DataFrame] = {}

    for category in requirements["required_categories"]:
        pool = get_theme_candidates(
            dataframe=dataframe,
            category=category,
            theme=theme,
            room_length_mm=room_length_mm,
            room_width_mm=room_width_mm,
            top_k=top_k_per_category,
        )

        # Only products with usable prototype price and dimensions enter
        # the deterministic optimizer.
        if not pool.empty:
            pool = pool[
                pool["price_inr"].notna()
                & pool["dimensions.width_mm"].notna()
                & pool["dimensions.depth_mm"].notna()
            ].copy()

        category_pools[category] = pool

    if any(pool.empty for pool in category_pools.values()):
        missing = [
            category
            for category, pool in category_pools.items()
            if pool.empty
        ]
        raise ValueError(
            "No usable catalog candidates were found for: "
            + ", ".join(missing)
        )

    bundles = []

    categories = list(category_pools.keys())
    pools = [category_pools[category].to_dict("records") for category in categories]

    for combination in cartesian_product(*pools):
        bundle = {
            category: product
            for category, product in zip(categories, combination)
        }
        bundles.append(bundle)

    return bundles


def evaluate_bundle(
    bundle: Dict[str, Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Dict[str, Any]:
    room_length_mm = feet_to_mm(requirements["length_ft"])
    room_width_mm = feet_to_mm(requirements["width_ft"])

    budget_result = validate_bundle_budget(
        bundle, requirements["budget_inr"]
    )

    space_result = validate_bundle_space(
        bundle,
        room_length_mm=room_length_mm,
        room_width_mm=room_width_mm,
        door_width_mm=requirements.get(
            "door_width_mm", DEFAULT_DOOR_WIDTH_MM
        ),
        door_clearance_mm=requirements.get(
            "door_clearance_mm", DEFAULT_DOOR_CLEARANCE_MM
        ),
    )

    compatibility_result = check_product_compatibility(bundle)

    theme_score = calculate_theme_score(
        bundle, requirements.get("theme", "")
    )
    feature_score = calculate_feature_score(
        bundle, requirements.get("preferred_features", [])
    )

    budget_score = status_to_score(budget_result["status"])
    space_score = status_to_score(space_result["status"])
    compatibility_score = status_to_score(
        compatibility_result["status"]
    )

    final_score = calculate_final_bundle_score(
        budget_score=budget_score,
        space_score=space_score,
        compatibility_score=compatibility_score,
        theme_score=theme_score,
        feature_score=feature_score,
    )

    feasible = (
        budget_result["status"] == "PASS"
        and space_result["status"] == "PASS"
        and compatibility_result["status"] != "FAIL"
    )

    return {
        "bundle": bundle,
        "budget": budget_result,
        "space": space_result,
        "compatibility": compatibility_result,
        "theme_score": theme_score,
        "feature_score": feature_score,
        "budget_score": budget_score,
        "space_score": space_score,
        "compatibility_score": compatibility_score,
        "final_score": final_score,
        "feasible": feasible,
    }


def result_to_row(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "final_score": round(float(result["final_score"]), 4),
        "feasible": bool(result["feasible"]),
        "budget_status": result["budget"]["status"],
        "space_status": result["space"]["status"],
        "compatibility_status": result["compatibility"]["status"],
        "budget_score": round(float(result["budget_score"]), 4),
        "space_score": round(float(result["space_score"]), 4),
        "compatibility_score": round(float(result["compatibility_score"]), 4),
        "theme_score": round(float(result["theme_score"]), 4),
        "feature_score": round(float(result["feature_score"]), 4),
        "total_price_inr": round(
            float(result["budget"]["total_price"]), 2
        ),
    }

    for category, product in result["bundle"].items():
        row[f"{category}_id"] = product.get("product_id")
        row[f"{category}_name"] = product.get("product_name")
        row[f"{category}_price_inr"] = get_product_price(product)

    return row


def bundle_signature(row: pd.Series) -> Tuple:
    categories = ["Toilet", "Sink", "Faucet", "Vanity", "Bathtub"]
    return tuple(
        str(row.get(f"{category}_id", ""))
        for category in categories
    )


def build_recommendation_dataframe(
    evaluated: List[Dict[str, Any]],
) -> pd.DataFrame:
    if not evaluated:
        return pd.DataFrame()

    df = pd.DataFrame([result_to_row(result) for result in evaluated])
    feasible = df[df["feasible"] == True].copy()

    if feasible.empty:
        return feasible

    # Lower price receives a positive normalized score.
    prices = pd.to_numeric(
        feasible["total_price_inr"], errors="coerce"
    )
    if prices.max() == prices.min():
        lower_price = pd.Series(
            0.5, index=feasible.index, dtype=float
        )
    else:
        normalized = (prices - prices.min()) / (
            prices.max() - prices.min()
        )
        lower_price = 1.0 - normalized

    feasible["lower_price_score"] = lower_price

    feasible["best_match_score"] = (
        0.45 * feasible["final_score"]
        + 0.25 * feasible["theme_score"]
        + 0.20 * feasible["feature_score"]
        + 0.10 * feasible["lower_price_score"]
    )

    feasible["budget_optimized_score"] = (
        0.55 * feasible["lower_price_score"]
        + 0.20 * feasible["final_score"]
        + 0.15 * feasible["theme_score"]
        + 0.10 * feasible["feature_score"]
    )

    feasible["premium_score"] = (
        0.45 * feasible["final_score"]
        + 0.30 * feasible["feature_score"]
        + 0.20 * feasible["theme_score"]
        + 0.05 * feasible["lower_price_score"]
    )

    return feasible.reset_index(drop=True)


def select_unique_recommendations(
    recommendation_df: pd.DataFrame,
) -> Dict[str, Optional[pd.Series]]:
    if recommendation_df.empty:
        return {
            "Best Match": None,
            "Budget Optimized": None,
            "Premium": None,
        }

    used = set()
    selections = {}

    score_columns = [
        ("Best Match", "best_match_score"),
        ("Budget Optimized", "budget_optimized_score"),
        ("Premium", "premium_score"),
    ]

    for name, score_column in score_columns:
        ordered = recommendation_df.sort_values(
            by=score_column,
            ascending=False,
        )

        selected = None
        for _, row in ordered.iterrows():
            signature = bundle_signature(row)
            if signature not in used:
                used.add(signature)
                selected = row
                break

        selections[name] = selected

    return selections


def row_to_bundle(
    row: pd.Series,
    catalog: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    bundle = {}

    category_columns = {
        "Toilet": "Toilet_id",
        "Sink": "Sink_id",
        "Faucet": "Faucet_id",
        "Vanity": "Vanity_id",
        "Bathtub": "Bathtub_id",
    }

    for category, id_column in category_columns.items():
        product_id = row.get(id_column)

        if is_missing(product_id):
            continue

        matches = catalog[
            catalog["product_id"].astype(str)
            == str(product_id)
        ]

        if not matches.empty:
            bundle[category] = matches.iloc[0].to_dict()

    return bundle


# ---------------------------------------------------------------------
# Public end-to-end API used by app.py
# ---------------------------------------------------------------------

def generate_recommendations(
    length_ft: float,
    width_ft: float,
    budget_inr: float,
    theme: str,
    required_fixtures: List[str],
) -> Dict[str, Any]:
    if safe_float(length_ft) <= 0 or safe_float(width_ft) <= 0:
        raise ValueError("Bathroom dimensions must be greater than zero.")

    if safe_float(budget_inr) <= 0:
        raise ValueError("Budget must be greater than zero.")

    if not required_fixtures:
        raise ValueError("Select at least one required fixture.")

    catalog = load_catalog()

    requirements = {
        "length_ft": float(length_ft),
        "width_ft": float(width_ft),
        "budget_inr": float(budget_inr),
        "theme": theme or "Minimalist Modern",
        "required_categories": [
            normalize_category(category)
            for category in required_fixtures
        ],
        "preferred_features": [
            "clean design",
            "minimalist",
            "water efficient",
        ],
        "door_width_mm": DEFAULT_DOOR_WIDTH_MM,
        "door_clearance_mm": DEFAULT_DOOR_CLEARANCE_MM,
    }

    # Remove duplicate categories while preserving order.
    requirements["required_categories"] = list(
        dict.fromkeys(requirements["required_categories"])
    )

    initialize_retrieval()

    bundles = generate_bundle_candidates(
        catalog,
        requirements,
        top_k_per_category=8,
    )

    evaluated = [
        evaluate_bundle(bundle, requirements)
        for bundle in bundles
    ]

    recommendation_df = build_recommendation_dataframe(evaluated)
    selections = select_unique_recommendations(recommendation_df)

    return {
        "requirements": requirements,
        "recommendation_df": recommendation_df,
        "recommendations": selections,
        "catalog_count": len(catalog),
        "bundle_count": len(bundles),
        "feasible_count": len(recommendation_df),
    }


def create_validation_markdown(
    row: pd.Series,
    requirements: Dict[str, Any],
    bundle: Dict[str, Dict[str, Any]],
) -> str:
    total = safe_float(row.get("total_price_inr"))
    budget = safe_float(requirements.get("budget_inr"))
    remaining = budget - total

    return "\n".join([
        "### Constraint Validation",
        "",
        f"- **Budget:** {row.get('budget_status', 'UNKNOWN')} "
        f"— ₹{total:,.0f} of ₹{budget:,.0f}",
        f"- **Budget remaining:** ₹{remaining:,.0f}",
        f"- **Space:** {row.get('space_status', 'UNKNOWN')}",
        f"- **Compatibility:** {row.get('compatibility_status', 'UNKNOWN')}",
        f"- **Theme score:** {safe_float(row.get('theme_score')):.2f}",
        f"- **Feature score:** {safe_float(row.get('feature_score')):.2f}",
        "",
        "The spatial result is a prototype-level 2D feasibility check, "
        "not a construction drawing.",
    ])


def create_bundle_markdown(
    row: pd.Series,
    bundle: Dict[str, Dict[str, Any]],
) -> str:
    lines = [
        f"### Total: ₹{safe_float(row.get('total_price_inr')):,.0f}",
        "",
    ]

    for category, product in bundle.items():
        lines.extend([
            f"**{category}**",
            str(product.get("product_name", "Product not specified")),
            f"Product ID: `{product.get('product_id', 'N/A')}`",
            f"Price: ₹{safe_float(get_product_price(product)):,.0f}",
            "",
        ])

    return "\n".join(lines)


def plot_bathroom_layout(
    layout: Dict[str, Any],
    room_length_mm: float,
    room_width_mm: float,
    door_width_mm: float = DEFAULT_DOOR_WIDTH_MM,
    door_clearance_mm: float = DEFAULT_DOOR_CLEARANCE_MM,
    title: str = "KOHLER AI Bathroom Designer — 2D Layout",
    show_clearance: bool = True,
    show_dimensions: bool = True,
):
    """
    Render the bathroom plan in the same visual style as the reference layout:

    - White/light plotting area with a strong black room boundary.
    - Toilet at the upper-left and sink at the upper-right when possible.
    - Faucet shown as a small red mounted component directly above the sink.
    - Blue toilet, green sink, red faucet visual language.
    - Light dashed clearance rectangles around floor fixtures.
    - Bottom-centered door with black swing arc and dotted door-clearance box.
    - Product IDs and dimensions displayed close to each fixture.
    """
    if layout is None or layout.get("status") != "PASS":
        return None

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # -----------------------------------------------------------------
    # Room boundary
    # -----------------------------------------------------------------
    ax.add_patch(
        Rectangle(
            (0, 0),
            room_width_mm,
            room_length_mm,
            fill=False,
            edgecolor="black",
            linewidth=3.0,
            zorder=5,
        )
    )

    # -----------------------------------------------------------------
    # Door + swing + clearance
    # -----------------------------------------------------------------
    door_x = (room_width_mm - door_width_mm) / 2

    ax.plot(
        [door_x, door_x + door_width_mm],
        [0, 0],
        color="#168ac4",
        linewidth=10,
        solid_capstyle="butt",
        zorder=8,
    )

    # Swing arc matches the reference: opens into the room.
    ax.add_patch(
        Arc(
            (door_x, 0),
            2 * door_width_mm,
            2 * door_width_mm,
            theta1=0,
            theta2=90,
            color="black",
            linewidth=2.0,
            linestyle="--",
            zorder=6,
        )
    )

    if show_clearance:
        ax.add_patch(
            Rectangle(
                (door_x, 0),
                door_width_mm,
                door_clearance_mm,
                fill=False,
                edgecolor="#777777",
                linewidth=1.5,
                linestyle=":",
                zorder=2,
            )
        )
        ax.plot(
            [door_x, door_x + door_width_mm],
            [0, door_clearance_mm],
            color="#ff8c00",
            linewidth=1.8,
            linestyle="--",
            zorder=3,
        )
        ax.text(
            door_x + door_width_mm / 2,
            door_clearance_mm / 2,
            "Door\nClearance",
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
            zorder=4,
        )

    ax.text(
        door_x + door_width_mm / 2,
        -105,
        "DOOR",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=10,
    )

    # -----------------------------------------------------------------
    # Fixture drawing
    # -----------------------------------------------------------------
    fixture_colors = {
        "Toilet": "#a9d9ea",
        "Sink": "#8fe58f",
        "Faucet": "#e53935",
        "Vanity": "#e8d7b7",
        "Bathtub": "#dca8df",
    }

    # Draw floor fixtures first. This preserves the reference ordering.
    floor_placements = [
        p for p in layout.get("placements", [])
        if not bool(p.get("mounted_fixture", False))
    ]

    mounted_placements = [
        p for p in layout.get("placements", [])
        if bool(p.get("mounted_fixture", False))
    ]

    # Put toilet before sink visually even if dictionary order changes.
    category_order = {
        "Toilet": 0,
        "Sink": 1,
        "Vanity": 2,
        "Bathtub": 3,
        "Other": 4,
    }
    floor_placements.sort(
        key=lambda p: category_order.get(normalize_category(p.get("category")), 99)
    )

    for placement in floor_placements:
        category = normalize_category(placement.get("category", "Fixture"))
        x = float(placement["x_mm"])
        y = float(placement["y_mm"])
        width = float(placement["width_mm"])
        depth = float(placement["depth_mm"])

        face = fixture_colors.get(category, "#b9c2cc")

        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                depth,
                facecolor=face,
                edgecolor="black",
                linewidth=2.0,
                alpha=0.78,
                zorder=10,
            )
        )

        # Clearance rectangle, visually similar to the reference image.
        if show_clearance:
            clearance = safe_float(
                placement.get("clearance_mm"),
                SPATIAL_CLEARANCE_MM,
            )
            ax.add_patch(
                Rectangle(
                    (x - clearance, y - clearance),
                    width + 2 * clearance,
                    depth + 2 * clearance,
                    fill=False,
                    edgecolor="#999999",
                    linewidth=1.2,
                    linestyle="--",
                    alpha=0.55,
                    zorder=4,
                )
            )

        # Main fixture label.
        label = f"{category}\n{width:.0f} × {depth:.0f} mm"
        ax.text(
            x + width / 2,
            y + depth / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=12,
        )

        # Product ID below the fixture, as in the reference image.
        product_id = str(placement.get("product_id", ""))
        if product_id:
            ax.text(
                x + width / 2,
                y - 35,
                product_id,
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
                zorder=12,
            )

        # Product name above the fixture, truncated so the plan remains clean.
        product_name = str(placement.get("product_name", ""))
        if product_name:
            max_chars = 34 if category == "Toilet" else 30
            short_name = (
                product_name[:max_chars - 1] + "…"
                if len(product_name) > max_chars
                else product_name
            )
            ax.text(
                x + width / 2,
                y + depth + 45,
                short_name,
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
                zorder=12,
            )

    # -----------------------------------------------------------------
    # Faucet mounted directly above the sink
    # -----------------------------------------------------------------
    for placement in mounted_placements:
        category = normalize_category(placement.get("category", "Faucet"))
        x = float(placement["x_mm"])
        y = float(placement["y_mm"])
        width = float(placement["width_mm"])
        depth = float(placement["depth_mm"])

        # Small red component matching the reference image.
        faucet_width = max(28.0, min(width, 55.0))
        faucet_depth = max(45.0, min(depth, 85.0))
        faucet_x = x + (width - faucet_width) / 2

        # Keep the visible faucet on the sink's upper/back edge.
        faucet_y = y + depth - faucet_depth

        ax.add_patch(
            Rectangle(
                (faucet_x, faucet_y),
                faucet_width,
                faucet_depth,
                facecolor="#e53935",
                edgecolor="black",
                linewidth=1.2,
                alpha=0.95,
                zorder=16,
            )
        )

        # Small vertical connector into the sink, making the attachment obvious.
        ax.plot(
            [faucet_x + faucet_width / 2, faucet_x + faucet_width / 2],
            [faucet_y, faucet_y - min(35.0, faucet_depth / 2)],
            color="#e53935",
            linewidth=3,
            zorder=15,
        )

        ax.text(
            faucet_x + faucet_width / 2,
            faucet_y + faucet_depth + 18,
            "Faucet",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#333333",
            zorder=17,
        )

        product_id = str(placement.get("product_id", ""))
        if product_id:
            ax.text(
                faucet_x + faucet_width / 2,
                faucet_y - 12,
                product_id,
                ha="center",
                va="top",
                fontsize=6.5,
                color="#333333",
                zorder=17,
            )

    # -----------------------------------------------------------------
    # Room dimensions
    # -----------------------------------------------------------------
    if show_dimensions:
        # Width dimension below the room.
        ax.annotate(
            "",
            xy=(0, -225),
            xytext=(room_width_mm, -225),
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": 1.4,
                "color": "black",
            },
        )
        ax.text(
            room_width_mm / 2,
            -315,
            f"{room_width_mm / FT_TO_MM:.1f} ft",
            ha="center",
            va="center",
            fontsize=11,
            color="black",
        )

        # Length dimension to the right of the room.
        ax.annotate(
            "",
            xy=(room_width_mm + 220, 0),
            xytext=(room_width_mm + 220, room_length_mm),
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": 1.4,
                "color": "black",
            },
        )
        ax.text(
            room_width_mm + 310,
            room_length_mm / 2,
            f"{room_length_mm / FT_TO_MM:.1f} ft",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            color="black",
        )

    # -----------------------------------------------------------------
    # Final theme / axes
    # -----------------------------------------------------------------
    ax.set_xlim(-420, room_width_mm + 520)
    ax.set_ylim(-470, room_length_mm + 260)
    ax.set_aspect("equal")
    ax.set_xlabel("Width (mm)", fontsize=10)
    ax.set_ylabel("Length (mm)", fontsize=10)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.grid(alpha=0.14, linewidth=0.7)

    # Keep the reference's clean presentation without a heavy legend.
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    return fig

def save_layout_image(
    layout: Dict[str, Any],
    requirements: Dict[str, Any],
    option_name: str,
) -> Optional[str]:
    if not layout or layout.get("status") != "PASS":
        return None

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        option_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    path = output_dir / f"{safe_name}_layout.png"

    fig = plot_bathroom_layout(
        layout,
        room_length_mm=feet_to_mm(requirements["length_ft"]),
        room_width_mm=feet_to_mm(requirements["width_ft"]),
        door_width_mm=requirements.get(
            "door_width_mm", DEFAULT_DOOR_WIDTH_MM
        ),
        door_clearance_mm=requirements.get(
            "door_clearance_mm", DEFAULT_DOOR_CLEARANCE_MM
        ),
        title=f"KOHLER AI Bathroom Designer — {option_name}",
    )

    if fig is None:
        return None

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def explain_design(
    row: pd.Series,
    bundle: Dict[str, Dict[str, Any]],
    requirements: Dict[str, Any],
    layout: Dict[str, Any],
) -> str:
    context = {
        "bathroom": {
            "length_ft": requirements["length_ft"],
            "width_ft": requirements["width_ft"],
            "budget_inr": requirements["budget_inr"],
            "theme": requirements["theme"],
        },
        "selected_products": [
            {
                "category": category,
                "product_id": product.get("product_id"),
                "product_name": product.get("product_name"),
                "price_inr": get_product_price(product),
                "dimensions_mm": {
                    "width": product.get("dimensions.width_mm"),
                    "depth": product.get("dimensions.depth_mm"),
                    "height": product.get("dimensions.height_mm"),
                },
                "material": product.get("material"),
                "color": product.get("color"),
                "installation": product.get("installation.type"),
                "features": product.get("features"),
            }
            for category, product in bundle.items()
        ],
        "calculated": {
            "total_price_inr": safe_float(row.get("total_price_inr")),
            "budget_status": row.get("budget_status"),
            "space_status": row.get("space_status"),
            "compatibility_status": row.get("compatibility_status"),
            "theme_score": safe_float(row.get("theme_score")),
            "feature_score": safe_float(row.get("feature_score")),
            "layout_reason": layout.get("reason") if layout else "Not available",
        },
    }

    api_key = os.getenv("GROQ_API_KEY")
    if load_dotenv is not None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        total = safe_float(row.get("total_price_inr"))
        budget = safe_float(requirements.get("budget_inr"))
        return (
            f"The selected {requirements['theme']} design contains "
            f"{', '.join(bundle.keys())}. The estimated bundle cost is "
            f"₹{total:,.0f} against a budget of ₹{budget:,.0f}. "
            f"Budget validation is {row.get('budget_status', 'UNKNOWN')} "
            f"and spatial validation is {row.get('space_status', 'UNKNOWN')}. "
            "The prototype does not have complete cross-product compatibility "
            "data, so compatibility is reported conservatively."
        )

    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=api_key,
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are the explanation component of a KOHLER AI Bathroom Designer.

Explain only the already-selected design using the verified context.
Never invent product specifications, prices, dimensions, features, or
compatibility. Never change the selection. If information is unavailable,
say it is not specified in the catalog data.

Explain:
- selected products
- budget fit
- spatial validation
- relation to the requested theme
- important limitations

Keep the explanation concise and customer-friendly.
Do not mention Python, prompts, databases, or internal implementation.
""",
            ),
            (
                "human",
                "Verified design context:\n{context}",
            ),
        ])

        response = (prompt | llm).invoke({
            "context": json.dumps(context, indent=2, default=str)
        })

        return str(response.content)

    except Exception as exc:
        total = safe_float(row.get("total_price_inr"))
        return (
            f"The design was validated with an estimated bundle cost of "
            f"₹{total:,.0f}. Budget status: {row.get('budget_status', 'UNKNOWN')}. "
            f"Space status: {row.get('space_status', 'UNKNOWN')}. "
            f"AI explanation is temporarily unavailable ({type(exc).__name__})."
        )


def create_final_design_result(
    recommendation_result: Dict[str, Any],
    selected_option: str = "Best Match",
) -> Dict[str, Any]:
    row = recommendation_result["recommendations"].get(selected_option)
    requirements = recommendation_result["requirements"]

    if row is None:
        return {
            "selected_option": selected_option,
            "available": False,
        }

    catalog = load_catalog()
    bundle = row_to_bundle(row, catalog)

    layout = build_layout_for_bundle(
        bundle=bundle,
        room_length_mm=feet_to_mm(requirements["length_ft"]),
        room_width_mm=feet_to_mm(requirements["width_ft"]),
        door_width_mm=requirements.get(
            "door_width_mm", DEFAULT_DOOR_WIDTH_MM
        ),
        door_clearance_mm=requirements.get(
            "door_clearance_mm", DEFAULT_DOOR_CLEARANCE_MM
        ),
        fixture_clearance_mm=SPATIAL_CLEARANCE_MM,
    )

    return {
        "available": True,
        "selected_option": selected_option,
        "requirements": requirements,
        "products": bundle,
        "total_price_inr": safe_float(row.get("total_price_inr")),
        "budget_inr": safe_float(requirements.get("budget_inr")),
        "remaining_budget_inr": (
            safe_float(requirements.get("budget_inr"))
            - safe_float(row.get("total_price_inr"))
        ),
        "budget_status": row.get("budget_status", "UNKNOWN"),
        "space_status": row.get("space_status", "UNKNOWN"),
        "compatibility_status": row.get(
            "compatibility_status", "UNKNOWN"
        ),
        "score": safe_float(row.get("final_score")),
        "layout": layout,
    }
