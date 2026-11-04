"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import InputRequiredResult

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311.server")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
READ_ONLY = {"readOnlyHint": True}

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
    "hour": lambda r: f"{r.opened.hour:02d}:00",
}
SOURCES = sorted({r.source for r in REQUESTS if r.source})

GroupBy = Literal["neighborhood", "service", "agency", "source", "weekday", "status", "hour"]

# --- helpers -----------------------------------------------------------------

HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def mask_address(address: str) -> str:
    """Reduce "994 HARRISON ST, SAN FRANCISCO, ..." to "900 block of HARRISON ST"."""
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def public(request: Request) -> dict:
    """A request as the model may see it: the address only to the block."""
    row = request.as_dict()
    row["address"] = mask_address(request.address)
    return row


def check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact names.")


def matching(
    service: str | None = None,
    neighborhood: str | None = None,
    status: str | None = None,
    source: str | None = None,
) -> list[Request]:
    check_name(service, SERVICES, "service")
    check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    check_name(source, SOURCES, "source")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
        and (source is None or r.source == source)
    ]


def resolve_place(name: str) -> tuple[str, str]:
    """Return ("official" | "local", exact name), matching case-insensitively."""
    lowered = name.strip().lower()
    for official in NEIGHBORHOODS:
        if official.lower() == lowered:
            return "official", official
    for local in LOCAL_NAMES:
        if local.lower() == lowered:
            return "local", local
    suggestions = closest(name, NEIGHBORHOODS + LOCAL_NAMES)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown neighborhood {name!r}.{hint} Call dataset_info for the official names.")


def in_local_area(
    place: str, ctx: Context, boundary: str | None, verb: str
) -> list[Request] | InputRequiredResult | str:
    """The requests in local area `place`, asking which part when it spans several official neighborhoods.

    Returns the requests, or something the tool should return as is: the
    question to ask, or a message (saying nothing was `verb`) when the user
    declines. Nothing is saved between calls: the retry re-derives everything.
    """
    in_area = [r for r in REQUESTS if r.local_name == place]
    parts = sorted({r.neighborhood for r in in_area if r.neighborhood})
    if len(parts) <= 1:
        return in_area

    all_option = f"All of {place}"
    if boundary is None:
        key = "boundary"
        choice = read_choice(ctx, key)
        if choice is None:
            return ask_choice(
                key,
                f"{place} overlaps {len(parts)} official neighborhoods. Which part do you mean?",
                parts + [all_option],
            )
        if not choice.accepted:
            return (
                f"Nothing was {verb}: no part of {place} was chosen. "
                f"Call again with boundary set to one of {parts} or 'all'."
            )
        boundary = choice.value

    if boundary.strip().lower() in ("all", all_option.lower()):
        return in_area
    if boundary not in parts:
        raise ToolError(f"boundary must be one of {parts} or 'all' for {place}.")
    return [r for r in in_area if r.neighborhood == boundary]


def ranked(counter: Counter) -> dict[str, int]:
    pairs = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return dict(pairs[:MAX_ROWS])


def medians_by(rows: list[Request], group_by: str) -> dict[str, float]:
    key = GROUPS[group_by]
    hours = defaultdict(list)
    for r in rows:
        if r.hours_to_close is not None and key(r):
            hours[key(r)].append(r.hours_to_close)
    return {
        group: round(median(values), 1)
        for group, values in hours.items()
        if len(values) >= MIN_CLOSED_FOR_MEDIAN
    }


# --- Exercise 1 --------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "sources": SOURCES,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


COLUMNS = {
    "request_id": "Unique ID of the request. Pass it to get_request.",
    "opened": "When the request was filed (ISO 8601, local time).",
    "closed": "When the city closed it (ISO 8601), or null while it is still open.",
    "status": "Open or Closed.",
    "service": "What the request is about, such as Encampment or Graffiti. Exact names are in dataset_info.",
    "subtype": "A finer category within the service.",
    "agency": "The city agency responsible for handling it.",
    "neighborhood": "One of the 41 official analysis neighborhoods, or empty when unknown.",
    "local_name": "The name people use for the area, such as Lower Nob Hill. It can span several official neighborhoods.",
    "address": "Where the problem is, shown only to the block (e.g. 900 block of HARRISON ST). Intersections are shown as is.",
    "source": "How it was filed, such as Phone, Web, or Mobile.",
}


@mcp.resource(
    "sf311://dataset/columns",
    name="dataset_columns",
    description="What each field of a 311 request means. Read this before interpreting tool results.",
    mime_type="application/json",
)
def dataset_columns() -> dict:
    """The fields every request row has, and what each one means."""
    return {
        "columns": COLUMNS,
        "derived": {"hours_to_close": "Hours from opened to closed, only on closed requests (close_time_extremes)."},
    }


# --- Exercise 2 --------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def search_requests(
    ctx: Context,
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
    boundary: str | None = None,
) -> list[dict] | str | InputRequiredResult:
    """List individual 311 requests, newest first, with addresses shown only to the block.

    Use this to show examples or look at a few specific requests. Do not use it
    to count or rank: use count_requests for totals and breakdowns, and
    median_close_hours for how fast requests close. `service` must be an exact
    name from dataset_info. `neighborhood` is an official neighborhood or a
    local name such as "Lower Nob Hill"; when a local name spans several
    official neighborhoods the user is asked which part, unless you pass
    `boundary` (one of those official neighborhoods, or "all"). `limit` is
    1 to 50.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}; use count_requests if you need totals.")
    check_name(service, SERVICES, "service")

    area = None
    if neighborhood is not None and neighborhood not in NEIGHBORHOODS:
        kind, place = resolve_place(neighborhood)
        if kind == "official":
            neighborhood = place
        else:
            area = in_local_area(place, ctx, boundary, "listed")
            if not isinstance(area, list):
                return area
            neighborhood = None

    rows = matching(service, neighborhood, status)
    if area is not None:
        ids = {r.request_id for r in area}
        rows = [r for r in rows if r.request_id in ids]
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [public(r) for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: GroupBy | None = None,
    status: Literal["Open", "Closed"] | None = None,
    source: str | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally broken down by one dimension.

    Without `group_by`, returns one number. With `group_by` (neighborhood,
    service, agency, source, weekday, status, or hour of day opened), returns
    a dict of group to count, largest first, at most 50 groups. Filter with
    exact `service`/`neighborhood` names from dataset_info, `status` (Open or
    Closed), and `source` (e.g. Phone, Web, Mobile). Use this for "how many",
    "which has the most", "busiest hour", and "still open" questions. For a
    local neighborhood name like "Lower Nob Hill", use count_in_neighborhood.
    """
    rows = matching(service, neighborhood, status, source)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    return ranked(Counter(key(r) for r in rows if key(r)))


@mcp.tool(annotations=READ_ONLY)
def median_close_hours(
    service: str | None = None,
    group_by: Literal["neighborhood", "agency"] | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests, rounded to 0.1.

    Without `group_by`, returns one number (None if nothing closed). With
    `group_by` (neighborhood or agency), returns a dict of group to median
    hours, fastest first, leaving out groups with fewer than 10 closed
    requests. Use close_time_extremes for the single longest or shortest
    close, and neighborhood_speed_gap to compare services by how unevenly
    they are handled across neighborhoods.
    """
    rows = [r for r in matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    medians = medians_by(rows, group_by)
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations=READ_ONLY)
def get_request(request_id: str) -> dict:
    """Return every field of one 311 request by its request_id, address shown only to the block.

    Use this after search_requests to look at one request in detail. To find
    requests, use search_requests; this tool needs an exact ID.
    """
    log.info("get_request %s by %s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with ID {request_id!r}. Use search_requests to find valid IDs.")
    return public(request)


# --- Exercise 3: extra analysis tools ----------------------------------------


@mcp.tool(annotations=READ_ONLY)
def close_time_extremes(
    service: str | None = None,
    neighborhood: str | None = None,
    which: Literal["longest", "shortest"] = "longest",
    limit: int = 5,
) -> list[dict]:
    """The closed 311 requests that took the longest (or shortest) time to close.

    Returns up to `limit` (1 to 50) requests with an `hours_to_close` field
    rounded to 0.1, addresses shown only to the block. Use this for "the
    longest any request took to close". For typical close times use
    median_close_hours instead.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}.")
    rows = [r for r in matching(service, neighborhood) if r.hours_to_close is not None]
    rows.sort(key=lambda r: r.hours_to_close, reverse=(which == "longest"))
    return [public(r) | {"hours_to_close": round(r.hours_to_close, 1)} for r in rows[:limit]]


@mcp.tool(annotations=READ_ONLY)
def neighborhood_speed_gap(service: str | None = None, limit: int = 10) -> list[dict]:
    """Rank services by the gap between their fastest and slowest neighborhood median close times.

    For each service (or only `service`), compares median hours-to-close
    across neighborhoods with at least 10 closed requests, and returns the
    fastest and slowest neighborhood, their medians, and `gap_hours`, widest
    gap first, at most `limit` (1 to 50) rows. Use this for "which service has
    the widest gap between neighborhoods". For one service's medians by
    neighborhood, use median_close_hours with group_by="neighborhood".
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}.")
    services = [service] if service else SERVICES
    check_name(service, SERVICES, "service")
    results = []
    for name in services:
        medians = medians_by([r for r in REQUESTS if r.service == name], "neighborhood")
        if len(medians) < 2:
            continue
        fastest = min(medians, key=lambda n: (medians[n], n))
        slowest = max(medians, key=lambda n: (medians[n], n))
        results.append(
            {
                "service": name,
                "gap_hours": round(medians[slowest] - medians[fastest], 1),
                "fastest_neighborhood": fastest,
                "fastest_median_hours": medians[fastest],
                "slowest_neighborhood": slowest,
                "slowest_median_hours": medians[slowest],
            }
        )
    results.sort(key=lambda row: (-row["gap_hours"], row["service"]))
    return results[:limit]


# --- Exercise 4 --------------------------------------------------------------


@mcp.tool(annotations=READ_ONLY)
def count_in_neighborhood(
    name: str,
    ctx: Context,
    boundary: str | None = None,
    breakdown: bool = False,
) -> int | dict[str, int] | str | InputRequiredResult:
    """Count 311 requests in a neighborhood, by official name or by the local name people use.

    Accepts any of the 41 official neighborhoods (e.g. "Mission") or a local
    name (e.g. "Lower Nob Hill", "Duboce Triangle"). When a local name spans
    several official neighborhoods, the user is asked which part to count;
    pass `boundary` (one of those official neighborhoods, or "all") to skip
    the question. Set `breakdown=true` to get a dict of official neighborhood
    to count within the local area, largest first, without asking. For
    filters by service or a breakdown by other dimensions, use count_requests.
    """
    kind, place = resolve_place(name)
    if kind == "official":
        return sum(1 for r in REQUESTS if r.neighborhood == place)

    if breakdown:
        return ranked(Counter(r.neighborhood for r in REQUESTS if r.local_name == place and r.neighborhood))
    rows = in_local_area(place, ctx, boundary, "counted")
    return len(rows) if isinstance(rows, list) else rows


if __name__ == "__main__":
    mcp.run()
