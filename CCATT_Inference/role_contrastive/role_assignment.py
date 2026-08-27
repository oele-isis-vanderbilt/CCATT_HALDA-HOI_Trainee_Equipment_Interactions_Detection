"""
Role stabilization, in two parts:

1. resolve_singleton_conflicts() — per-frame: refactored from Model_Training.ipynb cell 58.
   Nurse/Physician/RT are "singleton" roles (only one trainee should hold each per frame);
   keeps the highest-confidence detection for each and reassigns lower-confidence conflicts
   to the next-best non-conflicting class, falling back to "Additional Staff".

2. TrackRoleSmoother — across time: a per-track_id CUMULATIVE (unbounded) majority vote over
   every frame that track has ever appeared in, so one tracked bounding box gets a single,
   increasingly stable role label for its whole lifetime rather than a label that can drift
   or flicker based only on a recent sliding window. Old votes are never discarded, so once
   enough frames agree the label effectively "locks in" for the rest of that track's life —
   a handful of noisy frames later on can't flip it back. This uses ByteTrack track_id
   continuity only to group "the same detection over time" — it is NOT person
   re-identification; no identity embedding is involved (see role_contrastive/ plan).

3. resolve_global_track_roles() — across tracks that actually coexisted: resolve_singleton_
   conflicts() only checks conflicts *within a single frame*, so two different tracks can each
   independently accumulate a "Nurse" majority without ever being cross-checked (e.g. one
   track is misclassified "Nurse" only on frames where the real Nurse happens to be
   off-screen). This function resolves that — but ONLY between tracks that were ever visible
   in the same frame together (a real conflict). Tracks that never coexisted (e.g. the real
   Nurse's ByteTrack ID fragmenting into a new track_id after a brief occlusion, or one
   trainee's shift simply ending before another's begins) are NOT forced to compete for the
   same role — both can validly be "Nurse" since they were never shown as two Nurses at once.
   An earlier version of this function enforced a single video-wide winner per role regardless
   of temporal overlap, which incorrectly demoted legitimate track fragments of the same real
   person to "Additional Staff" whenever a different, non-overlapping track outvoted them —
   conflict-awareness (via track_conflicts) fixes that.
   Requires a two-pass pipeline: collect every track's votes AND frame-level co-occurrence
   across the whole video first, then render using these finalized labels — see demo_role.py /
   gradcam_role.py.
"""

from collections import Counter, defaultdict, deque
from typing import Optional

import numpy as np

CLASS_NAMES = ["Nurse", "Physician", "RT", "Additional Staff"]
SINGLETON_ROLES = {"Nurse", "Physician", "RT"}
FALLBACK_ROLE = "Additional Staff"


def resolve_singleton_conflicts(
    labels: list[str],
    confidences: list[float],
    probs: Optional[np.ndarray] = None,
    class_names: list[str] = CLASS_NAMES,
) -> list[str]:
    """
    labels: role label per detection in a single frame.
    confidences: detection confidence per detection, same order as `labels`.
    probs: optional (N, num_classes) per-class probabilities per detection, used to pick the
        next-best non-conflicting class when reassigning; without it, conflicts fall back
        directly to FALLBACK_ROLE.

    Returns adjusted labels, same order/length as `labels`.
    """
    n = len(labels)
    result = list(labels)
    used_roles: set[str] = set()
    order = np.argsort(-np.asarray(confidences, dtype=float))

    for i in order:
        i = int(i)
        label = labels[i]
        if label in SINGLETON_ROLES and label in used_roles:
            reassigned = FALLBACK_ROLE
            if probs is not None:
                for rid in np.argsort(-probs[i]):
                    cand = class_names[rid]
                    if cand in SINGLETON_ROLES and cand not in used_roles:
                        reassigned = cand
                        break
                    if cand not in SINGLETON_ROLES:
                        reassigned = cand
                        break
            result[i] = reassigned
            if reassigned in SINGLETON_ROLES:
                used_roles.add(reassigned)
        else:
            result[i] = label
            if label in SINGLETON_ROLES:
                used_roles.add(label)

    assert len(result) == n
    return result


class TrackRoleSmoother:
    """Per-track_id cumulative (unbounded) majority vote over every (post-singleton-resolved)
    role prediction that track has ever produced, so one tracked bounding box gets a single,
    stable role label for its whole lifetime — old evidence is never discarded/aged out, so
    the label only gets more locked-in over time, not more prone to drifting."""

    def __init__(self):
        self._history: dict[int, Counter] = {}

    def update(self, track_id: int, role_label: str) -> str:
        counts = self._history.setdefault(track_id, Counter())
        counts[role_label] += 1
        return counts.most_common(1)[0][0]

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._history.clear()
        else:
            self._history.pop(track_id, None)

    def all_votes(self) -> dict[int, Counter]:
        """Returns the full per-track vote history, for resolve_global_track_roles()."""
        return self._history


def build_track_conflicts(per_frame_track_ids: list[list[int]]) -> dict[int, set[int]]:
    """
    per_frame_track_ids: one list of track_ids per processed frame (the track_ids visible
        together in that frame).

    Returns {track_id: {other_track_ids it was ever visible in the same frame with}} — the
    conflict graph consumed by resolve_global_track_roles().
    """
    conflicts: dict[int, set[int]] = defaultdict(set)
    for ids_in_frame in per_frame_track_ids:
        for i, a in enumerate(ids_in_frame):
            for b in ids_in_frame[i + 1 :]:
                conflicts[a].add(b)
                conflicts[b].add(a)
    return conflicts


def resolve_global_track_roles(
    track_votes: dict[int, "Counter[str]"],
    track_conflicts: dict[int, set[int]],
    class_names: list[str] = CLASS_NAMES,
) -> dict[int, str]:
    """
    track_votes: {track_id: Counter({role_label: count, ...})} — each track's full per-frame
        vote history (already per-frame singleton-resolved via resolve_singleton_conflicts)
        accumulated across the whole video, e.g. from TrackRoleSmoother.all_votes().
    track_conflicts: {track_id: {other track_ids ever visible in the same frame}}, from
        build_track_conflicts() — defines which tracks actually compete for a role.

    Each track proposes its OWN current-best remaining label (ranked by its own vote counts,
    most-common first) and only ever competes against other tracks proposing that SAME label —
    a track whose own strongest signal is "Additional Staff" is never dragged into a Nurse/
    Physician/RT contest just because it has a handful of stray singleton votes lower in its
    own ranking. When multiple tracks do propose the same singleton role, only mutually
    CONFLICTING ones (ever visible in the same frame) compete — the strongest wins, losers
    drop to their own next-ranked label and re-propose. Non-conflicting tracks (e.g. the same
    real person's track fragmenting after an occlusion) can all keep the same role.

    Uses deferred acceptance (Gale-Shapley style): a role's holder is only ever *tentative* —
    if a stronger conflicting track proposes that same role later, it bumps the weakest
    conflicting current holder out (who then tries its own next-ranked label), rather than
    letting whichever track happened to propose first permanently lock the role in regardless
    of strength.

    Returns {track_id: final_role_label}.
    """
    prefs: dict[int, list[str]] = {tid: [r for r, _ in c.most_common()] for tid, c in track_votes.items()}
    next_idx: dict[int, int] = dict.fromkeys(track_votes, 0)

    final: dict[int, str] = {}
    role_holders: dict[str, set[int]] = defaultdict(set)  # role -> tentatively-accepted track_ids
    queue: deque = deque(track_votes.keys())

    while queue:
        tid = queue.popleft()
        idx = next_idx[tid]
        if idx >= len(prefs[tid]):
            final[tid] = FALLBACK_ROLE
            continue
        role = prefs[tid][idx]
        if role not in SINGLETON_ROLES:
            final[tid] = role
            continue

        my_conflicts = track_conflicts.get(tid, set())
        conflicting_holders = [h for h in role_holders[role] if h in my_conflicts]

        if not conflicting_holders:
            role_holders[role].add(tid)  # tentatively accepted; may still be bumped later
            continue

        weakest = min(conflicting_holders, key=lambda h: track_votes[h][role])
        if track_votes[tid][role] > track_votes[weakest][role]:
            role_holders[role].discard(weakest)
            role_holders[role].add(tid)
            next_idx[weakest] += 1
            queue.append(weakest)
        else:
            next_idx[tid] += 1
            queue.append(tid)

    # By the time the queue is empty, every track is either already in `final` (it proposed a
    # non-singleton label, or exhausted its preferences) or is a never-bumped tentative holder.
    for role, holders in role_holders.items():
        for tid in holders:
            final[tid] = role

    return final
