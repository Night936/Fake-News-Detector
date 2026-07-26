import math

from Utils.textFeatures import extractFeatures

VADER_COMPOUND_INDEX = 3  # position of VADER compound score in extractFeatures' output


def extractCommentSectionFeatures(bodies, ups=None, top_level=None):
    n_total = len(bodies)
    if n_total == 0:
        # no discussion at all is itself a signal (e.g. bot-posted, or too new) -- keep the
        # post, don't drop it, just mark it with a zero vector
        return [0.0] * 8

    ups = ups or [0] * n_total
    top_level = top_level if top_level is not None else [True] * n_total

    per_comment = [extractFeatures(b) for b in bodies]
    compounds = [f[VADER_COMPOUND_INDEX] for f in per_comment]

    avg_compound = sum(compounds) / len(compounds)
    spread = max(compounds) - min(compounds)          # disagreement / controversy signal
    avg_ups = sum(ups) / len(ups)
    top_level_frac = sum(1 for t in top_level if t) / len(top_level)
    downvoted_frac = sum(1 for u in ups if u < 0) / len(ups)

    return [
        math.log1p(n_total),   # comment volume, log-scaled to match textFeatures convention
        avg_compound,
        spread,
        avg_ups,
        top_level_frac,
        downvoted_frac,
        min(compounds),
        max(compounds),
    ]