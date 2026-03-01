# ============================================================
# Container dimensions — single source of truth
# All other files import from here; do not hardcode these elsewhere.
# ============================================================

CONTAINER_LENGTH_CM   = 1203   # internal usable depth (back wall to door)
CONTAINER_WIDTH_CM    = 235    # internal usable width
CONTAINER_HEIGHT_CM   = 270    # internal usable height (full)
CONTAINER_DOOR_HEIGHT_CM = 250 # door opening height (shorter than interior)
CONTAINER_MAX_WEIGHT_KG  = 18000

ROW_GAP_CM = 5  # gap left between consecutive row-blocks along container length

# ============================================================
# Recommendation system — fill leftover container space
# ============================================================
RECOMMEND_OBJECTIVE = "min_leftover"
# Options (primary objective):
#   "min_leftover"  — minimise remaining length after fill (default)
#   "min_pallets"   — minimise extra pallets ordered (prefer blocks with most cm per pallet)
#   "max_weight"    — prefer taller/larger blocks (volume proxy for weight utilisation)
#   "max_value"     — prefer highest FOB value (requires 'Item price FOB' column in Excel)
RECOMMEND_SECONDARY_OBJECTIVE = "min_pallets"  # tiebreaker when primary scores are equal
