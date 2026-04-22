import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent / "libs"))

import torch
import torch.nn.functional as F
from mjai import Bot

try:
    from model import MahjongDecisionNet
    from gamestate import (
        RoundState, pai_to_idx, idx_to_pai,
        CALL_KIND_TO_IDX, TSUMO_ACTION_TO_IDX, tile37_to_base34, NUM_TILES
    )
except ImportError:
    from model import MahjongDecisionNet
    from gamestate import (
        RoundState, pai_to_idx, idx_to_pai,
        CALL_KIND_TO_IDX, TSUMO_ACTION_TO_IDX, tile37_to_base34, NUM_TILES
    )

BASE_DIR = pathlib.Path(__file__).parent

# Thresholds: only act if probability exceeds threshold
# dahai_model output: 0=none,1=chi_low,2=chi_mid,3=chi_high,4=pon,5=kan,6=hora
DAHAI_CALL_THRESHOLDS = {
    1: 0.75,  # chi_low
    2: 0.75,  # chi_mid
    3: 0.75,  # chi_high
    4: 0.72,  # pon
    5: 0.96,  # kan
    6: 0.50,  # hora
}

# tsumo_model action output: 0=none,1=dahai,2=reach,3=kan,4=hora
TSUMO_ACTION_THRESHOLDS = {
    2: 0.60,  # reach
    3: 0.92,  # kan
    4: 0.50,  # hora
}


class MyBot(Bot):
    def __init__(self, player_id: int):
        super().__init__(player_id=player_id)
        self.device = "cpu"
        self._model = None
        self._round_state = RoundState()

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return
        model = MahjongDecisionNet().to(self.device)
        model.dahai_model.load_state_dict(
            torch.load(BASE_DIR / "best_dahai.pt", map_location=self.device)
        )
        model.tsumo_model.load_state_dict(
            torch.load(BASE_DIR / "best_tsumo.pt", map_location=self.device)
        )
        model.eval()
        self._model = model

    # ----------------------------------------------------------
    # Sync our RoundState from mjai.Bot's built-in events
    # mjai.Bot calls think() after updating its own state,
    # so we rebuild RoundState from the raw events via self.events
    # ----------------------------------------------------------
    def _sync_round_state(self):
        """Rebuild RoundState from mjai.Bot's event history."""
        try:
            events = self.player_state.events  # list of dicts
            self._round_state = RoundState()
            for e in events:
                self._round_state.apply_event(e)
        except Exception:
            pass

    # ----------------------------------------------------------
    # Get state tensors from our RoundState
    # ----------------------------------------------------------
    def _get_state_tensors(self):
        x = self._round_state.to_feature(self.player_id).unsqueeze(0).to(self.device)
        hist, hist_mask = self._round_state.get_history(self.player_id)
        hist = hist.unsqueeze(0).to(self.device)
        hist_mask = hist_mask.unsqueeze(0).to(self.device)
        return x, hist, hist_mask

    # ----------------------------------------------------------
    # Predict best discard tile index (37-tile index)
    # ----------------------------------------------------------
    @torch.no_grad()
    def _predict_discard_idx(self) -> int:
        x, hist, hist_mask = self._get_state_tensors()
        _, tile_logits, _ = self._model.tsumo_model(x, hist, hist_mask)
        tile_logits = tile_logits[0]  # [37]

        # Build discard mask from our hand
        hand_mask = self._round_state.legal_discard_mask(self.player_id).to(self.device)
        tile_logits = tile_logits.masked_fill(~hand_mask, -1e9)
        return int(torch.argmax(tile_logits).item())

    # ----------------------------------------------------------
    # Find best discard tile name from hand using model
    # ----------------------------------------------------------
    def _model_discard(self) -> str:
        hand = self.tehai_mjai  # e.g. ["1m", "2m", "5mr", ...]

        try:
            self._sync_round_state()
            best_idx = self._predict_discard_idx()
            best_pai = idx_to_pai(best_idx)      # e.g. "5mr" or "1m"
            best_base = best_pai[:2]             # e.g. "5m" or "1m"

            # 1. Try exact match (handles red fives correctly)
            if best_pai in hand and not self.forbidden_tiles.get(best_base, False):
                return best_pai

            # 2. Try base match (e.g. "5m" matches "5mr" or "5m")
            for t in hand:
                if t[:2] == best_base and not self.forbidden_tiles.get(t[:2], False):
                    return t

            # 3. Fallback: pick any non-forbidden tile from hand
            for t in hand:
                if not self.forbidden_tiles.get(t[:2], False):
                    return t

        except Exception as e:
            print(f"[model_discard error] {e}", file=sys.stderr)

        # Final fallback: tsumogiri or first tile
        return self.last_self_tsumo or hand[0]

    # ----------------------------------------------------------
    # Main decision function called by mjai.Bot framework
    # ----------------------------------------------------------
    def think(self) -> str:
        self._load_model()

        # ---- Winning conditions (use mjai.Bot's reliable detection) ----
        if self.can_tsumo_agari:
            return self.action_tsumo_agari()

        if self.can_ron_agari:
            return self.action_ron_agari()

        # ---- Riichi ----
        if self.can_riichi and not self.self_riichi_accepted:
            # Use mjai.Bot's riichi candidate finder
            try:
                candidates = self.find_improving_tiles()
                for c in candidates:
                    tile = c["discard_tile"]
                    if not self.forbidden_tiles.get(tile[:2], False):
                        return self.action_riichi()
            except Exception:
                pass

        # ---- Discard (own tsumo turn) ----
        if self.can_discard:
            # After riichi accepted, must tsumogiri
            if self.self_riichi_accepted:
                return self.action_discard(self.last_self_tsumo)

            # Use our model to pick the best tile to discard
            tile = self._model_discard()
            return self.action_discard(tile)

        # ---- Pon ----
        if self.can_pon:
            try:
                pons = self.find_pon_candidates()
                for pon in pons:
                    if pon["current_shanten"] > pon["next_shanten"]:
                        return self.action_pon(consumed=pon["consumed"])
            except Exception:
                pass

        # ---- Chi ----
        if self.can_chi:
            try:
                chis = self.find_chi_candidates()
                if chis:
                    best_ukeire = max(c["next_ukeire"] for c in chis)
                    for chi in chis:
                        if (chi["current_shanten"] > chi["next_shanten"]
                                and chi["next_ukeire"] == best_ukeire):
                            return self.action_chi(consumed=chi["consumed"])
            except Exception:
                pass

        # ---- Default: pass ----
        return self.action_nothing()


if __name__ == "__main__":
    MyBot(player_id=int(sys.argv[1])).start()