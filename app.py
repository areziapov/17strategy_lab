
from pathlib import Path
from itertools import combinations
import math
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = APP_DIR / "master_corrected_v3_to_2026-04-19.csv"
HARD_CUTOFF = pd.Timestamp("2026-04-19 23:59:59")

SIGNALS = [
    "main","stat","green_spi","blue_spi","red_spi","orange_hsr","red_hsr",
    "cloud4_long","cloud4_short"
]

FRIENDLY = {
    "main":"MAIN",
    "stat":"STAT",
    "green_spi":"Green SPI",
    "blue_spi":"Blue SPI",
    "red_spi":"Red SPI",
    "orange_hsr":"Orange HSR",
    "red_hsr":"Red HSR",
    "cloud4_long":"Market Cloud 4H Long",
    "cloud4_short":"Market Cloud 4H Short",
}

LONGISH = ["main","stat","green_spi","blue_spi","cloud4_long"]
SHORTISH = ["red_spi","orange_hsr","red_hsr","cloud4_short"]

st.set_page_config(page_title="Strategy Lab v1.4", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["bar_time_display"] = pd.to_datetime(df["bar_time_display"])
    df = df[df["bar_time_display"] <= HARD_CUTOFF].copy()
    for c in SIGNALS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 4H bars => 6 bars/day. Regime uses only trailing information.
    df["ret_30d"] = df["close"] / df["close"].shift(180) - 1
    df["ret_90d"] = df["close"] / df["close"].shift(540) - 1
    df["vol_30d"] = df["close"].pct_change().rolling(180).std() * np.sqrt(6 * 365)
    df["year"] = df["bar_time_display"].dt.year
    return df.reset_index(drop=True)

def apply_regime(df, threshold):
    x = df.copy()
    x["regime"] = np.select(
        [x["ret_90d"] >= threshold, x["ret_90d"] <= -threshold],
        ["bull", "bear"],
        default="transition",
    )
    return x

def count_feature(series, bars, include_current=False):
    s = series if include_current else series.shift(1)
    return s.rolling(bars, min_periods=1).sum().fillna(0)

def context_mask(df, context_rules):
    mask = pd.Series(True, index=df.index)
    for rule in context_rules:
        bars = max(1, int(math.ceil(rule["lookback_h"] / 4)))
        counts = count_feature(df[rule["signal"]], bars, include_current=False)
        mask &= counts >= int(rule["min_count"])
    return mask

def group_mask(df, group):
    """
    One logical group = current-bar signal logic AND that group's own memory conditions.

    Example:
      signals=[MAIN], mode=ANY
      contexts=[Green SPI >=1 in previous 24h]

    means:
      MAIN NOW AND Green SPI occurred in the previous 24h.
    """
    signals = group.get("signals", [])
    mode = group.get("mode", "ANY")
    k = int(group.get("k", 1))
    if not signals:
        return pd.Series(False, index=df.index)

    block = df[signals].astype(bool)
    if mode == "ALL":
        base = block.all(axis=1)
    elif mode == "ANY":
        base = block.any(axis=1)
    else:
        k = min(max(1, k), len(signals))
        base = block.sum(axis=1) >= k

    contexts = group.get("contexts", [])
    if contexts:
        base &= context_mask(df, contexts)
    return base

def grouped_logic_mask(df, groups, outer_operator):
    active = [g for g in groups if g.get("signals")]
    if not active:
        return pd.Series(False, index=df.index)

    masks = [group_mask(df, g) for g in active]
    out = masks[0].copy()
    for m in masks[1:]:
        if outer_operator == "AND":
            out &= m
        else:
            out |= m
    return out

def group_text(group):
    sigs = group.get("signals", [])
    if not sigs:
        return "—"
    mode = group.get("mode", "ANY")
    k = int(group.get("k", 1))
    names = [FRIENDLY[s] for s in sigs]

    if mode == "ALL":
        current = "(" + " AND ".join(names) + ")"
    elif mode == "ANY":
        current = "(" + " OR ".join(names) + ")"
    else:
        current = f"({k} OF {len(names)}: " + ", ".join(names) + ")"

    contexts = group.get("contexts", [])
    if contexts:
        ctx_parts = []
        for r in contexts:
            ctx_parts.append(
                f"{FRIENDLY[r['signal']]} >= {r['min_count']} in prev {r['lookback_h']}h"
            )
        current += " AND [" + " AND ".join(ctx_parts) + "]"
    return current

def grouped_logic_text(groups, outer_operator):
    parts = [group_text(g) for g in groups if g.get("signals")]
    return f" {outer_operator} ".join(parts) if parts else "—"

def calc_trade_path_stats(path, direction, entry_price):
    """
    Returns:
      MFE: best excursion from entry
      MAE: worst excursion from entry
      max_peak_to_trough_drawdown: worst peak-to-trough drawdown while trade is open
    """
    hi = path["high"].to_numpy(float)
    lo = path["low"].to_numpy(float)

    if direction == "Long":
        mfe = np.max(hi / entry_price - 1.0)
        mae = np.min(lo / entry_price - 1.0)

        # Conservative intratrade peak-to-trough DD:
        # running peak uses bar highs, drawdown uses subsequent/current lows.
        running_peak = np.maximum.accumulate(np.r_[entry_price, hi])[:-1]
        dd_series = lo / running_peak - 1.0
        max_dd = float(np.min(dd_series))
    else:
        # For a short, favorable move is downward, adverse move is upward.
        mfe = np.max(entry_price / lo - 1.0)
        mae = np.min(entry_price / hi - 1.0)

        # Short "equity" improves when price falls. Track running trough, then
        # adverse rebound to highs from that best trough.
        running_trough = np.minimum.accumulate(np.r_[entry_price, lo])[:-1]
        dd_series = running_trough / hi - 1.0
        max_dd = float(np.min(dd_series))

    return float(mfe), float(mae), max_dd

def backtest(
    df,
    direction,
    entry_groups,
    entry_outer,
    exit_groups,
    exit_outer,
    allowed_regimes,
    roundtrip_cost,
    cooldown_bars=0,
):
    # Important: year-filtered DataFrames can retain original row labels.
    # Backtest uses positional indexing, so always reset here.
    df = df.reset_index(drop=True).copy()

    entry_base = grouped_logic_mask(df, entry_groups, entry_outer)
    if allowed_regimes:
        entry_base &= df["regime"].isin(allowed_regimes)

    exit_base = grouped_logic_mask(df, exit_groups, exit_outer)

    trades = []
    in_pos = False
    entry_signal_i = None
    entry_exec_i = None
    next_allowed_i = 0

    for i in range(len(df) - 1):
        if not in_pos:
            if i < next_allowed_i:
                continue
            if bool(entry_base.iat[i]):
                entry_signal_i = i
                entry_exec_i = i + 1
                in_pos = True
        else:
            # Exit signal must occur no earlier than the executed entry bar.
            if i >= entry_exec_i and bool(exit_base.iat[i]):
                exit_exec_i = i + 1
                if exit_exec_i >= len(df):
                    continue

                ep = float(df.at[entry_exec_i, "open"])
                xp = float(df.at[exit_exec_i, "open"])

                gross = (xp / ep - 1.0) if direction == "Long" else (ep / xp - 1.0)
                net = gross - roundtrip_cost

                path = df.iloc[entry_exec_i:exit_exec_i + 1]
                mfe, mae, trade_max_dd = calc_trade_path_stats(path, direction, ep)

                trades.append({
                    "direction": direction,
                    "entry_signal_time": df.at[entry_signal_i, "bar_time_display"],
                    "entry_time": df.at[entry_exec_i, "bar_time_display"],
                    "entry_price": ep,
                    "entry_regime": df.at[entry_signal_i, "regime"],
                    "exit_signal_time": df.at[i, "bar_time_display"],
                    "exit_time": df.at[exit_exec_i, "bar_time_display"],
                    "exit_price": xp,
                    "hold_hours": (
                        df.at[exit_exec_i, "bar_time_display"] -
                        df.at[entry_exec_i, "bar_time_display"]
                    ).total_seconds() / 3600.0,
                    "gross_return": gross,
                    "net_return": net,
                    "win": int(net > 0),
                    "mfe": mfe,
                    "mae_from_entry": mae,
                    "max_drawdown_trade": trade_max_dd,
                    "year": int(df.at[entry_signal_i, "year"]),
                })

                in_pos = False
                next_allowed_i = exit_exec_i + cooldown_bars
                entry_signal_i = None
                entry_exec_i = None

    return pd.DataFrame(trades)

def max_drawdown_from_trade_returns(returns):
    if len(returns) == 0:
        return np.nan
    equity = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))

def profit_factor(returns):
    if len(returns) == 0:
        return np.nan
    r = np.asarray(returns, dtype=float)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return 99.0 if pos > 0 else np.nan
    return float(pos / neg)

def summarize(trades):
    if trades.empty:
        return {
            "trades":0,"win_rate":np.nan,"avg_return":np.nan,"median_return":np.nan,
            "profit_factor":np.nan,"comp_return":np.nan,"max_drawdown":np.nan,
            "avg_mfe":np.nan,"avg_mae":np.nan,"worst_trade_drawdown":np.nan
        }

    r = trades["net_return"].to_numpy(float)
    return {
        "trades": len(trades),
        "win_rate": float((r > 0).mean()),
        "avg_return": float(r.mean()),
        "median_return": float(np.median(r)),
        "profit_factor": profit_factor(r),
        "comp_return": float(np.prod(1 + r) - 1),
        "max_drawdown": max_drawdown_from_trade_returns(r),
        "avg_mfe": float(trades["mfe"].mean()),
        "avg_mae": float(trades["mae_from_entry"].mean()),
        "worst_trade_drawdown": float(trades["max_drawdown_trade"].min()),
    }

def metric_table(trades, group_col):
    rows = []
    for key, g in trades.groupby(group_col):
        rows.append({group_col:key, **summarize(g)})
    return pd.DataFrame(rows)

def fmt_perf_table(t):
    if t.empty:
        return t
    x = t.copy()
    for c in [
        "win_rate","avg_return","median_return","comp_return","max_drawdown",
        "avg_mfe","avg_mae","worst_trade_drawdown"
    ]:
        if c in x:
            x[c] = x[c].map(lambda v: None if pd.isna(v) else f"{v:.2%}")
    if "profit_factor" in x:
        x["profit_factor"] = x["profit_factor"].map(
            lambda v: None if pd.isna(v) else f"{v:.2f}"
        )
    return x

def render_summary_cards(summary):
    cols = st.columns(6)
    cols[0].metric("Trades", summary["trades"])
    cols[1].metric("Win rate", "—" if not np.isfinite(summary["win_rate"]) else f"{summary['win_rate']:.1%}")
    cols[2].metric("Avg trade", "—" if not np.isfinite(summary["avg_return"]) else f"{summary['avg_return']:.2%}")
    cols[3].metric("Profit factor", "—" if not np.isfinite(summary["profit_factor"]) else f"{summary['profit_factor']:.2f}")
    cols[4].metric("Strategy max DD", "—" if not np.isfinite(summary["max_drawdown"]) else f"{summary['max_drawdown']:.2%}")
    cols[5].metric("Worst trade DD", "—" if not np.isfinite(summary["worst_trade_drawdown"]) else f"{summary['worst_trade_drawdown']:.2%}")

def render_logic_group(prefix, i, default_signals=None, default_mode="ANY", default_contexts=None):
    st.markdown(f"#### Group {i}")
    cols = st.columns([2.4, 1, 0.8])
    sigs = cols[0].multiselect(
        f"Group {i} current-bar signals",
        SIGNALS,
        default=default_signals or [],
        key=f"{prefix}_group_{i}_signals",
        format_func=lambda x: FRIENDLY[x],
    )
    mode = cols[1].selectbox(
        f"Group {i} signal logic",
        ["ANY","ALL","K OF N"],
        index=["ANY","ALL","K OF N"].index(default_mode),
        key=f"{prefix}_group_{i}_mode",
    )
    k = cols[2].number_input(
        f"Group {i} K",
        min_value=1,
        max_value=max(1, len(sigs)),
        value=min(1, max(1, len(sigs))),
        disabled=(mode != "K OF N"),
        key=f"{prefix}_group_{i}_k",
    )

    defaults = default_contexts or []
    n_ctx = st.slider(
        f"Group {i}: number of memory conditions",
        0, 3, len(defaults),
        key=f"{prefix}_group_{i}_nctx"
    )

    contexts = []
    for j in range(1, n_ctx + 1):
        d = defaults[j-1] if j <= len(defaults) else {}
        a,b,c = st.columns([2,1,1])
        default_sig = d.get("signal", "blue_spi")
        default_look = d.get("lookback_h", 24)
        default_count = d.get("min_count", 1)

        sig = a.selectbox(
            f"Group {i} memory signal {j}",
            SIGNALS,
            index=SIGNALS.index(default_sig) if default_sig in SIGNALS else 0,
            key=f"{prefix}_group_{i}_ctx_{j}_sig",
            format_func=lambda x: FRIENDLY[x],
        )
        lookbacks = [8,12,24,48,72,168]
        look = b.selectbox(
            f"Group {i} lookback {j}",
            lookbacks,
            index=lookbacks.index(default_look) if default_look in lookbacks else 2,
            key=f"{prefix}_group_{i}_ctx_{j}_look",
        )
        cnt = c.number_input(
            f"Group {i} min count {j}",
            min_value=1, max_value=20,
            value=int(default_count),
            key=f"{prefix}_group_{i}_ctx_{j}_count",
        )
        contexts.append({"signal":sig, "lookback_h":int(look), "min_count":int(cnt)})

    return {"signals":sigs, "mode":mode, "k":int(k), "contexts":contexts}

def build_groups_ui(prefix, title, default_groups):
    st.markdown(f"### {title}")
    n = st.slider(
        f"Number of {title.lower()} groups",
        1, 3, len(default_groups),
        key=f"{prefix}_ngroups"
    )
    outer = st.radio(
        f"How to combine the {title.lower()} groups",
        ["AND","OR"],
        horizontal=True,
        key=f"{prefix}_outer"
    )

    groups = []
    for i in range(1, n + 1):
        if i <= len(default_groups):
            d = default_groups[i-1]
            groups.append(render_logic_group(
                prefix, i,
                d.get("signals", []),
                d.get("mode", "ANY"),
                d.get("contexts", [])
            ))
        else:
            groups.append(render_logic_group(prefix, i))
    return groups, outer

def candidate_rule_search(
    df,
    direction,
    anchors,
    contexts,
    exit_pool,
    regimes,
    cost,
    max_entry_terms,
    max_exit_terms,
    min_trades_dev,
    min_trades_val,
    top_n,
):
    """
    Automatic search remains intentionally simpler than the manual nested builder:
    - Entry = ALL of 1..N simultaneous anchors + optional single recent-context rule
    - Exit = ANY of 1..N exit signals
    - Ranking uses <=2024 development + 2025 validation only
    - 2026 is attached after ranking, not used to select candidates
    """
    dev = df[df["year"] <= 2024].copy()
    val = df[df["year"] == 2025].copy()
    final = df[df["year"] == 2026].copy()

    context_options = [None] + contexts
    results = []

    exit_sets = []
    for n in range(1, max_exit_terms + 1):
        exit_sets.extend(combinations(exit_pool, n))

    for n in range(1, max_entry_terms + 1):
        for combo in combinations(anchors, n):
            for ctx in context_options:
                ctx_rules = [] if ctx is None else [ctx]
                entry_groups = [{
                    "signals":list(combo), "mode":"ALL", "k":len(combo),
                    "contexts":ctx_rules
                }]
                for ex in exit_sets:
                    exit_groups = [{
                        "signals":list(ex), "mode":"ANY", "k":1,
                        "contexts":[]
                    }]

                    tr_dev = backtest(
                        dev,direction,entry_groups,"AND",
                        exit_groups,"OR",regimes,cost
                    )
                    tr_val = backtest(
                        val,direction,entry_groups,"AND",
                        exit_groups,"OR",regimes,cost
                    )

                    sd, sv = summarize(tr_dev), summarize(tr_val)
                    if sd["trades"] < min_trades_dev or sv["trades"] < min_trades_val:
                        continue
                    if not np.isfinite(sd["avg_return"]) or not np.isfinite(sv["avg_return"]):
                        continue
                    if sd["avg_return"] <= 0 or sv["avg_return"] <= 0:
                        continue

                    stability = min(sd["avg_return"], sv["avg_return"])
                    pf_term = min(
                        sd["profit_factor"] if np.isfinite(sd["profit_factor"]) else 0,
                        sv["profit_factor"] if np.isfinite(sv["profit_factor"]) else 0,
                        5
                    )
                    count_term = math.log1p(sd["trades"] + sv["trades"])
                    dd_penalty = (
                        1
                        + abs(min(sd["max_drawdown"], 0))
                        + abs(min(sv["max_drawdown"], 0))
                        + abs(min(sd["worst_trade_drawdown"], 0))
                        + abs(min(sv["worst_trade_drawdown"], 0))
                    )
                    score = stability * (1 + pf_term/5) * count_term / dd_penalty

                    tr_final = backtest(
                        final,direction,entry_groups,"AND",
                        exit_groups,"OR",regimes,cost
                    )
                    sf = summarize(tr_final)

                    ctx_text = ""
                    if ctx is not None:
                        ctx_text = (
                            f" + {FRIENDLY[ctx['signal']]} >= {ctx['min_count']} "
                            f"in prev {ctx['lookback_h']}h"
                        )

                    results.append({
                        "direction":direction,
                        "regimes":",".join(regimes) if regimes else "all",
                        "entry_rule":" AND ".join(FRIENDLY[x] for x in combo) + ctx_text,
                        "exit_rule":" OR ".join(FRIENDLY[x] for x in ex),
                        "dev_n":sd["trades"],
                        "dev_win_rate":sd["win_rate"],
                        "dev_avg_return":sd["avg_return"],
                        "dev_profit_factor":sd["profit_factor"],
                        "dev_max_drawdown":sd["max_drawdown"],
                        "dev_worst_trade_dd":sd["worst_trade_drawdown"],
                        "val_n":sv["trades"],
                        "val_win_rate":sv["win_rate"],
                        "val_avg_return":sv["avg_return"],
                        "val_profit_factor":sv["profit_factor"],
                        "val_max_drawdown":sv["max_drawdown"],
                        "val_worst_trade_dd":sv["worst_trade_drawdown"],
                        "final_n":sf["trades"],
                        "final_win_rate":sf["win_rate"],
                        "final_avg_return":sf["avg_return"],
                        "final_profit_factor":sf["profit_factor"],
                        "final_max_drawdown":sf["max_drawdown"],
                        "final_worst_trade_dd":sf["worst_trade_drawdown"],
                        "pretest_score":score,
                    })

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values("pretest_score", ascending=False).head(top_n).reset_index(drop=True)


def portfolio_backtest(
    df,
    direction,
    entry_groups,
    entry_outer,
    exit_groups,
    exit_outer,
    allowed_regimes,
    roundtrip_cost,
    cooldown_bars,
    starting_capital,
    allocation_pct,
    leverage,
    stop_loss_pct=None,
    simulate_liquidation=True,
):
    """
    One position at a time.

    Position notional = account equity * allocation_pct * leverage.
    Round-trip costs are charged on notional.
    Stop loss is expressed as an underlying-price move from entry.
    A simple liquidation proxy is optionally applied at an adverse move of 1/leverage.
    This is intentionally conservative and is NOT exchange-specific liquidation math.
    """
    df = df.reset_index(drop=True).copy()

    entry_base = grouped_logic_mask(df, entry_groups, entry_outer)
    if allowed_regimes:
        entry_base &= df["regime"].isin(allowed_regimes)
    exit_base = grouped_logic_mask(df, exit_groups, exit_outer)

    equity = float(starting_capital)
    initial_capital = float(starting_capital)
    allocation_pct = float(allocation_pct)
    leverage = float(leverage)

    trades = []
    equity_points = [{
        "time": df.at[0, "bar_time_display"],
        "equity": equity,
    }] if len(df) else []

    in_pos = False
    entry_signal_i = None
    entry_exec_i = None
    next_allowed_i = 0
    entry_equity = None
    margin = None
    notional = None
    entry_price = None

    def choose_protective_threshold():
        candidates = []
        if stop_loss_pct is not None and stop_loss_pct > 0:
            candidates.append(("stop_loss", float(stop_loss_pct)))
        if simulate_liquidation and leverage > 1:
            # Simplified zero-maintenance-margin liquidation proxy.
            candidates.append(("liquidation_proxy", 1.0 / leverage))
        if not candidates:
            return None, None
        # The tighter adverse threshold is hit first.
        return min(candidates, key=lambda x: x[1])

    protective_reason, protective_pct = choose_protective_threshold()

    for i in range(len(df) - 1):
        if equity <= 0:
            break

        if not in_pos:
            if i < next_allowed_i:
                continue
            if bool(entry_base.iat[i]):
                entry_signal_i = i
                entry_exec_i = i + 1
                if entry_exec_i >= len(df):
                    continue

                entry_equity = equity
                margin = entry_equity * allocation_pct
                notional = margin * leverage
                entry_price = float(df.at[entry_exec_i, "open"])
                in_pos = True

        else:
            # Check protective stop/liquidation on every bar starting with the entry bar.
            bar_i = max(i, entry_exec_i)
            exited = False
            exit_exec_i = None
            exit_signal_i = None
            exit_price = None
            exit_reason = None

            if protective_pct is not None and bar_i >= entry_exec_i:
                o = float(df.at[bar_i, "open"])
                hi = float(df.at[bar_i, "high"])
                lo = float(df.at[bar_i, "low"])

                if direction == "Long":
                    trigger_price = entry_price * (1.0 - protective_pct)
                    if lo <= trigger_price:
                        # If bar gaps below the trigger, fill at the worse open.
                        exit_price = min(o, trigger_price)
                        exit_exec_i = bar_i
                        exit_signal_i = bar_i
                        exit_reason = protective_reason
                        exited = True
                else:
                    trigger_price = entry_price * (1.0 + protective_pct)
                    if hi >= trigger_price:
                        # If bar gaps above the trigger, fill at the worse open.
                        exit_price = max(o, trigger_price)
                        exit_exec_i = bar_i
                        exit_signal_i = bar_i
                        exit_reason = protective_reason
                        exited = True

            # Event-driven exit: signal on closed bar, execute next bar open.
            if (not exited) and i >= entry_exec_i and bool(exit_base.iat[i]):
                exit_signal_i = i
                exit_exec_i = i + 1
                if exit_exec_i < len(df):
                    exit_price = float(df.at[exit_exec_i, "open"])
                    exit_reason = "signal_exit"
                    exited = True

            if exited:
                if direction == "Long":
                    underlying_return = exit_price / entry_price - 1.0
                else:
                    underlying_return = entry_price / exit_price - 1.0

                gross_pnl = notional * underlying_return
                fees = notional * roundtrip_cost

                if exit_reason == "liquidation_proxy":
                    # Cap the simplified liquidation loss at the allocated margin.
                    net_pnl = -margin
                    fees = 0.0
                else:
                    net_pnl = gross_pnl - fees

                equity_after = max(0.0, entry_equity + net_pnl)
                account_return = equity_after / entry_equity - 1.0 if entry_equity > 0 else -1.0

                path = df.iloc[entry_exec_i:exit_exec_i + 1]
                mfe, mae, trade_max_dd = calc_trade_path_stats(
                    path, direction, entry_price
                )

                trades.append({
                    "direction": direction,
                    "entry_signal_time": df.at[entry_signal_i, "bar_time_display"],
                    "entry_time": df.at[entry_exec_i, "bar_time_display"],
                    "entry_price": entry_price,
                    "entry_regime": df.at[entry_signal_i, "regime"],
                    "exit_reason": exit_reason,
                    "exit_signal_time": df.at[exit_signal_i, "bar_time_display"],
                    "exit_time": df.at[exit_exec_i, "bar_time_display"],
                    "exit_price": exit_price,
                    "hold_hours": (
                        df.at[exit_exec_i, "bar_time_display"]
                        - df.at[entry_exec_i, "bar_time_display"]
                    ).total_seconds() / 3600.0,
                    "capital_before": entry_equity,
                    "allocation_pct": allocation_pct,
                    "margin_used": margin,
                    "leverage": leverage,
                    "notional": notional,
                    "underlying_return": underlying_return,
                    "gross_pnl": gross_pnl,
                    "fees": fees,
                    "net_pnl": net_pnl,
                    "account_return_trade": account_return,
                    "capital_after": equity_after,
                    "win": int(net_pnl > 0),
                    "mfe": mfe,
                    "mae_from_entry": mae,
                    "max_drawdown_trade": trade_max_dd,
                    "year": int(df.at[entry_signal_i, "year"]),
                })

                equity = equity_after
                equity_points.append({
                    "time": df.at[exit_exec_i, "bar_time_display"],
                    "equity": equity,
                })

                in_pos = False
                next_allowed_i = exit_exec_i + cooldown_bars
                entry_signal_i = None
                entry_exec_i = None
                entry_equity = None
                margin = None
                notional = None
                entry_price = None

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)

    if equity_df.empty:
        max_equity_dd = np.nan
    else:
        vals = equity_df["equity"].to_numpy(float)
        peaks = np.maximum.accumulate(vals)
        dd = vals / peaks - 1.0
        max_equity_dd = float(np.min(dd))

    result = {
        "starting_capital": initial_capital,
        "ending_capital": equity,
        "total_return": equity / initial_capital - 1.0 if initial_capital > 0 else np.nan,
        "max_equity_drawdown": max_equity_dd,
        "trades": len(trades_df),
        "win_rate": (
            float((trades_df["net_pnl"] > 0).mean()) if not trades_df.empty else np.nan
        ),
        "stops": (
            int((trades_df["exit_reason"] == "stop_loss").sum())
            if not trades_df.empty else 0
        ),
        "liquidations": (
            int((trades_df["exit_reason"] == "liquidation_proxy").sum())
            if not trades_df.empty else 0
        ),
    }
    return trades_df, equity_df, result

def portfolio_breakdown(trades, group_col):
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for key, g in trades.groupby(group_col):
        capital_pnl = float(g["net_pnl"].sum())
        rows.append({
            group_col: key,
            "trades": len(g),
            "win_rate": float((g["net_pnl"] > 0).mean()),
            "net_pnl": capital_pnl,
            "avg_account_return": float(g["account_return_trade"].mean()),
            "avg_underlying_return": float(g["underlying_return"].mean()),
            "stops": int((g["exit_reason"] == "stop_loss").sum()),
            "liquidations": int((g["exit_reason"] == "liquidation_proxy").sum()),
            "worst_trade_drawdown": float(g["max_drawdown_trade"].min()),
        })
    return pd.DataFrame(rows)


# ---------------- UI ----------------
st.title("Strategy Lab v1.4")
st.caption(
    "Nested strategy builder + capital/leverage tester + per-group memory. "
    "Research dataset hard-stops on 19 Apr 2026."
)

df0 = load_data(DEFAULT_DATA)

with st.sidebar:
    st.header("Research settings")
    regime_threshold = st.slider(
        "90-day bull/bear threshold", 0.05, 0.30, 0.10, 0.01,
        help="Bull if trailing 90d return >= threshold; Bear if <= -threshold."
    )
    cost_bps = st.number_input(
        "Round-trip cost, bps", min_value=0.0, max_value=200.0,
        value=10.0, step=1.0
    )
    cooldown_h = st.number_input(
        "Cooldown after exit, hours", min_value=0, max_value=168,
        value=0, step=4
    )
    st.markdown("**Data cutoff:** 2026-04-19")

df = apply_regime(df0, regime_threshold)
cost = cost_bps / 10000.0
cooldown_bars = int(cooldown_h / 4)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Manual Strategy", "Strategy Search", "Strategy Tester", "Data / Regimes"]
)

with tab1:
    st.subheader("Nested entry / exit logic")

    top1, top2, top3 = st.columns(3)
    with top1:
        direction = st.radio("Direction", ["Long","Short"], horizontal=True)
    with top2:
        allowed_regimes = st.multiselect(
            "Allowed market regimes",
            ["bull","bear","transition"],
            default=["bull","bear","transition"]
        )
    with top3:
        st.write("Execution")
        st.caption("Closed 4H signal bar → trade at next 4H open")

    # Defaults directly match the user's requested example:
    # Long entry: (any SPI) AND (MAIN OR STAT)
    entry_defaults = [
        {
            "signals":["main"], "mode":"ANY",
            "contexts":[{"signal":"green_spi","lookback_h":24,"min_count":1}]
        },
        {
            "signals":["main"], "mode":"ANY",
            "contexts":[{"signal":"blue_spi","lookback_h":24,"min_count":1}]
        },
    ] if direction == "Long" else [
        {
            "signals":["red_hsr"], "mode":"ANY",
            "contexts":[{"signal":"red_spi","lookback_h":24,"min_count":1}]
        },
        {
            "signals":["cloud4_short"], "mode":"ANY",
            "contexts":[{"signal":"orange_hsr","lookback_h":24,"min_count":1}]
        },
    ]

    entry_groups, entry_outer = build_groups_ui(
        "entry", "Entry", entry_defaults
    )

    st.caption(
        "Each Entry group has its own memory conditions. "
        "This lets you build e.g. (MAIN + Green SPI in previous 24h) "
        "OR (MAIN + Blue SPI in previous 24h)."
    )

    exit_defaults = [
        {"signals":["red_spi","orange_hsr"], "mode":"ALL", "contexts":[]},
        {"signals":["red_hsr"], "mode":"ANY", "contexts":[]},
    ] if direction == "Long" else [
        {"signals":["main","stat"], "mode":"ANY", "contexts":[]},
        {"signals":["green_spi","blue_spi"], "mode":"ANY", "contexts":[]},
    ]

    exit_groups, exit_outer = build_groups_ui(
        "exit", "Exit", exit_defaults
    )

    st.caption(
        "Each Exit group also has its own memory conditions."
    )

    if st.button("Run backtest", type="primary", use_container_width=True):
        trades = backtest(
            df, direction,
            entry_groups, entry_outer,
            exit_groups, exit_outer,
            allowed_regimes, cost, cooldown_bars
        )
        st.session_state["manual_trades_v12"] = trades
        st.session_state["manual_desc_v12"] = {
            "entry": grouped_logic_text(entry_groups, entry_outer),
            "exit": grouped_logic_text(exit_groups, exit_outer),
            "direction": direction,
        }

    if "manual_trades_v12" in st.session_state:
        trades = st.session_state["manual_trades_v12"]
        desc = st.session_state["manual_desc_v12"]

        st.markdown(f"**{desc['direction']} entry:** {desc['entry']}")
        st.markdown(f"**Exit:** {desc['exit']}")

        summary = summarize(trades)
        render_summary_cards(summary)

        if not trades.empty:
            st.markdown("#### Performance by market regime")
            st.dataframe(
                fmt_perf_table(metric_table(trades, "entry_regime")),
                use_container_width=True, hide_index=True
            )

            st.markdown("#### Performance by year")
            st.dataframe(
                fmt_perf_table(metric_table(trades, "year")),
                use_container_width=True, hide_index=True
            )

            st.markdown("#### Equity curve")
            eq = trades[["exit_time","net_return"]].copy()
            eq["equity"] = np.cumprod(1 + eq["net_return"])
            st.line_chart(eq.set_index("exit_time")["equity"])

            st.markdown("#### Trade history")
            st.caption(
                "mae_from_entry = worst move against the entry price. "
                "max_drawdown_trade = worst peak-to-trough drawdown while that trade was open."
            )

            show = trades.copy()
            for c in [
                "gross_return","net_return","mfe",
                "mae_from_entry","max_drawdown_trade"
            ]:
                show[c] = show[c].map(lambda v:f"{v:.2%}")

            st.dataframe(show, use_container_width=True, hide_index=True)

            st.download_button(
                "Download trades CSV",
                trades.to_csv(index=False).encode("utf-8"),
                file_name="strategy_trades_v1_2.csv",
                mime="text/csv",
            )
        else:
            st.warning("No completed trades for this rule set.")

with tab2:
    st.subheader("Automatic interpretable strategy search")
    st.info(
        "The manual builder now supports nested groups. "
        "Automatic search intentionally stays simpler to reduce combinatorial overfitting."
    )
    st.warning(
        "Ranking uses 2022–2024 development + 2025 validation only. "
        "2026 is attached only after ranking."
    )

    q1,q2 = st.columns(2)
    with q1:
        search_direction = st.radio(
            "Search direction", ["Long","Short"],
            horizontal=True, key="sd"
        )
        search_regimes = st.multiselect(
            "Search regimes", ["bull","bear","transition"],
            default=["bull","bear","transition"], key="sr"
        )
        anchor_default = LONGISH if search_direction=="Long" else SHORTISH
        anchors = st.multiselect(
            "Allowed entry anchor signals", SIGNALS,
            default=anchor_default,
            format_func=lambda x:FRIENDLY[x]
        )
        max_terms = st.slider(
            "Max simultaneous entry signals", 1, 4, 2
        )

    with q2:
        exit_default = SHORTISH if search_direction=="Long" else LONGISH
        exit_pool = st.multiselect(
            "Allowed exit signals", SIGNALS,
            default=exit_default,
            format_func=lambda x:FRIENDLY[x]
        )
        max_exit_terms = st.slider("Max exit OR terms", 1, 3, 2)
        min_dev = st.number_input(
            "Minimum development trades", 1, 100, 8
        )
        min_val = st.number_input(
            "Minimum 2025 validation trades", 1, 50, 3
        )

    st.markdown("#### Optional recent-signal contexts included in search")
    ctx_signal_pool = st.multiselect(
        "Signals eligible as recent context", SIGNALS,
        default=(
            ["green_spi","blue_spi","main","stat"]
            if search_direction=="Long"
            else ["red_spi","orange_hsr","red_hsr"]
        ),
        format_func=lambda x:FRIENDLY[x]
    )
    lookbacks = st.multiselect(
        "Context lookbacks (hours)",
        [8,12,24,48,72,168],
        default=[24,72]
    )

    contexts = []
    for sig in ctx_signal_pool:
        for h in lookbacks:
            contexts.append({
                "signal":sig,"lookback_h":h,"min_count":1
            })

    top_n = st.slider("Return top candidates", 10, 200, 50, 10)

    if st.button("Search strategies", type="primary", use_container_width=True):
        with st.spinner("Testing strategy combinations..."):
            res = candidate_rule_search(
                df, search_direction, anchors, contexts, exit_pool,
                search_regimes, cost, max_terms, max_exit_terms,
                int(min_dev), int(min_val), top_n
            )
        st.session_state["search_results_v12"] = res

    if "search_results_v12" in st.session_state:
        res = st.session_state["search_results_v12"]
        if res.empty:
            st.warning("No candidates passed the development + validation gates.")
        else:
            display = res.copy()
            for c in [
                x for x in display.columns
                if "win_rate" in x or "avg_return" in x
                or "max_drawdown" in x or "worst_trade_dd" in x
            ]:
                display[c] = display[c].map(
                    lambda v: None if pd.isna(v) else f"{v:.2%}"
                )
            for c in [x for x in display.columns if "profit_factor" in x]:
                display[c] = display[c].map(
                    lambda v: None if pd.isna(v) else f"{v:.2f}"
                )

            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download search results CSV",
                res.to_csv(index=False).encode("utf-8"),
                file_name="strategy_search_results_v1_2.csv",
                mime="text/csv",
            )


with tab3:
    st.subheader("Capital / leverage strategy tester")
    st.caption(
        "Uses the same nested signal logic, but now simulates account equity, "
        "position sizing, leverage, fees and an optional stop loss."
    )

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        tester_direction = st.radio(
            "Tester direction", ["Long","Short"], horizontal=True, key="tester_direction"
        )
    with tc2:
        tester_regimes = st.multiselect(
            "Tester market regimes",
            ["bull","bear","transition"],
            default=["bull","bear","transition"],
            key="tester_regimes",
        )
    with tc3:
        start_date = st.date_input(
            "Test start date",
            value=df["bar_time_display"].min().date(),
            min_value=df["bar_time_display"].min().date(),
            max_value=df["bar_time_display"].max().date(),
            key="tester_start_date",
        )
        end_date = st.date_input(
            "Test end date",
            value=df["bar_time_display"].max().date(),
            min_value=df["bar_time_display"].min().date(),
            max_value=df["bar_time_display"].max().date(),
            key="tester_end_date",
        )

    st.markdown("### Account / risk settings")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        starting_capital = st.number_input(
            "Starting capital", min_value=1.0, value=10000.0, step=1000.0,
            key="tester_capital"
        )
    with rc2:
        allocation_pct_ui = st.number_input(
            "Capital allocated per trade, %", min_value=1.0, max_value=100.0,
            value=100.0, step=5.0, key="tester_allocation"
        )
    with rc3:
        tester_leverage = st.number_input(
            "Leverage", min_value=1.0, max_value=100.0,
            value=1.0, step=1.0, key="tester_leverage"
        )
    with rc4:
        use_stop = st.checkbox("Use stop loss", value=True, key="tester_use_stop")
        tester_stop_pct_ui = st.number_input(
            "Stop loss, underlying %", min_value=0.1, max_value=99.0,
            value=10.0, step=0.5, disabled=not use_stop,
            key="tester_stop_pct"
        )

    simulate_liq = st.checkbox(
        "Use simplified liquidation proxy",
        value=True,
        key="tester_liq",
        help=(
            "Approximation only: liquidation at an adverse underlying move of 1/leverage. "
            "Real exchange liquidation depends on maintenance margin and venue rules."
        )
    )

    if tester_leverage > 1:
        approx_liq = 100.0 / tester_leverage
        st.caption(
            f"Simplified liquidation distance at {tester_leverage:.1f}x is about "
            f"{approx_liq:.2f}% adverse underlying move."
        )
        if use_stop and tester_stop_pct_ui >= approx_liq and simulate_liq:
            st.warning(
                "Your stop is wider than the simplified liquidation distance, "
                "so the liquidation proxy can trigger before the stop."
            )

    tester_entry_defaults = [
        {
            "signals":["main"], "mode":"ANY",
            "contexts":[{"signal":"green_spi","lookback_h":24,"min_count":1}]
        },
        {
            "signals":["main"], "mode":"ANY",
            "contexts":[{"signal":"blue_spi","lookback_h":24,"min_count":1}]
        },
    ] if tester_direction == "Long" else [
        {
            "signals":["red_hsr"], "mode":"ANY",
            "contexts":[{"signal":"red_spi","lookback_h":24,"min_count":1}]
        },
        {
            "signals":["cloud4_short"], "mode":"ANY",
            "contexts":[{"signal":"orange_hsr","lookback_h":24,"min_count":1}]
        },
    ]

    tester_entry_groups, tester_entry_outer = build_groups_ui(
        "tester_entry", "Tester Entry", tester_entry_defaults
    )

    tester_exit_defaults = [
        {"signals":["red_spi","orange_hsr"], "mode":"ALL", "contexts":[]},
        {"signals":["red_hsr"], "mode":"ANY", "contexts":[]},
    ] if tester_direction == "Long" else [
        {"signals":["main","stat"], "mode":"ANY", "contexts":[]},
        {"signals":["green_spi","blue_spi"], "mode":"ANY", "contexts":[]},
    ]

    tester_exit_groups, tester_exit_outer = build_groups_ui(
        "tester_exit", "Tester Exit", tester_exit_defaults
    )

    if st.button("Run capital simulation", type="primary", use_container_width=True):
        if start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            mask = (
                (df["bar_time_display"].dt.date >= start_date)
                & (df["bar_time_display"].dt.date <= end_date)
            )
            test_df = df.loc[mask].copy()

            tester_trades, tester_equity, tester_result = portfolio_backtest(
                test_df,
                tester_direction,
                tester_entry_groups,
                tester_entry_outer,
                tester_exit_groups,
                tester_exit_outer,
                tester_regimes,
                cost,
                cooldown_bars,
                starting_capital,
                allocation_pct_ui / 100.0,
                tester_leverage,
                tester_stop_pct_ui / 100.0 if use_stop else None,
                simulate_liq,
            )
            st.session_state["tester_trades"] = tester_trades
            st.session_state["tester_equity"] = tester_equity
            st.session_state["tester_result"] = tester_result
            st.session_state["tester_desc"] = {
                "entry": grouped_logic_text(tester_entry_groups, tester_entry_outer),
                "exit": grouped_logic_text(tester_exit_groups, tester_exit_outer),
            }

    if "tester_result" in st.session_state:
        result = st.session_state["tester_result"]
        trades = st.session_state["tester_trades"]
        eq = st.session_state["tester_equity"]
        desc = st.session_state["tester_desc"]

        st.markdown(f"**Entry:** {desc['entry']}")
        st.markdown(f"**Exit:** {desc['exit']}")

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Starting capital", f"{result['starting_capital']:,.2f}")
        m2.metric("Ending capital", f"{result['ending_capital']:,.2f}")
        m3.metric(
            "Total return",
            "—" if not np.isfinite(result["total_return"]) else f"{result['total_return']:.2%}"
        )
        m4.metric(
            "Max equity DD",
            "—" if not np.isfinite(result["max_equity_drawdown"]) else f"{result['max_equity_drawdown']:.2%}"
        )
        m5.metric("Trades", result["trades"])
        m6.metric(
            "Win rate",
            "—" if not np.isfinite(result["win_rate"]) else f"{result['win_rate']:.1%}"
        )

        s1,s2 = st.columns(2)
        s1.metric("Stop-loss exits", result["stops"])
        s2.metric("Liquidation-proxy exits", result["liquidations"])

        if not eq.empty:
            st.markdown("#### Account equity curve")
            st.line_chart(eq.set_index("time")["equity"])

        if not trades.empty:
            st.markdown("#### Capital performance by market regime")
            by_regime = portfolio_breakdown(trades, "entry_regime")
            for c in ["win_rate","avg_account_return","avg_underlying_return","worst_trade_drawdown"]:
                if c in by_regime:
                    by_regime[c] = by_regime[c].map(
                        lambda v: None if pd.isna(v) else f"{v:.2%}"
                    )
            st.dataframe(by_regime, use_container_width=True, hide_index=True)

            st.markdown("#### Capital performance by year")
            by_year = portfolio_breakdown(trades, "year")
            for c in ["win_rate","avg_account_return","avg_underlying_return","worst_trade_drawdown"]:
                if c in by_year:
                    by_year[c] = by_year[c].map(
                        lambda v: None if pd.isna(v) else f"{v:.2%}"
                    )
            st.dataframe(by_year, use_container_width=True, hide_index=True)

            st.markdown("#### Strategy Tester trade history")
            show = trades.copy()
            for c in [
                "underlying_return","account_return_trade","mfe",
                "mae_from_entry","max_drawdown_trade"
            ]:
                show[c] = show[c].map(lambda v: f"{v:.2%}")
            st.dataframe(show, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Strategy Tester trades CSV",
                trades.to_csv(index=False).encode("utf-8"),
                file_name="strategy_tester_trades.csv",
                mime="text/csv",
            )
        else:
            st.warning("No completed trades for these tester settings.")


with tab4:
    st.subheader("Market state coverage")
    coverage = pd.crosstab(df["year"], df["regime"])
    st.dataframe(coverage, use_container_width=True)

    st.markdown("#### Trailing 90-day BTC return")
    chart = df[["bar_time_display","ret_90d"]].dropna().set_index("bar_time_display")
    st.line_chart(chart)

    st.markdown("#### Signal counts")
    counts = pd.DataFrame({
        "signal":[FRIENDLY[s] for s in SIGNALS],
        "count":[int(df[s].sum()) for s in SIGNALS]
    }).sort_values("count", ascending=False)
    st.dataframe(counts, use_container_width=True, hide_index=True)

    st.caption(
        "Regime labels use only trailing data and therefore do not look into the future."
    )
