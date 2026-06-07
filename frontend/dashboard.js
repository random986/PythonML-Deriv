/**
 * dashboard.js — Deriv AlgoBot Live Dashboard
 *
 * Connects to the Python bot's local WebSocket broadcast server
 * (ws://localhost:8765) and renders all live state in real time.
 *
 * Data flow:
 *   Python bot → JSON snapshot → WebSocket → here → DOM updates
 *
 * No external libraries required. Vanilla JS only.
 */

'use strict';

// ── Configuration ──────────────────────────────────────────────────────────
const WS_URL            = 'ws://127.0.0.1:8765';
const RECONNECT_DELAY   = 3000;   // ms before reconnect attempt
const MAX_LOG_ENTRIES   = 60;
const CONFIDENCE_GATE   = 0.58;   // must match config.py MIN_CONFIDENCE_THRESHOLD
const MAX_MG_STEPS      = 6;      // must match config.py MAX_MARTINGALE_STEPS

// ── State ──────────────────────────────────────────────────────────────────
let socket         = null;
let reconnectTimer = null;
let lastSnapshot   = null;
let bestOverSym    = null;
let bestUnderSym   = null;

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const elElapsed     = $('elapsed');
const elWsStatus    = $('ws-status');
const elReadyCount  = $('ready-count');
const elWrOver      = $('wr-over');
const elWrUnder     = $('wr-under');
const elPhaseBadge  = $('phase-badge');
const elMarketGrid  = $('market-grid');
const elTooltip     = $('tooltip');

// New Interactive UI refs
const elAccountSelect = $('account-select');
const elInputStake    = $('input-stake');
const elInputMg       = $('input-mg');

const elBtnStart      = $('btn-start');
const elBtnClearHist  = $('btn-clear-history');
const elHistoryRows   = $('history-rows');
const elHistoryEmpty  = $('history-empty');
const elAiLogsTerm    = $('ai-logs-terminal');
const elStatTradeCount = $('stat-trade-count');
const elStatSessionPnl = $('stat-session-pnl');
const elStatMaxLoss    = $('stat-max-loss');
const elStatMaxWin     = $('stat-max-win');
const elStatMaxStake   = $('stat-max-stake');
const elStatUptime     = $('stat-uptime');
const tabBtns         = document.querySelectorAll('.tab-btn');
const tabPanes        = document.querySelectorAll('.tab-pane');

let hasPopulatedAccounts = false;

// Martingale refs
const mgRefs = {
  over: {
    stake:  $('mg-over-stake'),
    step:   $('mg-over-step'),
    active: $('mg-over-active'),
    market: $('mg-over-market'),
    steps:  $('steps-over'),
    dot:    $('dot-over'),
  },
  under: {
    stake:  $('mg-under-stake'),
    step:   $('mg-under-step'),
    active: $('mg-under-active'),
    market: $('mg-under-market'),
    steps:  $('steps-under'),
    dot:    $('dot-under'),
  },
};

// Engine refs
const engRefs = {
  winsOver:    $('eng-wins-over'),
  lossesOver:  $('eng-losses-over'),
  winsUnder:   $('eng-wins-under'),
  lossesUnder: $('eng-losses-under'),
  guardOver:   $('eng-guard-over'),
  guardUnder:  $('eng-guard-under'),
};

// ═══════════════════════════════════════════════════════════
// WebSocket management
// ═══════════════════════════════════════════════════════════

function connect() {
  clearTimeout(reconnectTimer);
  setWsStatus('Connecting…', '#f0c14f');

  try {
    socket = new WebSocket(WS_URL);
  } catch (e) {
    scheduleReconnect('Failed to create WebSocket');
    return;
  }

  socket.addEventListener('open', () => {
    setWsStatus('Connected ✓', 'var(--accent-over)');
    addLog('Connected to Python bot', 'ok');
  });

  socket.addEventListener('message', evt => {
    try {
      const data = JSON.parse(evt.data);
      if (data.type === 'state_update') handleSnapshot(data);
    } catch (_) { /* ignore malformed */ }
  });

  socket.addEventListener('close', () => {
    setWsStatus('Disconnected', 'var(--accent-under)');
    addLog('Connection lost — reconnecting in 3s…', 'error');
    scheduleReconnect();
  });

  socket.addEventListener('error', () => {
    setWsStatus('Error', 'var(--accent-under)');
  });
}

function scheduleReconnect(reason) {
  if (reason) addLog(reason, 'error');
  reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
}

function setWsStatus(text, color) {
  elWsStatus.textContent = text;
  elWsStatus.style.color = color || '';
}

// ═══════════════════════════════════════════════════════════
// Snapshot processing
// ═══════════════════════════════════════════════════════════

function handleSnapshot(snap) {
  lastSnapshot = snap;

  updateHeader(snap);
  updateMartingale('over',  snap.martingale.over);
  updateMartingale('under', snap.martingale.under);
  updateEngine(snap.engine);
  updateMarketGrid(snap.markets);
  
  if (snap.accounts && snap.accounts.length > 0) {
    if (!hasPopulatedAccounts) {
      populateAccounts(snap.accounts, snap.current_account_id);
      hasPopulatedAccounts = true;
    } else {
      updateAccounts(snap.accounts, snap.current_account_id);
    }
  }
  
  if (snap.trade_history) {
    updateTradeHistory(snap.trade_history);
  }
  
  if (snap.learning_logs) {
    updateLearningLogs(snap.learning_logs, snap.engine);
  }

  // Toast notifications
  if (snap.toasts && snap.toasts.length > 0) {
    snap.toasts.forEach(t => showToast(t.message, t.type));
  }

  // Start/Stop Button Toggle (Only update if not just clicked to prevent UI jitter)
  if (!elBtnStart.dataset.optimistic) {
    if (snap.is_paused) {
      elBtnStart.textContent = "Start Trades";
      elBtnStart.className = "control-btn primary";
    } else {
      elBtnStart.textContent = "Stop Trades";
      elBtnStart.className = "control-btn danger";
    }
  }
}

// ── Header ─────────────────────────────────────────────────────────────────
function updateHeader(snap) {
  // Phase badge (Now shows the currently active account type)
  if (snap.account_type === 'demo') {
    elPhaseBadge.textContent = 'DEMO';
    elPhaseBadge.className   = 'phase-badge phase-demo';
  } else {
    elPhaseBadge.textContent = 'LIVE';
    elPhaseBadge.className   = 'phase-badge phase-live';
  }

  // Elapsed
  elElapsed.textContent = formatTime(snap.elapsed_seconds);

  // Ready count
  const markets  = snap.markets || [];
  const ready    = markets.filter(m => m.is_ready).length;
  elReadyCount.textContent = `${ready} / ${markets.length}`;

  // Win rates
  const eng = snap.engine || {};
  const wrOver  = eng.win_rate_over  != null ? (eng.win_rate_over  * 100).toFixed(1) + '%' : '—';
  const wrUnder = eng.win_rate_under != null ? (eng.win_rate_under * 100).toFixed(1) + '%' : '—';
  setIfChanged(elWrOver,  wrOver);
  setIfChanged(elWrUnder, wrUnder);
}

// ── Martingale panels ──────────────────────────────────────────────────────
function updateMartingale(side, mg) {
  const refs = mgRefs[side];
  if (!mg) return;

  setIfChanged(refs.stake,  `$${mg.stake != null ? mg.stake.toFixed(2) : '—'}`);
  setIfChanged(refs.step,   mg.step != null ? `${mg.step} / ${MAX_MG_STEPS}` : '—');
  setIfChanged(refs.market, mg.current_market || '—');

  // Active trade indicator
  if (mg.active) {
    refs.active.textContent = '● Active';
    refs.active.style.color = side === 'over' ? 'var(--accent-over)' : 'var(--accent-under)';
    refs.dot.classList.add('pulsing');
  } else {
    refs.active.textContent = '○ Idle';
    refs.active.style.color = 'var(--text-dim)';
    refs.dot.classList.remove('pulsing');
  }

  // Step progress dots
  renderStepDots(refs.steps, mg.step || 0, side);
}

function renderStepDots(container, step, side) {
  // Only re-render if step count changed (avoid DOM thrash)
  if (container.dataset.step === String(step)) return;
  container.dataset.step = step;
  container.innerHTML = '';

  for (let i = 0; i < MAX_MG_STEPS; i++) {
    const dot = document.createElement('div');
    dot.className = 'step-dot';
    if (i < step) {
      if (step >= MAX_MG_STEPS - 1)       dot.classList.add('filled-crit');
      else if (step >= MAX_MG_STEPS - 2)  dot.classList.add('filled-warn');
      else                                 dot.classList.add(`filled-${side}`);
    }
    container.appendChild(dot);
  }
}

// ── Engine stats ───────────────────────────────────────────────────────────
function updateEngine(eng) {
  if (!eng) return;
  setIfChanged(engRefs.winsOver,    eng.wins?.over    ?? 0);
  setIfChanged(engRefs.lossesOver,  eng.losses?.over  ?? 0);
  setIfChanged(engRefs.winsUnder,   eng.wins?.under   ?? 0);
  setIfChanged(engRefs.lossesUnder, eng.losses?.under ?? 0);

  const guardOverText  = eng.balance_guard_over_suppressed  ? '🔴 Suppressed' : '🟢 Active';
  const guardUnderText = eng.balance_guard_under_suppressed ? '🔴 Suppressed' : '🟢 Active';
  setIfChanged(engRefs.guardOver,  guardOverText);
  setIfChanged(engRefs.guardUnder, guardUnderText);
  engRefs.guardOver.style.color  = eng.balance_guard_over_suppressed  ? 'var(--accent-under)' : 'var(--accent-over)';
  engRefs.guardUnder.style.color = eng.balance_guard_under_suppressed ? 'var(--accent-under)' : 'var(--accent-over)';
}

// ═══════════════════════════════════════════════════════════
// Market Grid
// ═══════════════════════════════════════════════════════════

function updateMarketGrid(markets) {
  if (!markets || markets.length === 0) return;

  // Remove placeholder on first real data
  const placeholder = elMarketGrid.querySelector('.market-placeholder');
  if (placeholder) placeholder.remove();

  // Determine best over / best under markets right now
  let topOverConf  = -1, topUnderConf = -1;
  bestOverSym  = null;
  bestUnderSym = null;

  markets.forEach(m => {
    if (!m.is_ready) return;
    if (m.p_over  > topOverConf)  { topOverConf  = m.p_over;  bestOverSym  = m.symbol; }
    if (m.p_under > topUnderConf) { topUnderConf = m.p_under; bestUnderSym = m.symbol; }
  });

  markets.forEach(m => {
    let card = document.getElementById(`mc-${m.symbol}`);
    if (!card) {
      card = createMarketCard(m.symbol);
      elMarketGrid.appendChild(card);
    }
    updateMarketCard(card, m);
  });
}

function createMarketCard(symbol) {
  const card = document.createElement('div');
  card.id        = `mc-${symbol}`;
  card.className = 'market-card not-ready';
  card.innerHTML = `
    <div class="mc-ribbon">WARMING</div>
    <div class="mc-header">
      <span class="mc-symbol">${formatSymbol(symbol)}</span>
      <span class="mc-ready-badge" id="mc-badge-${symbol}">Warming up</span>
      <span class="mc-tick mono" id="mc-tick-${symbol}">0 ticks</span>
    </div>
    <div class="mc-bars">
      <div class="mc-bar-row">
        <span class="mc-bar-label" style="color:var(--accent-over)">▲ Over</span>
        <div class="mc-bar-track">
          <div class="mc-bar-fill over" id="mc-bar-over-${symbol}" style="width:50%"></div>
        </div>
        <span class="mc-bar-val over" id="mc-val-over-${symbol}">50%</span>
        <span class="mc-gate below" id="mc-gate-over-${symbol}">↓ gate</span>
      </div>
      <div class="mc-bar-row">
        <span class="mc-bar-label" style="color:var(--accent-under)">▼ Under</span>
        <div class="mc-bar-track">
          <div class="mc-bar-fill under" id="mc-bar-under-${symbol}" style="width:50%"></div>
        </div>
        <span class="mc-bar-val under" id="mc-val-under-${symbol}">50%</span>
        <span class="mc-gate below" id="mc-gate-under-${symbol}">↓ gate</span>
      </div>
    </div>
    <div class="mc-footer">
      <span class="mc-error-label">Virtual Error Rate</span>
      <span class="mc-error-val mid" id="mc-err-${symbol}">—</span>
    </div>
  `;

  // Tooltip on hover
  card.addEventListener('mouseenter', evt => showTooltip(evt, symbol));
  card.addEventListener('mousemove',  evt => moveTooltip(evt));
  card.addEventListener('mouseleave', () => hideTooltip());

  return card;
}

function updateMarketCard(card, m) {
  const sym = m.symbol;

  // Ready state
  card.classList.toggle('not-ready', !m.is_ready);

  // Best market highlights
  card.classList.toggle('best-over',  sym === bestOverSym  && m.p_over  >= CONFIDENCE_GATE);
  card.classList.toggle('best-under', sym === bestUnderSym && m.p_under >= CONFIDENCE_GATE);

  // Ribbon
  const ribbon = card.querySelector('.mc-ribbon');
  if (sym === bestOverSym && m.p_over >= CONFIDENCE_GATE)       ribbon.textContent = '★ BEST OVER';
  else if (sym === bestUnderSym && m.p_under >= CONFIDENCE_GATE) ribbon.textContent = '★ BEST UNDER';
  else if (!m.is_ready)                                          ribbon.textContent = 'WARMING';
  else                                                           ribbon.textContent = 'READY';

  // Badge
  const badge = $(`mc-badge-${sym}`);
  if (badge) {
    badge.textContent = m.is_ready ? 'Ready' : 'Warming up';
    badge.className   = `mc-ready-badge${m.is_ready ? ' ready' : ''}`;
  }

  // Tick count
  const tickEl = $(`mc-tick-${sym}`);
  if (tickEl) tickEl.textContent = `${(m.tick_count || 0).toLocaleString()} ticks`;

  // Over confidence bar
  const pOver = m.p_over ?? 0.5;
  const barOver = $(`mc-bar-over-${sym}`);
  const valOver = $(`mc-val-over-${sym}`);
  const gateOver = $(`mc-gate-over-${sym}`);
  if (barOver) barOver.style.width = `${(pOver * 100).toFixed(1)}%`;
  if (valOver) { valOver.textContent = `${(pOver * 100).toFixed(1)}%`; flash(valOver); }
  if (gateOver) {
    const above = pOver >= CONFIDENCE_GATE;
    gateOver.textContent  = above ? '↑ gate' : '↓ gate';
    gateOver.className    = `mc-gate ${above ? 'above' : 'below'}`;
  }

  // Under confidence bar
  const pUnder = m.p_under ?? 0.5;
  const barUnder = $(`mc-bar-under-${sym}`);
  const valUnder = $(`mc-val-under-${sym}`);
  const gateUnder = $(`mc-gate-under-${sym}`);
  if (barUnder) barUnder.style.width = `${(pUnder * 100).toFixed(1)}%`;
  if (valUnder) { valUnder.textContent = `${(pUnder * 100).toFixed(1)}%`; flash(valUnder); }
  if (gateUnder) {
    const above = pUnder >= CONFIDENCE_GATE;
    gateUnder.textContent = above ? '↑ gate' : '↓ gate';
    gateUnder.className   = `mc-gate ${above ? 'above' : 'below'}`;
  }

  // Error rate
  const errEl = $(`mc-err-${sym}`);
  if (errEl && m.virtual_error_rate != null) {
    const e = m.virtual_error_rate;
    errEl.textContent = (e * 100).toFixed(1) + '%';
    errEl.className   = `mc-error-val ${e < 0.45 ? 'low' : e < 0.52 ? 'mid' : 'high'}`;
  }
}

// ═══════════════════════════════════════════════════════════
// Event log
// ═══════════════════════════════════════════════════════════

function addLog(text, type = '') {
  const now  = new Date();
  const time = now.toTimeString().slice(0, 8);
  if (type === 'error') {
    console.error(`[${time}] ${text}`);
  } else {
    console.log(`[${time}] ${text}`);
  }
}

// ═══════════════════════════════════════════════════════════
// UI Interactive Features
// ═══════════════════════════════════════════════════════════

// Outbound Commands to Server
function sendCommand(cmd, payload = {}) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ cmd, ...payload }));
  } else {
    addLog('Cannot send command — WebSocket not connected', 'error');
  }
}

elAccountSelect.addEventListener('change', (e) => {
  sendCommand('set_account', { account_id: e.target.value });
  addLog(`Requested account switch to: ${e.target.value}`, 'info');
});

elInputStake.addEventListener('change', (e) => {
  const val = parseFloat(e.target.value);
  if (val >= 0.35) sendCommand('set_stake', { stake: val });
});

elInputMg.addEventListener('change', (e) => {
  const val = parseFloat(e.target.value);
  if (val >= 1.0) sendCommand('set_martingale', { multiplier: val });
});

// Button: Start/Stop
elBtnStart.addEventListener('click', () => {
  elBtnStart.dataset.optimistic = "true";
  if (elBtnStart.textContent.includes("Start")) {
    sendCommand('start_trades');
    addLog('Requested START trades...', 'info');
    elBtnStart.textContent = "Stop Trades";
    elBtnStart.className = "control-btn danger";
  } else {
    sendCommand('stop_trades');
    addLog('Requested STOP trades...', 'warn');
    elBtnStart.textContent = "Start Trades";
    elBtnStart.className = "control-btn primary";
  }
  setTimeout(() => delete elBtnStart.dataset.optimistic, 1500);
});

// Button: Clear History
if (elBtnClearHist) {
  elBtnClearHist.addEventListener('click', () => {
    sendCommand('clear_history');
    if (elHistoryRows) elHistoryRows.innerHTML = '';
    if (elHistoryEmpty) { elHistoryEmpty.style.display = 'block'; }
    updateHistoryStats([]);
  });
}

// Bottom Tabs
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => {
      b.classList.remove('active');
      b.style.borderBottom = '2px solid transparent';
    });
    tabPanes.forEach(p => p.style.display = 'none');
    
    btn.classList.add('active');
    btn.style.borderBottom = '2px solid var(--accent-over)';
    const pane = document.getElementById(btn.dataset.tab);
    if (pane) pane.style.display = 'flex';
  });
});

// Update UI from Snapshot
function populateAccounts(accounts, currentId) {
  elAccountSelect.innerHTML = '';
  accounts.forEach(acc => {
    const opt = document.createElement('option');
    opt.value = acc.account_id;
    opt.textContent = `${acc.account_type === 'demo' ? 'Demo' : 'Real'} - ${parseFloat(acc.balance).toFixed(2)} ${acc.currency}`;
    if (acc.account_id === currentId) opt.selected = true;
    elAccountSelect.appendChild(opt);
  });
}

function updateAccounts(accounts, currentId) {
  Array.from(elAccountSelect.options).forEach(opt => {
    const acc = accounts.find(a => a.account_id === opt.value);
    if (acc) {
      opt.textContent = `${acc.account_type === 'demo' ? 'Demo' : 'Real'} - ${parseFloat(acc.balance).toFixed(2)} ${acc.currency}`;
    }
  });
  if (currentId && elAccountSelect.value !== currentId) {
    elAccountSelect.value = currentId;
  }
}

function relativeTime(ts) {
  const diff = Math.floor((Date.now() / 1000) - ts);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function updateHistoryStats(trades) {
  const eng = lastSnapshot ? lastSnapshot.engine : null;
  const settled = trades.filter(t => t.status === 'sold');
  const totalPnl = eng ? eng.session_pnl : settled.reduce((sum, t) => sum + (t.profit || 0), 0);
  const maxStake = trades.length ? Math.max(...trades.map(t => t.stake || 0)) : 0;
  
  // Max loss from engine tracking (with timestamp)
  let maxLossDisplay = '$0.00';
  let maxWinDisplay = '$0.00';
  if (eng && eng.max_loss && eng.max_loss.amount < 0) {
    maxLossDisplay = `$${Math.abs(eng.max_loss.amount).toFixed(2)}`;
    if (elStatMaxLoss) elStatMaxLoss.title = `Max loss at ${eng.max_loss.timestamp} on ${eng.max_loss.symbol} (${eng.max_loss.side})`;
  }
  if (eng && eng.max_win && eng.max_win.amount > 0) {
    maxWinDisplay = `$${eng.max_win.amount.toFixed(2)}`;
    if (elStatMaxWin) elStatMaxWin.title = `Max win at ${eng.max_win.timestamp} on ${eng.max_win.symbol} (${eng.max_win.side})`;
  }
  
  if (elStatTradeCount) elStatTradeCount.textContent = trades.length;
  if (elStatSessionPnl) {
    elStatSessionPnl.textContent = `$${totalPnl >= 0 ? '' : ''}${totalPnl.toFixed(2)}`;
    elStatSessionPnl.style.color = totalPnl >= 0 ? 'var(--accent-over)' : 'var(--accent-under)';
  }
  if (elStatMaxLoss) {
    elStatMaxLoss.textContent = maxLossDisplay;
  }
  if (elStatMaxWin) {
    elStatMaxWin.textContent = maxWinDisplay;
  }
  if (elStatMaxStake) {
    const s = eng && eng.max_stake ? eng.max_stake : maxStake;
    elStatMaxStake.textContent = `$${s.toFixed(2)}`;
  }
}

function updateTradeHistory(trades) {
  if (!trades || trades.length === 0) {
    if (elHistoryEmpty) elHistoryEmpty.style.display = 'block';
    updateHistoryStats([]);
    return;
  }
  
  if (elHistoryEmpty) elHistoryEmpty.style.display = 'none';
  updateHistoryStats(trades);
  
  // Update uptime from snapshot
  if (lastSnapshot && lastSnapshot.elapsed_sec && elStatUptime) {
    const s = Math.floor(lastSnapshot.elapsed_sec);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    elStatUptime.textContent = `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
  }
  
  let html = '';
  trades.forEach(t => {
    const d = new Date(t.timestamp * 1000);
    const timeStr = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
    const rel = relativeTime(t.timestamp);
    const marketName = formatSymbol(t.symbol);
    const sideLabel = t.side === 'over' ? 'OVER5' : 'UNDER5';
    let sideColor = t.side === 'over' ? 'var(--accent-over)' : 'var(--accent-under)';
    
    let digitDisplay = '-';
    if (t.status === 'open') digitDisplay = '...';
    else if (t.digit) {
      const digitNum = parseInt(t.digit);
      if (t.side === 'over') {
        digitDisplay = `${sideLabel} (${t.digit})`;
      } else {
        digitDisplay = `${sideLabel} (${t.digit})`;
      }
    }
    
    let pnlDisplay = '...';
    let pnlColor = 'var(--text-muted)';
    let sideBg = t.side === 'over' ? 'rgba(0,208,156,0.15)' : 'rgba(255,75,75,0.15)';

    if (t.status === 'sold') {
      pnlDisplay = t.profit >= 0 ? `+${t.profit.toFixed(2)}` : t.profit.toFixed(2);
      pnlColor = t.profit >= 0 ? 'var(--accent-over)' : 'var(--accent-under)';
      sideColor = t.profit > 0 ? 'var(--accent-over)' : 'var(--accent-under)';
      sideBg = t.profit > 0 ? 'rgba(0,208,156,0.15)' : 'rgba(255,75,75,0.15)';
    } else if (t.status === 'failed') {
      pnlDisplay = 'FAIL';
      pnlColor = 'var(--accent-under)';
      sideColor = 'var(--accent-under)';
      sideBg = 'rgba(255,75,75,0.15)';
    }
    
    html += `
      <div style="display: grid; grid-template-columns: 1.3fr 1fr 1.1fr 0.9fr 0.8fr; padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.04); align-items: center; transition: background 0.15s;" onmouseenter="this.style.background='rgba(255,255,255,0.03)'" onmouseleave="this.style.background='none'">
        <div>
          <div style="font-size: 0.82rem; color: var(--text-secondary); font-weight: 500;">${timeStr}</div>
          <div style="font-size: 0.65rem; color: var(--text-muted);">${rel}</div>
        </div>
        <div style="font-weight: 700; color: var(--accent-blue); font-size: 0.85rem;">${marketName}</div>
        <div>
          <span style="display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; background: ${sideBg}; color: ${sideColor};">${t.status === 'sold' || t.status === 'failed' ? digitDisplay : sideLabel}</span>
        </div>
        <div style="font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #fff;">$${t.stake.toFixed(2)}</div>
        <div style="text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; color: ${pnlColor};">${pnlDisplay}</div>
      </div>
    `;
  });
  
  elHistoryRows.innerHTML = html;
}

function escapeHtml(unsafe) {
  return (unsafe || "").toString()
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function updateLearningLogs(logs, engineStats) {
  if (!logs || !logs.length) return;
  
  if (engineStats) {
    const elAccuracy = document.getElementById('stat-ai-accuracy');
    const elLabel = document.getElementById('stat-ai-accuracy-label');
    if (elLabel) elLabel.textContent = 'PRED. ACCURACY';

    if (elAccuracy) {
      const mlAcc = engineStats.ml_prediction_accuracy;
      const mlTotal = engineStats.ml_predictions_total || 0;

      if (mlAcc === null || mlAcc === undefined || mlTotal < 10) {
        elAccuracy.textContent = '--';
        elAccuracy.title = `Calibrating... ${mlTotal} predictions so far. Need at least 10.`;
        elAccuracy.style.color = 'var(--text-muted)';
      } else {
        const pct = (mlAcc * 100).toFixed(1);
        elAccuracy.textContent = `${pct}%`;
        elAccuracy.title = `Model predicted the correct digit side ${pct}% of the time over ${mlTotal} virtual trades.\nBaseline is ~50% (random). Digit 5 (house edge) is excluded.`;
        // Colour code: green if above baseline, red if below
        if (mlAcc >= 0.52) {
          elAccuracy.style.color = 'var(--accent-over)';
        } else if (mlAcc >= 0.48) {
          elAccuracy.style.color = 'var(--text-muted)';
        } else {
          elAccuracy.style.color = 'var(--accent-under)';
        }
      }
    }
  }

  // Update intelligence cards
  const elMktsReady = document.getElementById('stat-markets-ready');
  if (elMktsReady && engineStats) elMktsReady.textContent = engineStats.markets_ready || 0;

  // Virtual trades: total virtual wins + losses from engine
  const elVirtual = document.getElementById('stat-virtual-trades');
  if (elVirtual && engineStats) {
    const vw = (engineStats.virtual_wins?.over || 0) + (engineStats.virtual_wins?.under || 0);
    const vl = (engineStats.virtual_losses?.over || 0) + (engineStats.virtual_losses?.under || 0);
    elVirtual.textContent = (vw + vl).toLocaleString();
  }

  // Recovery mode: on if either martingale step > 0
  const elRecov = document.getElementById('stat-recovery-mode');
  if (elRecov && lastSnapshot && lastSnapshot.martingale) {
    const overStep = lastSnapshot.martingale.over?.step || 0;
    const underStep = lastSnapshot.martingale.under?.step || 0;
    const inRecovery = overStep > 0 || underStep > 0;
    elRecov.textContent = inRecovery ? `STEP ${Math.max(overStep, underStep)}` : 'OFF';
    elRecov.style.color = inRecovery ? '#ff4b4b' : '#00d09c';
  }
  
  let html = '';
  logs.forEach(msg => {
    let icon = '⚡';
    let typeClass = 'info';
    let badge = 'LOG';
    let cleanMsg = escapeHtml(msg);
    
    if (msg.includes('[TRADE]')) {
      icon = '🚀'; typeClass = 'trade'; badge = 'EXEC';
      cleanMsg = cleanMsg.replace('[TRADE]', '');
    } else if (msg.includes('WIN')) {
      icon = '✅'; typeClass = 'win'; badge = 'WIN';
      cleanMsg = cleanMsg.replace(/\[.*\]/g, '');
    } else if (msg.includes('LOSS')) {
      icon = '❌'; typeClass = 'loss'; badge = 'LOSS';
      cleanMsg = cleanMsg.replace(/\[.*\]/g, '');
    } else if (msg.includes('[VIRTUAL PENALTY]')) {
      icon = '🧠'; typeClass = 'penalty'; badge = 'LEARN';
      cleanMsg = cleanMsg.replace('[VIRTUAL PENALTY]', '');
    } else if (msg.includes('[RECOVERY]')) {
      icon = '🔄'; typeClass = 'recovery'; badge = 'RECOV';
      cleanMsg = cleanMsg.replace('[RECOVERY]', '');
    } else if (msg.includes('[Engine]')) {
      icon = '⚙️'; typeClass = 'engine'; badge = 'EVAL';
      cleanMsg = cleanMsg.replace('[Engine]', '');
    } else if (msg.includes('[DEMO]') || msg.includes('[LIVE-LEARN]')) {
      icon = '🔬'; typeClass = 'demo'; badge = 'NEURAL';
      cleanMsg = cleanMsg.replace(/\[DEMO\]|\[LIVE-LEARN\]/g, '');
    }
    
    html += `
      <div class="log-entry" style="display: flex; align-items: start; gap: 10px; margin-bottom: 8px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 6px; border-left: 3px solid var(--accent-${typeClass === 'win' ? 'over' : typeClass === 'loss' ? 'under' : typeClass === 'penalty' ? 'gold' : typeClass === 'demo' ? 'demo' : 'blue'});">
        <div style="min-width: 20px;">${icon}</div>
        <div style="flex: 1;">
          <span style="font-size: 0.65rem; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; margin-right: 8px;">${badge}</span>
          <span style="color: #ddd; line-height: 1.4;">${cleanMsg.trim()}</span>
        </div>
      </div>
    `;
  });
  
  const elLogs = document.getElementById('ai-logs-terminal');
  if (elLogs && elLogs.innerHTML !== html) {
    const isBottom = elLogs.scrollHeight - elLogs.clientHeight <= elLogs.scrollTop + 50;
    elLogs.innerHTML = html;
    if (isBottom || elLogs.scrollTop === 0) {
      elLogs.scrollTop = elLogs.scrollHeight;
    }
  }
}

// ═══════════════════════════════════════════════════════════
// Tooltip
// ═══════════════════════════════════════════════════════════

function showTooltip(evt, symbol) {
  if (!lastSnapshot) return;
  const m = (lastSnapshot.markets || []).find(x => x.symbol === symbol);
  if (!m) return;

  elTooltip.innerHTML = `
    <strong>${formatSymbol(symbol)}</strong><br>
    Ticks:        ${(m.tick_count || 0).toLocaleString()}<br>
    P(Over):      ${((m.p_over  || 0) * 100).toFixed(2)}%<br>
    P(Under):     ${((m.p_under || 0) * 100).toFixed(2)}%<br>
    Error rate:   ${((m.virtual_error_rate || 0) * 100).toFixed(2)}%<br>
    Loss streak ▲: ${m.loss_streak_over  ?? '—'}<br>
    Loss streak ▼: ${m.loss_streak_under ?? '—'}
  `;
  elTooltip.classList.remove('hidden');
  moveTooltip(evt);
}

function moveTooltip(evt) {
  const pad = 14;
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  const tw = elTooltip.offsetWidth;
  const th = elTooltip.offsetHeight;
  if (x + tw > window.innerWidth)  x = evt.clientX - tw - pad;
  if (y + th > window.innerHeight) y = evt.clientY - th - pad;
  elTooltip.style.left = x + 'px';
  elTooltip.style.top  = y + 'px';
}

function hideTooltip() { elTooltip.classList.add('hidden'); }

// ═══════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════

/** Only write to an element if the value changed (avoids layout thrash). */
function setIfChanged(el, value) {
  if (!el) return;
  const s = String(value);
  if (el.textContent !== s) {
    el.textContent = s;
    flash(el);
  }
}

/** Brief white flash to signal a value update. */
function flash(el) {
  el.classList.remove('num-flash');
  void el.offsetWidth; // reflow trick to restart animation
  el.classList.add('num-flash');
}

/** Seconds → MM:SS */
function formatTime(sec) {
  if (sec == null) return '00:00';
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

/** R_50 → "Vol 50", JD25 → "Jump 25", etc. */
function formatSymbol(sym) {
  if (!sym) return sym;
  if (sym.startsWith('R_'))  return `Vol ${sym.slice(2)}`;
  if (sym.startsWith('JD'))  return `Jump ${sym.slice(2)}`;
  return sym;
}

// ═══════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════

// Pre-populate step dots with empty state
['over', 'under'].forEach(side => renderStepDots(mgRefs[side].steps, 0, side));

// Connect immediately
connect();

// Visible connection hint if backend is not running
setTimeout(() => {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addLog('Tip: Make sure main.py is running first!', 'info');
  }
}, 4000);

// ═══════════════════════════════════════════════════════════
// Toast Notification System
// ═══════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
  // Create container if it doesn't exist
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 10000; display: flex; flex-direction: column; gap: 10px; pointer-events: none;';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  const bgColor = type === 'success' ? 'rgba(0, 208, 156, 0.95)' : type === 'error' ? 'rgba(255, 75, 75, 0.95)' : 'rgba(99, 102, 241, 0.95)';
  toast.style.cssText = `
    pointer-events: auto;
    padding: 16px 24px;
    background: ${bgColor};
    color: #fff;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transform: translateX(120%);
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s;
    opacity: 0;
    max-width: 450px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
  `;
  toast.textContent = message;
  container.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => {
    toast.style.transform = 'translateX(0)';
    toast.style.opacity = '1';
  });

  // Animate out after 6 seconds
  setTimeout(() => {
    toast.style.transform = 'translateX(120%)';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 400);
  }, 6000);
}
// ── Copy logs to clipboard ──────────────────────────────────────────────────
window._aiLogBuffer = window._aiLogBuffer || [];

function copyLogs() {
  const terminal = document.getElementById('ai-logs-terminal');
  if (!terminal) return;

  // Extract plain text from all log entries, preserving timestamps
  const lines = [];
  terminal.querySelectorAll('.log-entry, div').forEach(el => {
    const txt = el.innerText || el.textContent;
    if (txt && txt.trim()) lines.push(txt.trim());
  });

  const text = lines.join('\n');
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('btn-copy-logs');
    if (btn) {
      const orig = btn.innerHTML;
      btn.innerHTML = '✓ Copied!';
      btn.style.background = 'rgba(0,208,156,0.25)';
      btn.style.borderColor = 'rgba(0,208,156,0.5)';
      btn.style.color = '#00d09c';
      setTimeout(() => {
        btn.innerHTML = orig;
        btn.style.background = 'rgba(100,149,237,0.15)';
        btn.style.borderColor = 'rgba(100,149,237,0.3)';
        btn.style.color = '#6495ed';
      }, 2000);
    }
  }).catch(() => {
    // Fallback: select all text in terminal
    const range = document.createRange();
    range.selectNodeContents(terminal);
    window.getSelection().removeAllRanges();
    window.getSelection().addRange(range);
  });
}
