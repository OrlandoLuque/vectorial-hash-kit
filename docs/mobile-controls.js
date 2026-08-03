// Shared mobile control overlay for the demos (user 2026-07-23). On touch-primary
// devices it adds a ☰ button (top-LEFT, clear of the on-screen FPS/meters on the
// right) that opens a card of big finger-buttons. Each button dispatches a
// physical-`code` KeyboardEvent at BOTH `window` (winit/wgpu) and the <canvas>
// (macroquad/miniquad), so the demo's own key handlers fire — no per-demo plumbing.
// Camera stays on drag/pinch. Force it on desktop for testing with `?mobileui`.
//
// A key entry: { label, code, key, keyCode, sub?, cost?, start?, cycle? }.
//   cost: 0(light/green)…1(heavy/red) tints the button by performance impact.
//   cycle: [{ name, cost? }, …] — a button that steps through options on each tap,
//          updating its own caption + colour to the CURRENT option (start = index).
export function setupMobileControls(cfg) {
  const force = location.search.includes('mobileui');
  const mm = (q) => window.matchMedia && matchMedia(q).matches;
  // Touch-PRIMARY device: coarse pointer AND no hover (a phone/tablet). A desktop
  // with a touchscreen still reports hover:hover for its mouse, so it's left alone.
  const isMobile = force
    || (mm('(pointer: coarse)') && mm('(hover: none)'))
    || /Android|iPhone|iPad|iPod|Mobile|Silk/i.test(navigator.userAgent || '');
  if (!isMobile) return;

  const style = document.createElement('style');
  style.textContent = `
    #mc-burger { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 10px);
      left: calc(env(safe-area-inset-left, 0px) + 10px); z-index: 100000;
      width: 54px; height: 54px; border-radius: 15px; border: 1px solid #3c4c78;
      background: linear-gradient(160deg, #202a49, #141a2c); color: #e6ecfb;
      font-size: 25px; line-height: 54px; text-align: center; box-shadow: 0 3px 16px #000a;
      -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
      -webkit-tap-highlight-color: transparent; user-select: none; touch-action: manipulation;
      transition: transform .12s ease, box-shadow .12s ease; }
    #mc-burger:active { transform: scale(.93); box-shadow: 0 1px 8px #000a; }
    #mc-burger.open { background: linear-gradient(160deg, #2a3a63, #1a2138); }
    #mc-panel { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 74px);
      left: calc(env(safe-area-inset-left, 0px) + 10px); z-index: 100000; display: none;
      grid-template-columns: repeat(auto-fill, minmax(106px, 1fr)); gap: 9px; padding: 12px;
      width: min(88vw, 384px); max-height: 78vh; overflow-y: auto;
      background: linear-gradient(180deg, #121a2df4, #0c1120f7); border: 1px solid #2f3c5e;
      border-radius: 20px; box-shadow: 0 12px 38px #000c;
      -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
      -webkit-overflow-scrolling: touch; overscroll-behavior: contain; }
    #mc-panel.open { display: grid; animation: mc-in .16s ease; }
    /* A - / + pair is one grid item, so the auto-fill grid can never put the minus at the end
       of a row and the plus at the start of the next — which it did, and which is unusable on
       a phone where you are alternating between them by feel. */
    .mc-pair { grid-column: span 2; display: flex; flex-direction: column; gap: 7px; }
    .mc-pair-row { display: flex; gap: 9px; }
    .mc-pair-row > .mc-btn { flex: 1 1 0; min-width: 0; }
    .mc-slider { display: flex; align-items: center; gap: 8px; padding: 2px 4px 0; }
    .mc-slider input { flex: 1 1 auto; min-width: 0; accent-color: #6ea8ff; height: 26px; }
    .mc-slider .mc-val { color: #cfe0ff; font: 700 11px ui-monospace, monospace; min-width: 44px; text-align: right; }
    @keyframes mc-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }
    #mc-title { grid-column: 1/-1; margin: 1px 3px 3px; color: #93a3c8; font: 700 11px system-ui, sans-serif;
      letter-spacing: 1.1px; text-transform: uppercase; }
    .mc-btn { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 3px;
      min-height: 62px; padding: 8px 6px; border-radius: 14px; border: 1px solid #37436e; background: #1a2340;
      color: #eef3fc; text-align: center; cursor: pointer; box-shadow: inset 0 1px 0 #ffffff0f;
      -webkit-tap-highlight-color: transparent; user-select: none; touch-action: manipulation;
      transition: transform .07s ease, filter .07s ease; }
    .mc-btn.pressed { transform: scale(.93); filter: brightness(1.4); }
    .mc-name { font: 700 15px system-ui, sans-serif; line-height: 1.1; }
    .mc-sub { font: 600 11px system-ui, sans-serif; letter-spacing: .2px; padding: 1px 8px; border-radius: 999px;
      background: #ffffff1f; color: #f2f6ff; max-width: 96px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .mc-sub:empty { display: none; }
  `;
  document.head.appendChild(style);

  // cost 0 → green, .5 → amber, 1 → red (dark-UI friendly).
  const costStyle = (c) => {
    if (c == null) return null;
    const h = Math.round(120 * (1 - Math.max(0, Math.min(1, c))));
    return { bg: `hsl(${h} 44% 23%)`, bd: `hsl(${h} 48% 41%)` };
  };

  const burger = document.createElement('div');
  burger.id = 'mc-burger'; burger.textContent = '☰'; burger.setAttribute('aria-label', 'controls');
  const panel = document.createElement('div');
  panel.id = 'mc-panel';
  const title = document.createElement('div');
  title.id = 'mc-title'; title.textContent = cfg.title || 'Controls';
  panel.appendChild(title);

  // Macroquad shells keep a “← demos” link at top-left; drop the burger below it.
  if (document.getElementById('back')) {
    burger.style.top = 'calc(env(safe-area-inset-top, 0px) + 56px)';
    panel.style.top = 'calc(env(safe-area-inset-top, 0px) + 120px)';
  }
  burger.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    burger.classList.toggle('open', open);
    burger.textContent = open ? '✕' : '☰';
  });

  const mkEvent = (type, k) => new KeyboardEvent(type, {
    key: k.key, code: k.code, keyCode: k.keyCode || 0, which: k.keyCode || 0, bubbles: true, cancelable: true,
  });
  const send = (type, k) => {
    window.dispatchEvent(mkEvent(type, k));
    const cv = document.querySelector('canvas');
    if (cv) { if (cv.tabIndex < 0) cv.tabIndex = 0; cv.dispatchEvent(mkEvent(type, k)); }
  };
  const vibe = () => { try { navigator.vibrate && navigator.vibrate(8); } catch (e) {} };

  // Consecutive entries sharing a `group` become one unbreakable pair, optionally with a
  // slider above them.
  //
  // A slider here can only do what the rest of this overlay does: **press keys**. It has no way
  // to set a value, and no way to read one — the demo owns the number and never tells the page.
  // So a slider tracks its own position and sends the difference as that many presses, which is
  // honest as long as two things hold:
  //
  //   1. the range is only a couple of dozen presses wide (the horde steps 4 000 at a time from
  //      2k to 100k, so 25 — fine; the 2D critters step FIVE against a 40 000 cap, which would
  //      be 8 000 events, so that one deliberately has no slider), and
  //   2. **the presses are spaced by a frame.** The demos read input with edge detection once
  //      per frame, so two keydowns inside one frame are seen as one press and the slider would
  //      silently under-shoot. This is a correctness constraint, not politeness.
  //
  // If you also use the keyboard, the slider's idea of where it is will drift from the demo's.
  // That is a real limitation of driving an app you cannot query, and the readout says "step"
  // rather than pretending to know the population.
  const groups = new Map();

  function buildSlider(g) {
    const c = g.cfg;
    const row = document.createElement('div'); row.className = 'mc-slider';
    const inp = document.createElement('input');
    inp.type = 'range'; inp.min = '0'; inp.max = String(c.steps); inp.step = '1';
    inp.value = String(c.start != null ? c.start : 0);
    const val = document.createElement('span'); val.className = 'mc-val';
    const show = (i) => { val.textContent = c.fmt ? c.fmt(Number(i)) : `${i}/${c.steps}`; };
    show(inp.value);
    row.appendChild(inp); row.appendChild(val);
    g.insertBefore(row, g.row);

    // One press per animation frame. Anything faster is dropped by the demo's edge detection.
    let at = Number(inp.value), queue = 0, pumping = false;
    const pump = () => {
      if (!queue) { pumping = false; return; }
      const k = queue > 0 ? g.inc : g.dec;
      queue += queue > 0 ? -1 : 1;
      send('keydown', k); send('keyup', k);
      requestAnimationFrame(pump);
    };
    inp.addEventListener('input', () => {
      const want = Number(inp.value);
      show(want);
      queue += want - at; at = want;
      if (!pumping) { pumping = true; requestAnimationFrame(pump); }
    });
    return true;
  }

  for (const k of cfg.keys) {
    const b = document.createElement('button');
    b.className = 'mc-btn';
    const nm = document.createElement('span'); nm.className = 'mc-name'; nm.textContent = k.label;
    const sub = document.createElement('span'); sub.className = 'mc-sub';
    b.appendChild(nm); b.appendChild(sub);

    let idx = k.start || 0;
    const paint = () => {
      let cost = k.cost;
      if (k.cycle && k.cycle.length) {
        const st = k.cycle[idx % k.cycle.length];
        sub.textContent = st.name || '';
        if (st.cost != null) cost = st.cost;
      } else {
        sub.textContent = k.sub || '';
      }
      const cs = costStyle(cost);
      if (cs) { b.style.background = cs.bg; b.style.borderColor = cs.bd; }
      else { b.style.background = ''; b.style.borderColor = ''; }
    };
    paint();

    let held = false, timer = null;
    // Hold to repeat. A step button sends ONE event per tap, which is fine when the step is a
    // quarter of the range and useless when it is not: the 2D critters add five at a time out
    // of a 40 000 cap, i.e. 8 000 taps to fill the world. Repeating on hold turns that into a
    // few seconds of thumb without touching the simulation's step size, and it costs nothing on
    // the buttons where one press is the point.
    //
    // Cycle buttons (pause, index, brush) never repeat: they advance a state, and repeating one
    // just spins through the states under your finger.
    const repeats = !(k.cycle && k.cycle.length) && k.repeat !== false;
    const fire = () => { send('keydown', k); send('keyup', k); };
    const startRepeat = () => {
      if (!repeats) return;
      let gap = 260;                                   // a beat, so a deliberate single tap stays single
      const tick = () => {
        fire();
        gap = Math.max(45, gap * 0.82);                // accelerate: coarse at first, fast once you commit
        timer = setTimeout(tick, gap);
      };
      timer = setTimeout(tick, 420);
    };
    const press = (e) => {
      if (e) e.preventDefault();
      if (held) return; held = true;
      b.classList.add('pressed'); send('keydown', k); vibe();
      if (k.cycle && k.cycle.length) { idx = (idx + 1) % k.cycle.length; paint(); } // reflect the new state now
      startRepeat();
    };
    const release = (e) => {
      if (e) e.preventDefault();
      if (!held) return; held = false;
      if (timer) { clearTimeout(timer); timer = null; }
      b.classList.remove('pressed'); send('keyup', k);
    };
    b.addEventListener('touchstart', press, { passive: false });
    b.addEventListener('touchend', release, { passive: false });
    b.addEventListener('touchcancel', release, { passive: false });
    b.addEventListener('mousedown', press);
    b.addEventListener('mouseup', release);
    b.addEventListener('mouseleave', release);
    if (k.group) {
      let g = groups.get(k.group);
      if (!g) {
        g = document.createElement('div'); g.className = 'mc-pair';
        g.row = document.createElement('div'); g.row.className = 'mc-pair-row';
        groups.set(k.group, g); panel.appendChild(g);
        g.appendChild(g.row);
      }
      g.row.appendChild(b);
      if (k.slider) { g.dec = k; g.cfg = k.slider; } else if (g.cfg) { g.inc = k; }
      if (g.dec && g.inc && !g.slid) { g.slid = buildSlider(g); }
    } else {
      panel.appendChild(b);
    }
  }
  document.body.appendChild(burger);
  document.body.appendChild(panel);
}
