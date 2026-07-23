// Shared mobile control overlay for the demos (user 2026-07-23). On touch / coarse
// pointers it adds a ☰ hamburger that opens a grid of big buttons; each button
// dispatches a physical-`code` KeyboardEvent on `window`, so it triggers the demo's
// existing key handlers (winit maps `event.code` → KeyCode). Camera stays on
// drag/pinch. Force it on desktop for testing with `?mobileui`.
//
// Usage (after the wasm has started):
//   import { setupMobileControls } from './mobile-controls.js';
//   setupMobileControls({ keys: [{ label:'Pause', code:'KeyP', key:'p' }, …] });
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
    #mc-burger { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 8px); right: 8px; z-index: 100000;
      width: 54px; height: 54px; border-radius: 13px; border: 1px solid #3a4666;
      background: #141a2cd8; color: #cdd6ea; font-size: 26px; line-height: 52px; text-align: center;
      -webkit-tap-highlight-color: transparent; user-select: none; touch-action: manipulation;
      backdrop-filter: blur(3px); box-shadow: 0 2px 10px #0008; }
    #mc-panel { position: fixed; top: calc(env(safe-area-inset-top, 0px) + 70px); right: 8px; z-index: 100000;
      display: none; grid-template-columns: repeat(3, minmax(78px, 1fr)); gap: 8px; padding: 10px;
      max-width: min(76vw, 340px); max-height: 74vh; overflow-y: auto;
      background: #0e1424f0; border: 1px solid #3a4666; border-radius: 14px; box-shadow: 0 6px 22px #000a; }
    #mc-panel.open { display: grid; }
    .mc-btn { min-height: 52px; padding: 7px 8px; border-radius: 11px; border: 1px solid #34406a;
      background: #1b2440; color: #e6ecfa; font: 600 14px system-ui, sans-serif; text-align: center;
      -webkit-tap-highlight-color: transparent; user-select: none; touch-action: manipulation; }
    .mc-btn:active { background: #2f3f70; border-color: #5570c0; }
    .mc-btn small { display: block; margin-top: 2px; color: #93a3cc; font-weight: 400; font-size: 10.5px; }
  `;
  document.head.appendChild(style);

  const burger = document.createElement('div');
  burger.id = 'mc-burger'; burger.textContent = '☰'; burger.setAttribute('aria-label', 'controls');
  const panel = document.createElement('div');
  panel.id = 'mc-panel';
  burger.addEventListener('click', () => panel.classList.toggle('open'));

  // Fire at both `window` (winit/wgpu demos listen there) and the <canvas>
  // (macroquad/miniquad registers keydown/keyup on the canvas element).
  const mkEvent = (type, k) => new KeyboardEvent(type, {
    key: k.key, code: k.code, keyCode: k.keyCode || 0, which: k.keyCode || 0, bubbles: true, cancelable: true,
  });
  const send = (type, k) => {
    window.dispatchEvent(mkEvent(type, k));
    const cv = document.querySelector('canvas');
    if (cv) { if (cv.tabIndex < 0) cv.tabIndex = 0; cv.dispatchEvent(mkEvent(type, k)); }
  };

  for (const k of cfg.keys) {
    const b = document.createElement('div');
    b.className = 'mc-btn';
    b.innerHTML = k.label + (k.sub ? `<small>${k.sub}</small>` : '');
    // A tap = keydown then keyup. Hold-to-repeat (camera) works via press/release too.
    let down = false;
    const press = (e) => { if (e) e.preventDefault(); if (!down) { down = true; send('keydown', k); } };
    const release = (e) => { if (e) e.preventDefault(); if (down) { down = false; send('keyup', k); } };
    b.addEventListener('touchstart', press, { passive: false });
    b.addEventListener('touchend', release, { passive: false });
    b.addEventListener('touchcancel', release, { passive: false });
    b.addEventListener('mousedown', press);
    b.addEventListener('mouseup', release);
    b.addEventListener('mouseleave', release);
    panel.appendChild(b);
  }
  document.body.appendChild(burger);
  document.body.appendChild(panel);
}
