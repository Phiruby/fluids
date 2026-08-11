use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex};
use crate::physics::{MpmElasticParticle, MpmFluidParticle};
use winit::dpi::LogicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

const MAX_POINTS: usize = 2_000;
const PANEL_H: usize = 130;
const PAD: usize = 8;
const LEFT: usize = 58;
const BOTTOM: usize = 18;

#[derive(Default)]
struct PlotState {
    title: String,
    series: HashMap<String, Vec<[f64; 2]>>,
    samples_received: u64,
    last_t: f64,
}

/// Handle used by the sim to push samples (no UI thread of its own).
#[derive(Clone)]
pub struct LivePlot {
    state: Arc<Mutex<PlotState>>,
}

impl LivePlot {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            state: Arc::new(Mutex::new(PlotState {
                title: title.into(),
                ..Default::default()
            })),
        }
    }

    pub fn sample(&self, t: f64, values: impl IntoIterator<Item = (String, f64)>) {
        let Ok(mut g) = self.state.lock() else {
            return;
        };
        for (name, y) in values {
            let pts = g.series.entry(name).or_default();
            pts.push([t, y]);
            if pts.len() > MAX_POINTS {
                let drop = pts.len() - MAX_POINTS;
                pts.drain(0..drop);
            }
        }
        g.samples_received += 1;
        g.last_t = t;
    }

    pub fn reset(&self, title: impl Into<String>) {
        let Ok(mut g) = self.state.lock() else {
            return;
        };
        g.series.clear();
        g.samples_received = 0;
        g.last_t = 0.0;
        g.title = title.into();
    }
}

/// Software-rendered plot window owned by the main event loop.
pub struct PlotWindow {
    window: Arc<Window>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
    plot: LivePlot,
    cursor: Option<(f64, f64)>,
}

impl PlotWindow {
    pub fn open(event_loop: &ActiveEventLoop, plot: LivePlot) -> anyhow::Result<Self> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("MPM metrics")
                    .with_inner_size(LogicalSize::new(720.0, 560.0)),
            )?,
        );
        let context = softbuffer::Context::new(window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer context: {e}"))?;
        let surface = softbuffer::Surface::new(&context, window.clone())
            .map_err(|e| anyhow::anyhow!("softbuffer surface: {e}"))?;
        window.request_redraw();
        Ok(Self {
            window,
            surface,
            plot,
            cursor: None,
        })
    }

    pub fn window_id(&self) -> WindowId {
        self.window.id()
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn set_cursor(&mut self, x: f64, y: f64) {
        self.cursor = Some((x, y));
        self.window.request_redraw();
    }

    pub fn clear_cursor(&mut self) {
        self.cursor = None;
        self.window.request_redraw();
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        let _ = self.surface.resize(
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        );
    }

    pub fn paint(&mut self) {
        let size = self.window.inner_size();
        let w = size.width as usize;
        let h = size.height as usize;
        if w == 0 || h == 0 {
            return;
        }
        let _ = self.surface.resize(
            NonZeroU32::new(size.width).unwrap(),
            NonZeroU32::new(size.height).unwrap(),
        );

        let (title, series, samples, last_t) = {
            let Ok(g) = self.plot.state.lock() else {
                return;
            };
            (
                g.title.clone(),
                g.series.clone(),
                g.samples_received,
                g.last_t,
            )
        };
        let cursor = self.cursor;

        let Ok(mut buf) = self.surface.buffer_mut() else {
            return;
        };
        buf.fill(rgb(18, 18, 22));

        draw_text(
            &mut buf,
            w,
            10,
            10,
            &format!("{title}   samples={samples} series={} last_t={last_t:.3}s", series.len()),
            rgb(220, 220, 220),
        );

        if series.is_empty() {
            draw_text(
                &mut buf,
                w,
                10,
                36,
                "Waiting for samples… left-click a particle to track it.",
                rgb(200, 160, 60),
            );
            let _ = buf.present();
            return;
        }

        let mut names: Vec<_> = series.keys().cloned().collect();
        names.sort();
        let mut y0 = 40usize;
        let colors = [
            rgb(80, 180, 255),
            rgb(255, 140, 80),
            rgb(120, 220, 140),
            rgb(220, 120, 220),
            rgb(255, 220, 100),
            rgb(255, 100, 120),
        ];

        let mut hover: Option<String> = None;
        for (i, name) in names.iter().enumerate() {
            let Some(pts) = series.get(name) else {
                continue;
            };
            if y0 + PANEL_H > h {
                break;
            }
            let color = colors[i % colors.len()];
            if let Some(tip) = draw_panel(&mut buf, w, y0, name, pts, color, cursor) {
                hover = Some(tip);
            }
            y0 += PANEL_H + PAD;
        }

        if let Some(tip) = hover {
            // tooltip near cursor
            let (cx, cy) = cursor.unwrap_or((12.0, 12.0));
            let tx = (cx as usize + 12).min(w.saturating_sub(200));
            let ty = (cy as usize + 12).min(h.saturating_sub(20));
            let tw = (tip.len() * 6 + 10).min(w.saturating_sub(tx));
            fill_rect(
                &mut buf,
                w,
                tx,
                ty,
                tx + tw,
                ty + 16,
                rgb(40, 40, 48),
            );
            draw_text(&mut buf, w, tx + 4, ty + 4, &tip, rgb(255, 255, 230));
        }

        let _ = buf.present();
    }
}

fn rgb(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | b as u32
}

fn fmt_val(v: f64) -> String {
    let a = v.abs();
    if a >= 1000.0 || (a > 0.0 && a < 0.01) {
        format!("{v:.3e}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else if a >= 10.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

/// Returns a hover tooltip if the cursor is inside this panel's plot area.
fn draw_panel(
    buf: &mut [u32],
    w: usize,
    y0: usize,
    name: &str,
    pts: &[[f64; 2]],
    color: u32,
    cursor: Option<(f64, f64)>,
) -> Option<String> {
    let x0 = PAD;
    let x1 = w.saturating_sub(PAD);
    let y1 = y0 + PANEL_H;
    fill_rect(buf, w, x0, y0, x1, y1, rgb(28, 28, 34));
    draw_text(
        buf,
        w,
        x0 + 6,
        y0 + 4,
        &format!("{name} ({} pts)", pts.len()),
        color,
    );

    if pts.is_empty() || x1 <= x0 + LEFT + 4 {
        return None;
    }
    let plot_x0 = x0 + LEFT;
    let plot_x1 = x1 - 4;
    let plot_y0 = y0 + 22;
    let plot_y1 = y1 - BOTTOM;
    if plot_y1 <= plot_y0 + 2 {
        return None;
    }

    let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in pts {
        tmin = tmin.min(p[0]);
        tmax = tmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    if !(tmin.is_finite() && tmax.is_finite() && ymin.is_finite() && ymax.is_finite()) {
        return None;
    }
    if (tmax - tmin).abs() < 1e-12 {
        tmax = tmin + 1.0;
    }
    if (ymax - ymin).abs() < 1e-12 {
        ymax = ymin + 1.0;
        ymin -= 0.5;
    }
    let pad_y = (ymax - ymin) * 0.08;
    ymin -= pad_y;
    ymax += pad_y;

    let axis = rgb(70, 70, 80);
    let grid = rgb(45, 45, 55);
    hline(buf, w, plot_x0, plot_x1, plot_y1, axis);
    vline(buf, w, plot_x0, plot_y0, plot_y1, axis);

    // y-axis ticks: max / mid / min
    let y_ticks = [ymax, 0.5 * (ymin + ymax), ymin];
    for (i, &yv) in y_ticks.iter().enumerate() {
        let v = (yv - ymin) / (ymax - ymin);
        let py = (plot_y1 as f64 - v * (plot_y1 - plot_y0) as f64) as usize;
        if py >= plot_y0 && py <= plot_y1 {
            hline(buf, w, plot_x0, plot_x1, py, grid);
            let label = fmt_val(yv);
            let lx = plot_x0.saturating_sub(label.len() * 6 + 4);
            draw_text(buf, w, lx, py.saturating_sub(3), &label, rgb(160, 160, 170));
            // small tick mark
            hline(buf, w, plot_x0.saturating_sub(4), plot_x0, py, axis);
        }
        let _ = i;
    }

    let map = |t: f64, y: f64| -> (i32, i32) {
        let u = (t - tmin) / (tmax - tmin);
        let v = (y - ymin) / (ymax - ymin);
        let px = plot_x0 as f64 + u * (plot_x1 - plot_x0) as f64;
        let py = plot_y1 as f64 - v * (plot_y1 - plot_y0) as f64;
        (px as i32, py as i32)
    };

    for win in pts.windows(2) {
        let (x_a, y_a) = map(win[0][0], win[0][1]);
        let (x_b, y_b) = map(win[1][0], win[1][1]);
        line(buf, w, x_a, y_a, x_b, y_b, color);
    }
    if let Some(last) = pts.last() {
        let (x, y) = map(last[0], last[1]);
        dot(buf, w, x, y, color);
    }

    draw_text(
        buf,
        w,
        plot_x0,
        plot_y1 + 2,
        &format!("{}s - {}s", fmt_val(tmin), fmt_val(tmax)),
        rgb(140, 140, 150),
    );

    // hover: if cursor in this panel's plot rect, nearest sample by time
    let (cx, cy) = cursor?;
    if cx < plot_x0 as f64
        || cx > plot_x1 as f64
        || cy < plot_y0 as f64
        || cy > plot_y1 as f64
    {
        return None;
    }
    let u = ((cx - plot_x0 as f64) / (plot_x1 - plot_x0) as f64).clamp(0.0, 1.0);
    let t_at = tmin + u * (tmax - tmin);
    // nearest point by t
    let mut best = &pts[0];
    let mut best_d = (pts[0][0] - t_at).abs();
    for p in pts.iter().skip(1) {
        let d = (p[0] - t_at).abs();
        if d < best_d {
            best_d = d;
            best = p;
        }
    }
    let (hx, hy) = map(best[0], best[1]);
    // crosshair
    vline(buf, w, hx as usize, plot_y0, plot_y1, rgb(90, 90, 100));
    hline(buf, w, plot_x0, plot_x1, hy as usize, rgb(90, 90, 100));
    dot(buf, w, hx, hy, rgb(255, 255, 255));

    Some(format!(
        "{name}: t={}  y={}",
        fmt_val(best[0]),
        fmt_val(best[1])
    ))
}

fn fill_rect(buf: &mut [u32], w: usize, x0: usize, y0: usize, x1: usize, y1: usize, c: u32) {
    let h = buf.len() / w.max(1);
    let x1 = x1.min(w);
    let y1 = y1.min(h);
    for y in y0..y1 {
        for x in x0..x1 {
            buf[y * w + x] = c;
        }
    }
}

fn hline(buf: &mut [u32], w: usize, x0: usize, x1: usize, y: usize, c: u32) {
    let h = buf.len() / w.max(1);
    if y >= h {
        return;
    }
    for x in x0.min(w)..x1.min(w) {
        buf[y * w + x] = c;
    }
}

fn vline(buf: &mut [u32], w: usize, x: usize, y0: usize, y1: usize, c: u32) {
    let h = buf.len() / w.max(1);
    if x >= w {
        return;
    }
    for y in y0.min(h)..y1.min(h) {
        buf[y * w + x] = c;
    }
}

fn dot(buf: &mut [u32], w: usize, cx: i32, cy: i32, c: u32) {
    for dy in -2..=2 {
        for dx in -2..=2 {
            put(buf, w, cx + dx, cy + dy, c);
        }
    }
}

fn line(buf: &mut [u32], w: usize, mut x0: i32, mut y0: i32, x1: i32, y1: i32, c: u32) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(buf, w, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn put(buf: &mut [u32], w: usize, x: i32, y: i32, c: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let h = (buf.len() / w.max(1)) as i32;
    if x >= w as i32 || y >= h {
        return;
    }
    buf[y as usize * w + x as usize] = c;
}

/// Tiny 5x7 bitmap font (digits, letters, punctuation used in labels).
fn draw_text(buf: &mut [u32], w: usize, x: usize, y: usize, text: &str, c: u32) {
    let mut cx = x as i32;
    let cy = y as i32;
    for ch in text.chars() {
        if let Some(glyph) = glyph(ch) {
            for row in 0..7 {
                let bits = glyph[row];
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        put(buf, w, cx + col, cy + row as i32, c);
                    }
                }
            }
        }
        cx += 6;
    }
}

fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        ' ' => [0; 7],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        '+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        '.' => [0, 0, 0, 0, 0, 0b00100, 0],
        ':' => [0, 0b00100, 0, 0, 0b00100, 0, 0],
        '=' => [0, 0b11111, 0, 0b11111, 0, 0, 0],
        '(' => [0b00100, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00100],
        ')' => [0b00100, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b00100],
        ',' => [0, 0, 0, 0, 0b00100, 0b00100, 0b01000],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
        '…' | '—' => [0, 0, 0, 0b10101, 0, 0, 0],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        'a' | 'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'b' | 'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'c' | 'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'd' | 'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'e' | 'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'f' | 'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'g' | 'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'h' | 'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'i' | 'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'j' | 'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'k' | 'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'l' | 'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'm' | 'M' => [0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001],
        'n' | 'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'o' | 'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' | 'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'q' | 'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'r' | 'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        's' | 'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        't' | 'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'u' | 'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'v' | 'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'w' | 'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
        'x' | 'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'y' | 'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'z' | 'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '_' => [0, 0, 0, 0, 0, 0, 0b11111],
        '@' => [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110],
        _ => return None,
    })
}

pub fn mpm_particle_metrics(ps: &[MpmFluidParticle]) -> Vec<(String, f64)> {
    let mut max_speed = 0.0f64;
    let mut min_j = f64::MAX;
    let mut max_j = f64::MIN;
    let mut nan = 0.0;
    for p in ps {
        let sp = (p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]).sqrt() as f64;
        if !sp.is_finite() {
            nan += 1.0;
            continue;
        }
        max_speed = max_speed.max(sp);
        let f = p.deformation_gradient;
        let j = (f[0][0] * f[1][1] - f[0][1] * f[1][0]) as f64;
        if j.is_finite() {
            min_j = min_j.min(j);
            max_j = max_j.max(j);
        }
    }
    if min_j == f64::MAX {
        min_j = 0.0;
        max_j = 0.0;
    }
    vec![
        ("max_speed".into(), max_speed),
        ("min_j".into(), min_j),
        ("max_j".into(), max_j),
        ("nan_count".into(), nan),
    ]
}

pub fn mpm_one_particle_metrics(p: &MpmFluidParticle) -> Vec<(String, f64)> {
    let speed = (p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]).sqrt() as f64;
    let f = p.deformation_gradient;
    let j = (f[0][0] * f[1][1] - f[0][1] * f[1][0]) as f64;
    vec![
        ("speed".into(), speed),
        ("vx".into(), p.velocity[0] as f64),
        ("vy".into(), p.velocity[1] as f64),
        ("j".into(), j),
        ("x".into(), p.position[0] as f64),
        ("y".into(), p.position[1] as f64),
    ]
}

pub fn mpm_elastic_particle_metrics(ps: &[MpmElasticParticle]) -> Vec<(String, f64)> {
    let mut max_speed = 0.0f64;
    let mut min_j = f64::MAX;
    let mut max_j = f64::MIN;
    let mut min_jp = f64::MAX;
    let mut max_jp = f64::MIN;
    let mut sum_jp = 0.0f64;
    let mut n_jp = 0.0f64;
    let mut nan = 0.0;
    for p in ps {
        let sp = (p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]).sqrt() as f64;
        if !sp.is_finite() {
            nan += 1.0;
            continue;
        }
        max_speed = max_speed.max(sp);
        let f = p.elastic_deformation_gradient;
        let j = (f[0][0] * f[1][1] - f[0][1] * f[1][0]) as f64;
        if j.is_finite() {
            min_j = min_j.min(j);
            max_j = max_j.max(j);
        }
        let jp = p.plastic_deformation_determinant as f64;
        if jp.is_finite() {
            min_jp = min_jp.min(jp);
            max_jp = max_jp.max(jp);
            sum_jp += jp;
            n_jp += 1.0;
        }
    }
    if min_j == f64::MAX {
        min_j = 0.0;
        max_j = 0.0;
    }
    if min_jp == f64::MAX {
        min_jp = 0.0;
        max_jp = 0.0;
    }
    let mean_jp = if n_jp > 0.0 { sum_jp / n_jp } else { 0.0 };
    vec![
        ("max_speed".into(), max_speed),
        ("min_j".into(), min_j),
        ("max_j".into(), max_j),
        ("min_Jp".into(), min_jp),
        ("max_Jp".into(), max_jp),
        ("mean_Jp".into(), mean_jp),
        ("nan_count".into(), nan),
    ]
}

pub fn mpm_one_elastic_particle_metrics(p: &MpmElasticParticle) -> Vec<(String, f64)> {
    let speed = (p.velocity[0] * p.velocity[0] + p.velocity[1] * p.velocity[1]).sqrt() as f64;
    let f = p.elastic_deformation_gradient;
    let j = (f[0][0] * f[1][1] - f[0][1] * f[1][0]) as f64;
    vec![
        ("speed".into(), speed),
        ("vx".into(), p.velocity[0] as f64),
        ("vy".into(), p.velocity[1] as f64),
        ("j".into(), j),
        ("Jp".into(), p.plastic_deformation_determinant as f64),
        ("x".into(), p.position[0] as f64),
        ("y".into(), p.position[1] as f64),
    ]
}
