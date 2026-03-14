/* ═══════════════════════════════════════════════════════
   Forest Fire Recognition — Main JavaScript
   ═══════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    // AOS animations
    if (typeof AOS !== 'undefined') {
        AOS.init({ duration: 700, easing: 'ease-out-cubic', once: true });
    }

    // Highlight active nav link
    const path = window.location.pathname;
    document.querySelectorAll('.eco-nav .nav-link').forEach(link => {
        if (link.getAttribute('href') === path) {
            link.classList.add('active');
        }
    });

    // Theme toggle
    ThemeToggle.init();

    // Scroll-to-top button
    ScrollTop.init();

    // Navbar notification badge
    NavBadge.init();
});

/**
 * Show a Bootstrap-style toast notification.
 * @param {string} message
 * @param {'success'|'danger'|'warning'|'info'} type
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toasts');
    if (!container) return;

    const id = 'toast_' + Date.now();
    const bgMap = {
        success: 'bg-success',
        danger: 'bg-danger',
        warning: 'bg-warning text-dark',
        info: 'bg-info text-dark',
    };
    const bg = bgMap[type] || bgMap.info;

    const html = `
        <div id="${id}" class="toast align-items-center ${bg} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>`;

    container.insertAdjacentHTML('beforeend', html);
    const el = document.getElementById(id);
    const toast = new bootstrap.Toast(el, { delay: 4000 });
    toast.show();
    el.addEventListener('hidden.bs.toast', () => el.remove());
}

/* ═══════════════════════════════════════════════════════
   Geolocation helper
   ═══════════════════════════════════════════════════════ */
const EcoGeo = {
    lat: null,
    lng: null,
    address: 'Unknown',
    ready: false,

    init() {
        if (!navigator.geolocation) return;
        navigator.geolocation.getCurrentPosition(
            pos => {
                EcoGeo.lat = pos.coords.latitude;
                EcoGeo.lng = pos.coords.longitude;
                EcoGeo.ready = true;
                EcoGeo._reverseGeocode();
            },
            () => { /* permission denied — silently ignore */ },
            { enableHighAccuracy: true, timeout: 10000 }
        );
    },

    _reverseGeocode() {
        if (!EcoGeo.lat) return;
        fetch(`https://nominatim.openstreetmap.org/reverse?lat=${EcoGeo.lat}&lon=${EcoGeo.lng}&format=json`)
            .then(r => r.json())
            .then(d => { if (d.display_name) EcoGeo.address = d.display_name; })
            .catch(() => {});
    },

    appendToForm(fd) {
        if (EcoGeo.ready) {
            fd.append('latitude', EcoGeo.lat);
            fd.append('longitude', EcoGeo.lng);
            fd.append('address', EcoGeo.address);
        }
    },

    toParams() {
        if (!EcoGeo.ready) return {};
        return { latitude: EcoGeo.lat, longitude: EcoGeo.lng, address: EcoGeo.address };
    }
};

// Auto-init geolocation on page load
document.addEventListener('DOMContentLoaded', () => EcoGeo.init());

/* ═══════════════════════════════════════════════════════
   Fire Alarm System (Web Audio API)
   ═══════════════════════════════════════════════════════ */
const EcoAlarm = {
    _ctx: null,
    _playing: false,
    _oscillators: [],
    _interval: null,

    _ensureCtx() {
        if (!this._ctx) this._ctx = new (window.AudioContext || window.webkitAudioContext)();
        if (this._ctx.state === 'suspended') this._ctx.resume();
    },

    play() {
        if (this._playing) return;
        this._ensureCtx();
        this._playing = true;

        let high = true;
        const siren = () => {
            this.stopOscillators();
            const osc = this._ctx.createOscillator();
            const gain = this._ctx.createGain();
            osc.type = 'square';
            osc.frequency.value = high ? 880 : 660;
            gain.gain.value = 0.15;
            osc.connect(gain);
            gain.connect(this._ctx.destination);
            osc.start();
            this._oscillators.push({ osc, gain });
            high = !high;
        };

        siren();
        this._interval = setInterval(siren, 500);

        // Auto-stop after 8 seconds
        setTimeout(() => this.stop(), 8000);
    },

    stopOscillators() {
        this._oscillators.forEach(({ osc }) => { try { osc.stop(); } catch (e) {} });
        this._oscillators = [];
    },

    stop() {
        this._playing = false;
        if (this._interval) { clearInterval(this._interval); this._interval = null; }
        this.stopOscillators();
    },

    isPlaying() { return this._playing; }
};

/* ═══════════════════════════════════════════════════════
   Leaflet Map Helper
   ═══════════════════════════════════════════════════════ */
function createLocationMap(containerId, lat, lng, label) {
    if (!lat || !lng || typeof L === 'undefined') return null;
    const map = L.map(containerId).setView([lat, lng], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    const iconColor = (label === 'FIRE DETECTED') ? 'red' : 'green';
    const fireIcon = L.divIcon({
        className: 'eco-map-marker',
        html: `<i class="fa-solid fa-location-dot fa-2x" style="color:${iconColor}"></i>`,
        iconSize: [24, 36],
        iconAnchor: [12, 36]
    });
    L.marker([lat, lng], { icon: fireIcon }).addTo(map)
        .bindPopup(`<strong>${label || 'Detection'}</strong><br>Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}`);
    return map;
}

/* ═══════════════════════════════════════════════════════
   Animated Count-Up
   ═══════════════════════════════════════════════════════ */
function animateCountUp(el, target, suffix = '') {
    const duration = 1800;
    const start = performance.now();
    const isFloat = String(target).includes('.');
    const update = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
        const val = ease * target;
        el.textContent = (isFloat ? val.toFixed(1) : Math.floor(val)) + suffix;
        if (t < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
}

/* ═══════════════════════════════════════════════════════
   Dark / Light Theme Toggle
   ═══════════════════════════════════════════════════════ */
const ThemeToggle = {
    _btn: null,
    init() {
        this._btn = document.getElementById('themeToggleBtn');
        if (!this._btn) return;
        const saved = localStorage.getItem('eco-theme') || 'dark';
        this._apply(saved);
        this._btn.addEventListener('click', () => {
            const current = document.documentElement.getAttribute('data-bs-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            this._apply(next);
            localStorage.setItem('eco-theme', next);
        });
    },
    _apply(theme) {
        document.documentElement.setAttribute('data-bs-theme', theme);
        document.body.classList.toggle('light-mode', theme === 'light');
        if (this._btn) {
            this._btn.innerHTML = theme === 'dark'
                ? '<i class="fa-solid fa-sun"></i>'
                : '<i class="fa-solid fa-moon"></i>';
            this._btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
        }
    }
};

/* ═══════════════════════════════════════════════════════
   Scroll-to-Top Button
   ═══════════════════════════════════════════════════════ */
const ScrollTop = {
    _btn: null,
    init() {
        this._btn = document.getElementById('scrollTopBtn');
        if (!this._btn) return;
        window.addEventListener('scroll', () => {
            this._btn.classList.toggle('visible', window.scrollY > 400);
        });
        this._btn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
};

/* ═══════════════════════════════════════════════════════
   Navbar Notification Badge
   ═══════════════════════════════════════════════════════ */
const NavBadge = {
    init() {
        fetch('/api/stats').then(r => r.json()).then(d => {
            const badge = document.getElementById('navFireBadge');
            if (badge && d.fires > 0) {
                badge.textContent = d.fires > 99 ? '99+' : d.fires;
                badge.classList.remove('d-none');
            }
        }).catch(() => {});
    }
};

/* ═══════════════════════════════════════════════════════
   Image Zoom Modal
   ═══════════════════════════════════════════════════════ */
function initImageZoom() {
    document.querySelectorAll('.eco-zoomable').forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', () => {
            const modal = document.getElementById('imageZoomModal');
            const zoomImg = document.getElementById('zoomModalImg');
            if (modal && zoomImg) {
                zoomImg.src = img.src;
                new bootstrap.Modal(modal).show();
            }
        });
    });
}

/* ═══════════════════════════════════════════════════════
   Stepped Loading Indicator
   ═══════════════════════════════════════════════════════ */
const StepLoader = {
    _steps: ['Uploading image…', 'Preprocessing…', 'Running CNN model…', 'Analyzing colors…', 'Generating report…'],
    _el: null,
    _stepIdx: 0,
    _interval: null,

    start(el) {
        this._el = el;
        this._stepIdx = 0;
        if (!el) return;
        el.innerHTML = this._renderStep(0);
        this._interval = setInterval(() => {
            if (this._stepIdx < this._steps.length - 1) {
                this._stepIdx++;
                el.innerHTML = this._renderStep(this._stepIdx);
            }
        }, 800);
    },

    stop() {
        if (this._interval) { clearInterval(this._interval); this._interval = null; }
    },

    _renderStep(idx) {
        return `<div class="eco-step-loader">
            <div class="eco-spinner mb-2"></div>
            <div class="step-loader-steps">
                ${this._steps.map((s, i) => `<div class="step-item ${i < idx ? 'done' : ''} ${i === idx ? 'active' : ''}">
                    <i class="fa-solid ${i < idx ? 'fa-circle-check text-success' : i === idx ? 'fa-spinner fa-spin text-amber' : 'fa-circle text-muted'} me-1"></i>
                    <small>${s}</small>
                </div>`).join('')}
            </div>
        </div>`;
    }
};

/* ═══════════════════════════════════════════════════════
   CSV Export Utility
   ═══════════════════════════════════════════════════════ */
function exportHistoryCSV(records) {
    if (!records || !records.length) { showToast('No records to export', 'warning'); return; }
    const headers = ['#', 'Date/Time', 'Filename', 'Source', 'Result', 'Detailed Class', 'Confidence %', 'Risk Level', 'Fire %', 'Smoke %', 'No Fire %', 'Latitude', 'Longitude', 'Address'];
    const rows = records.map((r, i) => {
        const loc = r.location || {};
        return [
            i + 1, r.timestamp, r.filename, r.source || 'upload', r.label, r.detailed_label,
            r.confidence, r.risk_level,
            r.probabilities?.fire || '', r.probabilities?.smoke || '', r.probabilities?.no_fire || '',
            loc.lat || '', loc.lng || '', (loc.address || '').replace(/,/g, ';')
        ];
    });
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'fire_detection_history.csv'; a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported successfully', 'success');
}
