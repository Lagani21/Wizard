'use strict';

class WizApp {
    constructor() {
        this.results     = null;
        this.duration    = 0;
        this.tlScale     = 1;
        this.fileName    = '';
        this.taskId      = null;
        this._searchMode = 'topic';

        this._bindUpload();
        this._bindPlayer();
        this._bindTabs();
        this._bindTimeline();
        this._bindSearch();
    }

    // ── Upload ────────────────────────────────────────────────────
    _bindUpload() {
        const area   = document.getElementById('uploadArea');
        const input  = document.getElementById('videoInput');
        const browse = document.getElementById('browseBtn');
        const newBtn = document.getElementById('newVideoBtn');

        browse.addEventListener('click', () => input.click());
        area.addEventListener('click', () => input.click());
        area.addEventListener('dragover',  e => { e.preventDefault(); area.classList.add('drag-over'); });
        area.addEventListener('dragleave', () => area.classList.remove('drag-over'));
        area.addEventListener('drop', e => {
            e.preventDefault(); area.classList.remove('drag-over');
            const f = e.dataTransfer.files[0];
            if (f) this._startProcessing(f);
        });
        input.addEventListener('change', e => {
            const f = e.target.files[0];
            if (f) this._startProcessing(f);
        });
        newBtn.addEventListener('click', () => { this._clearSession(); this._showView('upload'); });
    }

    _bindPlayer() {
        const v = document.getElementById('videoPlayer');
        v.addEventListener('loadedmetadata', () => {
            this.duration = v.duration;
            this._buildTimeline();
        });
        v.addEventListener('timeupdate', () => this._updatePlayhead());
    }

    _bindTabs() {
        document.querySelectorAll('.dtab').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.dtab').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.dtab-content').forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById(btn.dataset.tab + 'Tab').classList.add('active');
            });
        });
    }

    _bindTimeline() {
        document.getElementById('zoomIn').addEventListener('click',    () => this._zoom(1.5));
        document.getElementById('zoomOut').addEventListener('click',   () => this._zoom(0.67));
        document.getElementById('resetZoom').addEventListener('click', () => { this.tlScale = 1; this._applyZoom(); });

        document.getElementById('tlTracks').addEventListener('click', e => {
            const content = e.target.closest('.tl-content');
            if (!content || !this.duration) return;
            const rect  = content.getBoundingClientRect();
            const pct   = (e.clientX - rect.left) / rect.width;
            document.getElementById('videoPlayer').currentTime = pct * this.duration;
        });
    }

    // ── Show/hide views ────────────────────────────────────────────
    _showView(name) {
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        document.getElementById(name === 'upload' ? 'viewUpload' : 'viewResults').classList.add('active');
    }

    // ── Session persistence ────────────────────────────────────────
    _saveSession(taskId, fileName) {
        try {
            sessionStorage.setItem('wiz_task_id',   taskId);
            sessionStorage.setItem('wiz_file_name', fileName);
        } catch (_) {}
    }

    _clearSession() {
        try {
            sessionStorage.removeItem('wiz_task_id');
            sessionStorage.removeItem('wiz_file_name');
        } catch (_) {}
    }

    async _tryRestoreSession() {
        const taskId   = sessionStorage.getItem('wiz_task_id');
        const fileName = sessionStorage.getItem('wiz_file_name');
        if (!taskId) return;

        try {
            const res = await fetch(`/api/progress/${taskId}`);
            if (!res.ok) { this._clearSession(); return; }
            const status = await res.json();
            if (status.status !== 'completed') { this._clearSession(); return; }

            // Restore metadata
            this.taskId   = taskId;
            this.fileName = fileName || 'video';
            document.getElementById('videoFileName').textContent = this.fileName;

            // Point the player at the server-side copy
            const player = document.getElementById('videoPlayer');
            player.src = `/api/uploads/${taskId}/video`;

            this._showView('results');
            this._showProcessing(true);
            this._setProgress(90, 'Restoring session…');

            const res2    = await fetch(`/api/results/${taskId}`);
            if (!res2.ok) { this._clearSession(); this._showView('upload'); return; }
            const results = await res2.json();
            this._displayResults(results);
        } catch (_) {
            this._clearSession();
        }
    }

    // ── Processing ─────────────────────────────────────────────────
    async _startProcessing(file) {
        this.fileName = file.name;
        document.getElementById('videoFileName').textContent = file.name;

        // Load video locally for immediate preview
        const player = document.getElementById('videoPlayer');
        player.src = URL.createObjectURL(file);

        this._showView('results');
        this._showProcessing(true);
        this._setProgress(5, 'Uploading…');

        try {
            const form = new FormData();
            form.append('video', file);

            const res = await fetch('/api/analyze', { method: 'POST', body: form });
            if (!res.ok) throw new Error('Upload failed');

            const { task_id } = await res.json();
            this.taskId = task_id;
            this._setProgress(15, 'Pipeline running…');
            await this._pollUntilDone(task_id);
        } catch (err) {
            this._setProgress(0, `Error: ${err.message}`);
        }
    }

    _showProcessing(show) {
        const ov = document.getElementById('processingOverlay');
        const rb = document.getElementById('resultsBody');
        if (show) {
            ov.style.display = 'flex';
            rb.style.display = 'none';
        } else {
            ov.style.display = 'none';
            rb.style.display = 'flex';
        }
        // processingOverlay needs to be positioned relative to viewResults
        document.getElementById('viewResults').style.position = 'relative';
    }

    _setProgress(pct, msg) {
        document.getElementById('processingFill').style.width = pct + '%';
        document.getElementById('processingText').textContent = msg;
    }

    async _pollUntilDone(taskId) {
        return new Promise((resolve, reject) => {
            const iv = setInterval(async () => {
                try {
                    const r    = await fetch(`/api/progress/${taskId}`);
                    const data = await r.json();
                    this._setProgress(data.progress, data.message || 'Processing…');

                    if (data.status === 'completed') {
                        clearInterval(iv);
                        this._saveSession(taskId, this.fileName);
                        const res2    = await fetch(`/api/results/${taskId}`);
                        const results = await res2.json();
                        this._displayResults(results);
                        resolve();
                    } else if (data.status === 'failed') {
                        clearInterval(iv);
                        reject(new Error(data.error || 'Processing failed'));
                    }
                } catch (e) {
                    clearInterval(iv);
                    reject(e);
                }
            }, 1200);
        });
    }

    // ── Display results ────────────────────────────────────────────
    _displayResults(results) {
        this.results = results;
        this._showProcessing(false);
        this._updateChips(results);
        this._populateTranscript(results);
        this._populateScenes(results);
        this._populateCaptions(results);
        this._populateEvents(results);
        // Timeline built after video metadata loads (onloadedmetadata)
        // but if already loaded, build now
        if (this.duration) this._buildTimeline();
    }

    _updateChips(r) {
        document.getElementById('blinkCount').textContent   = r.blink_events?.length ?? 0;
        document.getElementById('breathCount').textContent  = r.breath_events?.length ?? 0;
        document.getElementById('captionCount').textContent = r.video_captions?.length ?? 0;
        document.getElementById('sceneCount').textContent   = r.scene_summaries?.length ?? 0;

        const speakers = new Set();
        (r.aligned_segments || []).forEach(s => speakers.add(s.speaker_id));
        document.getElementById('speakerCount').textContent = speakers.size;

        // Populate speaker dropdown in search tab
        const sel = document.getElementById('searchSpeaker');
        sel.innerHTML = '<option value="">Any speaker</option>';
        speakers.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            opt.textContent = id;
            sel.appendChild(opt);
        });
    }

    // ── Timeline ───────────────────────────────────────────────────
    _buildTimeline() {
        if (!this.results || !this.duration) return;

        this._drawWaveform(this.results.audio_waveform);
        this._buildSpeechTrack(this.results.aligned_segments);
        this._buildToneTrack(this.results.tone_events);
        this._buildCaptionsTrack(this.results.video_captions);
        this._buildEventsTrack(this.results.blink_events, this.results.breath_events);
        this._buildRuler();
        this._applyZoom();
    }

    _pct(t) { return (t / this.duration) * 100; }

    _clearTrack(id) { document.getElementById(id).innerHTML = ''; }

    _drawWaveform(data) {
        const canvas = document.getElementById('waveformCanvas');
        if (!data || !canvas) return;
        canvas.width  = canvas.offsetWidth || canvas.parentElement.clientWidth;
        canvas.height = 44;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth   = 1;
        ctx.beginPath();
        data.forEach((v, i) => {
            const x = (i / data.length) * canvas.width;
            const y = canvas.height / 2 + v * (canvas.height / 2) * 0.85;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    _buildSpeechTrack(segs) {
        this._clearTrack('speechTrack');
        if (!segs) return;
        const track = document.getElementById('speechTrack');
        segs.forEach(seg => {
            const el = document.createElement('div');
            el.className = 'speech-seg';
            el.style.left  = this._pct(seg.start_time) + '%';
            el.style.width = this._pct(seg.end_time - seg.start_time) + '%';
            el.innerHTML = `<span class="seg-speaker">${seg.speaker_id}</span>
                            <span class="seg-text">${this._trunc(seg.text, 60)}</span>`;
            el.addEventListener('click', e => { e.stopPropagation(); this._seek(seg.start_time); });
            track.appendChild(el);
        });
    }

    _buildToneTrack(events) {
        this._clearTrack('toneTrack');
        if (!events) return;
        const track = document.getElementById('toneTrack');
        events.forEach(ev => {
            const el = document.createElement('div');
            el.className = `tone-seg tone-${ev.tone_label}`;
            el.style.left  = this._pct(ev.start_time) + '%';
            el.style.width = this._pct(ev.end_time - ev.start_time) + '%';
            el.textContent = ev.tone_label;
            el.title = `${ev.tone_label} (${Math.round(ev.confidence * 100)}%)`;
            el.addEventListener('click', e => { e.stopPropagation(); this._seek(ev.start_time); });
            track.appendChild(el);
        });
    }

    _buildCaptionsTrack(caps) {
        this._clearTrack('captionsTrack');
        if (!caps) return;
        const track = document.getElementById('captionsTrack');
        caps.forEach(cap => {
            const el = document.createElement('div');
            el.className = 'caption-seg';
            el.style.left  = this._pct(cap.start_time) + '%';
            el.style.width = this._pct(cap.end_time - cap.start_time) + '%';
            el.textContent = cap.caption;
            el.title = cap.caption;
            el.addEventListener('click', e => { e.stopPropagation(); this._seek(cap.start_time); });
            track.appendChild(el);
        });
    }

    _buildEventsTrack(blinks, breaths) {
        this._clearTrack('eventsTrack');
        const track = document.getElementById('eventsTrack');

        (blinks || []).forEach(ev => {
            const t  = ev.start_frame / (this.results.video_metadata?.fps || 30);
            const el = document.createElement('div');
            el.className = 'ev-marker ev-blink';
            el.style.left = this._pct(t) + '%';
            el.title = `Blink @ ${this._fmt(t)}`;
            el.addEventListener('click', e => { e.stopPropagation(); this._seek(t); });
            track.appendChild(el);
        });

        (breaths || []).forEach(ev => {
            const el = document.createElement('div');
            el.className = 'ev-marker ev-breath';
            el.style.left = this._pct(ev.start_time) + '%';
            el.title = `Breath @ ${this._fmt(ev.start_time)}`;
            el.addEventListener('click', e => { e.stopPropagation(); this._seek(ev.start_time); });
            track.appendChild(el);
        });
    }

    _buildRuler() {
        const ruler    = document.getElementById('timeRuler');
        ruler.innerHTML = '';
        const dur      = this.duration;
        const interval = dur <= 60 ? 5 : dur <= 300 ? 15 : dur <= 600 ? 30 : 60;

        for (let t = 0; t <= dur; t += interval) {
            const m = document.createElement('div');
            m.className = 'time-mark' + (t % (interval * 5) === 0 ? ' major' : '');
            m.style.left = this._pct(t) + '%';
            m.textContent = this._fmt(t);
            ruler.appendChild(m);
        }
    }

    _updatePlayhead() {
        const ph   = document.getElementById('timelinePlayhead');
        const v    = document.getElementById('videoPlayer');
        if (!this.duration) return;
        const frac = v.currentTime / this.duration;
        // Playhead lives inside #tlTracks (position:relative).
        // Content columns start at 72px (label width).
        // So: left = 72px + frac × (track_width − 72px)
        ph.style.left = `calc(72px + ${frac} * (100% - 72px))`;
    }

    _seek(t) { document.getElementById('videoPlayer').currentTime = t; }

    _zoom(f) { this.tlScale = Math.max(0.5, Math.min(10, this.tlScale * f)); this._applyZoom(); }

    _applyZoom() {
        document.querySelectorAll('.tl-content').forEach(el => {
            el.style.transform       = `scaleX(${this.tlScale})`;
            el.style.transformOrigin = 'left center';
        });
    }

    // ── Detail panels ──────────────────────────────────────────────
    _populateTranscript(r) {
        const pane = document.getElementById('transcriptContent');
        pane.innerHTML = '';
        const segs = r.aligned_segments;
        if (!segs || !segs.length) { pane.innerHTML = '<div class="empty-state">No transcript data.</div>'; return; }

        segs.forEach(seg => {
            const el = document.createElement('div');
            el.className = 't-item';
            el.innerHTML = `
                <div class="t-header">
                    <span class="t-speaker">${seg.speaker_id}</span>
                    <span class="t-time">${this._fmt(seg.start_time)} – ${this._fmt(seg.end_time)}</span>
                </div>
                <div class="t-text">${seg.text || '—'}</div>`;
            el.addEventListener('click', () => this._seek(seg.start_time));
            pane.appendChild(el);
        });
    }

    _populateScenes(r) {
        const pane = document.getElementById('scenesContent');
        pane.innerHTML = '';
        const scenes = r.scene_summaries;
        if (!scenes || !scenes.length) { pane.innerHTML = '<div class="empty-state">No scene summaries.</div>'; return; }

        scenes.forEach(sc => {
            const el = document.createElement('div');
            el.className = 's-item';
            el.innerHTML = `
                <div class="s-header">
                    <span class="s-title">${sc.scene_id}</span>
                    <span class="s-time">${this._fmt(sc.start_time)} – ${this._fmt(sc.end_time)}</span>
                </div>
                <span class="s-tone">${sc.tone_label || 'neutral'}</span>
                <div class="s-summary">${sc.summary_text || '—'}</div>`;
            el.addEventListener('click', () => this._seek(sc.start_time));
            pane.appendChild(el);
        });
    }

    _populateCaptions(r) {
        const pane = document.getElementById('captionsContent');
        pane.innerHTML = '';
        const caps = r.video_captions;
        if (!caps || !caps.length) { pane.innerHTML = '<div class="empty-state">No video captions.</div>'; return; }

        caps.forEach(cap => {
            const el = document.createElement('div');
            el.className = 'c-item';
            el.innerHTML = `
                <div class="c-header">
                    <span class="c-label">Window ${cap.window_id}</span>
                    <span class="c-time">${this._fmt(cap.start_time)} – ${this._fmt(cap.end_time)}</span>
                </div>
                <div class="c-text">${cap.caption}</div>`;
            el.addEventListener('click', () => this._seek(cap.start_time));
            pane.appendChild(el);
        });
    }

    _populateEvents(r) {
        const pane = document.getElementById('eventsContent');
        pane.innerHTML = '';

        const fps = r.video_metadata?.fps || 30;

        if (r.blink_events?.length) {
            const grp = document.createElement('div');
            grp.className = 'ev-group';
            grp.innerHTML = '<h4>Blink Events</h4>';
            r.blink_events.forEach(ev => {
                const t = ev.start_frame / fps;
                grp.appendChild(this._evItem('blink', t, ev.confidence));
            });
            pane.appendChild(grp);
        }

        if (r.breath_events?.length) {
            const grp = document.createElement('div');
            grp.className = 'ev-group';
            grp.innerHTML = '<h4>Breath Events</h4>';
            r.breath_events.forEach(ev => {
                grp.appendChild(this._evItem('breath', ev.start_time, ev.confidence));
            });
            pane.appendChild(grp);
        }

        if (!r.blink_events?.length && !r.breath_events?.length) {
            pane.innerHTML = '<div class="empty-state">No events detected.</div>';
        }
    }

    _evItem(type, t, conf) {
        const el = document.createElement('div');
        el.className = 'ev-item';
        el.innerHTML = `
            <div class="ev-dot ev-dot-${type}"></div>
            <span style="flex:1;margin-left:10px;color:#cbd5e1;font-size:0.85rem;text-transform:capitalize">${type}</span>
            <span class="ev-item-time">${this._fmt(t)}</span>
            <span class="ev-item-conf">${Math.round(conf * 100)}%</span>`;
        el.addEventListener('click', () => this._seek(t));
        return el;
    }

    // ── Search ─────────────────────────────────────────────────────
    _bindSearch() {
        // Mode pills
        document.querySelectorAll('.mode-pill').forEach(pill => {
            pill.addEventListener('click', () => {
                document.querySelectorAll('.mode-pill').forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
                this._searchMode = pill.dataset.mode;

                const queryInput    = document.getElementById('searchQuery');
                const speakerRow    = document.getElementById('searchSpeakerRow');
                const isSpeakerMode = this._searchMode === 'speaker_topic';
                const isSafeCuts    = this._searchMode === 'safe_cuts';

                speakerRow.style.display = isSpeakerMode ? 'flex' : 'none';
                queryInput.style.display = isSafeCuts    ? 'none' : '';

                const placeholders = {
                    topic:         'e.g. machine learning',
                    emotion:       'e.g. confident, excited, neutral',
                    speaker_topic: 'e.g. product launch',
                };
                queryInput.placeholder = placeholders[this._searchMode] || '';
            });
        });

        // Submit on button click or Enter
        document.getElementById('searchBtn').addEventListener('click', () => this._runSearch());
        document.getElementById('searchQuery').addEventListener('keydown', e => {
            if (e.key === 'Enter') this._runSearch();
        });
    }

    async _runSearch() {
        if (!this.taskId) {
            this._showSearchResults(null, 'No video processed yet.');
            return;
        }

        const query    = document.getElementById('searchQuery').value.trim();
        const speaker  = document.getElementById('searchSpeaker').value;
        const noBlink  = document.getElementById('searchNoBlink').checked;
        const mode     = this._searchMode;

        const params = new URLSearchParams();
        if (mode === 'safe_cuts') {
            params.set('safe_cuts', '1');
        } else if (mode === 'emotion') {
            if (!query) return;
            params.set('emotion', query);
        } else if (mode === 'speaker_topic') {
            if (speaker) params.set('speaker', speaker);
            if (query)   params.set('topic', query);
            if (noBlink) params.set('no_blink', '1');
            if (!speaker && !query) return;
        } else {
            if (!query) return;
            params.set('topic', query);
        }

        const btn = document.getElementById('searchBtn');
        btn.disabled = true;
        btn.textContent = '…';
        this._showSearchResults(null, 'Searching…');

        try {
            const res  = await fetch(`/api/results/${this.taskId}/search?${params}`);
            const data = await res.json();
            if (!res.ok) {
                this._showSearchResults([], data.error || 'Search failed.');
            } else {
                this._showSearchResults(data.results, null, data.count);
            }
        } catch (e) {
            this._showSearchResults([], 'Request failed.');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Search';
        }
    }

    _showSearchResults(results, statusMsg, count) {
        const pane = document.getElementById('searchResults');
        pane.innerHTML = '';

        const status = document.createElement('div');
        status.className = 'search-status';

        if (statusMsg) {
            status.textContent = statusMsg;
            pane.appendChild(status);
            return;
        }

        if (!results || !results.length) {
            status.textContent = 'No matches found.';
            pane.appendChild(status);
            return;
        }

        status.textContent = `${count} match${count === 1 ? '' : 'es'}`;
        pane.appendChild(status);

        results.forEach(r => {
            const el = document.createElement('div');
            el.className = 'sr-item';
            el.innerHTML = `
                <div class="sr-header">
                    <span class="sr-timecode">${r.timecode}</span>
                    ${r.speaker ? `<span class="sr-speaker">${r.speaker}</span>` : ''}
                    ${r.emotion ? `<span class="sr-emotion">${r.emotion}</span>` : ''}
                </div>
                ${r.transcript ? `<div class="sr-text">${this._trunc(r.transcript, 120)}</div>` : ''}
                <div class="sr-score">${this._fmt(r.time_start)} – ${this._fmt(r.time_end)} · score ${r.score.toFixed(2)}</div>`;
            el.addEventListener('click', () => this._seek(r.time_start));
            pane.appendChild(el);
        });
    }

    // ── Helpers ────────────────────────────────────────────────────
    _fmt(s) {
        const m = Math.floor(s / 60);
        const sec = Math.floor(s % 60);
        return `${m}:${String(sec).padStart(2, '0')}`;
    }

    _trunc(t, n) { return t && t.length > n ? t.slice(0, n) + '…' : (t || ''); }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new WizApp();
    app._tryRestoreSession();
});
window.addEventListener('resize', () => {
    const c = document.getElementById('waveformCanvas');
    if (c) {
        c.width = c.offsetWidth || c.parentElement.clientWidth;
    }
});