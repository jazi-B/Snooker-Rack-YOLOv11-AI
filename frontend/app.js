// Global State
let tables = [];
let calibrationPoints = [];
let activeTab = 'dashboard-tab';
let pollingInterval = null;
let occupancyChart = null;
let revenueChart = null;

// API Endpoints
const API_BASE = '/api';
const ENDPOINTS = {
    tables: `${API_BASE}/tables`,
    sessionsStart: `${API_BASE}/sessions/start`,
    sessionsAdjust: (id) => `${API_BASE}/sessions/${id}/adjust`,
    sessionsEnd: (id) => `${API_BASE}/sessions/${id}/end`,
    history: `${API_BASE}/sessions/history`,
    settings: `${API_BASE}/settings`
};

// --- INITIALIZATION ---
document.addEventListener('DOMContentLoaded', () => {
    initClock();
    initTabs();
    initModals();
    initSettings();
    initRoiDrawing();
    initAnalytics();
    
    // Initial fetch
    refreshDashboard();
    initAnalytics();
    
    // Set up polling interval to keep dashboard values synced (every 5 seconds)
    pollingInterval = setInterval(refreshDashboard, 5000);
});

// --- CLOCK CONTROLLER ---
function initClock() {
    const clockEl = document.getElementById('clock');
    const updateTime = () => {
        const now = new Date();
        let hours = now.getHours();
        let minutes = now.getMinutes();
        const ampm = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12;
        hours = hours ? hours : 12; // 12 instead of 0
        minutes = minutes < 10 ? '0' + minutes : minutes;
        clockEl.textContent = `${hours}:${minutes} ${ampm}`;
    };
    updateTime();
    setInterval(updateTime, 60000);
}

// --- TAB ROUTING ---
function initTabs() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const panels = document.querySelectorAll('.tab-panel');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            activeTab = targetTab;
            
            navButtons.forEach(b => b.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            if (targetTab === 'settings-tab') {
                loadCalibrationDropdown();
                loadHistory();
            } else if (targetTab === 'analytics-tab') {
                refreshAnalytics();
            }
            
            // Optimization: Stop MJPEG streams for tables that are not visible
            manageVideoStreams();
        });
    });
}

function manageVideoStreams() {
    // If feeds-tab is active, enable all camera streams.
    // If settings-tab is active and calibration selected, enable that stream.
    // Else, unload images to save bandwidth & GPU resource.
    const feedsGrid = document.getElementById('feeds-container');
    const calibImg = document.getElementById('calib-img');
    const selectedCalibTable = document.getElementById('calib-table-select').value;
    
    if (activeTab === 'feeds-tab') {
        renderFeedsTab();
    } else {
        // Clear feeds container
        feedsGrid.innerHTML = '';
    }
    
    if (activeTab !== 'settings-tab' || !selectedCalibTable) {
        calibImg.src = '';
        calibImg.style.display = 'none';
        document.getElementById('calib-placeholder').style.display = 'flex';
    }
}

// --- API ACTIONS ---

async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Network error occurred.');
        }
        return await response.json();
    } catch (e) {
        console.error(`[API Error] URL: ${url}`, e);
        alert(`Error: ${e.message}`);
        return null;
    }
}

async function refreshDashboard() {
    const data = await fetchAPI(ENDPOINTS.tables);
    if (data) {
        tables = data;
        renderDashboard();
        
        // If feeds tab is active, refresh the feeds (only if new tables added)
        if (activeTab === 'feeds-tab') {
            renderFeedsTab();
        }
    }
}

// --- RENDER DASHBOARD ---
function renderDashboard() {
    const container = document.getElementById('tables-container');
    if (tables.length === 0) {
        container.innerHTML = `
            <div class="loading-state">
                <i class="fa-solid fa-circle-question"></i>
                <p>No snooker tables configured. Use "Add New Table" to start.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    tables.forEach(table => {
        const activeSess = table.active_session;
        const card = document.createElement('div');
        card.className = `table-card ${activeSess ? 'active' : 'idle'}`;
        
        // Define statuses
        let statusBadge = '<span class="status-badge idle">Vacant</span>';
        if (activeSess) {
            if (!table.game_logic_started) {
                statusBadge = `<span class="status-badge waiting"><i class="fa-solid fa-hourglass-start"></i> Waiting for Rack</span>`;
            } else {
                if (table.rack_present) {
                    statusBadge = `<span class="status-badge set"><i class="fa-solid fa-circle-dot"></i> Rack Set</span>`;
                } else {
                    statusBadge = `<span class="status-badge in-progress"><i class="fa-solid fa-gamepad"></i> Game In Progress</span>`;
                }
            }
        }
        
        // Top section
        const deleteBtn = !activeSess ? `<button class="ctrl-btn" style="width: 32px; height: 32px; border-color: rgba(255, 59, 48, 0.2); color: #ff453a; background: rgba(255, 59, 48, 0.05); font-size: 13px; display: flex; align-items: center; justify-content: center;" onclick="deleteTable('${table.id}')" title="Delete Table"><i class="fa-solid fa-trash-can"></i></button>` : '';
        
        let cardContent = `
            <div class="card-top">
                <div class="table-title">
                    <h3>${table.name}</h3>
                    <span>Source: ${table.camera_source}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    ${deleteBtn}
                    ${statusBadge}
                </div>
            </div>
        `;
        
        // Middle session details
        if (activeSess) {
            // Calculate real-time duration
            const start = new Date(activeSess.start_time);
            const elapsedMins = Math.floor((new Date() - start) / 60000);
            const cost = (activeSess.games_played * activeSess.game_rate).toFixed(2);
            
            cardContent += `
                <div class="session-info">
                    <div class="customer-detail">
                        <span class="lbl">Customer:</span>
                        <span class="val name">${activeSess.customer_name}</span>
                    </div>
                    <div class="customer-detail">
                        <span class="lbl">Active Time:</span>
                        <span class="val time"><i class="fa-regular fa-clock"></i> ${elapsedMins} mins</span>
                    </div>
                    <div class="billing-detail">
                        <span>Rate: Rs. ${activeSess.game_rate.toFixed(2)}/game</span>
                        <span class="text-emerald text-bold">Total Bill: Rs. ${cost}</span>
                    </div>
                </div>
                
                <!-- Game counter adjustment panel -->
                <div class="games-controller">
                    <div class="games-label">
                        <span class="lbl">Frames Played</span>
                        <span class="val" id="count-${activeSess.id}">${activeSess.games_played}</span>
                    </div>
                    <div class="controller-btns">
                        <button class="ctrl-btn" onclick="adjustGames(${activeSess.id}, -1)">
                            <i class="fa-solid fa-minus"></i>
                        </button>
                        <button class="ctrl-btn" onclick="adjustGames(${activeSess.id}, 1)">
                            <i class="fa-solid fa-plus"></i>
                        </button>
                    </div>
                </div>
                
                <div class="card-actions">
                    <button class="btn btn-danger w-full" onclick="openReceiptModal('${table.id}')">
                        <i class="fa-solid fa-flag-checkered"></i> End Session & Bill
                    </button>
                </div>
            `;
        } else {
            cardContent += `
                <div class="session-info" style="align-items: center; justify-content: center; min-height: 150px; background: rgba(0,0,0,0.15)">
                    <i class="fa-solid fa-moon" style="font-size: 28px; color: var(--text-muted); margin-bottom: 8px;"></i>
                    <p style="color: var(--text-secondary); text-align: center; font-size: 13px;">Table is currently vacant</p>
                </div>
                <div class="card-actions">
                    <button class="btn btn-primary w-full" onclick="openStartSessionModal('${table.id}')">
                        <i class="fa-solid fa-play"></i> Start New Session
                    </button>
                </div>
            `;
        }
        
        card.innerHTML = cardContent;
        container.appendChild(card);
    });
}

// --- RENDER CAMERA FEEDS ---
function renderFeedsTab() {
    const container = document.getElementById('feeds-container');
    if (tables.length === 0) {
        container.innerHTML = `
            <div class="loading-state">
                <i class="fa-solid fa-video-slash"></i>
                <p>No feeds available. Add tables first.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    tables.forEach(table => {
        const feedCard = document.createElement('div');
        feedCard.className = 'feed-card';
        
        let feedStatusClass = 'idle';
        let feedStatusText = 'Vacant';
        if (table.active_session) {
            if (!table.game_logic_started) {
                feedStatusClass = 'waiting';
                feedStatusText = 'Waiting for Rack';
            } else if (table.rack_present) {
                feedStatusClass = 'set';
                feedStatusText = 'Rack Set';
            } else {
                feedStatusClass = 'in-progress';
                feedStatusText = 'Game In Progress';
            }
        }

        feedCard.innerHTML = `
            <div class="feed-title">
                <h4>${table.name} Live Feed</h4>
                <span class="status-badge ${feedStatusClass}">
                    ${feedStatusText}
                </span>
            </div>
            <div class="stream-frame-container">
                <img src="/api/tables/${table.id}/feed" alt="${table.name} Overhead Camera Feed" onerror="handleStreamError(this)">
            </div>
        `;
        container.appendChild(feedCard);
    });
}

function handleStreamError(img) {
    // Falls back gracefully if stream server connection drops
    const parent = img.parentElement;
    img.style.display = 'none';
    parent.innerHTML = `
        <div class="no-feed-msg">
            <i class="fa-solid fa-triangle-exclamation text-orange"></i>
            <p>Camera feed offline or loading...</p>
        </div>
    `;
}

// --- SESSIONS LOGIC ---

function openStartSessionModal(tableId) {
    document.getElementById('start-table-id').value = tableId;
    document.getElementById('customer-name-input').value = '';
    document.getElementById('custom-rate-input').value = '';
    openModal('start-session-modal');
}

async function adjustGames(sessionId, offset) {
    const countEl = document.getElementById(`count-${sessionId}`);
    let currentVal = parseInt(countEl.textContent);
    let newVal = Math.max(0, currentVal + offset);
    
    // Optimistic UI update
    countEl.textContent = newVal;
    
    const res = await fetchAPI(ENDPOINTS.sessionsAdjust(sessionId), {
        method: 'POST',
        body: JSON.stringify({ games_played: newVal })
    });
    
    if (res) {
        refreshDashboard();
    }
}

// Global modal triggers
async function submitStartSession() {
    const tableId = document.getElementById('start-table-id').value;
    const name = document.getElementById('customer-name-input').value.trim();
    const rateInput = document.getElementById('custom-rate-input').value;
    const rate = rateInput ? parseFloat(rateInput) : null;
    
    if (!name) {
        alert('Please enter a customer name.');
        return;
    }
    
    const res = await fetchAPI(ENDPOINTS.sessionsStart, {
        method: 'POST',
        body: JSON.stringify({
            table_id: tableId,
            customer_name: name,
            rate: rate
        })
    });
    
    if (res) {
        closeModal('start-session-modal');
        refreshDashboard();
    }
}

let activeEndingSession = null;

function openReceiptModal(tableId) {
    // Find the table and its active session
    const targetTable = tables.find(t => t.id === tableId || String(t.id) === String(tableId));
    if (!targetTable || !targetTable.active_session) return;
    
    const sess = targetTable.active_session;
    activeEndingSession = sess;
    
    // Time parsing
    const start = new Date(sess.start_time);
    const end = new Date();
    const durationMins = Math.floor((end - start) / 60000);
    
    const formatTime = (date) => {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };
    
    // Fill receipt modal details
    document.getElementById('receipt-customer').textContent = sess.customer_name;
    document.getElementById('receipt-table').textContent = targetTable.name;
    document.getElementById('receipt-start').textContent = formatTime(start);
    document.getElementById('receipt-end').textContent = formatTime(end);
    document.getElementById('receipt-duration').textContent = `${durationMins} minutes`;
    document.getElementById('receipt-games').textContent = `${sess.games_played} ${sess.games_played === 1 ? 'game' : 'games'}`;
    document.getElementById('receipt-rate').textContent = `Rs. ${sess.game_rate.toFixed(2)}`;
    document.getElementById('receipt-total').textContent = `Rs. ${(sess.games_played * sess.game_rate).toFixed(2)}`;
    
    openModal('receipt-modal');
}

async function submitEndSession() {
    if (!activeEndingSession) return;
    
    const res = await fetchAPI(ENDPOINTS.sessionsEnd(activeEndingSession.id), {
        method: 'POST'
    });
    
    if (res) {
        closeModal('receipt-modal');
        activeEndingSession = null;
        refreshDashboard();
        loadHistory();
    }
}

// --- ROI CALIBRATION CANVAS DRAWING ---

function initRoiDrawing() {
    const select = document.getElementById('calib-table-select');
    const svg = document.getElementById('calib-svg');
    const calibImg = document.getElementById('calib-img');
    const placeholder = document.getElementById('calib-placeholder');
    const clearBtn = document.getElementById('clear-roi-btn');
    const saveBtn = document.getElementById('save-roi-btn');
    
    select.addEventListener('change', () => {
        const tableId = select.value;
        calibrationPoints = [];
        drawCalibrationROI();
        
        if (tableId) {
            // Load feed
            calibImg.src = `/api/tables/${tableId}/feed`;
            calibImg.style.display = 'block';
            placeholder.style.display = 'none';
            
            // Load existing points
            const currentTable = tables.find(t => t.id === tableId);
            if (currentTable && currentTable.roi_polygon) {
                calibrationPoints = [...currentTable.roi_polygon];
                drawCalibrationROI();
            }
            
            clearBtn.disabled = false;
            saveBtn.disabled = false;
        } else {
            calibImg.src = '';
            calibImg.style.display = 'none';
            placeholder.style.display = 'flex';
            clearBtn.disabled = true;
            saveBtn.disabled = true;
        }
    });
    
    // Add point on click
    svg.addEventListener('click', (e) => {
        if (!select.value) return;
        
        const rect = svg.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const clickY = e.clientY - rect.top;
        
        // Calculate normalized float coordinates (0.0 to 1.0)
        const normX = parseFloat((clickX / rect.width).toFixed(4));
        const normY = parseFloat((clickY / rect.height).toFixed(4));
        
        calibrationPoints.push([normX, normY]);
        drawCalibrationROI();
    });
    
    clearBtn.addEventListener('click', () => {
        calibrationPoints = [];
        drawCalibrationROI();
    });
    
    saveBtn.addEventListener('click', async () => {
        const tableId = select.value;
        if (!tableId) return;
        
        const res = await fetchAPI(`${API_BASE}/tables/${tableId}/roi`, {
            method: 'POST',
            body: JSON.stringify({ points: calibrationPoints })
        });
        
        if (res) {
            alert('ROI Polygon calibrated and saved successfully!');
            // Refresh table configs locally
            refreshDashboard();
        }
    });
}

function drawCalibrationROI() {
    const svg = document.getElementById('calib-svg');
    const rect = svg.getBoundingClientRect();
    
    // Remove existing SVG tags
    svg.innerHTML = '';
    
    if (calibrationPoints.length === 0) return;
    
    // Draw polygon/polyline
    const pointsStr = calibrationPoints
        .map(pt => `${pt[0] * rect.width},${pt[1] * rect.height}`)
        .join(' ');
        
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", pointsStr);
    svg.appendChild(poly);
    
    // Draw individual anchor points
    calibrationPoints.forEach((pt, idx) => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", pt[0] * rect.width);
        circle.setAttribute("cy", pt[1] * rect.height);
        circle.setAttribute("r", "6");
        circle.setAttribute("title", `Point ${idx + 1}`);
        svg.appendChild(circle);
    });
}

// Window resizing adjustments for the SVG ROI overlay
window.addEventListener('resize', () => {
    if (activeTab === 'settings-tab' && document.getElementById('calib-table-select').value) {
        drawCalibrationROI();
    }
});

// --- SETTINGS CONFIGURATIONS ---

async function initSettings() {
    // Fetch default setting
    const res = await fetchAPI(ENDPOINTS.settings);
    if (res) {
        document.getElementById('game-rate-input').value = res.game_rate;
    }
    
    document.getElementById('save-settings-btn').addEventListener('click', async () => {
        const rate = parseFloat(document.getElementById('game-rate-input').value);
        if (isNaN(rate) || rate <= 0) {
            alert('Please enter a valid rate.');
            return;
        }
        
        const saveRes = await fetchAPI(ENDPOINTS.settings, {
            method: 'POST',
            body: JSON.stringify({ game_rate: rate })
        });
        
        if (saveRes) {
            alert('System default rates updated.');
            refreshDashboard();
        }
    });
}

async function loadHistory() {
    const container = document.getElementById('history-container');
    const res = await fetchAPI(ENDPOINTS.history);
    
    if (!res || res.length === 0) {
        container.innerHTML = '<div class="empty-history">No past sessions on record.</div>';
        return;
    }
    
    container.innerHTML = '';
    res.forEach(item => {
        // Format ISO times
        const date = new Date(item.end_time).toLocaleDateString();
        const start = new Date(item.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const end = new Date(item.end_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div class="history-info">
                <span class="name">${item.customer_name}</span>
                <span class="meta">${date} | ${start} - ${end} (${item.games_played} frames)</span>
            </div>
            <div class="history-bill">Rs. ${item.total_bill.toFixed(2)}</div>
        `;
        container.appendChild(div);
    });
}

function loadCalibrationDropdown() {
    const select = document.getElementById('calib-table-select');
    const currentVal = select.value;
    
    select.innerHTML = '<option value="">-- Choose Snooker Table --</option>';
    tables.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.textContent = t.name;
        select.appendChild(opt);
    });
    
    select.value = currentVal;
}

// --- ADD NEW TABLE ---
async function submitAddTable() {
    const tableId = document.getElementById('table-id-input').value.trim();
    const name = document.getElementById('table-name-input').value.trim();
    const source = document.getElementById('table-source-input').value.trim();
    
    if (!tableId || !name || !source) {
        alert('Please fill out all table details.');
        return;
    }
    
    const res = await fetchAPI(ENDPOINTS.tables, {
        method: 'POST',
        body: JSON.stringify({
            id: tableId,
            name: name,
            camera_source: source
        })
    });
    
    if (res) {
        closeModal('add-table-modal');
        refreshDashboard();
        loadCalibrationDropdown();
    }
}

// --- MODALS HELPER UTILITIES ---
function initModals() {
    // Open Add Table Trigger
    document.getElementById('open-add-table-btn').addEventListener('click', () => {
        document.getElementById('table-id-input').value = '';
        document.getElementById('table-name-input').value = '';
        document.getElementById('table-source-input').value = '0';
        openModal('add-table-modal');
    });
    
    // Close on overlay click or [data-close] button
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeModal(overlay.id);
            }
        });
    });
    
    document.querySelectorAll('.close-modal, [data-close]').forEach(btn => {
        btn.addEventListener('click', () => {
            const modalId = btn.getAttribute('data-close') || btn.closest('.modal-overlay').id;
            closeModal(modalId);
        });
    });
    
    // Confirm Actions
    document.getElementById('confirm-start-session-btn').addEventListener('click', submitStartSession);
    document.getElementById('confirm-end-session-btn').addEventListener('click', submitEndSession);
    document.getElementById('confirm-add-table-btn').addEventListener('click', submitAddTable);
}

function openModal(id) {
    document.getElementById(id).classList.add('open');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('open');
}

// --- DELETE TABLE SERVICE ---
async function deleteTable(tableId) {
    if (!confirm('Are you sure you want to delete this table? This will stop its active camera engine.')) return;
    const res = await fetchAPI(`${API_BASE}/tables/${tableId}`, {
        method: 'DELETE'
    });
    if (res) {
        refreshDashboard();
        loadCalibrationDropdown();
    }
}



// --- BUSINESS ANALYTICS LOGIC ---
function initAnalytics() {
    const rangeSelect = document.getElementById('analytics-time-range');
    const refreshBtn = document.getElementById('refresh-analytics-btn');
    
    if (rangeSelect) {
        rangeSelect.addEventListener('change', refreshAnalytics);
    }
    
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshAnalytics);
    }
}

async function refreshAnalytics() {
    const rangeSelect = document.getElementById('analytics-time-range');
    const range = rangeSelect ? rangeSelect.value : '7days';
    
    const refreshBtn = document.getElementById('refresh-analytics-btn');
    if (refreshBtn) {
        const icon = refreshBtn.querySelector('i');
        if (icon) icon.classList.add('fa-spin');
        refreshBtn.disabled = true;
    }
    
    try {
        const res = await fetchAPI(`/api/analytics?range=${range}`);
        if (res) {
            // Update metric cards
            document.getElementById('metric-revenue').textContent = `Rs. ${res.revenue_today.toFixed(2)}`;
            document.getElementById('metric-games').textContent = `${res.games_today} ${res.games_today === 1 ? 'Frame' : 'Frames'}`;
            document.getElementById('metric-duration').textContent = `${res.avg_duration_mins.toFixed(0)} Mins`;
            document.getElementById('metric-occupancy').textContent = `${res.occupancy_rate.toFixed(0)}%`;
            
            // Build / Update Charts
            renderOccupancyChart(res.hourly_occupancy);
            renderRevenueChart(res.revenue_share);
        }
    } catch (e) {
        console.error("Error fetching analytics", e);
    } finally {
        if (refreshBtn) {
            const icon = refreshBtn.querySelector('i');
            if (icon) icon.classList.remove('fa-spin');
            refreshBtn.disabled = false;
        }
    }
}

function renderOccupancyChart(hourlyData) {
    const canvasEl = document.getElementById('occupancy-chart');
    if (!canvasEl) return;
    const ctx = canvasEl.getContext('2d');
    
    if (occupancyChart) {
        occupancyChart.destroy();
    }
    
    // Create modern sleek area gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 250);
    gradient.addColorStop(0, 'rgba(0, 136, 255, 0.35)');
    gradient.addColorStop(1, 'rgba(0, 136, 255, 0.00)');
    
    const labels = Array.from({ length: 24 }, (_, i) => {
        const ampm = i >= 12 ? 'PM' : 'AM';
        const hour = i % 12 || 12;
        return `${hour} ${ampm}`;
    });
    
    occupancyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Occupancy Rate (%)',
                data: hourlyData,
                borderColor: '#0088ff',
                borderWidth: 2.5,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#0088ff',
                pointBorderColor: '#ffffff',
                pointBorderWidth: 1.5,
                pointRadius: 3,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#11111a',
                    titleColor: '#fff',
                    bodyColor: '#aaa',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `Occupancy: ${context.parsed.y}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)',
                        borderColor: 'transparent'
                    },
                    ticks: {
                        color: '#9aa0a6',
                        font: { family: 'Inter', size: 9 },
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)',
                        borderColor: 'transparent'
                    },
                    ticks: {
                        color: '#9aa0a6',
                        font: { family: 'Inter', size: 9 },
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

function renderRevenueChart(revenueShare) {
    const canvasEl = document.getElementById('revenue-chart');
    if (!canvasEl) return;
    const ctx = canvasEl.getContext('2d');
    
    if (revenueChart) {
        revenueChart.destroy();
    }
    
    const labels = Object.keys(revenueShare);
    const data = Object.values(revenueShare);
    
    const colors = [
        '#00e673', // Emerald Green
        '#0088ff', // Sky Blue
        '#ff9f1c', // Orange Amber
        '#8a2be2', // Purple Blue
        '#ff3b30'  // Coral Red
    ];
    
    const hasRevenue = data.some(val => val > 0);
    const chartData = hasRevenue ? data : [1];
    const chartLabels = hasRevenue ? labels : ['No Revenue Recorded'];
    const chartColors = hasRevenue ? colors.slice(0, labels.length) : ['rgba(255,255,255,0.08)'];
    
    revenueChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: chartLabels,
            datasets: [{
                data: chartData,
                backgroundColor: chartColors,
                borderWidth: 1.5,
                borderColor: '#11111a',
                hoverOffset: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#9aa0a6',
                        font: { family: 'Inter', size: 10 },
                        padding: 10,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: '#11111a',
                    titleColor: '#fff',
                    bodyColor: '#aaa',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            if (!hasRevenue) return 'No revenue registered';
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const val = context.parsed;
                            const percentage = ((val / total) * 100).toFixed(0);
                            return ` ${context.label}: Rs. ${val.toFixed(2)} (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '72%'
        }
    });
}
