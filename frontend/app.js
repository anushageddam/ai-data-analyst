/**
 * YOO PROJECT — Frontend Client Controller
 * Manages reactive state, dynamic cross-filtering, Plotly rendering, and conversational chat.
 */

// Application State
const state = {
  currentTab: 'dashboard',
  filters: {},
  filterOptions: {},
  dashboardData: null,
  profilerData: null,
  explorer: {
    page: 1,
    pageSize: 25,
    search: '',
    sortCol: '',
    sortDir: 'asc'
  },
  theme: 'dark'
};

// Initialize App on DOM Load
document.addEventListener('DOMContentLoaded', () => {
  feather.replace();
  setupNavigation();
  setupEventListeners();
  initApp();
});

async function initApp() {
  await fetchDashboard();
}

// --------------------------------------------------------------------------
// 1. DATA FETCHING & STATE SYNC
// --------------------------------------------------------------------------

async function fetchDashboard(filters = {}) {
  try {
    const res = await fetch('/api/dashboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filters: filters })
    });
    const data = await res.json();
    state.dashboardData = data;
    state.filterOptions = data.filter_options || {};

    renderHeaderStatus(data);
    renderFilters(data.filter_options || {});
    renderKPIs(data.kpis || []);
    renderVisuals(data.visuals || []);
    renderInsights(data.insights || []);
    populateForecastMeasures(data.semantic_summary?.semantic_roles?.all_measures || []);

    if (state.currentTab === 'profiler') renderProfiler();
    if (state.currentTab === 'explorer') fetchExplorerData();
    if (state.currentTab === 'forecast') fetchForecast();

    feather.replace();
  } catch (err) {
    console.error('Failed to load dashboard:', err);
  }
}

// --------------------------------------------------------------------------
// 2. HEADER & FILTERS
// --------------------------------------------------------------------------

function renderHeaderStatus(data) {
  const filename = data.metadata?.filename || 'Active Dataset';
  document.getElementById('datasetName').textContent = `${filename} (${data.active_rows} rows)`;
  document.getElementById('recordsPill').textContent = `Rows: ${data.active_rows} / ${data.total_rows}`;

  const domain = data.semantic_summary?.detected_domain || 'Business';
  document.getElementById('domainBadge').textContent = domain;

  const score = data.profiler_summary?.data_health_score ?? '--';
  const status = data.profiler_summary?.data_health_status ?? '';
  const healthBadge = document.getElementById('healthBadge');
  healthBadge.textContent = `Health: ${score}% (${status})`;
}

function renderFilters(filterOptions) {
  const container = document.getElementById('filterControls');
  container.innerHTML = '';

  const entries = Object.entries(filterOptions);
  if (entries.length === 0) {
    container.innerHTML = '<span class="filter-placeholder">No categorical filter dimensions detected</span>';
    return;
  }

  entries.forEach(([dim, values]) => {
    const group = document.createElement('div');
    group.className = 'filter-select-group';

    const label = document.createElement('label');
    label.textContent = dim.replace('_', ' ') + ':';

    const select = document.createElement('select');
    select.id = `filter_${dim}`;

    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = 'All';
    select.appendChild(defaultOpt);

    values.forEach(val => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = val;
      if (state.filters[dim] && state.filters[dim].includes(val)) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });

    select.addEventListener('change', (e) => {
      const selected = e.target.value;
      if (selected) {
        state.filters[dim] = [selected];
      } else {
        delete state.filters[dim];
      }
      fetchDashboard(state.filters);
    });

    group.appendChild(label);
    group.appendChild(select);
    container.appendChild(group);
  });
}

// --------------------------------------------------------------------------
// 3. KPI CARDS
// --------------------------------------------------------------------------

function renderKPIs(kpis) {
  const grid = document.getElementById('kpiGrid');
  grid.innerHTML = '';

  kpis.forEach(kpi => {
    const card = document.createElement('div');
    card.className = 'kpi-card';

    // Sparkline SVG generator
    let sparklineHtml = '';
    if (kpi.sparkline && kpi.sparkline.length >= 3) {
      const min = Math.min(...kpi.sparkline);
      const max = Math.max(...kpi.sparkline);
      const range = max - min || 1;
      const points = kpi.sparkline.map((val, idx) => {
        const x = (idx / (kpi.sparkline.length - 1)) * 70;
        const y = 22 - ((val - min) / range) * 20;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');

      sparklineHtml = `
        <svg class="sparkline-svg" viewBox="0 0 70 24">
          <polyline fill="none" stroke="#38bdf8" stroke-width="2" points="${points}" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      `;
    }

    // Delta badge
    let deltaHtml = '<span class="text-muted">Total</span>';
    if (kpi.delta_percentage !== null && kpi.delta_percentage !== undefined) {
      const dirClass = kpi.direction === 'positive' ? 'positive' : (kpi.direction === 'negative' ? 'negative' : '');
      const icon = kpi.direction === 'positive' ? 'trending-up' : 'trending-down';
      deltaHtml = `
        <span class="kpi-delta ${dirClass}">
          <i data-feather="${icon}" style="width:14px;height:14px;"></i>
          ${kpi.delta_percentage > 0 ? '+' : ''}${kpi.delta_percentage}% vs prev
        </span>
      `;
    }

    card.innerHTML = `
      <div class="kpi-header">
        <span class="kpi-title">${kpi.title}</span>
        <span class="kpi-badge">${kpi.badge || 'KPI'}</span>
      </div>
      <div class="kpi-value">${kpi.value}</div>
      <div class="kpi-footer">
        ${deltaHtml}
        ${sparklineHtml}
      </div>
    `;

    grid.appendChild(card);
  });
}

// --------------------------------------------------------------------------
// 4. PLOTLY VISUALIZATIONS
// --------------------------------------------------------------------------

function renderVisuals(visuals) {
  const grid = document.getElementById('visualsGrid');
  grid.innerHTML = '';

  visuals.forEach((vis, idx) => {
    const card = document.createElement('div');
    card.className = 'visual-card';
    if (vis.chart_type === 'line_area' || vis.chart_type === 'heatmap') {
      card.classList.add('span-12');
    } else if (idx === 1 || idx === 2) {
      card.classList.add('span-6');
    }

    const plotId = `plot_${vis.id || idx}`;
    card.innerHTML = `
      <div class="visual-header">
        <div>
          <div class="visual-title">${vis.title}</div>
          <div class="visual-subtitle">${vis.description || ''}</div>
        </div>
        <span class="badge">${vis.section || 'Analytics'}</span>
      </div>
      <div class="visual-plot-container" id="${plotId}"></div>
    `;
    grid.appendChild(card);

    // Render Plotly with responsive config
    setTimeout(() => {
      const spec = vis.plotly_spec || {};
      const config = { responsive: true, displayModeBar: false };
      if (document.getElementById(plotId)) {
        Plotly.newPlot(plotId, spec.data || [], spec.layout || {}, config);
      }
    }, 50);
  });
}

// --------------------------------------------------------------------------
// 5. EVIDENCE-BASED INSIGHTS
// --------------------------------------------------------------------------

function renderInsights(insights) {
  const countBadge = document.getElementById('insightsCountBadge');
  countBadge.textContent = insights.length;

  const masonry = document.getElementById('insightsMasonry');
  masonry.innerHTML = '';

  if (insights.length === 0) {
    masonry.innerHTML = '<p class="text-secondary">No automated insights available for this selection.</p>';
    return;
  }

  insights.forEach(item => {
    const card = document.createElement('div');
    card.className = 'insight-card';
    card.style.borderLeftColor = item.badge_color || '#6366f1';

    let evidencePills = '';
    if (item.evidence) {
      evidencePills = Object.entries(item.evidence).map(([k, v]) => `
        <span class="insight-evidence-pill">${k}: <strong>${v}</strong></span>
      `).join(' ');
    }

    card.innerHTML = `
      <div class="insight-top">
        <span class="insight-category">${item.category} • ${item.type}</span>
        <span class="badge" style="background:${item.badge_color}22; color:${item.badge_color}; border-color:${item.badge_color}44;">
          ${item.metric_value}
        </span>
      </div>
      <div class="insight-title">${item.title}</div>
      <div class="insight-desc">${item.description}</div>
      <div class="insight-evidence-row mt-2">${evidencePills}</div>
    `;

    masonry.appendChild(card);
  });
}

// --------------------------------------------------------------------------
// 6. PROFILER & DATA QUALITY
// --------------------------------------------------------------------------

async function renderProfiler() {
  try {
    const res = await fetch('/api/profiler');
    const data = await res.json();
    state.profilerData = data;

    const summary = data.summary || {};
    document.getElementById('healthScoreDisplay').textContent = `${summary.data_health_score}%`;
    document.getElementById('healthScoreDisplay').style.color = summary.data_health_color || '#10b981';

    const statusEl = document.getElementById('healthStatusDisplay');
    statusEl.textContent = summary.data_health_status || 'Good';
    statusEl.style.color = summary.data_health_color || '#10b981';

    const auditGrid = document.getElementById('auditGrid');
    auditGrid.innerHTML = `
      <div class="audit-item">
        <div class="audit-label">Total Rows</div>
        <div class="audit-value">${summary.total_rows?.toLocaleString() ?? 0}</div>
      </div>
      <div class="audit-item">
        <div class="audit-label">Total Columns</div>
        <div class="audit-value">${summary.total_columns ?? 0}</div>
      </div>
      <div class="audit-item">
        <div class="audit-label">Missing Cells</div>
        <div class="audit-value">${summary.total_null_cells} (${summary.overall_null_percentage}%)</div>
      </div>
      <div class="audit-item">
        <div class="audit-label">Duplicate Rows</div>
        <div class="audit-value">${summary.duplicate_rows} (${summary.duplicate_percentage}%)</div>
      </div>
      <div class="audit-item">
        <div class="audit-label">Detected Outliers</div>
        <div class="audit-value" style="color:#ec4899;">${summary.total_outliers_detected ?? 0}</div>
      </div>
      <div class="audit-item">
        <div class="audit-label">Empty Columns</div>
        <div class="audit-value">${summary.empty_columns_count ?? 0}</div>
      </div>
    `;

    renderColumnProfilesTable(data.columns || []);
  } catch (err) {
    console.error('Failed to load profiler details:', err);
  }
}

function renderColumnProfilesTable(columns) {
  const tbody = document.getElementById('columnsProfileTableBody');
  tbody.innerHTML = '';

  columns.forEach(col => {
    const tr = document.createElement('tr');
    tr.dataset.type = col.type;

    const topVal = col.top_frequencies?.[0] ? `${col.top_frequencies[0].value} (${col.top_frequencies[0].percentage}%)` : '--';
    const minMax = (col.min !== null && col.max !== null) ? `${col.min} / ${col.max}` : '--';
    const meanMedian = (col.mean !== null && col.median !== null) ? `${col.mean} / ${col.median}` : '--';

    tr.innerHTML = `
      <td><strong>${col.name}</strong></td>
      <td><span class="col-tag">${col.type}</span></td>
      <td>${col.null_percentage}%</td>
      <td>${col.unique_count}</td>
      <td>${minMax}</td>
      <td>${meanMedian}</td>
      <td>${col.outlier_count > 0 ? `<span style="color:#ec4899;font-weight:700;">${col.outlier_count} (${col.outlier_percentage}%)</span>` : '0'}</td>
      <td>${topVal}</td>
    `;
    tbody.appendChild(tr);
  });
}

// --------------------------------------------------------------------------
// 7. PREDICTIVE FORECASTING
// --------------------------------------------------------------------------

function populateForecastMeasures(measures) {
  const select = document.getElementById('forecastMeasureSelect');
  select.innerHTML = '';
  measures.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m.replace('_', ' ').toUpperCase();
    select.appendChild(opt);
  });
}

async function fetchForecast() {
  const measure = document.getElementById('forecastMeasureSelect').value;
  const horizon = parseInt(document.getElementById('forecastHorizonSelect').value, 10) || 3;

  try {
    const res = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ measure: measure, horizon: horizon })
    });
    const result = await res.json();

    if (!result.is_suitable) {
      document.getElementById('forecastNarrative').innerHTML = `<span style="color:#f59e0b;">⚠️ ${result.reason}</span>`;
      document.getElementById('forecastPlotlyChart').innerHTML = '<div style="padding:40px;text-align:center;color:#64748b;">Forecasting not reliable for this measure.</div>';
      document.getElementById('modelMetricsGrid').innerHTML = '';
      return;
    }

    document.getElementById('forecastNarrative').textContent = result.narrative;
    const m = result.metrics || {};
    document.getElementById('modelMetricsGrid').innerHTML = `
      <div class="model-metric-item">
        <div class="label">Historical Mean</div>
        <div class="val">${m.historical_mean?.toLocaleString()}</div>
      </div>
      <div class="model-metric-item">
        <div class="label">Forecast Mean</div>
        <div class="val">${m.forecast_mean?.toLocaleString()}</div>
      </div>
      <div class="model-metric-item">
        <div class="label">Model Error (MAE)</div>
        <div class="val">${m.mae}</div>
      </div>
      <div class="model-metric-item">
        <div class="label">Reliability (MAPE)</div>
        <div class="val">${m.mape}%</div>
      </div>
    `;

    // Render forecast plot
    Plotly.newPlot('forecastPlotlyChart', result.plotly_spec.data, result.plotly_spec.layout, { responsive: true, displayModeBar: false });
  } catch (err) {
    console.error('Forecast error:', err);
  }
}

// --------------------------------------------------------------------------
// 8. DATA EXPLORER
// --------------------------------------------------------------------------

async function fetchExplorerData() {
  const { page, pageSize, search, sortCol, sortDir } = state.explorer;
  const url = `/api/explorer?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(search)}&sort_col=${encodeURIComponent(sortCol)}&sort_dir=${sortDir}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    const thead = document.getElementById('explorerTableHead');
    const tbody = document.getElementById('explorerTableBody');

    // Headers
    thead.innerHTML = '<tr>' + data.columns.map(c => `<th onclick="sortExplorer('${c}')" style="cursor:pointer;">${c} ${state.explorer.sortCol === c ? (state.explorer.sortDir === 'asc' ? '▲' : '▼') : ''}</th>`).join('') + '</tr>';

    // Rows
    tbody.innerHTML = '';
    data.data.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = data.columns.map(c => `<td>${row[c] !== null && row[c] !== undefined ? row[c] : ''}</td>`).join('');
      tbody.appendChild(tr);
    });

    // Pagination
    const totalPages = Math.ceil(data.total_records / pageSize) || 1;
    document.getElementById('pageIndicator').textContent = `Page ${data.page} of ${totalPages} (${data.total_records} rows)`;
  } catch (err) {
    console.error('Explorer fetch error:', err);
  }
}

function sortExplorer(col) {
  if (state.explorer.sortCol === col) {
    state.explorer.sortDir = state.explorer.sortDir === 'asc' ? 'desc' : 'asc';
  } else {
    state.explorer.sortCol = col;
    state.explorer.sortDir = 'asc';
  }
  fetchExplorerData();
}

// --------------------------------------------------------------------------
// 9. CONVERSATIONAL AI BI ANALYST CHATBOT
// --------------------------------------------------------------------------

async function sendChat(queryText) {
  const input = document.getElementById('chatInput');
  const q = queryText || input.value.trim();
  if (!q) return;

  input.value = '';
  appendChatMessage('user', q);

  // Assistant typing bubble
  const typingId = 'typing_' + Date.now();
  appendChatMessage('assistant', 'Analyzing dataset...', typingId);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q })
    });
    const resp = await res.json();

    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    appendChatMessage('assistant', resp.answer, null, resp.chart);
  } catch (err) {
    console.error('Chat error:', err);
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.textContent = 'Sorry, could not process that question right now.';
  }
}

function askChat(sampleQuery) {
  document.querySelector('[data-tab="chat"]').click();
  sendChat(sampleQuery);
}

function appendChatMessage(role, text, id, chartSpec = null) {
  const container = document.getElementById('chatMessages');
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}-bubble`;
  if (id) bubble.id = id;

  const avatarIcon = role === 'user' ? 'user' : 'cpu';
  const plotBoxId = chartSpec ? `chat_plot_${Date.now()}` : '';

  bubble.innerHTML = `
    <div class="avatar"><i data-feather="${avatarIcon}"></i></div>
    <div class="message-body">
      <div>${text.replace(/\n/g, '<br>')}</div>
      ${chartSpec ? `<div class="chat-chart-box" id="${plotBoxId}" style="height:280px;"></div>` : ''}
    </div>
  `;

  container.appendChild(bubble);
  feather.replace();
  container.scrollTop = container.scrollHeight;

  if (chartSpec && plotBoxId) {
    setTimeout(() => {
      Plotly.newPlot(plotBoxId, chartSpec.data, chartSpec.layout, { responsive: true, displayModeBar: false });
      container.scrollTop = container.scrollHeight;
    }, 60);
  }
}

// --------------------------------------------------------------------------
// 10. UPLOAD & SAMPLE DATA LOADERS
// --------------------------------------------------------------------------

async function loadSample(sampleId) {
  closeSampleDropdown();
  showUploadSpinner('Loading enterprise dataset...');
  try {
    const res = await fetch('/api/load-sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId })
    });
    const data = await res.json();
    state.filters = {};
    await fetchDashboard();
    hideUploadSpinner();
  } catch (err) {
    alert('Error loading sample dataset: ' + err.message);
    hideUploadSpinner();
  }
}

async function startSampleAnalysis(sampleId) {
  await loadSample(sampleId);
  const dashTab = document.querySelector('[data-tab="dashboard"]');
  if (dashTab) dashTab.click();
}

async function uploadFile(file, autoSwitch = false) {
  const formData = new FormData();
  formData.append('file', file);

  showUploadSpinner('Uploading & parsing workbook...');

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    if (data.status === 'multiple_sheets') {
      hideUploadSpinner();
      const select = document.getElementById('excelSheetSelect');
      select.innerHTML = '';
      data.sheets.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;
        select.appendChild(opt);
      });
      document.getElementById('sheetSelectorContainer').style.display = 'block';
    } else {
      state.filters = {};
      await fetchDashboard();
      hideUploadModal();
      if (autoSwitch || state.currentTab === 'home') {
        const dashTab = document.querySelector('[data-tab="dashboard"]');
        if (dashTab) dashTab.click();
      }
    }
  } catch (err) {
    alert('Upload error: ' + err.message);
    hideUploadSpinner();
  }
}

async function confirmExcelSheet() {
  const select1 = document.getElementById('excelSheetSelect');
  const select2 = document.getElementById('homeExcelSheetSelect');
  const sheetName = (select1 && select1.value) || (select2 && select2.value);

  showUploadSpinner('Analyzing selected sheet...');
  try {
    await fetch('/api/select-sheet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sheet_name: sheetName })
    });
    state.filters = {};
    await fetchDashboard();
    hideUploadModal();
    const dashTab = document.querySelector('[data-tab="dashboard"]');
    if (dashTab) dashTab.click();
  } catch (err) {
    alert('Sheet load error: ' + err.message);
    hideUploadSpinner();
  }
}

// --------------------------------------------------------------------------
// 11. NAVIGATION & EVENT LISTENERS
// --------------------------------------------------------------------------

function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      const target = tab.dataset.tab;
      state.currentTab = target;
      const content = document.getElementById(`tab-${target}`);
      if (content) content.classList.add('active');

      if (target === 'profiler') renderProfiler();
      if (target === 'explorer') fetchExplorerData();
      if (target === 'forecast') fetchForecast();

      // Trigger window resize so Plotly graphs re-align to new dimensions
      window.dispatchEvent(new Event('resize'));
    });
  });
}

function setupEventListeners() {
  // Sample dropdown
  const sampleBtn = document.getElementById('sampleDropdownBtn');
  const sampleMenu = document.getElementById('sampleDropdownMenu');
  sampleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    sampleMenu.classList.toggle('show');
  });
  document.addEventListener('click', () => sampleMenu.classList.remove('show'));

  // Upload modal
  document.getElementById('openUploadModalBtn').addEventListener('click', showUploadModal);
  document.getElementById('closeUploadModalBtn').addEventListener('click', hideUploadModal);

  // Modal File dropzone
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) uploadFile(e.target.files[0]);
  });

  document.getElementById('confirmSheetBtn').addEventListener('click', confirmExcelSheet);

  // Homepage Dropzone
  const homeDropzone = document.getElementById('homeDropzone');
  const homeFileInput = document.getElementById('homeFileInput');
  if (homeDropzone && homeFileInput) {
    homeDropzone.addEventListener('dragover', (e) => { e.preventDefault(); homeDropzone.classList.add('dragover'); });
    homeDropzone.addEventListener('dragleave', () => homeDropzone.classList.remove('dragover'));
    homeDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      homeDropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0], true);
    });
    homeFileInput.addEventListener('change', (e) => {
      if (e.target.files.length) uploadFile(e.target.files[0], true);
    });
  }

  const homeConfirmBtn = document.getElementById('homeConfirmSheetBtn');
  if (homeConfirmBtn) {
    homeConfirmBtn.addEventListener('click', confirmExcelSheet);
  }

  // Filter Reset
  document.getElementById('resetFiltersBtn').addEventListener('click', () => {
    state.filters = {};
    fetchDashboard({});
  });

  // Export CSV
  document.getElementById('exportBtn').addEventListener('click', () => {
    window.location.href = '/api/export';
  });

  // Forecast Run
  document.getElementById('runForecastBtn').addEventListener('click', fetchForecast);

  // Explorer Search & Pagination
  document.getElementById('explorerSearchInput').addEventListener('input', (e) => {
    state.explorer.search = e.target.value;
    state.explorer.page = 1;
    fetchExplorerData();
  });
  document.getElementById('prevPageBtn').addEventListener('click', () => {
    if (state.explorer.page > 1) {
      state.explorer.page--;
      fetchExplorerData();
    }
  });
  document.getElementById('nextPageBtn').addEventListener('click', () => {
    state.explorer.page++;
    fetchExplorerData();
  });

  // Chat send
  document.getElementById('sendChatBtn').addEventListener('click', () => sendChat());
  document.getElementById('chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendChat();
  });

  // Theme switch
  document.getElementById('themeToggleBtn').addEventListener('click', toggleTheme);

  // Profiler column type filter chips
  document.getElementById('typeFilterChips').addEventListener('click', (e) => {
    if (e.target.classList.contains('chip')) {
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      const type = e.target.dataset.type;
      const rows = document.querySelectorAll('#columnsProfileTableBody tr');
      rows.forEach(r => {
        if (type === 'all' || r.dataset.type === type) {
          r.style.display = '';
        } else {
          r.style.display = 'none';
        }
      });
    }
  });
}

function showUploadModal() {
  document.getElementById('uploadModal').classList.add('show');
  document.getElementById('sheetSelectorContainer').style.display = 'none';
  document.getElementById('uploadSpinner').style.display = 'none';
}

function hideUploadModal() {
  document.getElementById('uploadModal').classList.remove('show');
  hideUploadSpinner();
}

function showUploadSpinner(text) {
  document.getElementById('uploadSpinner').style.display = 'block';
  document.getElementById('spinnerText').textContent = text;
}

function hideUploadSpinner() {
  document.getElementById('uploadSpinner').style.display = 'none';
}

function closeSampleDropdown() {
  document.getElementById('sampleDropdownMenu').classList.remove('show');
}

function toggleTheme() {
  const body = document.body;
  const icon = document.getElementById('themeIcon');
  if (body.classList.contains('dark-theme')) {
    body.classList.remove('dark-theme');
    body.classList.add('light-theme');
    icon.setAttribute('data-feather', 'moon');
    state.theme = 'light';
  } else {
    body.classList.remove('light-theme');
    body.classList.add('dark-theme');
    icon.setAttribute('data-feather', 'sun');
    state.theme = 'dark';
  }
  feather.replace();
  window.dispatchEvent(new Event('resize'));
}
