/**
 * Dashboard JavaScript for Insider Trading Detection System
 * Handles all frontend interactions, data visualization, and API communication
 */

// Global variables
let currentData = null;
let currentSymbol = 'all';
let refreshInterval = null;
let chartInstances = {};

// Configuration
const CONFIG = {
    refreshInterval: 30000, // 30 seconds
    animationDuration: 300,
    maxRetries: 3,
    retryDelay: 1000,
    toastTimeout: 5000
};

// API endpoints
const API_ENDPOINTS = {
    analyze: '/api/analyze',
    report: '/api/report',
    status: '/api/status',
    visualizations: '/api/visualizations',
    symbol: '/api/symbol'
};

/**
 * Initialize dashboard on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startAutoRefresh();
});

/**
 * Initialize dashboard components
 */
function initializeDashboard() {
    console.log('Initializing dashboard...');
    
    // Load initial data
    loadDashboardData();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize loading states
    showLoadingStates();
}

/**
 * Setup event listeners for interactive elements
 */
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshData');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            this.disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Refreshing...';
            loadDashboardData().finally(() => {
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-sync-alt me-1"></i>Refresh';
            });
        });
    }
    
    // Symbol filter
    const symbolFilter = document.getElementById('symbolFilter');
    if (symbolFilter) {
        symbolFilter.addEventListener('change', function() {
            currentSymbol = this.value;
            loadVisualizationsForSymbol(currentSymbol);
        });
    }
    
    // Symbol selection in analysis
    const symbolSelect = document.getElementById('symbolSelect');
    if (symbolSelect) {
        // Select all by default
        Array.from(symbolSelect.options).forEach(option => {
            option.selected = true;
        });
    }
    
    // Window events
    window.addEventListener('resize', debounce(handleWindowResize, 250));
    window.addEventListener('beforeunload', cleanup);
}

/**
 * Load dashboard data from API
 */
async function loadDashboardData() {
    try {
        showLoadingStates();
        
        // Load report data
        const reportData = await fetchWithRetry(API_ENDPOINTS.report);
        
        if (reportData.success && reportData.report) {
            currentData = reportData.report;
            updateDashboardComponents(currentData);
            showSuccessToast('Dashboard data loaded successfully');
        } else {
            throw new Error(reportData.error || 'No analysis data available');
        }
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showErrorState('Error loading dashboard data. Please run analysis first.');
        showErrorToast('Failed to load dashboard data: ' + error.message);
    } finally {
        hideLoadingStates();
    }
}

/**
 * Update all dashboard components with new data
 */
function updateDashboardComponents(data) {
    if (!data) return;
    
    // Update status cards
    updateStatusCards(data.summary);
    
    // Update symbol filter
    updateSymbolFilter(data.summary.symbols_analyzed);
    
    // Update symbol table
    updateSymbolTable(data.symbols);
    
    // Update alerts
    updateAlerts(data.alerts);
    
    // Load visualizations
    loadVisualizationsForSymbol(currentSymbol);
    
    // Animate updates
    animateUpdates();
}

/**
 * Update status cards with summary data
 */
function updateStatusCards(summary) {
    const elements = {
        totalObservations: document.getElementById('totalObservations'),
        totalAnomalies: document.getElementById('totalAnomalies'),
        anomalyRate: document.getElementById('anomalyRate'),
        symbolsCount: document.getElementById('symbolsCount')
    };
    
    if (elements.totalObservations) {
        animateCounter(elements.totalObservations, summary.total_observations || 0);
    }
    
    if (elements.totalAnomalies) {
        animateCounter(elements.totalAnomalies, summary.total_anomalies || 0);
    }
    
    if (elements.anomalyRate) {
        const rate = ((summary.anomaly_rate || 0) * 100).toFixed(2);
        animateCounter(elements.anomalyRate, rate, '%');
    }
    
    if (elements.symbolsCount) {
        const count = summary.symbols_analyzed ? summary.symbols_analyzed.length : 0;
        animateCounter(elements.symbolsCount, count);
    }
}

/**
 * Update symbol filter dropdown
 */
function updateSymbolFilter(symbols) {
    const select = document.getElementById('symbolFilter');
    if (!select || !symbols) return;
    
    // Store current selection
    const currentSelection = select.value;
    
    // Clear existing options except "All Symbols"
    while (select.children.length > 1) {
        select.removeChild(select.lastChild);
    }
    
    // Add symbol options
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol;
        option.textContent = symbol;
        select.appendChild(option);
    });
    
    // Restore selection if it still exists
    if (symbols.includes(currentSelection)) {
        select.value = currentSelection;
    } else {
        select.value = 'all';
        currentSymbol = 'all';
    }
}

/**
 * Update symbol summary table
 */
function updateSymbolTable(symbolsData) {
    const tbody = document.querySelector('#symbolTable tbody');
    if (!tbody || !symbolsData) return;
    
    tbody.innerHTML = '';
    
    Object.entries(symbolsData).forEach(([symbol, data]) => {
        const row = createSymbolTableRow(symbol, data);
        tbody.appendChild(row);
    });
    
    // Add hover effects
    tbody.addEventListener('mouseover', function(e) {
        if (e.target.closest('tr')) {
            e.target.closest('tr').classList.add('table-hover-highlight');
        }
    });
    
    tbody.addEventListener('mouseout', function(e) {
        if (e.target.closest('tr')) {
            e.target.closest('tr').classList.remove('table-hover-highlight');
        }
    });
}

/**
 * Create a table row for symbol data
 */
function createSymbolTableRow(symbol, data) {
    const row = document.createElement('tr');
    const riskLevel = data.latest_risk_level || 'UNKNOWN';
    const riskClass = getRiskBadgeClass(riskLevel);
    
    row.innerHTML = `
        <td><strong>${symbol}</strong></td>
        <td>${formatNumber(data.total_observations || 0)}</td>
        <td>${formatNumber(data.anomalies || 0)}</td>
        <td>${formatDecimal(data.avg_ensemble_score || 0, 3)}</td>
        <td>${formatDecimal(data.max_ensemble_score || 0, 3)}</td>
        <td>${formatNumber(data.high_risk_days || 0)}</td>
        <td><span class="badge ${riskClass}">${riskLevel}</span></td>
        <td>
            <div class="btn-group" role="group">
                <button class="btn btn-sm btn-outline-primary" onclick="viewSymbolDetails('${symbol}')" title="View Details">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-info" onclick="loadSymbolVisualization('${symbol}')" title="View Charts">
                    <i class="fas fa-chart-line"></i>
                </button>
            </div>
        </td>
    `;
    
    // Add fade-in animation
    row.classList.add('fade-in');
    
    return row;
}

/**
 * Update alerts section
 */
function updateAlerts(alerts) {
    const alertSection = document.getElementById('alertSection');
    const alertsList = document.getElementById('alertsList');
    
    if (!alerts || alerts.length === 0) {
        if (alertSection) alertSection.style.display = 'none';
        return;
    }
    
    if (alertSection) alertSection.style.display = 'block';
    if (!alertsList) return;
    
    alertsList.innerHTML = '';
    
    // Group alerts by risk level
    const groupedAlerts = groupAlertsByRisk(alerts);
    
    Object.entries(groupedAlerts).forEach(([riskLevel, riskAlerts]) => {
        const alertGroup = createAlertGroup(riskLevel, riskAlerts);
        alertsList.appendChild(alertGroup);
    });
}

/**
 * Group alerts by risk level
 */
function groupAlertsByRisk(alerts) {
    return alerts.reduce((groups, alert) => {
        const risk = determineAlertRisk(alert);
        if (!groups[risk]) groups[risk] = [];
        groups[risk].push(alert);
        return groups;
    }, {});
}

/**
 * Create alert group element
 */
function createAlertGroup(riskLevel, alerts) {
    const group = document.createElement('div');
    group.className = 'alert-group mb-3';
    
    const header = document.createElement('h6');
    header.className = `text-${getRiskColor(riskLevel)}`;
    header.innerHTML = `<i class="fas fa-${getRiskIcon(riskLevel)} me-2"></i>${riskLevel} Risk Alerts`;
    group.appendChild(header);
    
    alerts.slice(0, 5).forEach(alert => { // Limit to 5 alerts per group
        const alertElement = createAlertElement(alert);
        group.appendChild(alertElement);
    });
    
    if (alerts.length > 5) {
        const moreInfo = document.createElement('small');
        moreInfo.className = 'text-muted';
        moreInfo.textContent = `... and ${alerts.length - 5} more alerts`;
        group.appendChild(moreInfo);
    }
    
    return group;
}

/**
 * Create individual alert element
 */
function createAlertElement(alert) {
    const alertDiv = document.createElement('div');
    const alertClass = getAlertClass(alert.type);
    
    alertDiv.className = `alert ${alertClass} alert-dismissible fade show mb-2`;
    alertDiv.innerHTML = `
        <div class="d-flex justify-content-between align-items-start">
            <div>
                <strong>${alert.symbol}</strong> - ${alert.message}
                <br><small class="text-muted">${formatDate(alert.date)}</small>
            </div>
            <span class="badge bg-secondary">
                Score: ${formatDecimal(alert.ensemble_score || 0, 2)}
            </span>
        </div>
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    return alertDiv;
}

/**
 * Load visualizations for specified symbol
 */
async function loadVisualizationsForSymbol(symbol) {
    try {
        showVisualizationLoading();
        
        const response = await fetchWithRetry(`${API_ENDPOINTS.visualizations}/${symbol}`);
        
        if (response.success && response.visualizations) {
            updateVisualizationPlots(response.visualizations);
            showSuccessToast(`Visualizations loaded for ${symbol === 'all' ? 'all symbols' : symbol}`);
        } else {
            throw new Error(response.error || 'Failed to load visualizations');
        }
        
    } catch (error) {
        console.error('Error loading visualizations:', error);
        showVisualizationError('Error loading visualizations: ' + error.message);
        showErrorToast('Failed to load visualizations');
    } finally {
        hideVisualizationLoading();
    }
}

/**
 * Update visualization plots
 */
function updateVisualizationPlots(visualizations) {
    const plotContainers = {
        timeseries: document.getElementById('timeseriesPlot'),
        distribution: document.getElementById('distributionPlot'),
        correlation: document.getElementById('correlationPlot'),
        riskLevels: document.getElementById('riskLevelsPlot')
    };
    
    Object.entries(visualizations).forEach(([plotType, htmlContent]) => {
        const container = plotContainers[plotType];
        if (container && htmlContent) {
            container.innerHTML = htmlContent;
            
            // Add fade-in animation
            container.classList.add('fade-in');
            
            // Store chart reference if needed
            if (window.Plotly) {
                const plotDiv = container.querySelector('.plotly-graph-div');
                if (plotDiv) {
                    chartInstances[plotType] = plotDiv;
                }
            }
        }
    });
}

/**
 * View detailed analysis for a symbol
 */
async function viewSymbolDetails(symbol) {
    try {
        const response = await fetchWithRetry(`${API_ENDPOINTS.symbol}/${symbol}`);
        
        if (response.success && response.analysis) {
            displaySymbolDetailsModal(response.analysis);
        } else {
            throw new Error(response.error || 'Failed to load symbol details');
        }
        
    } catch (error) {
        console.error('Error loading symbol details:', error);
        showErrorToast('Failed to load symbol details: ' + error.message);
    }
}

/**
 * Load visualization for specific symbol
 */
function loadSymbolVisualization(symbol) {
    const symbolFilter = document.getElementById('symbolFilter');
    if (symbolFilter) {
        symbolFilter.value = symbol;
        currentSymbol = symbol;
        loadVisualizationsForSymbol(symbol);
    }
}

/**
 * Display symbol details in modal
 */
function displaySymbolDetailsModal(analysis) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('symbolDetailsModal');
    if (!modal) {
        modal = createSymbolDetailsModal();
        document.body.appendChild(modal);
    }
    
    // Update modal content
    updateSymbolDetailsModal(modal, analysis);
    
    // Show modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

/**
 * Create symbol details modal
 */
function createSymbolDetailsModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'symbolDetailsModal';
    modal.tabIndex = -1;
    
    modal.innerHTML = `
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Symbol Analysis Details</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div id="symbolDetailsContent">
                        <div class="text-center py-4">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" onclick="exportSymbolData()">Export Data</button>
                </div>
            </div>
        </div>
    `;
    
    return modal;
}

/**
 * Update symbol details modal content
 */
function updateSymbolDetailsModal(modal, analysis) {
    const content = modal.querySelector('#symbolDetailsContent');
    if (!content) return;
    
    content.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6><i class="fas fa-chart-bar me-2"></i>${analysis.symbol} Summary</h6>
                    </div>
                    <div class="card-body">
                        <ul class="list-unstyled">
                            <li><strong>Total Observations:</strong> ${formatNumber(analysis.summary.total_observations)}</li>
                            <li><strong>Anomalies:</strong> ${formatNumber(analysis.summary.anomalies)}</li>
                            <li><strong>Average Score:</strong> ${formatDecimal(analysis.summary.avg_score, 3)}</li>
                            <li><strong>Maximum Score:</strong> ${formatDecimal(analysis.summary.max_score, 3)}</li>
                            <li><strong>Current Risk:</strong> 
                                <span class="badge ${getRiskBadgeClass(analysis.summary.current_risk)}">
                                    ${analysis.summary.current_risk}
                                </span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h6><i class="fas fa-chart-line me-2"></i>Recent Activity</h6>
                    </div>
                    <div class="card-body">
                        <div id="recentActivityChart" style="height: 300px;"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-3">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h6><i class="fas fa-table me-2"></i>Recent Data Points</h6>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Close Price</th>
                                        <th>Volume</th>
                                        <th>Returns</th>
                                        <th>Ensemble Score</th>
                                        <th>Risk Level</th>
                                    </tr>
                                </thead>
                                <tbody id="symbolDataTable">
                                    ${generateSymbolDataRows(analysis.data)}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Load symbol-specific visualizations if available
    if (analysis.visualizations) {
        setTimeout(() => {
            Object.entries(analysis.visualizations).forEach(([plotType, htmlContent]) => {
                const container = content.querySelector(`#${plotType}Plot`);
                if (container && htmlContent) {
                    container.innerHTML = htmlContent;
                }
            });
        }, 100);
    }
}

/**
 * Generate symbol data table rows
 */
function generateSymbolDataRows(data) {
    if (!data || !Array.isArray(data)) return '<tr><td colspan="6">No data available</td></tr>';
    
    return data.slice(-10).map(row => { // Show last 10 rows
        const riskLevel = row.Risk_Level || 'UNKNOWN';
        const riskClass = getRiskBadgeClass(riskLevel);
        
        return `
            <tr>
                <td>${formatDate(row.Date || row.index)}</td>
                <td>${formatCurrency(row.Close || 0)}</td>
                <td>${formatNumber(row.Volume || 0)}</td>
                <td class="${(row.Returns || 0) >= 0 ? 'text-success' : 'text-danger'}">
                    ${formatPercentage(row.Returns || 0)}
                </td>
                <td>${formatDecimal(row.Ensemble_Score || 0, 3)}</td>
                <td><span class="badge ${riskClass}">${riskLevel}</span></td>
            </tr>
        `;
    }).join('');
}

/**
 * Start auto-refresh functionality
 */
function startAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
    
    refreshInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            loadDashboardData();
        }
    }, CONFIG.refreshInterval);
}

/**
 * Stop auto-refresh
 */
function stopAutoRefresh() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }
}

/**
 * Fetch data with retry logic
 */
async function fetchWithRetry(url, options = {}, retries = CONFIG.maxRetries) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
        
    } catch (error) {
        if (retries > 0) {
            console.warn(`Fetch failed, retrying... (${retries} attempts left)`);
            await delay(CONFIG.retryDelay);
            return fetchWithRetry(url, options, retries - 1);
        } else {
            throw error;
        }
    }
}

/**
 * Utility Functions
 */

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatDecimal(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function formatPercentage(num) {
    return (num * 100).toFixed(2) + '%';
}

function formatCurrency(num, currency = '$') {
    return currency + parseFloat(num).toFixed(2);
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString();
}

function getRiskBadgeClass(riskLevel) {
    const classes = {
        'CRITICAL': 'bg-danger',
        'HIGH': 'bg-warning text-dark',
        'MEDIUM': 'bg-info',
        'LOW': 'bg-secondary',
        'NORMAL': 'bg-success',
        'UNKNOWN': 'bg-light text-dark'
    };
    return classes[riskLevel] || classes['UNKNOWN'];
}

function getRiskColor(riskLevel) {
    const colors = {
        'CRITICAL': 'danger',
        'HIGH': 'warning',
        'MEDIUM': 'info',
        'LOW': 'secondary',
        'NORMAL': 'success'
    };
    return colors[riskLevel] || 'secondary';
}

function getRiskIcon(riskLevel) {
    const icons = {
        'CRITICAL': 'exclamation-triangle',
        'HIGH': 'exclamation-circle',
        'MEDIUM': 'info-circle',
        'LOW': 'check-circle',
        'NORMAL': 'check-circle'
    };
    return icons[riskLevel] || 'question-circle';
}

function determineAlertRisk(alert) {
    const score = alert.ensemble_score || 0;
    if (score >= 6.0) return 'CRITICAL';
    if (score >= 4.0) return 'HIGH';
    if (score >= 2.5) return 'MEDIUM';
    if (score >= 1.5) return 'LOW';
    return 'NORMAL';
}

function getAlertClass(alertType) {
    const classes = {
        'HIGH_RISK_ANOMALY': 'alert-danger',
        'RECENT_ANOMALY': 'alert-warning',
        'PATTERN_DETECTED': 'alert-info',
        'VOLUME_SPIKE': 'alert-warning'
    };
    return classes[alertType] || 'alert-info';
}

/**
 * Animation and UI Functions
 */

function animateCounter(element, target, suffix = '') {
    if (!element) return;
    
    const start = parseInt(element.textContent.replace(/[^\d]/g, '')) || 0;
    const duration = 1000;
    const step = (target - start) / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += step;
        if ((step > 0 && current >= target) || (step < 0 && current <= target)) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current).toLocaleString() + suffix;
    }, 16);
}

function animateUpdates() {
    const elements = document.querySelectorAll('.card, .alert, .table');
    elements.forEach((element, index) => {
        setTimeout(() => {
            element.classList.add('fade-in');
        }, index * 50);
    });
}

function showLoadingStates() {
    const loadingElements = document.querySelectorAll('.spinner-border');
    loadingElements.forEach(element => {
        element.style.display = 'block';
    });
}

function hideLoadingStates() {
    const loadingElements = document.querySelectorAll('.spinner-border');
    loadingElements.forEach(element => {
        if (element.closest('.toast') === null) {
            element.style.display = 'none';
        }
    });
}

function showVisualizationLoading() {
    const plotContainers = [
        'timeseriesPlot', 'distributionPlot', 
        'correlationPlot', 'riskLevelsPlot'
    ];
    
    plotContainers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-3 text-muted">Loading visualization...</p>
                </div>
            `;
        }
    });
}

function hideVisualizationLoading() {
    // Loading will be hidden when content is updated
}

function showVisualizationError(message) {
    const plotContainers = [
        'timeseriesPlot', 'distributionPlot', 
        'correlationPlot', 'riskLevelsPlot'
    ];
    
    plotContainers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container && container.innerHTML.includes('spinner-border')) {
            container.innerHTML = `
                <div class="text-center py-5">
                    <i class="fas fa-exclamation-triangle fa-3x text-warning mb-3"></i>
                    <p class="text-muted">${message}</p>
                    <button class="btn btn-outline-primary btn-sm" onclick="loadVisualizationsForSymbol('${currentSymbol}')">
                        <i class="fas fa-retry me-1"></i>Retry
                    </button>
                </div>
            `;
        }
    });
}

function showErrorState(message) {
    const container = document.querySelector('.container-fluid');
    if (container) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger alert-dismissible fade show';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        container.insertBefore(errorDiv, container.firstChild);
    }
}

/**
 * Toast Notifications
 */

function showSuccessToast(message) {
    showToast(message, 'success');
}

function showErrorToast(message) {
    showToast(message, 'danger');
}

function showInfoToast(message) {
    showToast(message, 'info');
}

function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '1080';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${getToastIcon(type)} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    // Show toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: CONFIG.toastTimeout
    });
    bsToast.show();
    
    // Remove toast element after it's hidden
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function getToastIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

/**
 * Event Handlers
 */

function handleWindowResize() {
    // Resize charts if Plotly is available
    if (window.Plotly && chartInstances) {
        Object.values(chartInstances).forEach(chart => {
            if (chart) {
                window.Plotly.Plots.resize(chart);
            }
        });
    }
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function cleanup() {
    // Stop auto-refresh
    stopAutoRefresh();
    
    // Clear chart instances
    chartInstances = {};
    
    // Clear any running timers
    if (window.dashboardTimers) {
        window.dashboardTimers.forEach(timer => clearTimeout(timer));
    }
}

/**
 * Export Functions
 */

function exportSymbolData() {
    if (!currentData) {
        showErrorToast('No data available to export');
        return;
    }
    
    const dataStr = JSON.stringify(currentData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `insider_trading_data_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showSuccessToast('Data exported successfully');
}

// Global exports for HTML onclick handlers
window.viewSymbolDetails = viewSymbolDetails;
window.loadSymbolVisualization = loadSymbolVisualization;
window.exportSymbolData = exportSymbolData;
window.loadVisualizationsForSymbol = loadVisualizationsForSymbol;
