// Missing Chart Management Functions for Display.html
// Add these functions to your main.js file or include this as a separate script

// Global object to store chart instances
let chartInstances = {
    transitChart: null,
    periodogramChart: null,
    phaseChart: null,
    binnedPhaseChart: null
};

// Function to destroy existing charts before creating new ones
function destroyExistingCharts() {
    console.log('🧹 Destroying existing charts...');
    
    Object.keys(chartInstances).forEach(chartKey => {
        if (chartInstances[chartKey]) {
            try {
                chartInstances[chartKey].destroy();
                console.log(`✅ Destroyed ${chartKey}`);
            } catch (error) {
                console.warn(`⚠️ Error destroying ${chartKey}:`, error);
            }
            chartInstances[chartKey] = null;
        }
    });
    
    console.log('✅ All existing charts destroyed');
}

// Hash function to generate unique identifiers
function hashCode(str) {
    let hash = 0;
    if (str.length === 0) return hash;
    
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    
    return hash;
}

// Enhanced error handling for chart creation
function createChartSafely(canvasId, chartConfig, chartName) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.error(`❌ Canvas element '${canvasId}' not found for ${chartName}`);
            return null;
        }
        
        // Get 2D context to ensure canvas is ready
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.error(`❌ Could not get 2D context for ${canvasId}`);
            return null;
        }
        
        console.log(`🎨 Creating ${chartName} chart on canvas ${canvasId}`);
        const chart = new Chart(ctx, chartConfig);
        
        console.log(`✅ Successfully created ${chartName} chart`);
        return chart;
        
    } catch (error) {
        console.error(`❌ Error creating ${chartName} chart:`, error);
        return null;
    }
}

// Debug function to check chart container visibility
function debugChartContainers() {
    console.log('🔍 Debugging chart containers...');
    
    const containers = [
        'transit-curve',
        'periodogram', 
        'phaseChart',
        'binnedPhaseChart'
    ];
    
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            const rect = container.getBoundingClientRect();
            console.log(`📊 Container ${containerId}:`, {
                exists: true,
                visible: rect.width > 0 && rect.height > 0,
                dimensions: `${rect.width}x${rect.height}`,
                position: `${rect.left},${rect.top}`
            });
        } else {
            console.log(`❌ Container ${containerId}: NOT FOUND`);
        }
    });
}

// Function to ensure Chart.js is loaded
function ensureChartJSLoaded() {
    return new Promise((resolve, reject) => {
        if (typeof Chart !== 'undefined') {
            console.log('✅ Chart.js is already loaded');
            resolve();
            return;
        }
        
        console.log('⏳ Waiting for Chart.js to load...');
        let attempts = 0;
        const maxAttempts = 50; // 5 seconds
        
        const checkChart = setInterval(() => {
            attempts++;
            if (typeof Chart !== 'undefined') {
                console.log('✅ Chart.js loaded successfully');
                clearInterval(checkChart);
                resolve();
            } else if (attempts >= maxAttempts) {
                console.error('❌ Chart.js failed to load within 5 seconds');
                clearInterval(checkChart);
                reject(new Error('Chart.js not loaded'));
            }
        }, 100);
    });
}

// Enhanced initialization function for display page
async function initializeDisplayCharts() {
    console.log('🚀 Initializing display charts...');
    
    try {
        // Ensure Chart.js is loaded
        await ensureChartJSLoaded();
        
        // Debug chart containers
        debugChartContainers();
        
        // Destroy any existing charts
        destroyExistingCharts();
        
        console.log('✅ Display charts initialization complete');
        
    } catch (error) {
        console.error('❌ Display charts initialization failed:', error);
    }
}

// Call initialization when DOM is ready (for display.html)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDisplayCharts);
} else {
    initializeDisplayCharts();
}

console.log('📊 Chart management functions loaded successfully!');