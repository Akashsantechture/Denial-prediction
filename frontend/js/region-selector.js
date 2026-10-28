/**
 * region-selector.js
 * ------------------
 * Multi-region toggle for UAE (CHI) and KSA (CCHI/Nphies).
 * 
 * Manages:
 * - Region selection and persistence (localStorage)
 * - UI updates (currency, regulator name)
 * - API header injection (X-Region)
 */

class RegionManager {
  constructor() {
    this.regions = {
      uae: {
        code: 'uae',
        name: 'UAE',
        fullName: 'United Arab Emirates',
        regulator: 'CHI',
        regulatorFull: 'Council of Health Insurance Abu Dhabi',
        currency: 'AED',
        currencySymbol: 'د.إ',
        flag: '🇦🇪',
        claimSystem: 'Thynk',
        productionReady: true
      },
      ksa: {
        code: 'ksa',
        name: 'KSA',
        fullName: 'Kingdom of Saudi Arabia',
        regulator: 'CCHI',
        regulatorFull: 'Council of Cooperative Health Insurance',
        currency: 'SAR',
        currencySymbol: 'ر.س',
        flag: '🇸🇦',
        claimSystem: 'Nphies',
        productionReady: false
      }
    };

    // Load saved region or default to UAE
    this.currentRegion = localStorage.getItem('claimai-region') || 'uae';
    
    // Validate saved region
    if (!this.regions[this.currentRegion]) {
      this.currentRegion = 'uae';
      localStorage.setItem('claimai-region', 'uae');
    }

    this.init();
  }

  init() {
    this.renderSelector();
    this.attachEventListeners();
    this.updateUI();
    
    // Log current region on page load
    console.log(`[Region] Active: ${this.getCurrentRegion().name} (${this.getCurrentRegion().regulator})`);
  }

  renderSelector() {
    const container = document.getElementById('region-selector-container');
    if (!container) {
      console.warn('[Region] Container #region-selector-container not found');
      return;
    }

    const config = this.getCurrentRegion();
    const otherCode = this.currentRegion === 'uae' ? 'ksa' : 'uae';
    const otherConfig = this.regions[otherCode];

    container.innerHTML = `
      <div class="region-toggle-wrap">
        <span class="region-label">Region:</span>
        <div class="region-switch">
          <input type="checkbox" id="region-toggle-input" ${this.currentRegion === 'ksa' ? 'checked' : ''} />
          <label for="region-toggle-input" class="region-slider">
            <span class="region-slider-option region-uae">${this.regions.uae.flag} ${this.regions.uae.name}</span>
            <span class="region-slider-option region-ksa">${this.regions.ksa.flag} ${this.regions.ksa.name}</span>
          </label>
        </div>
        ${!config.productionReady ? '' : ''}
      </div>
    `;
  }

  attachEventListeners() {
    const toggle = document.getElementById('region-toggle-input');
    if (toggle) {
      toggle.addEventListener('change', (e) => {
        const newRegion = e.target.checked ? 'ksa' : 'uae';
        this.switchRegion(newRegion);
      });
    }
  }

  switchRegion(newRegion) {
    if (!this.regions[newRegion]) {
      console.error(`[Region] Invalid region: ${newRegion}`);
      return;
    }

    if (this.currentRegion === newRegion) {
      return; // No change
    }

    const config = this.regions[newRegion];

    // Save and reload
    this.currentRegion = newRegion;
    localStorage.setItem('claimai-region', newRegion);
    
    console.log(`[Region] Switching to ${config.name} (${config.regulator})`);
    
    // Reload page to fetch region-specific data
    window.location.reload();
  }

  updateUI() {
    const config = this.getCurrentRegion();
    
    // Update page title
    document.title = `ClaimAI — Denial Prediction (${config.regulator})`;
    
    // Update any currency displays (can be expanded)
    this.updateCurrencyLabels(config.currencySymbol);
    
    // Store region config globally for other scripts
    window.CURRENT_REGION = config;
  }

  updateCurrencyLabels(symbol) {
    // Update all elements with data-currency attribute
    document.querySelectorAll('[data-currency]').forEach(el => {
      el.textContent = symbol;
    });
    
    // Update input placeholders that mention currency
    document.querySelectorAll('input[placeholder*="AED"], input[placeholder*="SAR"]').forEach(input => {
      input.placeholder = input.placeholder.replace(/AED|SAR/, symbol);
    });
  }

  getCurrentRegion() {
    return this.regions[this.currentRegion];
  }

  getRegionCode() {
    return this.currentRegion;
  }

  /**
   * Get the X-Region header value for API calls
   */
  getRegionHeader() {
    return this.currentRegion;
  }

  /**
   * Check if current region is production-ready
   */
  isProductionReady() {
    return this.getCurrentRegion().productionReady;
  }
}

// Global singleton instance
window.regionManager = new RegionManager();

// Export for module systems (optional)
if (typeof module !== 'undefined' && module.exports) {
  module.exports = RegionManager;
}
