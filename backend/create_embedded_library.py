#!/usr/bin/env python3
"""
Create a self-contained library.html with embedded real NASA data sample
"""

import json

def create_embedded_library():
    # Read the full catalog
    try:
        with open('real_nasa_planet_catalog.json', 'r') as f:
            catalog = json.load(f)
    except:
        print("Error: real_nasa_planet_catalog.json not found")
        return
    
    # Get a representative sample of real planets
    all_planets = catalog['planets']
    
    # Sample strategy: get diverse examples
    sample_planets = []
    
    # Get first 50 TESS planets
    tess_planets = [p for p in all_planets if p['mission'] == 'TESS'][:50]
    sample_planets.extend(tess_planets)
    
    # Get first 50 Kepler planets
    kepler_planets = [p for p in all_planets if p['mission'] == 'Kepler'][:50]
    sample_planets.extend(kepler_planets)
    
    # Create sample catalog
    sample_catalog = {
        "planets": sample_planets,
        "statistics": {
            "tess_total": len(tess_planets),
            "kepler_total": len(kepler_planets), 
            "total_confirmed": len([p for p in sample_planets if p['disposition'] in ['CONFIRMED', 'PC']]),
            "habitable_zone": len([p for p in sample_planets if p.get('equilibrium_temp') and 200 <= p['equilibrium_temp'] <= 400])
        }
    }
    
    # Convert to JavaScript
    js_data = f"const EMBEDDED_NASA_CATALOG = {json.dumps(sample_catalog, indent=2)};"
    
    # Read the library template
    with open('library_fixed.html', 'r') as f:
        html_content = f.read()
    
    # Insert the embedded data before the main script
    script_insert = f"""
    <script>
        // Embedded Real NASA Data Sample (100 real planets)
        {js_data}
    </script>
    <script src="assets/js/main.js"></script>"""
    
    # Replace the script tag
    html_content = html_content.replace('<script src="assets/js/main.js"></script>', script_insert)
    
    # Update the initialization function to use embedded data first
    new_init = '''        async function initializeLibrary() {
            // First try embedded data
            if (typeof EMBEDDED_NASA_CATALOG !== 'undefined' && EMBEDDED_NASA_CATALOG.planets) {
                console.log('📦 Using embedded real NASA data sample');
                
                planetCatalog = EMBEDDED_NASA_CATALOG.planets;
                filteredPlanets = [...planetCatalog];
                
                updateStatistics(EMBEDDED_NASA_CATALOG.statistics);
                setupEventListeners();
                displayPlanets();
                
                document.getElementById('loading-section').style.display = 'none';
                document.getElementById('planet-grid').style.display = 'grid';
                document.getElementById('pagination-section').style.display = 'block';
                
                console.log(`✅ Loaded ${planetCatalog.length} real NASA planets (embedded sample)`);
                return;
            }
            
            // Try multiple paths to find the full NASA catalog
            const possiblePaths = [
                'real_nasa_planet_catalog.json',
                '../real_nasa_planet_catalog.json',
                'user_input_files/real_nasa_planet_catalog.json',
                './real_nasa_planet_catalog.json'
            ];
            
            let catalogData = null;
            let loadedFrom = '';
            
            for (const path of possiblePaths) {
                try {
                    console.log(`🔍 Trying to load full NASA catalog from: ${path}`);
                    const response = await fetch(path);
                    
                    if (response.ok) {
                        catalogData = await response.json();
                        loadedFrom = path;
                        console.log(`✅ Successfully loaded full catalog from: ${path}`);
                        break;
                    }
                } catch (error) {
                    console.log(`❌ Failed to load from ${path}:`, error.message);
                    continue;
                }
            }
            
            if (catalogData && catalogData.planets) {
                planetCatalog = catalogData.planets;
                filteredPlanets = [...planetCatalog];
                
                updateStatistics(catalogData.statistics);
                setupEventListeners();
                displayPlanets();
                
                document.getElementById('loading-section').style.display = 'none';
                document.getElementById('planet-grid').style.display = 'grid';
                document.getElementById('pagination-section').style.display = 'block';
                
                console.log(`✅ Loaded ${planetCatalog.length} REAL NASA planets from ${loadedFrom}`);
                console.log(`📊 TESS: ${catalogData.statistics.tess_total}, Kepler: ${catalogData.statistics.kepler_total}`);
                
            } else {
                console.error('❌ Could not load real NASA catalog from any path');
                
                // Fallback sample data if embedded data also fails
                generateSampleRealData();
                
                document.getElementById('loading-section').innerHTML = `
                    <div class="loading-spinner">
                        <i class="fas fa-info-circle" style="color: #ffa726;"></i>
                    </div>
                    <h3>Using Fallback Sample Data</h3>
                    <p>Could not load full NASA catalog. Showing sample real data structure.</p>
                `;
                
                setTimeout(() => {
                    document.getElementById('loading-section').style.display = 'none';
                    document.getElementById('planet-grid').style.display = 'grid';
                    document.getElementById('pagination-section').style.display = 'block';
                }, 2000);
            }
        }'''
    
    # Replace the initialization function
    html_content = html_content.replace(
        html_content[html_content.find('async function initializeLibrary()'):html_content.find('function updateStatistics(')],
        new_init + '\n        '
    )
    
    # Write the self-contained version
    with open('library_self_contained.html', 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created library_self_contained.html with {len(sample_planets)} embedded real NASA planets")
    print(f"   - {len(tess_planets)} TESS planets")
    print(f"   - {len(kepler_planets)} Kepler planets")
    print(f"   - {sample_catalog['statistics']['total_confirmed']} confirmed planets")

if __name__ == "__main__":
    create_embedded_library()