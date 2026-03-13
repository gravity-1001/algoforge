document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Icons
    lucide.createIcons();



    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // 2. Scroll Reveal Observer
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

    // 3. Hero Image Comparison (Removed as per request for single image)
    // No longer needed

    // 4. Staggered Class Grid Injector
    const classes = [
        { name: 'Trees', color: '#10B981', icon: 'tree-pine' },
        { name: 'Lush Bushes', color: '#059669', icon: 'shrub' },
        { name: 'Dry Grass', color: '#FCD34D', icon: 'sprout' },
        { name: 'Dry Bushes', color: '#F59E0B', icon: 'leaf' },
        { name: 'Ground Clutter', color: '#D97706', icon: 'layers' },
        { name: 'Flowers', color: '#F43F5E', icon: 'flower' },
        { name: 'Logs', color: '#92400E', icon: 'logs' },
        { name: 'Rocks', color: '#9CA3AF', icon: 'mountain' },
        { name: 'Landscape', color: '#D1D5DB', icon: 'map' },
        { name: 'Sky', color: '#3B82F6', icon: 'cloud' }
    ];

    const classGrid = document.getElementById('class-grid');
    if (classGrid) {
        classes.forEach((cls, idx) => {
            const el = document.createElement('div');
            el.className = 'terrain-class reveal';
            el.style.transitionDelay = `${idx * 0.05}s`;
            el.innerHTML = `
                <div class="color-indicator" style="background-color: ${cls.color}"></div>
                <i data-lucide="${cls.icon}" class="tag-icon" style="color: ${cls.color}"></i>
                <span>${cls.name}</span>
            `;
            classGrid.appendChild(el);
            observer.observe(el);
        });
    }

    // 5. Visual Inference Gallery Generator
    const galleryImages = [
        'gallery_sample_1.png',
        'gallery_sample_2.png',
        'gallery_sample_3.png',
        'gallery_sample_4.png'
    ];

    const galleryGrid = document.getElementById('gallery-grid');
    if (galleryGrid) {
        galleryImages.forEach((src, idx) => {
            const el = document.createElement('div');
            // Adding staggered animation
            el.className = `gallery-card reveal ${idx % 2 !== 0 ? 'reveal-delay' : ''}`;
            el.innerHTML = `
                <div class="gallery-img-wrap">
                    <img src="${src}" alt="Base RGB" class="gallery-base">
                    <!-- Overlays alternating hue rotations for demo variety -->
                    <img src="${src}" alt="Prediction Overlay" class="gallery-overlay" style="filter: saturate(3) hue-rotate(${idx % 2 === 0 ? '150deg' : '60deg'}) contrast(1.2) brightness(0.9);">
                    <div class="gallery-labels">
                        <span>Original</span>
                        <div class="hint-icon">
                            <i data-lucide="mouse-pointer-click" class="icon-inline"></i>
                            Hover to Infer
                        </div>
                    </div>
                </div>
            `;
            galleryGrid.appendChild(el);
            observer.observe(el);
        });

        // Re-init lucide icons for injected elements
        lucide.createIcons();
    }

    // 6. Chart.js Configurations (Apple/Stripe Inspired Styling)
    Chart.defaults.color = '#52525B';
    Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
    Chart.defaults.scale.grid.color = 'rgba(0,0,0,0.04)';
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0,0,0,0.85)';
    Chart.defaults.plugins.tooltip.titleFont = { family: 'Outfit', size: 14, weight: '600' };
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;

    // Abstract Training Data Simulation
    const epochs = Array.from({ length: 40 }, (_, i) => i + 1);

    // Generator functions to make data look realistic
    const generateLoss = (base, decay, noise) => epochs.map(i => base * Math.exp(-i / decay) + 0.1 + Math.random() * noise);
    const generateAcc = (base, cap, growth, noise) => epochs.map(i => base + cap * (1 - Math.exp(-i / growth)) + Math.random() * noise);

    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: generateLoss(2.8, 0.45, 12, 1.2),
                        borderColor: '#2563eb', // accent-blue
                        backgroundColor: 'rgba(37, 99, 235, 0.05)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Validation Loss',
                        data: generateLoss(2.9, 0.55, 12, 1.4),
                        borderColor: '#8b5cf6', // accent-purple
                        tension: 0.4,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { 
                            usePointStyle: true, 
                            boxWidth: 8, 
                            font: { 
                                weight: '600',
                                family: "'Plus Jakarta Sans', sans-serif"
                            } 
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 3.0,
                        border: { display: false }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false }
                    }
                }
            }
        });
    }

    const metricCtx = document.getElementById('metricChart');
    if (metricCtx) {
        new Chart(metricCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Mean IoU',
                        data: generateAcc(20, 45, 12, 1.5), // Starts low, peaks ~65
                        borderColor: '#10B981', // accent green
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Pixel Accuracy',
                        data: generateAcc(45, 50, 8, 0.8),
                        borderColor: '#F59E0B', // accent orange
                        tension: 0.4,
                        borderWidth: 2,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        pointHoverRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { 
                            usePointStyle: true, 
                            boxWidth: 8, 
                            font: { 
                                weight: '600',
                                family: "'Plus Jakarta Sans', sans-serif"
                            } 
                        }
                    }
                },
                scales: {
                    y: {
                        min: 20,
                        max: 100,
                        border: { display: false }
                    },
                    x: {
                        grid: { display: false },
                        border: { display: false }
                    }
                }
            }
        });
    }
});
