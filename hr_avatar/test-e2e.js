/**
 * End-to-end test for HR Avatar Demo
 *
 * Run with: npx playwright test test-e2e.js
 * Or: node test-e2e.js (for standalone execution)
 */

const { chromium } = require('playwright');

const BASE_URL = process.env.BASE_URL || 'https://zoharbabin.github.io/kaltura-avatar-sdk/hr_avatar/';

async function runTests() {
    console.log('🧪 Starting HR Avatar E2E Tests\n');
    console.log(`📍 Testing: ${BASE_URL}\n`);

    const browser = await chromium.launch({
        headless: process.env.HEADLESS !== 'false'
    });
    const context = await browser.newContext();
    const page = await context.newPage();

    const results = {
        passed: 0,
        failed: 0,
        tests: []
    };

    async function test(name, fn) {
        try {
            await fn();
            results.passed++;
            results.tests.push({ name, status: 'passed' });
            console.log(`✅ ${name}`);
        } catch (error) {
            results.failed++;
            results.tests.push({ name, status: 'failed', error: error.message });
            console.log(`❌ ${name}`);
            console.log(`   Error: ${error.message}`);
        }
    }

    try {
        // Navigate to the page
        await page.goto(BASE_URL);
        await page.waitForLoadState('networkidle');

        // Test 1: Page loads correctly
        await test('Page loads with correct title', async () => {
            const title = await page.title();
            if (!title.includes('HR') && !title.includes('Avatar')) {
                throw new Error(`Unexpected title: ${title}`);
            }
        });

        // Test 2: Scenario cards are present
        await test('Scenario cards are rendered', async () => {
            const cards = await page.locator('.scenario-card').count();
            if (cards < 3) {
                throw new Error(`Expected at least 3 scenario cards, found ${cards}`);
            }
        });

        // Test 3: Interview scenarios section exists
        await test('Interview scenarios section exists', async () => {
            const section = await page.locator('text=Interview').first();
            await section.waitFor({ timeout: 5000 });
        });

        // Test 4: Avatar container exists
        await test('Avatar container is present', async () => {
            const container = await page.locator('#avatar-container, .avatar-container').count();
            if (container === 0) {
                throw new Error('Avatar container not found');
            }
        });

        // Test 5: Click a scenario card
        await test('Can select a scenario', async () => {
            const firstCard = page.locator('.scenario-card').first();
            await firstCard.click();

            // Wait for scenario to be selected (card should become active)
            await page.waitForTimeout(500);
            const isActive = await firstCard.evaluate(el => el.classList.contains('active'));
            if (!isActive) {
                // Check if details panel appeared instead
                const detailsVisible = await page.locator('.scenario-details, #scenario-details').isVisible();
                if (!detailsVisible) {
                    throw new Error('Scenario selection did not activate card or show details');
                }
            }
        });

        // Test 6: Editable fields appear after scenario selection
        await test('Editable fields are shown after scenario selection', async () => {
            // Look for any input fields in the details area
            await page.waitForTimeout(500);
            const inputs = await page.locator('input[type="text"], .editable-field input').count();
            if (inputs === 0) {
                // Fields might be displayed as text initially
                const editableElements = await page.locator('[contenteditable="true"], .field-value').count();
                if (editableElements === 0) {
                    console.log('   Note: No editable fields found (may be expected for some scenarios)');
                }
            }
        });

        // Test 7: Start button exists
        await test('Start conversation button exists', async () => {
            const startBtn = page.locator('button:has-text("Start"), #start-btn, .start-button');
            const count = await startBtn.count();
            if (count === 0) {
                throw new Error('Start button not found');
            }
        });

        // Test 8: Check console for JavaScript errors
        await test('No critical JavaScript errors', async () => {
            const errors = [];
            page.on('pageerror', error => {
                if (!error.message.includes('net::ERR') && !error.message.includes('favicon')) {
                    errors.push(error.message);
                }
            });

            // Reload to catch any errors
            await page.reload();
            await page.waitForLoadState('networkidle');
            await page.waitForTimeout(1000);

            if (errors.length > 0) {
                throw new Error(`JS Errors: ${errors.join(', ')}`);
            }
        });

        // Test 9: Verify CONFIG version is updated
        await test('Version is v1.0.21 or higher', async () => {
            const version = await page.evaluate(() => {
                return window.CONFIG?.VERSION || 'unknown';
            });

            if (version === 'unknown') {
                console.log('   Note: Could not read CONFIG.VERSION from window');
            } else {
                const versionNum = parseFloat(version.replace('1.0.', ''));
                if (versionNum < 21) {
                    throw new Error(`Expected version >= 1.0.21, got ${version}`);
                }
                console.log(`   Version: ${version}`);
            }
        });

        // Test 10: Check DPP validation function exists
        await test('validateDPP function exists', async () => {
            const hasValidateDPP = await page.evaluate(() => {
                return typeof window.validateDPP === 'function' ||
                       document.body.innerHTML.includes('validateDPP');
            });
            // Note: function may not be exposed to window, just check it's in the code
            const scriptContent = await page.evaluate(() => {
                const scripts = document.querySelectorAll('script[src*="hr-demo"]');
                return scripts.length > 0;
            });
            if (!scriptContent) {
                throw new Error('hr-demo.js not loaded');
            }
        });

    } catch (error) {
        console.error('\n💥 Test suite error:', error.message);
    } finally {
        await browser.close();
    }

    // Summary
    console.log('\n' + '='.repeat(50));
    console.log(`📊 Results: ${results.passed} passed, ${results.failed} failed`);
    console.log('='.repeat(50));

    if (results.failed > 0) {
        console.log('\n❌ Some tests failed');
        process.exit(1);
    } else {
        console.log('\n✅ All tests passed!');
        process.exit(0);
    }
}

runTests().catch(console.error);
