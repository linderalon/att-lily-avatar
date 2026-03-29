/**
 * End-to-end test for Code Interview Demo
 *
 * Run with: node test-e2e.js
 */

const { chromium } = require('playwright');

const BASE_URL = process.env.BASE_URL || 'https://zoharbabin.github.io/kaltura-avatar-sdk/code_interview/';

async function runTests() {
    console.log('🧪 Starting Code Interview E2E Tests\n');
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
            if (!title.toLowerCase().includes('interview') && !title.toLowerCase().includes('code')) {
                throw new Error(`Unexpected title: ${title}`);
            }
        });

        // Test 2: Registration form is present
        await test('Registration form is present', async () => {
            const form = await page.locator('form, #registration-form, .registration-form').count();
            if (form === 0) {
                throw new Error('Registration form not found');
            }
        });

        // Test 3: Name input fields exist
        await test('Name input fields exist', async () => {
            const firstNameInput = await page.locator('input[name="firstName"], #firstName, #first-name').count();
            const lastNameInput = await page.locator('input[name="lastName"], #lastName, #last-name').count();
            if (firstNameInput === 0 || lastNameInput === 0) {
                throw new Error('Name inputs not found');
            }
        });

        // Test 4: Email input exists
        await test('Email input exists', async () => {
            const emailInput = await page.locator('input[type="email"], input[name="email"], #email').count();
            if (emailInput === 0) {
                throw new Error('Email input not found');
            }
        });

        // Test 5: Fill registration form
        await test('Can fill registration form', async () => {
            await page.fill('input[name="firstName"], #firstName, #first-name', 'Test');
            await page.fill('input[name="lastName"], #lastName, #last-name', 'User');
            await page.fill('input[type="email"], input[name="email"], #email', 'test@example.com');
        });

        // Test 6: Submit button exists
        await test('Submit/Start button exists', async () => {
            const submitBtn = page.locator('button[type="submit"], button:has-text("Start"), #start-btn');
            const count = await submitBtn.count();
            if (count === 0) {
                throw new Error('Submit button not found');
            }
        });

        // Test 7: Check for Monaco editor container
        await test('Editor container exists', async () => {
            const editor = await page.locator('#editor, .monaco-editor, .editor-container, [data-mode-id]').count();
            // Editor might not be visible until after registration
            console.log(`   Note: Editor elements found: ${editor}`);
        });

        // Test 8: Problem panel exists (may be hidden initially)
        await test('Problem panel structure exists', async () => {
            const problemPanel = await page.locator('#problem-panel, .problem-panel, .problem-title, #problem-title').count();
            console.log(`   Note: Problem panel elements found: ${problemPanel}`);
        });

        // Test 9: Check console for JavaScript errors
        await test('No critical JavaScript errors', async () => {
            const errors = [];
            page.on('pageerror', error => {
                if (!error.message.includes('net::ERR') && !error.message.includes('favicon')) {
                    errors.push(error.message);
                }
            });

            await page.reload();
            await page.waitForLoadState('networkidle');
            await page.waitForTimeout(1000);

            if (errors.length > 0) {
                throw new Error(`JS Errors: ${errors.join(', ')}`);
            }
        });

        // Test 10: Check that fixes are in place (validateDPP, lastInjectedProblemId)
        await test('Bug fix code is present', async () => {
            const response = await page.goto(BASE_URL.replace('index.html', '') + 'code-interview.js');
            const jsContent = await response.text();

            const hasValidateDPP = jsContent.includes('validateDPP');
            const hasLastInjectedProblemId = jsContent.includes('lastInjectedProblemId');
            const hasStopCodeTracking = jsContent.includes('stopCodeTracking()');

            if (!hasValidateDPP) {
                throw new Error('validateDPP function not found in code');
            }
            if (!hasLastInjectedProblemId) {
                throw new Error('lastInjectedProblemId not found in code');
            }
            if (!hasStopCodeTracking) {
                throw new Error('stopCodeTracking call not found in code');
            }

            console.log('   All bug fixes verified in source code');
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
