// Takes dashboard screenshots by actually navigating the sidebar selectbox
// (the previous script used URL paths, which Streamlit ignores for a
// selectbox-driven app - so every screenshot was the Overview page).
//
// For the New Prediction page it uploads a sample CSV and clicks
// "Generate Prediction" so the screenshot shows a real prediction with a
// confidence interval and feature-importance chart.
//
//   streamlit run webapp/dashboard.py   # must already be running on :8501
//   node take_screenshots.js

const puppeteer = require('puppeteer-core');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

const BASE = 'http://localhost:8501';
const OUT_DIR = '/home/louis/GitHub_Repos/NASA-Turbofan-Twin/docs/screenshots';
const SAMPLE_CSV = path.join(OUT_DIR, 'sample_upload.csv');

// Generates the sample CSV used for the New Prediction screenshot from the
// local cleaned dataset (engines 1 and 2, first 60 cycles each - early life,
// so the prediction is non-trivial). Not committed; created on demand.
function ensureSampleCsv() {
  if (fs.existsSync(SAMPLE_CSV)) return;
  execSync(
    `python3 -c "import pandas as pd; df=pd.read_csv('data/silver/cmapss/FD001_cleaned.csv'); keep=['engine_id','time_cycles','operational_setting_1','operational_setting_2','operational_setting_3']+[f'sensor_{i}' for i in [2,3,4,7,8,9,11,12,13,15,17,20,21]]; pd.concat([df[df.engine_id==1].head(60), df[df.engine_id==2].head(60)])[keep].to_csv('${SAMPLE_CSV}', index=False)"`,
    { stdio: 'inherit' }
  );
}

const PAGES = [
  { name: 'Overview', file: '01-overview', header: 'System Overview' },
  { name: 'New Prediction', file: '02-new-prediction', header: 'New Engine Prediction', special: 'csv' },
  { name: 'Engine Analysis', file: '03-engine-analysis', header: 'Individual Engine Analysis' },
  { name: 'Model Comparison', file: '04-model-comparison', header: 'Model Performance Comparison' },
  // Fleet Management loops all 100 engines with a progress bar; wait for the
  // summary that only appears once the loop finishes before screenshotting.
  { name: 'Fleet Management', file: '05-fleet-management', header: 'Fleet-Wide Risk Assessment', waitForText: 'Fleet Summary' },
  { name: 'Performance Metrics', file: '06-performance-metrics', header: 'Detailed Performance Analysis' },
];

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function selectPage(page, pageName) {
  // Open the sidebar "Select Page" selectbox and click the matching option.
  // Uses evaluate-based native clicks so overlays / visibility checks don't
  // interfere (puppeteer's .click() rejects covered elements).
  await page.keyboard.press('Escape');
  await sleep(300);

  const opened = await page.evaluate(() => {
    const sb = document.querySelector('[data-testid="stSidebar"]');
    if (!sb) return 'no sidebar';
    const sel = sb.querySelector('[data-testid="stSelectbox"]');
    if (!sel) return 'no selectbox';
    const trigger = sel.querySelector('button[aria-haspopup="listbox"]') || sel.querySelector('button');
    if (!trigger) return 'no trigger';
    trigger.scrollIntoView({ block: 'center' });
    trigger.click();
    return 'ok';
  });
  if (opened !== 'ok') throw new Error(opened);

  // The first open after load can be swallowed while the app is still
  // settling; retry clicking the trigger a few times until options appear.
  let optionsReady = false;
  for (let attempt = 0; attempt < 4 && !optionsReady; attempt++) {
    try {
      await page.waitForSelector('[role="option"]', { timeout: 3000 });
      optionsReady = true;
    } catch (e) {
      await page.keyboard.press('Escape');
      await sleep(300);
      await page.evaluate(() => {
        const sel = document.querySelector('[data-testid="stSidebar"] [data-testid="stSelectbox"]');
        const trigger = sel && (sel.querySelector('button[aria-haspopup="listbox"]') || sel.querySelector('button'));
        if (trigger) trigger.click();
      });
    }
  }
  if (!optionsReady) throw new Error('Selectbox dropdown did not open');
  await sleep(400);
  const clicked = await page.evaluate((name) => {
    const opts = [...document.querySelectorAll('[role="option"]')];
    const match = opts.find(o => o.textContent.trim() === name);
    if (match) { match.click(); return true; }
    return false;
  }, pageName);
  if (!clicked) throw new Error(`Option "${pageName}" not found in selectbox`);
}

async function waitForHeader(page, headerText, timeout = 30000) {
  await page.waitForFunction(
    (h) => {
      const headers = [...document.querySelectorAll('h1, [data-testid="stHeader"] h1, h2')];
      return headers.some(el => el.textContent.includes(h));
    },
    { timeout },
    headerText
  ).catch(() => console.log(`  (header "${headerText}" not detected, screenshotting anyway)`));
}

async function waitForContentStable(page) {
  // Wait until the main area has rendered and stopped growing.
  await page.waitForFunction(() => {
    const main = document.querySelector('[data-testid="stMain"]');
    return main && main.offsetHeight > 200;
  }, { timeout: 30000 }).catch(() => {});
  let prev = 0;
  for (let i = 0; i < 10; i++) {
    const h = await page.evaluate(() => document.querySelector('[data-testid="stMain"]')?.offsetHeight || 0);
    if (h === prev && h > 200) break;
    prev = h;
    await sleep(700);
  }
}

async function captureFull(page, outPath) {
  // Streamlit's app root is a fixed viewport-height container with an internal
  // scroll area, so fullPage only captures the viewport. Resize the viewport to
  // the page's full content height so everything is visible at once, then the
  // fullPage screenshot captures it without internal scrolling or clipping.
  for (let i = 0; i < 3; i++) {
    const h = await page.evaluate(() => {
      const main = document.querySelector('[data-testid="stMain"]');
      const sb = document.querySelector('[data-testid="stSidebar"]');
      const block = document.querySelector('[data-testid="stMainBlockContainer"]');
      // scrollHeight of the scroll container = full content height; fall back
      // to the block container's offsetHeight.
      const mainH = Math.max(main?.scrollHeight || 0, block?.offsetHeight || 0);
      const sbH = sb?.scrollHeight || 0;
      return Math.max(mainH, sbH);
    });
    const target = Math.max(h + 60, 800);
    const cur = page.viewport().height;
    if (Math.abs(cur - target) <= 40) break;
    await page.setViewport({ width: 1600, height: target });
    await sleep(1200);
  }
  await page.screenshot({ path: outPath, fullPage: true });
}

async function runNewPredictionFlow(page) {
  // The CSV Upload tab is first/active by default. Upload the sample CSV.
  const fileInput = await page.$('input[type="file"]');
  if (!fileInput) { console.log('  (file input not found, skipping upload)'); return; }
  await fileInput.uploadFile(SAMPLE_CSV);
  console.log('  uploaded sample CSV');

  // Wait for the "Generate Prediction" button to appear after the file is parsed.
  await page.waitForFunction(
    () => [...document.querySelectorAll('button')].some(b => b.textContent.includes('Generate Prediction')),
    { timeout: 30000 }
  );
  await sleep(1500);

  // Click it (Streamlit buttons are <button> with role/kind; pick by text).
  const clicked = await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Generate Prediction'));
    if (btn) { btn.click(); return true; }
    return false;
  });
  if (!clicked) { console.log('  (could not click Generate Prediction)'); return; }

  // Wait for the prediction result to render.
  await page.waitForFunction(
    () => document.body.textContent.includes('Predicted RUL') || document.body.textContent.includes('Prediction Results'),
    { timeout: 60000 }
  ).catch(() => console.log('  (prediction result not detected in time)'));
  await sleep(3000);
}

async function main() {
  ensureSampleCsv();
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--window-size=1600,1000'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });

  await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 90000 });
  // Wait for the sidebar and main area to exist before doing anything; on a
  // fresh server start the first render takes a while (models load on demand).
  await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 60000 });
  // Streamlit's app container is position:absolute with a viewport-height
  // scroll area inside stMain, so body height is 0 and fullPage only captures
  // the viewport. Make the container flow normally and disable the internal
  // scroll so the body grows to the full content height.
  await page.addStyleTag({ content: `
    [data-testid="stAppViewContainer"] {
      position: static !important; height: auto !important; max-height: none !important; overflow: visible !important;
    }
    [data-testid="stMain"] {
      height: auto !important; max-height: none !important; overflow: visible !important;
    }
    [data-testid="stSidebar"] {
      position: static !important; height: auto !important; max-height: none !important; overflow: visible !important;
    }
  `});
  await waitForContentStable(page);
  await sleep(8000);

  for (let i = 0; i < PAGES.length; i++) {
    const p = PAGES[i];
    try {
      // The first page (Overview) is the selectbox default, so it is already
      // rendered on load - no need to open the dropdown for it.
      if (i > 0) {
        await selectPage(page, p.name);
        await sleep(1500);
        await waitForContentStable(page);
      }
      await waitForHeader(page, p.header);

      if (p.waitForText) {
        await page.waitForFunction(
          (t) => document.querySelector('[data-testid="stMain"]').innerText.includes(t),
          { timeout: 120000 }, p.waitForText
        ).catch(() => console.log(`  (wait text "${p.waitForText}" not found, screenshotting anyway)`));
        await sleep(2000);
      }

      if (p.special === 'csv') {
        await runNewPredictionFlow(page);
        await waitForContentStable(page);
      }

      const out = path.join(OUT_DIR, `${p.file}.png`);
      await captureFull(page, out);
      console.log(`saved ${p.file}.png`);
    } catch (e) {
      console.log(`FAILED on ${p.name}: ${e.message}`);
      await page.screenshot({ path: path.join(OUT_DIR, `${p.file}.png`), fullPage: true }).catch(() => {});
    }
  }

  await browser.close();
}

main().catch(e => { console.error(e); process.exit(1); });
