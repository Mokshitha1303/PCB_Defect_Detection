/* PCB Defect Inspector — frontend logic */

const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const preview   = document.getElementById('preview-img');
const inspectBtn = document.getElementById('inspect-btn');
const spinner   = document.getElementById('spinner');
const results   = document.getElementById('results');
const confSlider = document.getElementById('conf-slider');
const confValue  = document.getElementById('conf-value');

let selectedFile = null;

/* ── Confidence slider ─────────────────────────────────────── */
confSlider.addEventListener('input', () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
});

/* ── Drag-and-drop ─────────────────────────────────────────── */
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = 'block';
  inspectBtn.disabled = false;
  results.style.display = 'none';
  document.getElementById('drop-zone-hint').style.display = 'none';
}

/* ── Inference ─────────────────────────────────────────────── */
inspectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  inspectBtn.disabled = true;
  spinner.style.display = 'block';
  results.style.display = 'none';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      alert('Error: ' + (err.detail || resp.statusText));
      return;
    }
    const data = await resp.json();
    renderResults(data);
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    inspectBtn.disabled = false;
    spinner.style.display = 'none';
  }
});

/* ── Render ────────────────────────────────────────────────── */
function renderResults(data) {
  // Status banner
  const banner = document.getElementById('status-banner');
  banner.textContent = data.board_status;
  banner.className = 'status-banner status-' + data.board_status;

  // Summary
  document.getElementById('summary-text').textContent = data.natural_language_summary;

  // Meta
  document.getElementById('meta-latency').innerHTML = `<b>${data.inference_time_ms.toFixed(1)} ms</b>`;
  document.getElementById('meta-size').innerHTML = `<b>${data.image_width} × ${data.image_height}</b>`;
  document.getElementById('meta-total').innerHTML = `<b>${data.total_defects}</b>`;

  // Annotated image
  if (data.annotated_image_b64) {
    document.getElementById('annotated-img').src = 'data:image/png;base64,' + data.annotated_image_b64;
  }

  // Defect table
  const tbody = document.getElementById('defect-tbody');
  tbody.innerHTML = '';
  const entries = Object.entries(data.counts_by_class);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--muted)">No defects detected</td></tr>';
  } else {
    entries
      .sort((a, b) => b[1] - a[1])
      .forEach(([cls, cnt]) => {
        const maxConf = data.max_conf_by_class[cls] ?? 0;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${cls}</td><td>${cnt}</td><td>${(maxConf * 100).toFixed(1)}%</td>`;
        tbody.appendChild(tr);
      });
  }

  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth' });
}
