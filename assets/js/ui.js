// ui.js - Interfaz de usuario
export class UI {
  constructor() {
    this.plateText = document.getElementById('plate-text');
    this.plateBox = document.getElementById('plate-box');
    this.resultBadge = document.getElementById('result-badge');
    this.hudFps = document.getElementById('hud-fps');
    this.hudBackend = document.getElementById('hud-backend');
    this.status = document.getElementById('status');
    this.historyList = document.getElementById('history-list');
    this.registryList = document.getElementById('registry-list');
  }

  setPlate(plate) {
    this.plateText.textContent = plate || '------------';
  }

  setResult(match) {
    if (match === true) {
      this.resultBadge.textContent = 'Patente registrada';
      this.resultBadge.className = 'result-badge ok';
      this.plateBox.className = 'plate-box match';
    } else if (match === false) {
      this.resultBadge.textContent = 'No registrada';
      this.resultBadge.className = 'result-badge fail';
      this.plateBox.className = 'plate-box no-match';
    } else {
      this.resultBadge.textContent = '\u00a0';
      this.resultBadge.className = 'result-badge';
      this.plateBox.className = 'plate-box';
    }
  }

  setFPS(fps) {
    this.hudFps.textContent = `FPS ${fps}`;
  }

  setBackend(backend) {
    this.hudBackend.textContent = backend || '--';
  }

  setStatus(msg) {
    this.status.textContent = msg;
  }

  updateHistory(items, registrySet) {
    this.historyList.innerHTML = '';
    items.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'list-group-item d-flex justify-content-between align-items-center';
      const inReg = registrySet && registrySet.has(item.plate);
      li.innerHTML = `<span>${item.plate} <span class="text-muted small">${new Date(item.timestamp).toLocaleTimeString()}</span></span>
        <button class="btn btn-sm ${inReg ? 'btn-warning' : 'btn-outline-info'} toggle-reg" data-plate="${item.plate}">${inReg ? 'Quitar' : 'Registrar'}</button>`;
      this.historyList.appendChild(li);
    });
  }

  updateRegistry(plates) {
    this.registryList.innerHTML = '';
    document.getElementById('registry-count').textContent = plates.length;
    if (plates.length === 0) {
      const li = document.createElement('li');
      li.className = 'list-group-item text-muted small';
      li.textContent = 'Sin patentes registradas';
      this.registryList.appendChild(li);
      return;
    }
    plates.forEach((plate) => {
      const li = document.createElement('li');
      li.className = 'list-group-item d-flex justify-content-between align-items-center';
      li.innerHTML = `<span class="font-monospace">${plate}</span>
        <button class="btn btn-sm btn-outline-danger remove-reg" data-plate="${plate}">Quitar</button>`;
      this.registryList.appendChild(li);
    });
  }
}