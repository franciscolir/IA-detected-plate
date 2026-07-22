// history.js - Buffer temporal con mayoría
export class HistoryBuffer {
  constructor(size = 20) {
    this.size = size;
    this.buffer = [];
  }

  push(plate) {
    this.buffer.push(plate);
    if (this.buffer.length > this.size) {
      this.buffer.shift();
    }
  }

  getMajority() {
    if (this.buffer.length === 0) return null;
    const counts = {};
    this.buffer.forEach((p) => {
      counts[p] = (counts[p] || 0) + 1;
    });
    let max = 0;
    let majority = null;
    for (const [plate, count] of Object.entries(counts)) {
      if (count > max) {
        max = count;
        majority = plate;
      }
    }
    return majority;
  }

  clear() {
    this.buffer = [];
  }
}