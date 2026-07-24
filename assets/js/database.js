// database.js - Persistencia de config con Dexie (IndexedDB)
import Dexie from 'https://cdn.jsdelivr.net/npm/dexie@4.0.8/dist/dexie.min.mjs';

const DB_NAME = 'PlateDetector';
let db = null;

function getDB() {
  if (db) return db;
  try {
    db = new Dexie(DB_NAME);
    db.version(1).stores({ config: 'key' });
    return db;
  } catch (e) {
    return null;
  }
}

export async function initDB() {
  const d = getDB();
  if (d) await d.open();
}

// Si Dexie/IndexedDB no esta disponible, devuelve el valor por defecto
export async function getConfig(key, defaultVal) {
  const d = getDB();
  if (!d) return defaultVal;
  const row = await d.config.get(key);
  return row ? row.value : defaultVal;
}

export async function setConfig(key, value) {
  const d = getDB();
  if (d) await d.config.put({ key, value });
}