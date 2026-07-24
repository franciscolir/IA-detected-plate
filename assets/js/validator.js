// validator.js - Validacion de formatos de patentes chilenas
const PLATE_FORMATS = [
  /^[A-Z]{4}[0-9]{2}$/,           // ABCD12 (actual desde 2007)
  /^[A-Z]{2}[0-9]{4}$/,           // AB1234 (antiguo, sin guion)
  /^[A-Z]{2}-[0-9]{4}$/,          // AB-1234 (antiguo, con guion)
  /^[A-Z]{2}-[0-9]{2}-[0-9]{2}$/, // AB-12-34
  /^[A-Z][0-9]{4}$/,              // A1234 (motos)
  /^C[DJ][0-9]{4}$/,              // CD1234 (diplomatico/consular, sin guion)
  /^C[DJ]-[0-9]{4}$/              // CD-1234 (diplomatico/consular, con guion)
];

export function validatePlate(plate) {
  return PLATE_FORMATS.some((r) => r.test(plate));
}

// Normalizar: uppercase + eliminar todo lo que no sea alfanumerico
export function normalizePlate(text) {
  return text.toUpperCase().replace(/[^A-Z0-9]/g, '');
}