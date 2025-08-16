# 🕵️‍♂️ Antofagasta TV “Policial” Scraper (Node.js + Puppeteer)

Scraper robusto para extraer artículos de la categoría **Policial** en Antofagasta TV. El script:
1) abre la página de listado,  
2) realiza **scroll infinito** para cargar más tarjetas,  
3) **detecta y visita** cada enlace de detalle que coincida con el prefijo permitido,  
4) **extrae la fecha** de publicación (y se puede extender a título, autor, cuerpo),  
5) retorna al listado y continúa hasta agotar el feed.  

> Pensado para ejecutarse en entornos locales o CI, con **configuración por constantes** y salida en **JSON/CSV**.

---

## 🧰 Tecnologías

| Categoría              | Herramienta / Librería |
|-----------------------:|------------------------|
| Runtime                | Node.js (>= 18) |
| Automatización         | Puppeteer |
| Navegador              | Chromium (provisionado por Puppeteer) |
| Logs (opcional)        | console / pino |
| Persistencia (opcional)| fs (JSON/CSV), SQLite/SQLite3 |

---

## ✨ Características clave

- **Scroll infinito** con detección de “nuevos elementos” por incremento de `scrollHeight`.
- **De-duplicación** de URLs con `Set` para evitar reprocesos.
- **Filtro de prefijo** para restringir navegación a la ruta `.../categoria/policial/`.
- **Extracción de fecha** desde el detalle (selector configurable).
- **Modo headless** conmutables (`headless: true | false | "new"`).
- Diseño listo para **persistencia** de resultados y **reintentos** básicos.

---

## 🗂️ Estructura mínima

```
.
├─ index.js           # Script principal
├─ package.json
└─ output/
   ├─ results.json    # (opcional) Resultados en JSON
   └─ results.csv     # (opcional) Resultados en CSV
```

---

## ⚙️ Configuración

Edita las **constantes** en `index.js` para ajustar el comportamiento:

```js
const URL = "https://www.antofagasta.tv/categoria/policial";
const HEADLESS = true;    // "new" | true | false
const SCROLL_PAUSE_MS = 3000;
const ALLOWED_PREFIX = "https://www.antofagasta.tv/categoria/policial/";
const SELECTORS = {
  listLinksA: ".MuiGrid-root.MuiGrid-item.widget.typography.widget-a0ebd116-8931-4bc4-8e90-5e45a4c60405 a.layerTwo",
  listLinksB: ".MuiGrid-root.MuiGrid-item.widget.typography.widget-f33bdd76-40cb-413e-945d-1a3599f68b47 a.layerTwo",
  date: ".MuiTypography-root.MuiTypography-body1.layerOne.date.css-iqx1c8",
};
```

> Si la web cambia sus clases `Mui*`, actualiza `SELECTORS` por selectores más resilientes (por ejemplo, combinaciones de jerarquías u otros atributos).

---

## 🚀 Quickstart

```bash
# 1) Inicializar proyecto (opcional)
mkdir scraper-antofagasta && cd scraper-antofagasta
npm init -y

# 2) Instalar Puppeteer
npm i puppeteer

# 3) Crear index.js con el script
# (pega el contenido del ejemplo de abajo)

# 4) Ejecutar
node index.js
```

## 📤 Formato de salida (JSON)

```json
[
  {
    "link": "https://www.antofagasta.tv/categoria/policial/...",
    "date": "13 de agosto de 2025",
    "title": "Título de ejemplo",
    "body": "Contenido resumido del artículo…"
  }
]
```

---

## 🧪 Pruebas y CI (sugerido)

- Línter y formato: `eslint`, `prettier`.
- Test simple de “selectores vivos” con `jest` + `puppeteer` en CI nocturno (cron) para alertar si cambian.
- Snapshot de DOM (parcial) para detectar cambios de estructura.

---

## 🧯 Troubleshooting

- **No carga más tarjetas:** incrementa `SCROLL_PAUSE_MS` o usa `page.waitForSelector` de un “footer/sentinel”.
- **Selectores rotos:** inspecciona el DOM actual y ajusta `SELECTORS` (evita depender solo de clases auto–generadas).
- **Bloqueos / detección:** usa `HEADLESS: "new"`, agrega *headers* (UA / Accept-Language), añade delays aleatorios o `puppeteer-extra-plugin-stealth`.
- **Tiempo de espera:** eleva `timeout` en `page.goto` según tu red/CI.

---

## ⚖️ Consideraciones legales y éticas

- Revisa **robots.txt** y **Términos de Servicio** del sitio.  
- Limita la tasa de requests; evita impacto en la plataforma.  
- No recolectes ni publiques información personal sensible.  
- Ofrece atribución de fuente cuando corresponda.

---

## 🗺️ Roadmap

- Soporte para múltiples categorías y paginación paralela (varias páginas).
- Extracción de **título/autor/cuerpo** más robusta (fallbacks, normalización).
- Persistencia en base de datos (SQLite/Postgres) y panel de control.
- Dockerfile para despliegue reproducible.
- Migración a **Playwright** (alternativa moderna con mejores herramientas de test).

---

## 📄 Licencia

MIT (o la que definas para tu proyecto).
