# Web Scraping Antofagasta TV - Automated Data Extraction Project

Este proyecto implementa una solución automatizada de web scraping para extraer información de noticias policiales del sitio web de Antofagasta TV utilizando Puppeteer y Node.js. El sistema navega automáticamente por la página, hace scroll infinito para cargar contenido dinámico, y extrae datos específicos de cada artículo de manera eficiente.

## 🧠 Descripción del Proyecto

El proyecto utiliza **Puppeteer** para automatizar la navegación web y **técnicas de scraping avanzadas** para extraer información de un sitio web con contenido dinámico. A través de scroll automático, detección de nuevos elementos y navegación programática, se construye un extractor robusto capaz de recopilar datos de múltiples páginas de manera automatizada.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Runtime** | Node.js | 14.x+ | Entorno de ejecución JavaScript |
| **Web Automation** | Puppeteer | ^19.0.0 | Control programático del navegador |
| **Browser Engine** | Chromium | - | Motor de navegación automatizada |
| **Web Standards** | DOM API | - | Manipulación de elementos HTML |
| **Async Programming** | ES2017+ | - | Programación asíncrona con async/await |
| **Data Structures** | Set | - | Almacenamiento de enlaces únicos |
| **HTTP Protocol** | - | - | Navegación y carga de páginas |

## 📄 Pipeline de Desarrollo

### 1. **Configuración e Inicialización**
```javascript
const puppeteer = require("puppeteer");

const URL = "https://www.antofagasta.tv/categoria/policial";

const browser = await puppeteer.launch({
  headless: false,  // Modo visible para debugging
});

const page = await browser.newPage();
await page.goto(URL, { waitUntil: "networkidle2" });
```

**Configuración principal:**
- **URL objetivo:** Sección policial de Antofagasta TV
- **Modo headless:** Deshabilitado para monitoreo visual
- **Wait strategy:** NetworkIdle2 para asegurar carga completa
- **Browser instance:** Chromium controlado por Puppeteer

### 2. **Extracción de Enlaces Dinámicos**

#### Función de Detección de Enlaces
```javascript
async function getCardLinks(page) {
  return page.evaluate(() => {
    // Selector principal para tarjetas de noticias
    const links = Array.from(document.querySelectorAll(
      ".MuiGrid-root.MuiGrid-item.widget.typography.widget-a0ebd116-8931-4bc4-8e90-5e45a4c60405 a.layerTwo"
    )).map(link => link.href);
    
    // Selector secundario para artículos detallados
    const detailedLinks = Array.from(document.querySelectorAll(
      ".MuiGrid-root.MuiGrid-item.widget.typography.widget-f33bdd76-40cb-413e-945d-1a3599f68b47 a.layerTwo"
    )).map(link => link.href);
    
    return [...links, ...detailedLinks];
  });
}
```

**Estrategia de extracción:**
- **Selectores múltiples:** Captura diferentes tipos de tarjetas de noticias
- **Array spreading:** Combinación de resultados de múltiples selectores
- **Link extraction:** Extracción directa de URLs desde elementos `<a>`
- **DOM evaluation:** Ejecución de código en el contexto del navegador

### 3. **Scroll Infinito y Carga Dinámica**

#### Sistema de Scroll Automático
```javascript
async function autoScrollAndVisitLinks(page) {
  let hasMoreCards = true;
  let visitedLinks = new Set();

  while (hasMoreCards) {
    const links = await getCardLinks(page);

    // Procesar enlaces únicos
    for (const link of links) {
      if (link.startsWith("https://www.antofagasta.tv/categoria/policial/") && 
          !visitedLinks.has(link)) {
        visitedLinks.add(link);
        console.log(`Navegando a: ${link}`);
        await visitLink(page, link);
      }
    }

    // Scroll down con detección de nuevo contenido
    hasMoreCards = await page.evaluate(() => {
      const initialHeight = document.body.scrollHeight;
      window.scrollBy(0, window.innerHeight);
      
      return new Promise((resolve) => {
        setTimeout(() => {
          const newHeight = document.body.scrollHeight;
          resolve(newHeight > initialHeight);
        }, 3000); // Buffer de tiempo para carga AJAX
      });
    });
  }
}
```

**Características del scroll automático:**
- **Detección de duplicados:** Set para evitar visitas repetidas
- **Validación de URL:** Filtrado por categoría específica
- **Scroll dinámico:** Detección automática de nuevo contenido
- **Timeout inteligente:** 3 segundos para carga de contenido AJAX

### 4. **Navegación y Extracción de Datos**

#### Función de Visita de Enlaces
```javascript
async function visitLink(page, link) {
  const originalPage = page.url();
  
  // Navegación a página de detalle
  await page.goto(link, { waitUntil: "networkidle2" });
  console.log(`Visité la página: ${link}`);

  // Extracción de fecha específica
  const dateText = await page.evaluate(() => {
    const dateElement = document.querySelector(
      '.MuiTypography-root.MuiTypography-body1.layerOne.date.css-iqx1c8'
    );
    return dateElement ? dateElement.innerText : null;
  });

  console.log(`Fecha encontrada: ${dateText}`);

  // Regreso a página original
  await page.goto(originalPage, { waitUntil: "networkidle2" });
}
```

**Estrategia de navegación:**
- **URL preservation:** Almacenamiento de página original
- **Data extraction:** Extracción selectiva de elementos específicos
- **Error handling:** Validación de existencia de elementos
- **Return navigation:** Regreso automático a página principal

## 🗂️ Estructura del Proyecto

### Archivos y Configuración (VS Code Workspace):

```
DATA-SCIENCE-PORTFOLIO/
└── utils/
    └── WebScrappingCriminalistico/
        ├── node_modules/            # Dependencias instaladas
        ├── .gitignore              # Archivos a ignorar en Git
        ├── index.js                # Script principal de scraping
        ├── package-lock.json       # Lock de versiones exactas
        ├── package.json            # Dependencias y scripts de Node.js
        └── README_Puppeteer_Scraper.md  # Documentación del proyecto
```

### Configuración de package.json (Actual):
```json
{
  "name": "webscrapping-criminalistico",
  "version": "1.0.0",
  "description": "Web scraper para noticias policiales de Antofagasta TV",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js"
  },
  "dependencies": {
    "puppeteer": "^19.0.0"
  },
  "devDependencies": {
    "nodemon": "^2.0.20"
  }
}
```

### Archivos de Configuración VS Code:
```
.vscode/
├── settings.json          # Configuraciones del workspace
├── launch.json           # Configuraciones de debugging
└── tasks.json            # Tareas automatizadas
```

## 🚀 Instalación y Configuración

### Prerrequisitos
```bash
# Verificar versión de Node.js
node --version  # Debe ser 14.x o superior
npm --version   # Debe ser 6.x o superior
```

### Instalación de Dependencias
```bash
# Inicializar proyecto
npm init -y

# Instalar Puppeteer (incluye Chromium)
npm install puppeteer

# Dependencias de desarrollo (opcional)
npm install --save-dev nodemon
```

### Configuración del Entorno (Workspace Structure)
```bash
# Navegar al directorio del proyecto
cd data-science-portfolio/utils/WebScrappingCriminalistico

# Verificar archivos del proyecto
ls -la
# index.js
# package.json
# package-lock.json
# node_modules/
# README_Puppeteer_Scraper.md
```

## 🛠️ Cómo Ejecutar el Proyecto

### Ejecución en VS Code Workspace
```bash
# Desde el directorio raíz del workspace
cd utils/WebScrappingCriminalistico

# Ejecutar scraper una vez
node index.js

# Ejecutar con debugging en VS Code
# F5 o Run > Start Debugging

# Usar terminal integrado de VS Code
# Ctrl+` para abrir terminal
npm start
```

### Configuraciones Avanzadas

#### Modo Headless (Producción)
```javascript
const browser = await puppeteer.launch({
  headless: true,          // Ocultar navegador
  args: ['--no-sandbox']   // Para servidores Linux
});
```

#### Configuración de Timeouts
```javascript
await page.goto(URL, { 
  waitUntil: "networkidle2",
  timeout: 30000  // 30 segundos máximo
});
```

#### Configuración de Viewport
```javascript
await page.setViewport({
  width: 1920,
  height: 1080,
  deviceScaleFactor: 1
});
```

## 📈 Funcionalidades Implementadas

### Sistema de Navegación Automática
- **Scroll infinito:** Carga automática de contenido dinámico
- **Detección de enlaces:** Identificación de tarjetas de noticias
- **Navegación programática:** Visita automática de páginas de detalle
- **Control de flujo:** Regreso inteligente a página principal

### Extracción de Datos
- **Selectores específicos:** Targeting de elementos Material-UI
- **Manejo de estados:** Gestión de elementos que pueden no existir
- **Logging completo:** Registro de URLs visitadas y datos extraídos
- **Validación de contenido:** Verificación de estructura esperada

### Optimizaciones de Rendimiento
- **Reutilización de página:** Una sola instancia para toda la navegación
- **Control de memoria:** Gestión eficiente de recursos del navegador
- **Timeouts inteligentes:** Esperas apropiadas para carga AJAX
- **Deduplicación:** Evita procesar enlaces repetidos

## 🔧 Personalización y Extensiones

### Modificación de Selectores
```javascript
// Actualizar selectores para diferentes layouts
const customSelectors = {
  newsCards: ".custom-news-card a",
  dateElement: ".custom-date-selector",
  titleElement: ".custom-title-selector"
};
```

### Extracción de Datos Adicionales
```javascript
// Función extendida para extraer más información
async function extractArticleData(page) {
  return page.evaluate(() => {
    return {
      title: document.querySelector('.title-selector')?.innerText,
      date: document.querySelector('.date-selector')?.innerText,
      content: document.querySelector('.content-selector')?.innerText,
      author: document.querySelector('.author-selector')?.innerText,
      tags: Array.from(document.querySelectorAll('.tag-selector'))
        .map(tag => tag.innerText)
    };
  });
}
```

### Almacenamiento de Datos
```javascript
const fs = require('fs');

// Guardar datos extraídos en JSON
function saveData(data, filename) {
  fs.writeFileSync(
    `./data/${filename}`, 
    JSON.stringify(data, null, 2)
  );
}
```

## 🎯 Casos de Uso y Aplicaciones

### Monitoreo de Noticias
- **Alertas automáticas:** Notificaciones de nuevas noticias policiales
- **Análisis de tendencias:** Tracking de frecuencia de incidentes
- **Base de datos histórica:** Archivo de noticias para análisis temporal
- **Dashboard en tiempo real:** Visualización de datos extraídos

### Análisis de Contenido
- **Extracción de patrones:** Identificación de tipos de incidentes
- **Análisis temporal:** Distribución de eventos por fechas/horarios
- **Clasificación automática:** Categorización de noticias por tipo
- **Detección de anomalías:** Identificación de picos de actividad

### Integración con Otros Sistemas
- **APIs REST:** Exposición de datos extraídos via API
- **Bases de datos:** Almacenamiento en PostgreSQL/MongoDB
- **Sistemas de notificación:** Slack, Discord, Email alerts
- **Análisis ML:** Input para modelos de procesamiento de texto

## 🔍 Consideraciones Técnicas

### Robustez y Manejo de Errores
```javascript
// Implementación de retry logic
async function safePageVisit(page, url, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await page.goto(url, { waitUntil: "networkidle2" });
      return true;
    } catch (error) {
      console.log(`Intento ${i + 1} fallido: ${error.message}`);
      if (i === maxRetries - 1) throw error;
      await page.waitForTimeout(2000); // Esperar antes de reintentar
    }
  }
}
```

### Optimización de Memoria
```javascript
// Limpieza periódica de memoria
async function clearBrowserMemory(page) {
  await page.evaluate(() => {
    // Limpiar cache del navegador
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
      });
    }
  });
}
```

### Rate Limiting y Ética
```javascript
// Implementar delays para ser respetuoso con el servidor
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Usar entre requests
await delay(1000); // 1 segundo entre requests
```

## 🎯 Mejoras Futuras

### Funcionalidades Avanzadas
1. **Scraping distribuido:** Múltiples instancias paralelas
2. **Detección de cambios:** Monitoring de actualizaciones de contenido
3. **Clasificación automática:** ML para categorizar noticias
4. **Análisis de sentimiento:** Procesamiento NLP del contenido

### Infraestructura
1. **Containerización:** Docker para deployment
2. **Scheduling:** Cron jobs para ejecución periódica
3. **Monitoring:** Métricas de performance y errores
4. **Escalabilidad:** Cluster de scrapers distribuidos

### Integración de Datos
1. **Data pipelines:** ETL automático hacia data warehouses
2. **APIs GraphQL:** Interface flexible para consultas
3. **Real-time streaming:** WebSockets para datos en vivo
4. **Machine Learning:** Modelos predictivos sobre los datos

## ⚖️ Consideraciones Legales y Éticas

### Términos de Servicio
- **Revisar robots.txt:** Verificar políticas de scraping del sitio
- **Rate limiting:** Implementar delays apropiados
- **User-Agent:** Identificación transparente del bot
- **Respeto al ancho de banda:** Evitar sobrecarga del servidor

### Buenas Prácticas
```javascript
// Configurar User-Agent identificable
await page.setUserAgent('AntofagastaTV-Scraper/1.0 (+contacto@example.com)');

// Implementar delays entre requests
const DELAY_BETWEEN_REQUESTS = 2000; // 2 segundos

// Respetar señales de stop del servidor
if (response.status() === 429) { // Too Many Requests
  await delay(60000); // Esperar 1 minuto
}
```

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos de web scraping, automatización web, o implementación de sistemas de monitoreo de noticias, no dudes en contactar.

## 📗 Referencias y Recursos

- **Puppeteer Documentation:** Guía oficial y API reference
- **Web Scraping Best Practices:** Técnicas éticas y eficientes
- **CSS Selectors:** Documentación de selectores avanzados
- **Node.js Async Patterns:** Programación asíncrona avanzada

---

*Este proyecto representa una implementación robusta de web scraping automatizado utilizando tecnologías modernas de Node.js y Puppeteer, diseñado para extraer información de manera eficiente y ética desde sitios web con contenido dinámico, proporcionando una base sólida para sistemas de monitoreo y análisis de noticias.*