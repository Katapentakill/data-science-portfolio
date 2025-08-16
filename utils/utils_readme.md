# Utils Collection - Automation and Data Extraction Tools

Esta carpeta contiene herramientas especializadas de automatización y extracción de datos que complementan los proyectos principales de ciencia de datos del portfolio.

## 🛠️ Descripción General

Las utilities proporcionan capacidades de **web scraping** y **automatización web** utilizando tecnologías modernas para recopilar información de fuentes externas que requieren interacción automatizada.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnologías | Proyectos |
|-----------|-------------|-----------|
| **Runtime** | Node.js 14.x+ | WebScrapping |
| **Web Automation** | Puppeteer ^19.0.0 | WebScrapping |
| **Browser Engine** | Chromium | WebScrapping |
| **Programming** | ES2017+ Async/Await | WebScrapping |
| **Data Structures** | JavaScript Set, Array | WebScrapping |
| **Development** | VS Code, Nodemon | WebScrapping |
| **Package Management** | NPM, package.json | WebScrapping |

## 🏗️ Estructura

```
utils/
│
├── Terremoto/                          # Herramientas de análisis sísmico
│
├── WebScrappingCriminalistico/         # Web scraping automatizado
│   ├── node_modules/                   # Dependencias Node.js
│   ├── index.js                        # Script principal
│   ├── package.json                    # Configuración del proyecto
│   └── README_Puppeteer_Scraper.md     # Documentación técnica
│
└── README.md                           # Este archivo
```

## 🏆 Herramientas Disponibles

### **WebScrappingCriminalistico - Automated News Extraction** 📰

- **Tecnología:** Puppeteer + Node.js
- **Objetivo:** Extracción automatizada de noticias policiales de Antofagasta TV
- **Características:** Scroll infinito, navegación programática, extracción selectiva
- **Estado:** Completamente funcional

#### Funcionalidades Principales
- **Navegación automática** con scroll infinito para contenido dinámico
- **Extracción selectiva** de fechas y contenido específico
- **Manejo de duplicados** para evitar procesamiento repetido
- **Sistema robusto** con reintentos y manejo de errores

#### Tecnologías Utilizadas
- Node.js 14.x+ como runtime
- Puppeteer ^19.0.0 para automatización web
- Chromium como motor de navegación
- ES2017+ async/await para programación asíncrona

## 📊 Casos de Uso

- **Monitoreo de noticias** policiales en tiempo real
- **Análisis de tendencias** de incidentes por fecha
- **Base de datos histórica** para análisis temporal
- **Integración con pipelines** de procesamiento de datos

## 🔧 Consideraciones Técnicas

- **Ética de scraping:** Implementa delays apropiados y respeta robots.txt
- **Robustez:** Sistema de reintentos con manejo de errores
- **Performance:** Optimización de memoria y recursos del navegador
- **Configurabilidad:** Modos headless y visible para desarrollo/producción

---

*Las herramientas de utils complementan el portfolio de ciencia de datos proporcionando capacidades de extracción automatizada de información desde fuentes web dinámicas.*