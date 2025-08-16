const puppeteer = require("puppeteer");

(async () => {
  const URL = "https://www.antofagasta.tv/categoria/policial"; // Reemplaza con la URL de la página que quieres scrapear

  const browser = await puppeteer.launch({
    headless: false,
  });

  const page = await browser.newPage();

  await page.goto(URL, { waitUntil: "networkidle2" });

  // Función para obtener enlaces de las tarjetas
  async function getCardLinks(page) {
    return page.evaluate(() => {
      const links = Array.from(document.querySelectorAll(".MuiGrid-root.MuiGrid-item.widget.typography.widget-a0ebd116-8931-4bc4-8e90-5e45a4c60405 a.layerTwo"))
        .map(link => link.href);
      const detailedLinks = Array.from(document.querySelectorAll(".MuiGrid-root.MuiGrid-item.widget.typography.widget-f33bdd76-40cb-413e-945d-1a3599f68b47 a.layerTwo"))
        .map(link => link.href);
      return [...links, ...detailedLinks];
    });
  }

  // Función para hacer scroll down hasta que no se carguen más tarjetas
  async function autoScrollAndVisitLinks(page) {
    let hasMoreCards = true;
    let visitedLinks = new Set();

    while (hasMoreCards) {
      const links = await getCardLinks(page);

      for (const link of links) {
        if (link.startsWith("https://www.antofagasta.tv/categoria/policial/") && !visitedLinks.has(link)) {
          visitedLinks.add(link);
          console.log(`Navegando a: ${link}`);
          await visitLink(page, link);
        }
      }

      // Hacer scroll hacia abajo y esperar a que se carguen nuevas tarjetas
      hasMoreCards = await page.evaluate(() => {
        const initialHeight = document.body.scrollHeight;
        window.scrollBy(0, window.innerHeight);
        return new Promise((resolve) => {
          setTimeout(() => {
            const newHeight = document.body.scrollHeight;
            resolve(newHeight > initialHeight);
          }, 3000); // Esperar a que se carguen nuevas tarjetas
        });
      });
    }
  }

  // Función para visitar una página de detalle y luego regresar
  async function visitLink(page, link) {
    const originalPage = page.url();
    await page.goto(link, { waitUntil: "networkidle2" });
    console.log(`Visité la página: ${link}`);

    // Extraer la fecha
    const dateText = await page.evaluate(() => {
      const dateElement = document.querySelector('.MuiTypography-root.MuiTypography-body1.layerOne.date.css-iqx1c8');
      return dateElement ? dateElement.innerText : null;
    });

    console.log(`Fecha encontrada: ${dateText}`);

    await page.goto(originalPage, { waitUntil: "networkidle2" });
  }

  // Hacer scroll y visitar enlaces de las tarjetas
  await autoScrollAndVisitLinks(page);

  await browser.close();
})();