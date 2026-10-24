// Applies site-wide config: product name, contact address, page title.
(function () {
  const cfg = window.SITE_CONFIG;
  if (!cfg) return;

  document.querySelectorAll("[data-product-name]").forEach((el) => {
    el.textContent = cfg.productName;
  });

  if (document.body.dataset.pageTitle) {
    document.title = `${cfg.productName} · ${document.body.dataset.pageTitle}`;
  } else {
    document.title = cfg.productName;
  }

  const email = cfg.contactEmail;
  document.querySelectorAll("[data-contact-email]").forEach((el) => {
    if (!email) {
      el.textContent = "[E-posta adresi]";
      el.removeAttribute("href");
      return;
    }
    el.textContent = email;
    if (el.tagName === "A") el.href = `mailto:${email}`;
  });
})();
