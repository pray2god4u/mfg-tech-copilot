(() => {
  const field = document.getElementById('evidenceFilter');
  field.addEventListener('input', () => {
    const q = field.value.trim().toLowerCase();
    for (const section of document.querySelectorAll('.evidence-section')) section.hidden = Boolean(q) && !section.textContent.toLowerCase().includes(q);
  });
  function reveal() {
    const id = decodeURIComponent(location.hash.slice(1));
    const target = document.getElementById(id);
    if (target) { field.value = ''; for (const section of document.querySelectorAll('.evidence-section')) section.hidden=false; target.scrollIntoView(); }
  }
  window.addEventListener('hashchange', reveal); reveal();
})();
