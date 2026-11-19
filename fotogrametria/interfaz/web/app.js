"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

// Conversión UTM -> WGS84 (series de Krüger) para la lectura de coordenadas bajo el cursor
function utmALatLon(e, n, epsg) {
  if (!epsg || !((epsg > 32600 && epsg < 32661) || (epsg > 32700 && epsg < 32761))) return null;
  const zona = epsg % 100, sur = epsg > 32700;
  const a = 6378137, f = 1 / 298.257223563, k0 = 0.9996;
  const n1 = f / (2 - f), A = a / (1 + n1) * (1 + n1 * n1 / 4 + n1 ** 4 / 64);
  const b = [n1 / 2 - 2 / 3 * n1 ** 2 + 37 / 96 * n1 ** 3, 1 / 48 * n1 ** 2 + 1 / 15 * n1 ** 3, 17 / 480 * n1 ** 3];
  const d = [2 * n1 - 2 / 3 * n1 ** 2 - 2 * n1 ** 3, 7 / 3 * n1 ** 2 - 8 / 5 * n1 ** 3, 56 / 15 * n1 ** 3];
  const xi = (sur ? n - 1e7 : n) / (k0 * A), eta = (e - 5e5) / (k0 * A);
  let xp = xi, ep = eta;
  for (let j = 1; j <= 3; j++) { xp -= b[j - 1] * Math.sin(2 * j * xi) * Math.cosh(2 * j * eta); ep -= b[j - 1] * Math.cos(2 * j * xi) * Math.sinh(2 * j * eta); }
  const chi = Math.asin(Math.sin(xp) / Math.cosh(ep));
  let lat = chi; for (let j = 1; j <= 3; j++) lat += d[j - 1] * Math.sin(2 * j * chi);
  const lon = (zona - 1) * 6 - 180 + 3 + Math.atan2(Math.sinh(ep), Math.cos(xp)) * 180 / Math.PI;
  return [lat * 180 / Math.PI, lon];
}

const fmt = {
  t(s) { s = Math.max(0, Math.round(s)); const m = Math.floor(s / 60); return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; },
  num(v, d = 2) { return v.toLocaleString("es", { minimumFractionDigits: d, maximumFractionDigits: d }); },
};

const CALIDADES = [
  // s/disparo medidos en vuelos reales; "alta" es una estimación (3x estándar). [RGB, RGB + multiespectral]
  { v: "express", n: "Express", d: "Solo el mapa, lo más rápido posible", s: [0.25, 0.75] },
  { v: "rapida", n: "Rápida", d: "Mapa y nube de puntos para revisar", s: [1.2, 2.5] },
  { v: "normal", n: "Estándar", d: "El mejor equilibrio para la mayoría", s: [1.6, 4.3], rec: true },
  { v: "alta", n: "Alta", d: "Máximo detalle, para entregas finales", s: [5, 13] },
];
// Express no crea nube densa: reutiliza los puntos de la alineación
const ETAPA_EXPRESS = { "Creando nube de puntos densa": "Preparando el terreno" };
const DESC_INDICE = { NDVI: "Vigor de la vegetación", GNDVI: "Clorofila", NDRE: "Nitrógeno y clorofila", LCI: "Clorofila foliar" };
const PALETAS = {
  dem: "linear-gradient(90deg,#264653,#2a746e 25%,#8aa06e 50%,#d6be8c 72%,#a87854 88%,#f0ece4)",
  indice: "linear-gradient(90deg,#963c2d,#cd9650 35%,#e1d28c 55%,#78a550 75%,#236937)",
};

const app = {
  carpeta: null, analisis: null, info: null, capa: null, calidad: "normal", etapas: [], inicio: 0, reloj: null, trabajando: false,

  async iniciar() {
    const est = await pywebview.api.estado();
    this.etapas = est.etapas;
    $("#pasos").innerHTML = this.etapas.map((n, i) =>
      `<li data-i="${i}"><span class="marca"></span><span class="n">${n}</span><span class="t"></span></li>`).join("");
    this.pintarCalidades();
    $$(".pestanas button").forEach((b) => (b.onclick = () => this.pestana(b.dataset.p)));
    $("#btn-imagenes").onclick = () => this.elegirCarpeta();
    $("#btn-proyecto").onclick = () => this.abrirResultados();
    $("#btn-procesar").onclick = () => this.procesar();
    $("#btn-pausa").onclick = () => this.alternarPausa();
    $("#btn-cancelar").onclick = () => this.cancelar();
    $("#btn-reg-error").onclick = $("#btn-registro").onclick = () => this.guardarRegistro();
    $("#btn-abrir").onclick = () => this.info && pywebview.api.abrir_carpeta(this.info.carpeta);
    $("#btn-informe").onclick = () => this.info && pywebview.api.abrir_archivo(this.info.carpeta + "\\informe.html");
    $("#btn-tema").onclick = () => this.tema(document.documentElement.dataset.tema === "oscuro" ? "claro" : "oscuro");
    $("#btn-instalar-gpu").onclick = () => pywebview.api.instalar_gpu();
    $("#zoom-mas").onclick = () => (this.capa === "nube" ? this.nube.acercar(0.8) : this.mapa.zoomIn());
    $("#zoom-menos").onclick = () => (this.capa === "nube" ? this.nube.acercar(1.25) : this.mapa.zoomOut());
    $("#encuadrar").onclick = () => (this.capa === "nube" ? this.nube.vistaInicial() : this.encuadrar());
    $("#sel-color").onchange = (e) => { if (this.nube) { this.nube.modo = +e.target.value; this.nube.redibujar(); } };
    $("#tam-punto").oninput = (e) => { if (this.nube) { this.nube.tam = +e.target.value; this.nube.redibujar(); } };
    this.tema(document.documentElement.dataset.tema);
    this.estadoGpu();
  },

  pestana(p) {
    $$(".pestanas button").forEach((b) => b.classList.toggle("activa", b.dataset.p === p));
    $("#tab-nuevo").classList.toggle("oculto", p !== "nuevo");
    $("#tab-res").classList.toggle("oculto", p !== "res");
  },

  duracion(s) {
    if (s < 90) return `≈ ${Math.max(5, Math.round(s / 5) * 5)} s`;
    if (s < 5400) return `≈ ${Math.round(s / 60)} min`;
    return `≈ ${fmt.num(s / 3600, 1)} h`;
  },

  pintarCalidades() {
    const a = this.analisis, ms = a && a.bandas.length ? 1 : 0;
    $("#calidades").innerHTML = CALIDADES.map((c) => `<button class="calidad${c.v === this.calidad ? " activa" : ""}" data-v="${c.v}">
      <b>${c.n}${c.rec ? "<em>RECOMENDADA</em>" : ""}</b><small>${c.d}</small>
      <span class="t">${a ? this.duracion(c.s[ms] * a.disparos) : ""}</span></button>`).join("");
    $$(".calidad").forEach((b) => (b.onclick = () => { this.calidad = b.dataset.v; this.pintarCalidades(); }));
  },

  tema(t) {
    document.documentElement.dataset.tema = t;
    try { localStorage.setItem("tema", t); } catch (e) { /* sin almacenamiento: vale sólo para esta sesión */ }
    $("#ic-tema").setAttribute("href", t === "oscuro" ? "#i-sol" : "#i-luna");
    if (this.nube) { this.nube.fondo = this.colorVisor(); this.nube.redibujar(); }
  },

  colorVisor() {
    const c = getComputedStyle(document.documentElement).getPropertyValue("--visor").trim();
    return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255);
  },

  aviso(texto, tipo = "") {
    const t = $("#toast"); t.textContent = texto; t.className = "toast " + tipo;
    clearTimeout(this._toast); this._toast = setTimeout(() => t.classList.add("oculto"), 6000);
  },

  async estadoGpu() { this.mostrarGpu(await pywebview.api.estado_gpu()); },

  mostrarGpu(g) {
    const txt = $("#gpu-txt"), aviso = $("#gpu-aviso");
    $("#btn-instalar-gpu").classList.toggle("oculto", !(g.nvidia && !g.completo));
    if (!g.nvidia) {
      txt.textContent = "No hay tarjeta NVIDIA: se usará la CPU.";
      $("#sel-gpu option[value=si]").disabled = true;
      return;
    }
    txt.textContent = `${g.nombre.replace("NVIDIA ", "")} · ${fmt.num(g.libre_mb / 1024, 1)} de ${fmt.num(g.memoria_mb / 1024, 1)} GB libres`;
    const ajenos = (g.ajenos || []).filter(([, u]) => u >= 20);
    if (ajenos.length) {
      const lista = ajenos.map(([n]) => n.replace(/\.exe$/i, "")).join(", ");
      aviso.classList.remove("oculto");
      aviso.querySelector("span").textContent = `${lista} está usando la tarjeta gráfica. Ciérralo para terminar antes.`;
    } else aviso.classList.add("oculto");
  },

  async elegirCarpeta() {
    if (this.trabajando) return;
    const c = await pywebview.api.elegir_carpeta();
    if (!c) return;
    this.carpeta = c; this.analisis = null;
    $("#btn-imagenes").classList.add("lista");
    $("#z-titulo").textContent = c.nombre; $("#z-titulo").title = c.ruta;
    $("#z-sub").textContent = `${c.imagenes} imágenes · ${fmt.num(c.megabytes / 1024, 1)} GB`;
    $("#btn-procesar").disabled = true;
    const t = $("#analisis"); t.classList.remove("oculto"); t.innerHTML = '<span class="tenue">Revisando las fotos…</span>';
    this.pintarCalidades();
    pywebview.api.analizar_carpeta(c.ruta);
  },

  mostrarAnalisis(a) {
    const t = $("#analisis");
    if (this.carpeta && a.ruta !== this.carpeta.ruta) return;
    if (a.error) { t.innerHTML = `<div class="mal">${a.error}</div>`; return; }
    this.analisis = a;
    const chips = (a.rgb ? ['<span class="chip">RGB</span>'] : []).concat(a.bandas.map((b) => `<span class="chip ms">${b}</span>`));
    const avisos = [];
    if (a.gps === 0) avisos.push("Las fotos no tienen GPS: el mapa no estará georreferenciado.");
    if (a.ignoradas.length) avisos.push(`Se ignoran: ${[...new Set(a.ignoradas)].join(", ")}.`);
    t.innerHTML = `<div class="fila"><svg class="ic"><use href="#i-camara"/></svg>${a.camara}</div>
      <div class="chips">${chips.join("")}</div>
      <div class="tenue">${a.disparos} disparos${a.bandas.length ? ` · ${a.total} fotos` : ""}${a.gps ? " · con GPS" : ""}</div>
      ${avisos.map((x) => `<div class="mal">${x}</div>`).join("")}`;
    this.pintarCalidades();
    $("#btn-procesar").disabled = a.disparos < 3;
  },

  async procesar() {
    if (!this.carpeta || this.trabajando) return;
    this.trabajando = true;
    $("#consola").textContent = "";
    this.calidadEnCurso = this.calidad;
    $$("#pasos li").forEach((li, i) => {
      li.className = ""; li.querySelector(".t").textContent = "";
      li.querySelector(".n").textContent = this.nombreEtapa(this.etapas[i]);
    });
    $("#btn-procesar").disabled = true; $("#btn-imagenes").disabled = true; $("#btn-proyecto").disabled = true;
    $("#avance").classList.remove("oculto");
    this.pct(0); $("#av-titulo").textContent = "Preparando…"; $("#av-tiempo").textContent = "";
    this.controles(true); this.pausadoDesde = 0; this.confirmaCancelar = false;
    this.inicio = performance.now(); this.etapaActual = -1; this.inicioEtapa = this.inicio; this.total = 0;
    clearInterval(this.reloj); this.reloj = setInterval(() => this.tick(), 500);
    await pywebview.api.iniciar(this.carpeta.ruta, this.calidad, $("#sel-gpu").value);
    setTimeout(() => this.estadoGpu(), 3000);
  },

  nombreEtapa(n) { return (this.calidadEnCurso === "express" && ETAPA_EXPRESS[n]) || n; },

  controles(activos) {
    $("#btn-pausa").classList.toggle("oculto", !activos);
    $("#btn-cancelar").classList.toggle("oculto", !activos);
    $("#btn-pausa").disabled = $("#btn-cancelar").disabled = false;
    $("#btn-pausa span").textContent = "Pausar"; $("#ic-pausa").setAttribute("href", "#i-pausa");
    $("#btn-cancelar").textContent = "Cancelar";
    $("#btn-reg-error").classList.add("oculto");
    $("#av-nota").classList.add("oculto");
  },

  async alternarPausa() {
    const b = $("#btn-pausa"); b.disabled = true;
    if (!this.pausadoDesde) {
      if (await pywebview.api.pausar()) {
        this.pausadoDesde = performance.now();
        $("#btn-pausa span").textContent = "Reanudar"; $("#ic-pausa").setAttribute("href", "#i-play");
        const n = $("#av-nota"); n.textContent = "En pausa. El equipo queda libre, pero la memoria sigue reservada. Si cierras el programa se pierde el avance.";
        n.classList.remove("oculto");
        $("#est-texto").textContent = "En pausa";
      }
    } else {
      await pywebview.api.reanudar();
      const d = performance.now() - this.pausadoDesde;   // el reloj no cuenta el tiempo en pausa
      this.inicio += d; this.inicioEtapa += d; this.pausadoDesde = 0;
      $("#btn-pausa span").textContent = "Pausar"; $("#ic-pausa").setAttribute("href", "#i-pausa");
      $("#av-nota").classList.add("oculto");
    }
    b.disabled = false;
  },

  cancelar() {
    const b = $("#btn-cancelar");
    if (!this.confirmaCancelar) {
      this.confirmaCancelar = true; b.textContent = "¿Seguro? Pulsa otra vez";
      setTimeout(() => { if (this.trabajando && !b.disabled) { this.confirmaCancelar = false; b.textContent = "Cancelar"; } }, 4000);
      return;
    }
    b.disabled = true; $("#btn-pausa").disabled = true; b.textContent = "Cancelando…";
    pywebview.api.cancelar();
  },

  async guardarRegistro() {
    const r = await pywebview.api.guardar_registro();
    if (r && r.error) this.aviso(r.error, "error");
    else if (r) this.aviso("Registro guardado. Puedes enviarlo a quien te dio el programa.", "ok");
  },

  pct(f) {
    $("#pct").textContent = `${Math.floor(f * 100)}%`;
    $("#anillo-v").style.strokeDashoffset = 327 * (1 - f);
  },

  tick() {
    if (this.pausadoDesde) return;
    const s = (performance.now() - this.inicio) / 1000;
    const resta = this.calidadEnCurso !== "express" && this.total > 0.05 ? ` · quedan ${fmt.t(s / this.total - s)}` : "";
    $("#av-tiempo").textContent = fmt.t(s) + resta;
    const li = $(`#pasos li[data-i="${this.etapaActual}"]`);
    if (li) li.querySelector(".t").textContent = fmt.t((performance.now() - this.inicioEtapa) / 1000);
  },

  evento(e) {
    if (e.tipo === "log") {
      const c = $("#consola"); c.textContent += e.texto + "\n"; c.scrollTop = c.scrollHeight;
    } else if (e.tipo === "analisis") {
      this.mostrarAnalisis(e);
    } else if (e.tipo === "progreso") {
      const i = this.etapas.indexOf(e.etapa);
      if (i !== this.etapaActual && i >= 0) {
        $$("#pasos li").forEach((li, k) => {
          if (k < i) li.className = "hecho";
          if (k === i) li.className = "actual";
        });
        this.etapaActual = i; this.inicioEtapa = performance.now();
      }
      this.total = e.total;
      $("#av-titulo").textContent = this.nombreEtapa(e.etapa); this.pct(e.total);
      $("#est-texto").textContent = `Procesando: ${Math.floor(e.total * 100)} %`;
    } else if (e.tipo === "fin") {
      this.controles(false);
      const dur = fmt.t((performance.now() - this.inicio) / 1000);
      this.terminar(`Listo en ${dur}`);
      this.mostrarResultados(e.info, e.resumen);
      this.aviso(`¡Tu mapa está listo! Tardó ${dur}.`, "ok");
    } else if (e.tipo === "cancelado") {
      this.controles(false); this.pausadoDesde = 0;
      this.terminar("Cancelado");
      this.aviso("Procesamiento cancelado.", "ok");
    } else if (e.tipo === "error") {
      this.controles(false);
      $("#btn-pausa").classList.add("oculto"); $("#btn-cancelar").classList.add("oculto");
      $("#btn-reg-error").classList.remove("oculto");
      const li = $(`#pasos li[data-i="${this.etapaActual}"]`); if (li) li.className = "error";
      clearInterval(this.reloj); this.trabajando = false;
      $("#av-titulo").textContent = "No se pudo terminar";
      $("#av-tiempo").textContent = e.mensaje;
      $("#consola").textContent += "\nERROR: " + e.mensaje + "\n" + (e.detalle || "");
      $(".registro").open = true;
      this.terminar("Error en el procesamiento", true);
      this.aviso(e.mensaje, "error");
    } else if (e.tipo === "gpu") {
      this.mostrarGpu(e.estado);
    }
  },

  terminar(texto, mantener = false) {
    this.trabajando = false; clearInterval(this.reloj);
    $("#est-texto").textContent = texto;
    if (!mantener) $("#avance").classList.add("oculto");
    else $("#avance").onclick = (e) => { if (e.target.id === "avance") $("#avance").classList.add("oculto"); };
    $("#btn-procesar").disabled = !this.analisis; $("#btn-imagenes").disabled = false; $("#btn-proyecto").disabled = false;
    this.estadoGpu();
  },

  async abrirResultados() {
    const info = await pywebview.api.abrir_resultados();
    if (!info) return;
    if (info.error) return this.aviso(info.error, "error");
    this.mostrarResultados(info, info.resumen);
  },

  // ------------------------------------------------------------------ visor
  mostrarResultados(info, resumen) {
    this.info = info;
    $("#vacio").classList.add("oculto");
    $("#herr-vista").classList.remove("oculto");
    $("#pest-res").disabled = false;
    $("#btn-informe").disabled = !info.informe;
    $("#c-crs").textContent = info.epsg ? `EPSG:${info.epsg}` : "";
    this.listaCapas(info);
    this.resumen(resumen || {});
    $("#archivos").innerHTML = (info.archivos || []).map((f) =>
      `<li>${f.nombre}<span>${f.mb >= 1000 ? fmt.num(f.mb / 1000, 1) + " GB" : Math.ceil(f.mb) + " MB"}</span></li>`).join("");
    this.crearMapa();
    if (this.nube) { this.nube.n = 0; this.nubeCargada = null; }
    this.pestana("res");
    const c = info.capas;
    this.mostrarCapa(c.orto ? "orto" : c.multi ? "multi" : Object.keys(c)[0]);
  },

  listaCapas(info) {
    const c = info.capas, fila = (v, ic, n, sub, valor) =>
      `<button class="capa" data-v="${v}"><svg class="ic"><use href="#i-${ic}"/></svg><span>${n}${sub ? `<small>${sub}</small>` : ""}</span>${valor ? `<u>${valor}</u>` : ""}</button>`;
    const mapas = [], ind = [], terreno = [];
    if (c.orto) mapas.push(fila("orto", "orto", "Ortomosaico", "Color real"));
    if (c.multi) mapas.push(fila("multi", "orto", "Falso color", "Infrarrojo: la vegetación sana se ve roja"));
    for (const [n, d] of Object.entries(c)) {
      if (d.tipo === "indice") {
        const m = (info.indices || {})[n];
        ind.push(fila(n, "hoja", n, DESC_INDICE[n] || "", m ? fmt.num(m.mediana, 2) : ""));
      }
    }
    if (c.dem) terreno.push(fila("dem", "dem", "Modelo de elevación", ""));
    if (info.nube) terreno.push(fila("nube", "nube", "Nube de puntos 3D", ""));
    const grupo = (t, l) => (l.length ? `<div class="grupo-capas"><small>${t}</small>${l.join("")}</div>` : "");
    $("#capas").innerHTML = grupo("Mapas", mapas) + grupo("Salud del cultivo (valor típico)", ind) + grupo("Terreno", terreno);
    $$("#capas .capa").forEach((b) => (b.onclick = () => this.mostrarCapa(b.dataset.v)));
  },

  resumen(r) {
    const filas = [["Fotos alineadas", r["Cámaras alineadas"]], ["Resolución", r["Resolución ortomosaico (cm)"] && r["Resolución ortomosaico (cm)"] + " cm/píxel"],
      ["Área", r["Área (ha)"] && r["Área (ha)"] + " ha"], ["Puntos 3D", r["Puntos nube densa"]],
      ["Precisión vs GPS", r["Error cámaras vs GPS XY / Z (m)"] && r["Error cámaras vs GPS XY / Z (m)"] + " m"],
      ["Sistema", r["Sistema de coordenadas"]], ["Calidad", (CALIDADES.find((c) => c.v === r["Calidad"]) || {}).n],
      ["Procesador", r["Procesador"] && r["Procesador"].replace("NVIDIA GeForce ", "")]].filter((f) => f[1]);
    $("#resultados").innerHTML = filas.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join("");
  },

  crearMapa() {
    const g = this.info.rejilla, l = this.info.limites;
    if (this.mapa) { this.mapa.remove(); this.mapa = null; }
    const crs = L.extend({}, L.CRS.Simple, { transformation: new L.Transformation(1 / g.res0, -g.e0 / g.res0, -1 / g.res0, g.n1 / g.res0) });
    $("#mapa").classList.remove("oculto");
    this.mapa = L.map("mapa", { crs, minZoom: -3, maxZoom: g.zmax + 2, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90,
      attributionControl: false, zoomControl: false });
    this.limites = L.latLngBounds([[l[1], l[0]], [l[3], l[2]]]);
    L.control.scale({ imperial: false, maxWidth: 140, position: "bottomright" }).addTo(this.mapa);
    this.capasMapa = {};
    for (const nombre of Object.keys(this.info.capas)) {
      this.capasMapa[nombre] = L.tileLayer(`t/${this.info.version}/${nombre}/{z}/{x}/{y}.webp`, {
        tileSize: 256, minNativeZoom: 0, maxNativeZoom: g.zmax, minZoom: -3, maxZoom: g.zmax + 2, noWrap: true,
        bounds: this.limites, keepBuffer: 3, updateWhenZooming: false });
    }
    this.encuadrar();
    let pendiente = null;
    this.mapa.on("mousemove", (ev) => {
      const e = ev.latlng.lng, n = ev.latlng.lat;
      $("#c-coord").textContent = `E ${fmt.num(e, 2)}   N ${fmt.num(n, 2)}`;
      const ll = utmALatLon(e, n, this.info.epsg);
      $("#c-ll").textContent = ll ? `${ll[0].toFixed(7)}°, ${ll[1].toFixed(7)}°` : "";
      clearTimeout(pendiente);
      pendiente = setTimeout(async () => {
        const v = await (await fetch(`valor?e=${e}&n=${n}`)).json();
        const activa = this.info.capas[this.capa];
        $("#c-z").textContent = [v.dem !== undefined ? `Z ${fmt.num(v.dem, 2)} m` : "",
          activa && activa.tipo === "indice" && v[this.capa] !== undefined ? `${this.capa} ${v[this.capa].toFixed(3)}` : ""].filter(Boolean).join("   ");
      }, 60);
    });
  },

  encuadrar() { if (this.mapa) this.mapa.fitBounds(this.limites, { padding: [40, 40], animate: false }); },

  mostrarCapa(v) {
    this.capa = v;
    $$("#capas .capa").forEach((b) => b.classList.toggle("activa", b.dataset.v === v));
    const es3d = v === "nube";
    $("#mapa").classList.toggle("oculto", es3d);
    $("#nube3d").classList.toggle("oculto", !es3d);
    $("#herr-nube").classList.toggle("oculto", !es3d);
    if (es3d) { $("#leyenda").classList.add("oculto"); this.mostrarNube(); return; }
    $("#carga-nube").classList.add("oculto");
    // Leaflet pierde teselas si se añade una capa durante una animación de zoom: se espera a que acabe
    const cambiar = () => {
      for (const [n, capa] of Object.entries(this.capasMapa)) {
        if (n === v) capa.addTo(this.mapa); else this.mapa.removeLayer(capa);
      }
      this.mapa.invalidateSize();
    };
    if (this.mapa._animatingZoom) this.mapa.once("zoomend", cambiar); else cambiar();
    const c = this.info.capas[v], ley = $("#leyenda");
    if (c.tipo === "dem" || c.tipo === "indice") {
      const dem = c.tipo === "dem";
      ley.classList.remove("oculto");
      $("#leyenda-titulo").textContent = dem ? "Elevación (m)" : `${v} · ${DESC_INDICE[v] || ""}`;
      $("#leyenda-barra").style.background = PALETAS[c.tipo];
      $("#leyenda-min").textContent = fmt.num(c.rango[0], dem ? 1 : 2);
      $("#leyenda-max").textContent = fmt.num(c.rango[1], dem ? 1 : 2);
      $("#leyenda-nota").textContent = dem ? "" : "Rojo: menos vigor · Verde: más vigor";
    } else ley.classList.add("oculto");
  },

  async mostrarNube() {
    if (!this.nube) {
      try { this.nube = new Nube3D.Visor($("#nube3d")); }
      catch (err) { this.aviso("No se pudo iniciar el visor 3D: " + err.message, "error"); return; }
      this.nube.tam = +$("#tam-punto").value;
      this.nube.fondo = this.colorVisor();
    }
    if (this.nubeCargada === this.info.version) { this.nube.redibujar(); return; }
    const carga = $("#carga-nube"); carga.classList.remove("oculto");
    try {
      const meta = await (await fetch("nube.json")).json();
      const r = await fetch("nube.bin");
      const total = +r.headers.get("Content-Length") || meta.puntos * 9;
      const buf = new Uint8Array(total); let n = 0;
      const lector = r.body.getReader();
      for (;;) {
        const { done, value } = await lector.read(); if (done) break;
        buf.set(value, n); n += value.length;
        $("#carga-pct").textContent = Math.round(100 * n / total) + "%";
        $("#carga-nube .prog i").style.width = (100 * n / total) + "%";
      }
      this.nube.cargar(meta, buf.buffer);
      this.nubeCargada = this.info.version;
      $("#c-z").textContent = `${meta.puntos.toLocaleString("es")} puntos`;
    } catch (err) {
      this.aviso("No se pudo cargar la nube de puntos: " + err.message, "error");
    } finally { carga.classList.add("oculto"); }
  },
};

window.app = app;
window.addEventListener("pywebviewready", () => app.iniciar());
