# Curso: RAG en producción, explicado desde un sistema real

> Material de estudio construido sobre el RAG del chatbot de AMICANA.
> No es teoría abstracta: cada concepto se explica con el código que efectivamente
> corrió, y cada error documentado acá se rompió de verdad antes de arreglarse.


**El sistema:** un asistente conversacional para el Instituto Cultural Argentino
Norteamericano (AMICANA), en Mendoza. Recupera sobre un corpus de preguntas
frecuentes, cursos y avisos, y responde fundamentado únicamente en lo recuperado.

**El stack:** embeddings de Google Gemini (768 dimensiones) · Postgres con
pgvector sobre Supabase · Groq para la generación · n8n como orquestador.

**El código:** los workflows están en
[MartinZ18/Amicana-v2.0/n8n](https://github.com/MartinZ18/Amicana-v2.0/tree/main/n8n)
— pipeline de ingesta, pipeline de consulta y el esquema SQL de la base vectorial.

**Por qué existe este documento:** porque casi todo lo que se escribe sobre RAG
explica cómo armarlo y nada explica cómo se rompe. Las seis fallas de la
sección 8 pasaron de verdad, y cinco de las seis devolvían `200 OK`.

---

## Índice

1. [El problema que resuelve RAG](#1-el-problema-que-resuelve-rag)
2. [Anatomía del sistema](#2-anatomía-del-sistema)
3. [Embeddings: convertir significado en números](#3-embeddings-convertir-significado-en-números)
4. [La base vectorial](#4-la-base-vectorial)
5. [Chunking: cómo se parte la información](#5-chunking-cómo-se-parte-la-información)
6. [El pipeline de ingesta](#6-el-pipeline-de-ingesta)
7. [El pipeline de consulta](#7-el-pipeline-de-consulta)
8. [Seis fallas reales y qué enseña cada una](#8-seis-fallas-reales-y-qué-enseña-cada-una)
9. [Metodología: cómo se depura lo que no se ve](#9-metodología-cómo-se-depura-lo-que-no-se-ve)
10. [Límites del sistema actual](#10-límites-del-sistema-actual)
11. [Glosario](#11-glosario)
12. [Preguntas de entrevista](#12-preguntas-de-entrevista)

---

## 1. El problema que resuelve RAG

### El punto de partida

Un modelo de lenguaje sabe lo que había en su corpus de entrenamiento. Nada más.
No sabe cuánto cuesta la cuota de un curso del Instituto AMICANA, qué exámenes se
rinden ahí, ni qué avisos se publicaron esta semana. Esa información no existía
cuando el modelo se entrenó, y aunque hubiera existido, no habría forma de
actualizarla sin reentrenar.

Frente a eso hay tres caminos:

**Camino 1: escribir el conocimiento en el prompt.**
Es lo que hacía el chatbot antes de este trabajo. En el system prompt había un
bloque de texto fijo con los datos del instituto. Funciona hasta que:

- alguien agrega un curso y el prompt queda desactualizado en silencio;
- la información crece y no entra en la ventana de contexto;
- se paga por procesar todo ese texto en cada consulta, incluso cuando el usuario
  pregunta algo que no tiene nada que ver.

**Camino 2: fine-tuning.**
Reentrenar el modelo con los datos propios. Caro, lento, y el conocimiento queda
congelado en los pesos: para actualizar un precio hay que volver a entrenar.
Además, el fine-tuning enseña *estilo y formato* mucho mejor que *hechos*. Un
modelo afinado con datos del instituto va a sonar como el instituto, pero puede
inventar precios con total soltura.

**Camino 3: RAG.**
Guardar el conocimiento afuera, buscar en cada consulta solo los fragmentos
relevantes, y pasárselos al modelo como contexto. El modelo no "sabe" nada nuevo:
lee lo que le damos y redacta.

### La definición

**RAG (Retrieval-Augmented Generation)** = recuperación + generación.

```
pregunta del usuario
      ↓
[RECUPERACIÓN]  buscar en una base de conocimiento los fragmentos más relevantes
      ↓
[AUMENTO]       inyectar esos fragmentos en el prompt como contexto
      ↓
[GENERACIÓN]    el LLM redacta una respuesta usando SOLO ese contexto
      ↓
respuesta fundamentada
```

Lo importante conceptualmente: **RAG separa el conocimiento del razonamiento.**
El LLM aporta comprensión del lenguaje y capacidad de redacción. La base vectorial
aporta los hechos. Cambiar un precio es un `UPDATE`, no un reentrenamiento.

### Por qué importa la palabra "fundamentada"

Un LLM sin contexto que no sabe algo, lo inventa. No miente por malicia: predice el
token más probable, y "la cuota es de $12.000" es una secuencia perfectamente
probable aunque sea falsa.

Un RAG bien armado le da al modelo dos cosas: el contexto, y **el permiso explícito
de decir que no sabe**. Ese permiso va en el system prompt:

```
Responde SOLO con la información del contexto de abajo.
Si el contexto no alcanza para responder, indica que consulten en secretaría.
```

Esa segunda oración es la que separa un asistente confiable de un generador de
mentiras con formato profesional. Un RAG que no puede decir "no sé" es peor que no
tener RAG, porque suena más creíble.

---

## 2. Anatomía del sistema

### Los dos pipelines

Todo sistema RAG tiene dos flujos que corren en momentos distintos:

```
┌─────────────────────── INGESTA (offline, ocasional) ───────────────────────┐
│                                                                            │
│  Fuentes de datos          Embeddings              Base vectorial          │
│  ┌──────────────┐         ┌──────────┐            ┌─────────────┐          │
│  │ /chatbot/faq │────┐    │          │            │  documents  │          │
│  │ /cursos/info │────┼───▶│  Gemini  │───────────▶│  (pgvector) │          │
│  │ /avisos      │────┘    │  768 dim │            │   9 filas   │          │
│  └──────────────┘         └──────────┘            └─────────────┘          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── CONSULTA (online, cada mensaje) ────────────────────┐
│                                                                            │
│  "¿qué exámenes puedo rendir?"                                             │
│         ↓                                                                  │
│    [embedding de la pregunta]  ──── Gemini, mismo modelo, 768 dim          │
│         ↓                                                                  │
│    [match_documents(vector, 4)]  ── Supabase, distancia coseno             │
│         ↓                                                                  │
│    4 fragmentos más parecidos                                              │
│         ↓                                                                  │
│    [prompt: sistema + contexto + pregunta]                                 │
│         ↓                                                                  │
│    [Groq / gpt-oss-120b]                                                   │
│         ↓                                                                  │
│    "Puede rendir ECECE (A2), TOEIC Bridge (A2), TOEIC (B1)..."             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Regla de oro:** el modelo de embeddings de la ingesta y el de la consulta tienen
que ser **el mismo**, con **la misma dimensionalidad**. Un vector generado con un
modelo no es comparable con uno generado por otro: viven en espacios distintos.
Comparar sus distancias da números que parecen válidos y son basura. Es uno de los
errores más difíciles de detectar porque no lanza ninguna excepción.

### El stack elegido y por qué

| Capa | Herramienta | Razón |
|---|---|---|
| Embeddings | Google Gemini `gemini-embedding-001` | Capa gratuita generosa, soporta dimensionalidad configurable |
| Base vectorial | Supabase (Postgres + pgvector) | Postgres de verdad: se consulta con SQL, se hacen joins, hay transacciones |
| LLM | Groq `openai/gpt-oss-120b` | Inferencia muy rápida, capa gratuita, soporta modo JSON |
| Orquestación | n8n | Visual, versionable como JSON, sin build |

Sobre Supabase: mucha gente arranca con una base vectorial dedicada (Pinecone,
Weaviate, Qdrant). Para un volumen chico o mediano, usar Postgres con pgvector es
casi siempre mejor decisión. Los datos viven al lado del resto de la aplicación,
se pueden filtrar con `WHERE` común y corriente, y no hay un servicio más que
mantener. Migrar a algo dedicado tiene sentido cuando el volumen o la latencia lo
justifiquen — no antes.

---

## 3. Embeddings: convertir significado en números

### Qué es un embedding

Un embedding es un vector de números que representa el **significado** de un texto.
En este sistema, cada texto se convierte en 768 números decimales:

```
"¿Qué exámenes internacionales puedo rendir?"
      ↓ modelo de embeddings
[0.0234, -0.1871, 0.0455, ..., 0.0912]   ← 768 valores
```

La propiedad clave: **textos con significado parecido producen vectores cercanos en
el espacio**, aunque no compartan una sola palabra.

```
"¿qué exámenes puedo rendir?"        ┐
                                     ├─ vectores cercanos entre sí
"¿qué certificaciones puedo sacar?"  ┘

"¿cuánto sale la cuota?"             ← vector lejano de los dos anteriores
```

Esto es lo que separa la búsqueda semántica de la búsqueda por palabras clave. Un
`LIKE '%examen%'` no encuentra un documento que habla de "certificaciones
internacionales". Un embedding sí, porque captura el concepto y no la cadena de
caracteres.

### La dimensionalidad

768 no es un número mágico: es un parámetro. Los modelos modernos de Google usan
una técnica llamada **Matryoshka Representation Learning**, que permite pedir el
mismo embedding en distintos tamaños:

```javascript
body: {
  model: 'models/gemini-embedding-001',
  content: { parts: [{ text: doc.content }] },
  outputDimensionality: 768,     // por default devolvería 3072
}
```

El nombre viene de las muñecas rusas: el vector de 768 dimensiones es literalmente
el prefijo del de 3072, entrenado para que las primeras dimensiones concentren la
mayor parte de la información.

**El compromiso:**

| Dimensiones | Precisión | Almacenamiento | Velocidad de búsqueda |
|---|---|---|---|
| 3072 | máxima | 4x | más lenta |
| 768 | muy buena | 1x | más rápida |
| 256 | aceptable | 0.33x | muy rápida |

Para un corpus de 9 documentos, 768 sobra. La decisión de diseño real acá no fue de
precisión sino de **compatibilidad**: la tabla se declaró como `vector(768)`, así
que el modelo debe producir exactamente 768. Si mañana se cambia el modelo, o se
mantiene la dimensionalidad, o hay que migrar la tabla y reindexar todo.

### Detalle sobre normalización

Al truncar un embedding de 3072 a 768, la magnitud del vector cambia. La
documentación de Google recomienda normalizarlo antes de compararlo.

En este sistema **no hace falta**, y vale entender por qué: usamos **distancia
coseno**, que mide el ángulo entre dos vectores e ignora completamente su magnitud.
Si usáramos distancia euclidiana (L2), la normalización sería obligatoria.

Es un buen ejemplo de por qué conviene entender la métrica que se usa en vez de
copiar configuración: la misma decisión es crítica o irrelevante según el operador
de distancia que elijas.

---

## 4. La base vectorial

### El esquema

```sql
create extension if not exists vector;

create table if not exists documents (
  id          bigint generated always as identity primary key,
  source      text not null,        -- 'faq' | 'curso' | 'aviso'
  external_id text not null,        -- id del registro de origen
  content     text not null,        -- el texto que se embebió
  embedding   vector(768),          -- el vector
  updated_at  timestamptz not null default now(),
  unique (source, external_id)      -- clave natural: permite upsert
);
```

Tres decisiones para mirar de cerca:

**`content` guarda el texto original.** El embedding no es reversible: no se puede
reconstruir el texto desde los 768 números. Si solo guardáramos el vector, la
búsqueda encontraría el documento correcto y no habría nada que pasarle al modelo.

**`unique (source, external_id)`** es lo que hace la ingesta idempotente. Sin esa
restricción, cada corrida duplicaría todo y el retrieval devolvería el mismo
documento cuatro veces, desperdiciando el presupuesto de contexto.

**`source`** permite filtrar por tipo. Hoy no se usa, pero habilita cosas como
"buscar solo en avisos" sin cambiar el esquema.

### La función de búsqueda

```sql
create or replace function match_documents(
  query_embedding vector(768),
  match_count int default 4
)
returns table (source text, content text, similarity float)
language sql stable
as $$
  select source, content, 1 - (embedding <=> query_embedding) as similarity
  from documents
  order by embedding <=> query_embedding
  limit match_count;
$$;
```

El operador `<=>` es **distancia coseno** en pgvector. Devuelve 0 para vectores
idénticos y 2 para opuestos. Como es más intuitivo hablar de "parecido" que de
"distancia", se convierte con `1 - distancia`, dando un valor donde 1 es idéntico.

Los otros operadores de pgvector:

| Operador | Métrica | Cuándo |
|---|---|---|
| `<=>` | Distancia coseno | Texto. Ignora magnitud. **El default sensato** |
| `<->` | Distancia L2 (euclidiana) | Cuando la magnitud significa algo |
| `<#>` | Producto interno negativo | Vectores ya normalizados, es más rápido |

En una consulta real de este sistema:

```
[0.777] (faq)   Pregunta: ¿Qué modalidades ofrecen? / Presencial, virtual e híbrida...
[0.737] (faq)   Pregunta: ¿Cómo me inscribo? / Podés acercarte a la sede...
[0.730] (faq)   Pregunta: ¿Cómo pago la cuota? / Por MercadoPago o generando...
[0.710] (curso) Curso: Conversation Class. Práctica oral intensiva...
```

Fijate que ninguno llega a 0.8. Eso es normal y esperable: dos textos distintos que
hablan del mismo tema rondan 0.7–0.8. Similitudes de 0.95+ aparecen cuando el texto
es casi idéntico. Si esperabas números cercanos a 1, el modelo mental estaba mal
calibrado, no el sistema.

### El índice, o cómo optimizar de más rompe todo

Este es el error más instructivo de todo el proyecto.

El setup original creaba un índice:

```sql
create index documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);
```

Parece prolijo. Es exactamente lo que sale en cualquier tutorial. **Y rompía el
sistema por completo.**

**Cómo funciona ivfflat:** agrupa los vectores en `lists` grupos por cercanía
(k-means). Al buscar, en vez de comparar contra todos los vectores, compara solo
contra los grupos más prometedores. Cuántos grupos revisa lo define el parámetro
`ivfflat.probes`, **que vale 1 por default**.

Con 100 listas y 9 documentos:

```
lista 1:  [doc]         ← la búsqueda entra acá y solo mira este
lista 2:  []
lista 3:  [doc, doc]
...
lista 100: []
```

La búsqueda escanea **una sola lista**, encuentra un documento, y devuelve uno.
Pidas 4, 9 o 50. Siempre uno.

**Cómo se veía el síntoma:** el chatbot contestaba *"el contexto disponible no
incluye información sobre modalidades ni cuotas"* — con los cuatro documentos de
cursos cargados en la tabla. El modelo se portaba impecable: no alucinaba, decía la
verdad sobre lo que le habían dado. Lo estaban dejando sin datos.

**La solución fue borrar el índice.**

```sql
drop index if exists documents_embedding_idx;
```

Con menos de mil filas, el escaneo secuencial es **exacto** y prácticamente
instantáneo. El índice ivfflat es una búsqueda **aproximada**: cambia precisión por
velocidad. Sobre 9 filas no hay velocidad que ganar, solo precisión que perder.

**La lección general:** un índice vectorial es una optimización, y como toda
optimización, tiene un umbral por debajo del cual solo hace daño. Los índices
aproximados degradan el *recall* — la fracción de resultados relevantes que
realmente encontrás. Y lo hacen en silencio: no hay error, no hay warning, solo
resultados peores.

Cuándo sí crearlo:

```sql
-- recién arriba de unos miles de filas
create index documents_embedding_idx
  on documents using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);      -- regla práctica: filas / 1000
set ivfflat.probes = 4;   -- y subir probes en la sesión
```

---

## 5. Chunking: cómo se parte la información

### El problema

Los documentos suelen ser más largos que lo que conviene embeber. Un PDF de 40
páginas convertido en un solo vector produce un embedding que no representa nada:
es el promedio de cuarenta temas distintos. Al buscar, no matchea bien con nada.

La solución es partir el documento en fragmentos (*chunks*) y embeber cada uno por
separado.

### Las estrategias

**Por tamaño fijo.** Cortar cada N caracteres. Simple y brutal: parte oraciones al
medio y separa una pregunta de su respuesta.

**Por tamaño con solapamiento.** Igual, pero cada fragmento repite las últimas N
palabras del anterior. El solapamiento evita perder información en los bordes. Es
el default razonable para texto corrido.

**Recursiva por estructura.** Intenta cortar primero por párrafos; si el fragmento
sigue siendo grande, por oraciones; después por palabras. Respeta la estructura del
documento. Es lo que usan la mayoría de las librerías serias.

**Semántica.** Embeber oración por oración y cortar donde el significado cambia
bruscamente. Es la más costosa y la que mejores resultados da en textos largos.

### Qué se hizo acá, y por qué

**Nada de eso.** Este sistema usa **un documento = un chunk**, porque las fuentes ya
vienen atomizadas:

```javascript
for (const item of faqResp.data.faq || []) {
  docs.push({
    source: 'faq',
    external_id: item.q,
    content: `Pregunta: ${item.q}\nRespuesta: ${item.a}`,
  });
}

for (const c of cursosResp.data.cursos || []) {
  docs.push({
    source: 'curso',
    external_id: String(c.id),
    content: `Curso: ${c.nombre}. ${c.descripcion || ''} Modalidad: ${c.modalidad}. ` +
             `Categoria: ${c.categoria}. Cuota mensual: $${c.monto_cuota}.`,
  });
}
```

Un FAQ ya es una unidad de significado completa: pregunta más respuesta. Partirlo
sería destruirlo. Un curso también: nombre, modalidad, categoría y precio forman un
bloque que se consulta junto.

**La lección:** el chunking no es un paso obligatorio que hay que cumplir. Es una
respuesta a un problema — documentos demasiado largos. Si tus datos ya vienen en
unidades semánticas naturales, aplicar chunking los empeora.

### El detalle de la redacción del chunk

Mirá cómo se construye el texto de un curso. No es un volcado de campos:

```
Curso: English A1 - Beginners. Curso inicial de inglés. Modalidad: presencial.
Categoria: idiomas. Cuota mensual: $8500.
```

Está escrito como una oración, con las etiquetas adentro del texto. Eso importa
porque el modelo de embeddings fue entrenado con lenguaje natural. Un chunk que
dice `{"nombre":"English A1","modalidad":"presencial"}` produce un embedding peor
que el mismo dato redactado como prosa.

**Regla:** el chunk se escribe para que lo lea un humano, no para que lo parsee una
máquina. Después ese mismo texto es el que ve el LLM como contexto, así que
cualquier esfuerzo de redacción se aprovecha dos veces.

---

## 6. El pipeline de ingesta

### Estructura

```
[Trigger manual] → [Fetch + Embed + Upsert] → { upserted: 9, total: 9, errors: [] }
```

Tres nodos, uno con toda la lógica. Para un volumen chico, concentrar la lógica en
un nodo de código es más legible que quince nodos visuales encadenados.

### El código, comentado

```javascript
const helpers = this.helpers;              // cliente HTTP de n8n
const BASE = $env.FASTAPI_BASE_URL;        // backend, por variable de entorno

// 1. TRAER — las tres fuentes en paralelo
const [faqResp, cursosResp, avisosResp] = await Promise.all([
  get('/chatbot/faq'),
  get('/chatbot/cursos/info'),
  get('/chatbot/avisos'),
]);
```

`Promise.all` en vez de tres `await` secuenciales: las tres peticiones son
independientes, no hay razón para esperarlas en fila. Detalle chico, hábito
correcto.

```javascript
// 2. NORMALIZAR — cada fuente a la misma forma
const docs = [];
// ... (ver sección de chunking)

// 3. EMBEBER Y GUARDAR — uno por uno
let upserted = 0;
const errors = [];

for (const doc of docs) {
  try {
    const embedResp = await helpers.httpRequest({
      method: 'POST',
      url: `https://generativelanguage.googleapis.com/v1beta/models/` +
           `gemini-embedding-001:embedContent?key=${geminiKey}`,
      body: {
        model: 'models/gemini-embedding-001',
        content: { parts: [{ text: doc.content }] },
        outputDimensionality: 768,
      },
      json: true,
    });
    const embedding = embedResp.embedding.values;

    await helpers.httpRequest({
      method: 'POST',
      url: `${supabaseUrl}/rest/v1/documents?on_conflict=source,external_id`,
      headers: {
        apikey: supabaseKey,
        Authorization: `Bearer ${supabaseKey}`,
        Prefer: 'resolution=merge-duplicates',   // ← el upsert
      },
      body: { ...doc, embedding, updated_at: new Date().toISOString() },
      json: true,
    });
    upserted++;
  } catch (e) {
    errors.push({ doc: doc.external_id, error: e.message });   // ← no aborta
  }
}

return [{ json: { upserted, total: docs.length, errors } }];
```

### Las tres propiedades que hacen esto operable

**Idempotencia.** `Prefer: resolution=merge-duplicates` sobre la restricción única
convierte el `INSERT` en un `UPSERT`. Correrlo diez veces deja la tabla igual que
correrlo una. Sin esto, cada ejecución duplicaría el corpus. Un pipeline de datos
que no se puede re-ejecutar sin miedo no es operable: obliga a razonar sobre el
estado previo antes de cada corrida.

**Tolerancia parcial a fallas.** El `try/catch` está **adentro** del bucle. Si el
documento 5 falla, los otros 8 se guardan igual. Si estuviera afuera, un solo
error dejaría la ingesta a medias y sin saber dónde quedó.

**Observabilidad.** Devuelve `{ upserted, total, errors }`. No un `"ok"`. Si
`upserted < total`, el array `errors` dice cuál falló y por qué. Un pipeline que
devuelve "listo" es un pipeline que no se puede monitorear.

Esas tres propiedades no son opcionales en un pipeline de datos. Son la diferencia
entre algo que corre en tu máquina y algo que corre todos los días sin que nadie lo
mire.

---

## 7. El pipeline de consulta

### El grafo completo

```
Webhook
  ↓
Cargar sesión (HTTP → backend)
  ↓
Hidratar sesión (historial + estado)
  ↓
Router LLM (Groq, modo JSON) ──── clasifica intención
  ↓
Parsear intención
  ↓
Switch ─┬─ AUTH        → buscar alumno
        ├─ ESTADO      → cuotas del alumno
        ├─ PAGAR       → crear pago en MercadoPago
        ├─ CONFIRMAR   → registrar pago
        ├─ CERRAR      → despedida
        └─ FUERA_SCOPE → ► RAG ◄ → formatear
                                      ↓
                              Guardar sesión (HTTP)
                                      ↓
                              Responder al webhook
```

Observación de arquitectura: **el RAG no atiende todas las consultas.** Solo la
rama `FUERA_SCOPE`. Si el alumno pregunta por sus cuotas, eso se resuelve con una
consulta directa a la base transaccional, que es exacta. Meter datos personales de
cuotas en un pipeline de similitud semántica sería reemplazar una respuesta exacta
por una aproximada. RAG es para conocimiento general, no para consultar registros.

### El router

Antes del RAG hay un LLM que clasifica la intención y devuelve JSON estricto:

```json
{
  "model": "openai/gpt-oss-120b",
  "messages": "...",
  "temperature": 0.1,
  "response_format": { "type": "json_object" }
}
```

Dos parámetros para entender:

**`temperature: 0.1`.** La temperatura controla cuánta aleatoriedad hay al elegir
el siguiente token. Para clasificar se quiere el resultado más probable siempre —
la misma pregunta debe dar la misma intención. Para redactar la respuesta final se
usa un valor más alto, porque ahí sí interesa que suene natural.

**`response_format: json_object`.** Fuerza al modelo a producir JSON válido. Sin
eso, un modelo puede contestar "Claro, acá va el JSON:" seguido del objeto, y el
`JSON.parse` explota. Esta restricción fue la que condicionó qué modelo se podía
elegir cuando el anterior fue dado de baja: no todos los soportan.

### El nodo RAG

```javascript
const helpers = this.helpers;
const prev = $input.first().json;
let ragError = null;

// respuesta de reserva: la que ya generó el router
let response = prev.response || 'Puedo ayudarle con información sobre cursos...';

try {
  // 1. embeber la pregunta — MISMO modelo y dimensión que la ingesta
  const embedResp = await helpers.httpRequest({ /* gemini-embedding-001, 768 */ });
  const embedding = embedResp.embedding.values;

  // 2. recuperar los 4 fragmentos más cercanos
  const matches = await helpers.httpRequest({
    url: `${supabaseUrl}/rest/v1/rpc/match_documents`,
    body: { query_embedding: embedding, match_count: 4 },
  });

  // 3. armar el contexto
  const context = (matches || []).map(m => `- ${m.content}`).join('\n');

  // 4. generar SOLO si hay contexto
  if (context) {
    const groqResp = await helpers.httpRequest({
      body: {
        model: 'openai/gpt-oss-120b',
        messages: [
          { role: 'system', content:
            `Sos Ianna, asistente del Instituto AMICANA. Responde SOLO con la ` +
            `informacion del contexto de abajo, en espanol formal (trato de usted), ` +
            `1 a 3 oraciones. Si el contexto no alcanza para responder, indica que ` +
            `consulten en secretaria.\n\nContexto:\n${context}` },
          { role: 'user', content: prev.message },
        ],
        temperature: 0.2,
        max_tokens: 300,
      },
    });
    response = groqResp.choices?.[0]?.message?.content || response;
  }
} catch (e) {
  ragError = e.message;     // el retrieval es best-effort, pero deja rastro
}

return [{ json: { ...prev, response, ...(ragError ? { _rag_error: ragError } : {}) } }];
```

### Anatomía del prompt

Cada cláusula del system prompt hace un trabajo específico:

| Fragmento | Qué controla |
|---|---|
| `Sos Ianna, asistente del Instituto AMICANA` | Identidad y tono |
| `Responde SOLO con la informacion del contexto` | **Antialucinación.** La instrucción central |
| `en espanol formal (trato de usted)` | Registro |
| `1 a 3 oraciones` | Longitud — sin esto se va a cinco párrafos |
| `Si el contexto no alcanza... consulten en secretaria` | **La salida honesta.** Le da permiso de no saber |
| `Contexto:\n${context}` | Los datos recuperados |

El orden importa. Las instrucciones van **antes** del contexto, no después. Los
modelos les prestan más atención a las instrucciones que abren el prompt, y si el
contexto va primero, en corpus largos las reglas quedan sepultadas.

### La prueba de que funciona

No es que responda bien. Es que responde bien **y también responde mal cuando
corresponde**:

| Pregunta | Respuesta | Qué demuestra |
|---|---|---|
| "¿qué exámenes internacionales puedo rendir?" | Cita ECECE, TOEIC Bridge, TOEIC, TOEFL iBT | Recupera y fundamenta |
| "¿qué modalidades y cuánto sale la cuota?" | Combina FAQ + datos de un curso con precio | Integra múltiples fragmentos |
| "¿tienen clases de alemán los sábados?" | Deriva a secretaría | **No alucina** |
| "y de esas tres, ¿cuál me conviene?" | Entiende a qué se refiere "esas tres" | Mantiene contexto conversacional |

La tercera fila es la más valiosa. Cualquier RAG mal armado aprueba las dos
primeras. La tercera es la que separa un sistema confiable de uno peligroso.

---

## 8. Seis fallas reales y qué enseña cada una

Todo lo que sigue rompió de verdad. Ninguno era un problema de configuración.

### Falla 1: el `catch` silencioso

```javascript
} catch (e) {
  // Si falla el retrieval, se queda con la respuesta de fallback.
}
```

Este bloque, aparentemente prudente, escondía **tres bugs simultáneos**. El sistema
devolvía `200 OK` con castellano formal impecable. Parecía andar.

El nodo tardaba 12 milisegundos. Dos llamadas HTTP a servicios externos no se hacen
en 12 milisegundos. Ese número era la única evidencia visible de que algo estaba
mal, y solo si uno lo miraba.

**El arreglo:**

```javascript
} catch (e) {
  // El retrieval es best-effort: si falla se conserva el fallback del router,
  // pero el motivo queda expuesto en _rag_error para poder diagnosticarlo.
  ragError = e.message;
}
return [{ json: { ...prev, response, ...(ragError ? { _rag_error: ragError } : {}) } }];
```

Fijate que **el comportamiento no cambió**: sigue degradando con elegancia, sigue
sin tirar el pipeline abajo. Lo único que cambió es que ahora deja rastro.

> **Lección.** Degradar con elegancia y ocultar el error son cosas distintas. Un
> `catch` vacío no es manejo de errores: es esconder la basura abajo de la
> alfombra. Si una falla es tolerable, registrala. Si no es tolerable, propagala.
> Lo que nunca es válido es tragártela.

### Falla 2: la API que ya no existía

El código llamaba a `$helpers.httpRequest`, que era la forma correcta en n8n 1.x.
En n8n 2.x, esa variable **no existe**.

En vez de adivinar cuál era el reemplazo, se instrumentó una sonda dentro del
entorno real:

```javascript
probe = 'fetch=' + (typeof fetch)
      + ' | globalThis.fetch=' + (typeof globalThis.fetch)
      + ' | $http=' + (typeof $http)
      + ' | this.helpers=' + (typeof (this && this.helpers))
      + ' | require=' + (typeof require);
```

Resultado:

```
fetch=undefined | globalThis.fetch=undefined | $http=undefined |
this.helpers=object | require=function
```

Respuesta definitiva en un solo intento: `this.helpers`. Ni `fetch` global (el
sandbox no lo expone) ni `$http`.

> **Lección.** Cuando no sabés qué hay disponible en un entorno, no busques en foros
> ni pruebes a ciegas: preguntale al entorno. Una sonda que imprime `typeof` de los
> candidatos resuelve en una iteración lo que la prueba y error resuelve en seis.

### Falla 3 y 4: modelos dados de baja

```
404: models/text-embedding-004 is not found for API version v1beta
"The model `llama-3.3-70b-versatile` does not exist or you do not have access to it"
```

Ambos modelos existían cuando se escribió el workflow. Los proveedores los
retiraron.

Se resolvió listando lo que había disponible en ese momento:

```bash
GET https://generativelanguage.googleapis.com/v1beta/models
GET https://api.groq.com/openai/v1/models
```

Y verificando que el reemplazo cumpliera los requisitos concretos: para embeddings,
que soportara `outputDimensionality: 768`; para el router, que soportara
`response_format: json_object`.

> **Lección.** Los identificadores de modelo son dependencias externas con ciclo de
> vida propio, y más cortas que las de una librería. Un `model:` hardcodeado en
> quince nodos es un punto de falla esperando fecha. Convienen en una variable de
> entorno, igual que una URL de base de datos. Y el endpoint de listado de modelos
> es la fuente de verdad — no la documentación, que puede estar desactualizada.

### Falla 5: el nodo que descartaba el trabajo del anterior

El nodo RAG generaba la respuesta correcta. El nodo siguiente la tiraba a la basura:

```javascript
const prev = $('Parse Intent').first().json;   // ← lee un nodo ANTERIOR al RAG
```

En n8n, `$('Nombre')` accede a la salida de cualquier nodo por nombre, salteando el
flujo. El cableado visual mostraba `RAG → Formatear`, pero el código de Formatear
leía de otro lado. **La flecha en el canvas mentía.**

```javascript
const prev = $input.first().json;              // ← lee su propia entrada
```

Lo más incómodo: el validador de n8n lo había marcado.

```
Fuera Scope — Format Response: "Code doesn't reference input data"
```

Ese warning estaba en una lista de 37, junto a treinta y seis avisos cosméticos
sobre `typeVersion` desactualizados. Se despachó todo el bloque como ruido.

> **Lección doble.** Primera: en herramientas visuales, el diagrama muestra el
> cableado, no el flujo de datos. Un nodo puede ignorar su entrada y leer de
> cualquier lado; la única verdad es el código. Segunda, y más incómoda: cuando un
> validador tira 37 warnings, la tentación de tratarlos como decoración es enorme.
> Ahí adentro había uno que describía exactamente el bug. Los warnings no se
> descartan en bloque: se descartan de a uno, con una razón.

### Falla 6: el índice que estrangulaba el recall

Explicada en detalle en la sección 4. El resumen: un índice `ivfflat` con
`lists = 100` sobre una tabla de 9 filas limitaba toda búsqueda a un solo
resultado.

> **Lección.** Toda optimización tiene un umbral por debajo del cual perjudica. Los
> índices aproximados cambian precisión por velocidad — sobre un corpus chico no hay
> velocidad que ganar y sí precisión que perder. Y lo peor: degradan en silencio.
> No hay error ni warning, solo resultados peores que nadie relaciona con el índice.

### El patrón detrás de las seis

Cinco de las seis fallas devolvían `200 OK`.

Ninguna tiró una excepción visible. Ninguna apareció como error en un log. El
sistema respondía en castellano correcto y con formato profesional mientras estaba
roto de tres formas al mismo tiempo.

> **La lección que engloba a todas: en sistemas con LLMs, "responde" y "funciona"
> son propiedades independientes.** Un servicio tradicional que se rompe tira un
> 500. Uno con LLM devuelve un párrafo bien redactado que suena perfectamente
> razonable. La verificación no puede ser "¿contestó?". Tiene que ser "¿contestó
> **esto**, que es lo que corresponde según los datos que tiene?".

---

## 9. Metodología: cómo se depura lo que no se ve

### Aislar cada capa

Cuando el chatbot devolvía una respuesta genérica, había cinco sospechosos: el
embedding, la base vectorial, el retrieval, el LLM, o el cableado del workflow.
Probarlos juntos es adivinar. Se probaron por separado:

```
1. ¿La API de embeddings responde?        → llamada directa, contar dimensiones
2. ¿Supabase tiene los documentos?        → SELECT count(*)
3. ¿El retrieval devuelve algo?           → llamar match_documents a mano
4. ¿Cuántas filas devuelve?               → probar con match_count 3, 9 y 50
5. ¿El LLM responde con ese contexto?     → llamada directa con el prompt armado
6. ¿El workflow los conecta bien?         → inspeccionar la ejecución nodo por nodo
```

El paso 4 fue el que encontró el problema del índice: pedir 3, 9 y 50 y recibir
siempre 1 es una firma inconfundible. Ninguna de las otras cinco pruebas lo habría
detectado.

> **Regla:** cuando el resultado final está mal y hay N capas, no adivines cuál es.
> Probá cada una por separado, de la más profunda a la más superficial.

### Buscar la fuente que no puede mentir

Se necesitaba saber si un workflow se había ejecutado. Se consultaron tres fuentes:

| Fuente | Qué dijo | Confiabilidad |
|---|---|---|
| Logs del contenedor | Nada | **Engañosa** — en nivel default no registra requests REST |
| API pública de n8n | Sin ejecuciones | Buena, pero podía tener filtros |
| Tabla `execution_entity` de su base | 9 ejecuciones, todas de tipo webhook, cero manuales | **Definitiva** |

La ausencia de logs no probaba nada: n8n en nivel default simplemente no los
escribe. Confundir "no hay registro" con "no pasó" es un error clásico.

> **Regla:** el silencio no es evidencia. Antes de concluir "no pasó nada",
> verificá que el instrumento que estás mirando **registraría** ese evento si
> hubiera ocurrido.

### Buscar corroboración independiente

Para confirmar que la ingesta no había corrido, se usaron tres señales que no
dependen entre sí:

1. n8n no tenía ninguna ejecución en modo manual;
2. el `updated_at` de Supabase no se había movido;
3. la cantidad de filas no había cambiado.

Tres fuentes distintas apuntando a lo mismo. Con una sola, siempre queda la duda de
si el instrumento falla. Con tres independientes, la conclusión se sostiene.

### Trabajar sobre hipótesis falsables

Cada diagnóstico se formuló como una afirmación que se podía refutar:

> *"Si el problema es que el índice limita las probes a una lista, entonces pedir
> más resultados va a seguir devolviendo uno."*

Se probó con 3, 9 y 50. Siempre uno. Hipótesis confirmada, y confirmada de una
forma que la habría descartado si fuera falsa.

Compará con: *"debe ser algo del índice, probemos borrarlo"*. Si se borra y anda,
uno se queda sin saber por qué. Y el "por qué" es lo que evita repetir el error.

### Instrumentar antes que adivinar

Dos veces la respuesta llegó de instrumentar el entorno real en vez de razonar
desde afuera:

- la sonda de `typeof` que resolvió qué cliente HTTP existía;
- el campo `_rag_error` que expuso el error que el `catch` se comía.

En los dos casos, una iteración de instrumentación reemplazó varias de prueba y
error.

---

## 10. Límites del sistema actual

Ser honesto sobre lo que no está resuelto vale más que exagerar lo que sí.

**Corpus muy chico.** Nueve documentos. Suficiente para validar el pipeline
completo, insuficiente para evaluar calidad de retrieval de verdad. Con nueve
documentos, casi cualquier configuración parece funcionar.

**Sin umbral de similitud.** Se recuperan siempre los 4 más cercanos, sin importar
qué tan lejos estén. Si alguien pregunta por el clima, igual entran cuatro
documentos del instituto al contexto. Hoy el prompt lo compensa, pero lo correcto
es filtrar:

```sql
where 1 - (embedding <=> query_embedding) > 0.65
```

**Sin re-ranking.** Los sistemas maduros recuperan 20 candidatos con búsqueda
vectorial (rápida, aproximada) y después los reordenan con un modelo *cross-encoder*
más preciso pero más lento. Mejora bastante la precisión del top-4.

**Sin búsqueda híbrida.** La búsqueda semántica es mala con nombres propios,
códigos y siglas. Preguntar por "TOEIC" funciona por casualidad. Lo robusto es
combinar búsqueda vectorial con búsqueda léxica (BM25 o full-text de Postgres) y
fusionar los rankings.

**Sin evaluación automatizada.** No hay un conjunto de preguntas con respuestas
esperadas que corra en CI. Hoy cualquier cambio de prompt o de modelo se valida a
mano. Es lo primero que agregaría un equipo serio: veinte pares pregunta/respuesta
y una métrica de si el documento correcto aparece en el top-k.

**Ingesta manual.** Se dispara a mano. Debería correr por cron o dispararse cuando
se publica un aviso nuevo.

**`avisos` está vacío.** El endpoint existe, el pipeline lo indexa, la tabla no
tiene datos. La rama funciona pero hoy no aporta nada.

---

## 11. Glosario

**Embedding** — Vector de números que representa el significado de un texto. Textos
parecidos producen vectores cercanos.

**Dimensionalidad** — Cantidad de números del vector. Acá, 768. Más dimensiones dan
más precisión y cuestan más espacio y tiempo.

**Matryoshka (MRL)** — Técnica que entrena un embedding para que sus primeras N
dimensiones sean, por sí solas, una representación válida. Permite truncar sin
reentrenar.

**Base vectorial** — Base de datos capaz de indexar y buscar por proximidad entre
vectores. Acá, Postgres con la extensión pgvector.

**pgvector** — Extensión de Postgres que agrega el tipo `vector` y operadores de
distancia.

**Distancia coseno** — Mide el ángulo entre dos vectores, ignorando su magnitud. El
default sensato para texto. Operador `<=>` en pgvector.

**Similitud** — `1 - distancia`. Uno es idéntico, cero es sin relación. Entre textos
distintos sobre el mismo tema, esperar 0.7–0.8.

**Chunk** — Fragmento de documento que se embebe como unidad. Acá, cada FAQ y cada
curso es un chunk.

**Retrieval** — La fase de búsqueda: dada una pregunta, encontrar los chunks
relevantes.

**Recall** — Fracción de los documentos relevantes que la búsqueda efectivamente
encontró. Es lo que degradaba el índice mal dimensionado.

**Top-k** — Cuántos chunks se recuperan. Acá, 4.

**ivfflat** — Índice vectorial aproximado. Agrupa vectores en listas y solo revisa
las más prometedoras. Rápido, pero pierde recall si está mal configurado.

**probes** — Cuántas listas revisa ivfflat. Default: 1.

**Alucinación** — Cuando un modelo genera información falsa con tono seguro. El
objetivo central de RAG es reducirla.

**Grounding (fundamentación)** — Que la respuesta esté anclada en documentos reales
recuperados, no en el conocimiento paramétrico del modelo.

**Temperatura** — Aleatoriedad al elegir el siguiente token. Baja para clasificar,
más alta para redactar.

**Upsert** — Insertar o actualizar si ya existe. Lo que hace idempotente a la
ingesta.

**Idempotente** — Que ejecutarlo N veces deja el mismo resultado que ejecutarlo una.

**Re-ranking** — Reordenar los candidatos recuperados con un modelo más preciso.

**Búsqueda híbrida** — Combinar búsqueda semántica con búsqueda por palabras clave.

---

## 12. Preguntas de entrevista

Preguntas frecuentes sobre RAG, con la respuesta anclada en este sistema.

**¿Qué es RAG y qué problema resuelve?**
Recuperación aumentada por generación. Resuelve que un LLM solo conoce su corpus de
entrenamiento. En vez de reentrenar o meter todo el conocimiento en el prompt, se
guarda afuera, se busca lo relevante en cada consulta y se inyecta como contexto.
Separa el conocimiento del razonamiento: actualizar un dato es un `UPDATE`, no un
reentrenamiento.

**¿Por qué no fine-tuning?**
El fine-tuning enseña estilo y formato mucho mejor que hechos, es caro, y congela
el conocimiento en los pesos. Para información que cambia — precios, horarios,
avisos — RAG es la herramienta correcta. No son excluyentes: se puede afinar el
tono y usar RAG para los datos.

**¿Cómo elegiste la dimensionalidad?**
La restringió el esquema: la tabla es `vector(768)`, así que el modelo debe producir
768. `gemini-embedding-001` devuelve 3072 por default, pero soporta
`outputDimensionality` gracias a Matryoshka. Para un corpus chico, 768 sobra y
ocupa cuatro veces menos.

**¿Qué estrategia de chunking usaste?**
Ninguna, y a propósito. Las fuentes ya venían atomizadas: un FAQ es pregunta más
respuesta, un curso es un registro. Partirlas habría destruido unidades de
significado completas. El chunking responde a documentos largos; si no tenés ese
problema, aplicarlo empeora las cosas.

**¿Cómo evitás las alucinaciones?**
Tres capas. El prompt restringe explícitamente al contexto recuperado. El prompt le
da una salida honesta — derivar a secretaría — para que no tenga que inventar. Y se
verifica con preguntas cuya respuesta *no* está en el corpus: si contesta algo
concreto sobre clases de alemán, el sistema está roto.

**¿Cómo medís si el retrieval funciona?**
Hoy, manualmente, y es una deuda reconocida. Lo correcto es un conjunto de
evaluación con pares pregunta/documento esperado y medir si el documento correcto
aparece en el top-k. Sin eso, cada cambio de prompt o modelo es una apuesta.

**¿Por qué Postgres y no una base vectorial dedicada?**
Los datos viven al lado del resto de la aplicación, se filtran con SQL común, hay
transacciones, y no hay un servicio más que mantener. Para este volumen es la
decisión correcta. Migrar tiene sentido cuando el volumen o la latencia lo pidan.

**Contame de un bug difícil de este proyecto.**
Un índice `ivfflat` con `lists = 100` sobre 9 filas. Con `probes = 1`, la búsqueda
escaneaba una sola lista y devolvía un resultado, pidiera los que pidiera. El
síntoma era el chatbot diciendo que no tenía información de cursos con los cursos
cargados. Lo confirmé pidiendo 3, 9 y 50 resultados y recibiendo siempre uno. La
solución fue borrar el índice: con menos de mil filas el escaneo secuencial es
exacto e instantáneo. La lección es que las optimizaciones tienen un umbral, y los
índices aproximados degradan el recall en silencio.

**¿Qué mejorarías si tuvieras más tiempo?**
En orden: un conjunto de evaluación automatizado, un umbral de similitud para
descartar contexto irrelevante, búsqueda híbrida para que los nombres propios y
siglas funcionen de forma confiable, y re-ranking. Y automatizar la ingesta, que
hoy es manual.

---

## Cierre

Si tuviera que quedarme con tres ideas de todo esto:

**Uno.** RAG no es magia ni es difícil. Son dos pipelines: uno que convierte
documentos en vectores y los guarda, otro que convierte la pregunta en un vector,
busca los más cercanos y se los pasa a un modelo. Todo lo demás son detalles de
implementación.

**Dos.** La parte difícil no es construirlo, es **verificar que funciona**. Cinco de
las seis fallas de este proyecto devolvían `200 OK` con texto impecable. En
sistemas con LLMs, "responde" y "funciona" son propiedades independientes, y hay que
verificar la segunda explícitamente.

**Tres.** Los errores más caros fueron de exceso de prolijidad, no de descuido. El
índice vectorial estaba puesto porque "es lo que se hace". El `catch` vacío estaba
puesto porque "hay que manejar los errores". Las dos decisiones se veían
profesionales y las dos rompían el sistema.

> Copiar la forma sin entender la razón produce código que parece correcto y no lo
> está. Entender por qué existe cada pieza es lo que permite decidir cuándo no
> usarla.
