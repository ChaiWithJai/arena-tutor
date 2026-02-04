/**
 * Curriculum Embedding & Retrieval Module
 *
 * Embeds curriculum content using nomic-embed-text via Ollama.
 * Provides vector search for RAG-enhanced tutoring.
 *
 * Design Principles (from the curriculum):
 * - Dimensionality Model: Embeddings compress knowledge into fixed dimensions
 * - Routing Model: Vector search routes queries to relevant content
 * - Thermostat Model: Retrieval → Generation feedback loop
 */

const fs = require('fs');
const path = require('path');
const http = require('http');

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const EMBED_MODEL = 'nomic-embed-text';
const EMBEDDINGS_PATH = path.join(__dirname, '..', 'data', 'embeddings.json');
const CURRICULUM_PATH = path.join(__dirname, '..', 'data', 'curriculum.json');

// In-memory cache of embeddings
let embeddingsCache = null;

/**
 * Call Ollama embedding API
 */
async function embed(text) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify({
      model: EMBED_MODEL,
      prompt: text,
    });

    const url = new URL(`${OLLAMA_HOST}/api/embeddings`);
    const options = {
      hostname: url.hostname,
      port: url.port || 11434,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
      },
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const result = JSON.parse(data);
          resolve(result.embedding);
        } catch (e) {
          reject(new Error(`Failed to parse embedding response: ${e.message}`));
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(30000, () => {
      req.destroy();
      reject(new Error('Embedding request timeout'));
    });

    req.write(payload);
    req.end();
  });
}

/**
 * Chunk curriculum into embeddable pieces
 * Each chunk has: text, metadata (chapter, section, type, mental_model)
 */
function chunkCurriculum(curriculum) {
  const chunks = [];

  // Add capstone context
  if (curriculum.capstone_domain) {
    chunks.push({
      id: 'capstone',
      text: `Capstone Domain: ${curriculum.capstone_domain}. The course builds toward detecting and mitigating AI sycophancy - when models optimize for user approval over truth.`,
      metadata: {
        type: 'capstone',
        chapter: -1,
        section: -1,
      }
    });
  }

  // Add mental models (object with keys like 'collapse', 'thermostat', etc.)
  if (curriculum.mental_models) {
    Object.entries(curriculum.mental_models).forEach(([key, mm]) => {
      // Core mental model chunk
      chunks.push({
        id: `mental-model-${key}`,
        text: `Mental Model: ${mm.name}. Core insight: ${mm.core}. Arena application: ${mm.arena || mm.application || ''}`,
        metadata: {
          type: 'mental_model',
          name: mm.name,
          key: key,
          chapter: -1,
          section: -1,
        }
      });

      // Deep explanation (if present)
      if (mm.deep_explanation) {
        chunks.push({
          id: `mental-model-${key}-deep`,
          text: `${mm.name} - Deep Explanation: ${mm.deep_explanation}`,
          metadata: {
            type: 'mental_model_deep',
            name: mm.name,
            key: key,
            chapter: -1,
            section: -1,
          }
        });
      }

      // JavaScript analogy (for bootcamp devs)
      if (mm.js_analogy) {
        chunks.push({
          id: `mental-model-${key}-js`,
          text: `${mm.name} - JavaScript Analogy: ${mm.js_analogy}`,
          metadata: {
            type: 'js_analogy',
            name: mm.name,
            key: key,
            chapter: -1,
            section: -1,
          }
        });
      }

      // Unified view / progression (for relationships)
      if (mm.unified_view) {
        chunks.push({
          id: `mental-model-${key}-unified`,
          text: `How Mental Models Connect: ${mm.unified_view}`,
          metadata: {
            type: 'mental_model_synthesis',
            name: mm.name,
            key: key,
            chapter: -1,
            section: -1,
          }
        });
      }
    });
  }

  // Add supplementary concepts (Q/K/V intuition, numerical gradients, etc.)
  if (curriculum.supplementary_concepts) {
    Object.entries(curriculum.supplementary_concepts).forEach(([key, concept]) => {
      // Main explanation
      chunks.push({
        id: `concept-${key}`,
        text: `${concept.name}: ${concept.core}. ${concept.explanation}`,
        metadata: {
          type: 'supplementary_concept',
          key: key,
          name: concept.name,
          chapter: -1,
          section: -1,
        }
      });

      // JavaScript analogy
      if (concept.js_analogy) {
        chunks.push({
          id: `concept-${key}-js`,
          text: `${concept.name} - JavaScript Code: ${concept.js_analogy}`,
          metadata: {
            type: 'js_analogy',
            key: key,
            name: concept.name,
            chapter: -1,
            section: -1,
          }
        });
      }

      // Math intuition (for CS undergrads)
      if (concept.math_intuition) {
        chunks.push({
          id: `concept-${key}-math`,
          text: `${concept.name} - Mathematical Intuition: ${concept.math_intuition}`,
          metadata: {
            type: 'math_intuition',
            key: key,
            name: concept.name,
            chapter: -1,
            section: -1,
          }
        });
      }

      // Capstone connection
      if (concept.capstone_connection) {
        chunks.push({
          id: `concept-${key}-capstone`,
          text: `${concept.name} for Sycophancy Detection: ${concept.capstone_connection}`,
          metadata: {
            type: 'capstone_connection',
            key: key,
            name: concept.name,
            chapter: -1,
            section: -1,
          }
        });
      }
    });
  }

  // Add real-world context (ICE/Palantir case, sycophancy spectrum)
  if (curriculum.real_world_context) {
    const rwc = curriculum.real_world_context;
    if (rwc.primary_case) {
      chunks.push({
        id: 'real-world-case',
        text: `Real-World Sycophancy Case: ${rwc.primary_case.title}. ${rwc.primary_case.description}. Alignment Lesson: ${rwc.primary_case.alignment_lesson}`,
        metadata: {
          type: 'real_world_context',
          chapter: -1,
          section: -1,
        }
      });
    }
    if (rwc.sycophancy_spectrum) {
      const spectrum = rwc.sycophancy_spectrum.map(l =>
        `Level ${l.level} (${l.name}): ${l.severity} severity - ${l.harm}`
      ).join('. ');
      chunks.push({
        id: 'sycophancy-spectrum',
        text: `Sycophancy Danger Spectrum: ${spectrum}`,
        metadata: {
          type: 'real_world_context',
          chapter: -1,
          section: -1,
        }
      });
    }
  }

  // Add chapter content
  curriculum.chapters.forEach((chapter, chIdx) => {
    // Chapter milestone
    if (chapter.milestone) {
      chunks.push({
        id: `ch${chIdx}-milestone`,
        text: `Chapter ${chIdx} (${chapter.title}) milestone: ${chapter.milestone}`,
        metadata: {
          type: 'milestone',
          chapter: chIdx,
          section: -1,
        }
      });
    }

    // Sections
    chapter.sections.forEach((section, secIdx) => {
      const sectionPrefix = `Chapter ${chIdx}, Section ${secIdx} (${section.title}):`;

      // Learning objectives (each as separate chunk for precision)
      section.learning_objectives?.forEach((obj, objIdx) => {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-obj${objIdx}`,
          text: `${sectionPrefix} Learning objective: ${obj}`,
          metadata: {
            type: 'learning_objective',
            chapter: chIdx,
            section: secIdx,
            objective_index: objIdx,
          }
        });
      });

      // Worked examples
      section.worked_examples?.forEach((ex, exIdx) => {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-ex${exIdx}`,
          text: `${sectionPrefix} Worked example: ${ex}`,
          metadata: {
            type: 'worked_example',
            chapter: chIdx,
            section: secIdx,
          }
        });
      });

      // Code snippet
      if (section.code_snippet) {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-code`,
          text: `${sectionPrefix} Code example: ${section.code_snippet}`,
          metadata: {
            type: 'code_snippet',
            chapter: chIdx,
            section: secIdx,
          }
        });
      }

      // Capstone connection
      if (section.capstone_connection) {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-capstone`,
          text: `${sectionPrefix} Capstone connection: ${section.capstone_connection}`,
          metadata: {
            type: 'capstone_connection',
            chapter: chIdx,
            section: secIdx,
          }
        });
      }

      // Mental model reference (section.mental_model is a string key)
      if (section.mental_model && curriculum.mental_models) {
        const mmKey = section.mental_model;
        const mm = curriculum.mental_models[mmKey];
        if (mm) {
          chunks.push({
            id: `ch${chIdx}-sec${secIdx}-mm`,
            text: `${sectionPrefix} Uses mental model "${mm.name}": ${mm.core}. Arena application: ${mm.arena}`,
            metadata: {
              type: 'mental_model_usage',
              chapter: chIdx,
              section: secIdx,
              mental_model: mm.name,
              mental_model_key: mmKey,
            }
          });
        }
      }

      // Deep explanation (expanded content for understanding)
      if (section.deep_explanation) {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-deep`,
          text: `${sectionPrefix} Deep explanation: ${section.deep_explanation}`,
          metadata: {
            type: 'deep_explanation',
            chapter: chIdx,
            section: secIdx,
          }
        });
      }

      // JavaScript analogy (for bootcamp developers)
      if (section.js_analogy) {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-js`,
          text: `${sectionPrefix} JavaScript analogy: ${section.js_analogy}`,
          metadata: {
            type: 'js_analogy',
            chapter: chIdx,
            section: secIdx,
          }
        });
      }

      // Math intuition (for CS undergrads)
      if (section.math_intuition) {
        chunks.push({
          id: `ch${chIdx}-sec${secIdx}-math`,
          text: `${sectionPrefix} Mathematical intuition: ${section.math_intuition}`,
          metadata: {
            type: 'math_intuition',
            chapter: chIdx,
            section: secIdx,
          }
        });
      }
    });
  });

  return chunks;
}

/**
 * Embed all curriculum chunks and save to file
 */
async function embedCurriculum() {
  console.log('Loading curriculum...');
  const curriculum = JSON.parse(fs.readFileSync(CURRICULUM_PATH, 'utf-8'));

  console.log('Chunking curriculum...');
  const chunks = chunkCurriculum(curriculum);
  console.log(`Created ${chunks.length} chunks`);

  console.log('Embedding chunks (this may take a minute)...');
  const embeddings = [];

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    try {
      const embedding = await embed(chunk.text);
      embeddings.push({
        ...chunk,
        embedding,
      });
      process.stdout.write(`\r  Embedded ${i + 1}/${chunks.length}`);
    } catch (e) {
      console.error(`\nFailed to embed chunk ${chunk.id}: ${e.message}`);
    }
  }
  console.log('\n');

  console.log(`Saving ${embeddings.length} embeddings to ${EMBEDDINGS_PATH}...`);
  fs.writeFileSync(EMBEDDINGS_PATH, JSON.stringify(embeddings, null, 2));

  embeddingsCache = embeddings;
  console.log('Done!');

  return embeddings;
}

/**
 * Load embeddings from file (or cache)
 */
function loadEmbeddings() {
  if (embeddingsCache) return embeddingsCache;

  if (fs.existsSync(EMBEDDINGS_PATH)) {
    embeddingsCache = JSON.parse(fs.readFileSync(EMBEDDINGS_PATH, 'utf-8'));
    return embeddingsCache;
  }

  return null;
}

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Search for similar chunks
 * @param {string} query - User's question
 * @param {object} options - Search options
 * @returns {Array} Top-k similar chunks with scores
 */
async function search(query, options = {}) {
  const {
    topK = 5,
    minScore = 0.3,
    filterChapter = null,  // Only search within a chapter
    filterTypes = null,    // Only search certain types
  } = options;

  const embeddings = loadEmbeddings();
  if (!embeddings) {
    throw new Error('Embeddings not loaded. Run embedCurriculum() first.');
  }

  // Embed the query
  const queryEmbedding = await embed(query);

  // Calculate similarities
  let results = embeddings.map(chunk => ({
    ...chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));

  // Apply filters
  if (filterChapter !== null) {
    results = results.filter(r =>
      r.metadata.chapter === filterChapter || r.metadata.chapter === -1
    );
  }

  if (filterTypes) {
    results = results.filter(r => filterTypes.includes(r.metadata.type));
  }

  // Sort by score and filter by minimum
  results = results
    .filter(r => r.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  // Remove embedding from results (save memory)
  return results.map(({ embedding, ...rest }) => rest);
}

/**
 * Build context string from search results
 */
function buildContext(results) {
  if (!results.length) return '';

  const contextParts = results.map(r => {
    const typeLabel = r.metadata.type.replace(/_/g, ' ');
    return `[${typeLabel}] ${r.text}`;
  });

  return contextParts.join('\n\n');
}

// Export for use in tutor API
module.exports = {
  embed,
  embedCurriculum,
  loadEmbeddings,
  search,
  buildContext,
  cosineSimilarity,
};

// CLI: Run embedding if called directly
if (require.main === module) {
  embedCurriculum().catch(console.error);
}
