/**
 * ARENA Tutor API Server
 *
 * Provides chat interface using Ollama (Gemma) with optional LangSmith tracing.
 * Following Node.js server patterns - no framework dependencies.
 *
 * Usage:
 *   node server/tutor-api.js
 *
 * Environment:
 *   OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
 *   OLLAMA_MODEL - Model to use (default: gemma3:12b)
 *   LANGSMITH_API_KEY - Optional: Enable LangSmith tracing
 *   LANGSMITH_PROJECT - Optional: LangSmith project name
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// RAG module for curriculum retrieval
const { search, buildContext, loadEmbeddings } = require('./embeddings');

// Configuration
const PORT = process.env.PORT || 3002;
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma3:12b';
const LANGSMITH_API_KEY = process.env.LANGSMITH_API_KEY;
const LANGSMITH_PROJECT = process.env.LANGSMITH_PROJECT || 'arena-tutor';
const USE_RAG = process.env.USE_RAG !== 'false'; // Enable RAG by default

// Compact system prompt for RAG mode (reduces token usage)
const RAG_SYSTEM_PROMPT = `You are a tutor for ARENA 3.0 (AI Safety). Answer using ONLY the retrieved context below.

**Style**: Concise, focused, use code examples. Connect to the capstone (sycophancy detection).
**Constraint**: Keep responses under 200 words. If context doesn't cover the question, say so.

RETRIEVED CONTEXT:
{context}

---
Answer the student's question based on the context above.`;

// Full system prompt (used when RAG is disabled)
const BASE_SYSTEM_PROMPT = `You are an expert tutor for ARENA 3.0: AI Safety Fundamentals.

**Capstone**: Sycophancy Detection - understanding when AI optimizes for approval over truth.

**Mental Models** (use these to explain):
- Collapse: Linear compositions stay linear; ReLU breaks the chain
- Thermostat: Training is a feedback loop; loss measures gap, gradients signal direction
- Routing: Backprop routes error signals; vanishing gradients break the loop
- Dimensionality: Tensor shapes are system boundaries

Keep responses under 200 words. Use code examples. Connect concepts to the capstone.`;

// Persona-specific prompts
const PERSONA_PROMPTS = {
  tyla: `\n\n**Student Profile - Tyla (CS Undergrad)**:
- Has intermediate Python and some PyTorch from ML class
- Needs explicit "why" before "how"
- Blocker: Can do exercises but doesn't understand WHY
- Provide research context and mathematical intuition`,

  aaliyah: `\n\n**Student Profile - Aaliyah (Bootcamp Dev)**:
- Strong JavaScript developer, learning Python, high school math only
- Needs code-first explanations and JS analogies
- Blocker: Math notation makes no sense
- Translate math to code, avoid notation when possible`,

  maneesha: `\n\n**Student Profile - Maneesha (Instructional Designer)**:
- 8 years L&D experience, basic Python, strong conceptual thinking
- Needs meta-level insights and concept frameworks first
- Blocker: Gets lost in implementation details
- Focus on the learning design principles, not just the code`,
};

function buildSystemPrompt(currentWeek, persona = 'tyla', retrievedContext = null) {
  const personaPrompt = PERSONA_PROMPTS[persona] || PERSONA_PROMPTS.tyla;

  // If we have retrieved context, use the compact RAG prompt
  if (retrievedContext && USE_RAG) {
    return RAG_SYSTEM_PROMPT.replace('{context}', retrievedContext) + personaPrompt;
  }

  // Fallback to full system prompt
  return BASE_SYSTEM_PROMPT + personaPrompt + `\n\nStudent is on Week ${currentWeek}.`;
}

// LangSmith tracing (if configured)
let langsmithClient = null;
let traceable = null;

async function initLangSmith() {
  if (!LANGSMITH_API_KEY) {
    console.log('LangSmith tracing disabled (no API key)');
    return;
  }

  try {
    // Dynamic imports for LangSmith
    const { Client } = await import('langsmith');
    const traceableModule = await import('langsmith/traceable');

    langsmithClient = new Client({
      apiUrl: 'https://api.smith.langchain.com',
      apiKey: LANGSMITH_API_KEY,
    });

    traceable = traceableModule.traceable;

    // Set env vars for automatic tracing
    process.env.LANGSMITH_TRACING = 'true';
    process.env.LANGSMITH_PROJECT = LANGSMITH_PROJECT;

    console.log(`LangSmith tracing enabled (project: ${LANGSMITH_PROJECT})`);
  } catch (e) {
    console.log('LangSmith not available:', e.message);
  }
}

// Flush pending traces (call before exit)
async function flushTraces() {
  if (langsmithClient) {
    try {
      console.log('[LangSmith] Flushing pending traces...');
      await langsmithClient.awaitPendingTraceBatches();
      console.log('[LangSmith] Traces flushed.');
    } catch (e) {
      console.error('[LangSmith] Flush error:', e.message);
    }
  }
}

// Call Ollama API (raw, untraced version)
async function callOllamaRaw(messages, systemPrompt) {
  const url = `${OLLAMA_HOST}/api/chat`;

  const payload = JSON.stringify({
    model: OLLAMA_MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      ...messages
    ],
    stream: false,
    options: {
      temperature: 0.7,
      num_predict: 500,
    }
  });

  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || 11434,
      path: urlObj.pathname,
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
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Failed to parse Ollama response: ${data}`));
        }
      });
    });

    req.on('error', (e) => {
      reject(new Error(`Ollama connection failed: ${e.message}. Is Ollama running?`));
    });

    req.setTimeout(60000, () => {
      req.destroy();
      reject(new Error('Ollama request timeout (60s)'));
    });

    req.write(payload);
    req.end();
  });
}

// Traced wrapper for Ollama LLM calls
async function callOllama(messages, currentWeek, persona = 'tyla', retrievedContext = null) {
  const systemPrompt = buildSystemPrompt(currentWeek, persona, retrievedContext);

  // If tracing enabled, wrap with traceable
  if (traceable) {
    const tracedCall = traceable(
      async (input) => {
        const result = await callOllamaRaw(input.messages, input.systemPrompt);
        return {
          response: result.message?.content || '',
          model: result.model,
          done: result.done,
          eval_count: result.eval_count,
          prompt_eval_count: result.prompt_eval_count,
        };
      },
      {
        name: 'ollama-llm',
        run_type: 'llm',
        client: langsmithClient,
        metadata: {
          model: OLLAMA_MODEL,
          persona,
          week: currentWeek,
          has_rag_context: !!retrievedContext,
          system_prompt_length: systemPrompt.length,
        },
      }
    );

    const traced = await tracedCall({
      messages,
      systemPrompt,
      userQuery: messages[messages.length - 1]?.content || '',
    });

    // Return in original format
    return {
      message: { content: traced.response },
      model: traced.model,
      done: traced.done,
    };
  }

  // Fallback: no tracing
  return callOllamaRaw(messages, systemPrompt);
}

// Parse request body
function parseBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const body = Buffer.concat(chunks).toString();
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve(body);
      }
    });
    req.on('error', reject);
  });
}

// Send JSON response with CORS
function sendJSON(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  });
  res.end(JSON.stringify(data));
}

// Create HTTP server
const server = http.createServer(async (req, res) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const pathname = url.pathname;

  try {
    // Health check
    if (pathname === '/api/health' && req.method === 'GET') {
      // Check Ollama connectivity
      try {
        const ollamaCheck = await new Promise((resolve, reject) => {
          const checkReq = http.get(`${OLLAMA_HOST}/api/tags`, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(JSON.parse(data)));
          });
          checkReq.on('error', reject);
          checkReq.setTimeout(5000, () => reject(new Error('timeout')));
        });

        const models = ollamaCheck.models?.map(m => m.name) || [];
        const embeddings = loadEmbeddings();
        sendJSON(res, {
          status: 'ok',
          ollama: 'connected',
          model: OLLAMA_MODEL,
          available_models: models,
          langsmith: !!langsmithClient,
          rag: {
            enabled: USE_RAG,
            embeddings_loaded: embeddings ? embeddings.length : 0,
          },
        });
      } catch (e) {
        sendJSON(res, {
          status: 'degraded',
          ollama: 'disconnected',
          error: e.message,
        }, 503);
      }
    }

    // Chat endpoint with RAG
    else if (pathname === '/api/chat' && req.method === 'POST') {
      const body = await parseBody(req);
      const { messages, currentWeek, persona, currentChapter } = body;

      if (!messages || !Array.isArray(messages)) {
        sendJSON(res, { error: 'messages array required' }, 400);
        return;
      }

      const lastMessage = messages[messages.length - 1]?.content || '';
      console.log(`[Chat] Week ${currentWeek || '?'} (${persona || 'tyla'}): "${lastMessage.slice(0, 50)}..."`);

      // Traced RAG retrieval function
      async function ragRetrieval(query, options) {
        const results = await search(query, options);
        return {
          chunks: results,
          context: results.length > 0 ? buildContext(results) : null,
        };
      }

      // Wrap RAG retrieval with tracing if available
      const tracedRagRetrieval = traceable
        ? traceable(ragRetrieval, {
            name: 'rag-retrieval',
            run_type: 'retriever',
            client: langsmithClient,
            metadata: {
              topK: 5,
              minScore: 0.25,
              filterChapter: currentChapter,
            },
          })
        : ragRetrieval;

      // Main chat handler (parent trace)
      async function handleChat() {
        // RAG: Retrieve relevant curriculum context
        let retrievedContext = null;
        let searchResults = [];

        if (USE_RAG && lastMessage) {
          try {
            const ragResult = await tracedRagRetrieval(lastMessage, {
              topK: 5,
              minScore: 0.25,
              filterChapter: currentChapter !== undefined ? currentChapter : null,
            });

            searchResults = ragResult.chunks;
            retrievedContext = ragResult.context;

            if (searchResults.length > 0) {
              console.log(`[RAG] Retrieved ${searchResults.length} chunks (scores: ${searchResults.map(r => r.score.toFixed(2)).join(', ')})`);
            } else {
              console.log('[RAG] No relevant chunks found');
            }
          } catch (e) {
            console.error('[RAG] Search failed:', e.message);
          }
        }

        const result = await callOllama(
          messages,
          currentWeek || 1,
          persona || 'tyla',
          retrievedContext
        );

        return {
          response: result.message?.content || 'No response',
          model: result.model,
          done: result.done,
          rag: {
            enabled: USE_RAG,
            chunks_retrieved: searchResults.length,
            top_scores: searchResults.slice(0, 3).map(r => ({
              type: r.metadata.type,
              score: r.score.toFixed(3),
              chunk_id: r.id,
            })),
            chunk_ids: searchResults.map(r => r.id),
          },
        };
      }

      // Wrap entire chat handler with parent trace
      const tracedHandleChat = traceable
        ? traceable(handleChat, {
            name: 'arena-tutor-chat',
            run_type: 'chain',
            client: langsmithClient,
            metadata: {
              persona: persona || 'tyla',
              week: currentWeek || 1,
              chapter: currentChapter,
              rag_enabled: USE_RAG,
            },
            tags: ['arena-tutor', `persona:${persona || 'tyla'}`, `week:${currentWeek || 1}`],
          })
        : handleChat;

      const chatResult = await tracedHandleChat();
      sendJSON(res, chatResult);
    }

    // Curriculum data
    else if (pathname === '/api/curriculum' && req.method === 'GET') {
      const curriculumPath = path.join(__dirname, '..', 'data', 'curriculum.json');
      if (fs.existsSync(curriculumPath)) {
        const data = JSON.parse(fs.readFileSync(curriculumPath, 'utf-8'));
        sendJSON(res, data);
      } else {
        sendJSON(res, { error: 'Curriculum not found' }, 404);
      }
    }

    // 404
    else {
      sendJSON(res, { error: 'Not found' }, 404);
    }

  } catch (err) {
    console.error('Server error:', err);
    sendJSON(res, { error: err.message }, 500);
  }
});

// Start server
async function main() {
  await initLangSmith();

  // Load embeddings for RAG
  if (USE_RAG) {
    const embeddings = loadEmbeddings();
    if (embeddings) {
      console.log(`📚 RAG enabled: ${embeddings.length} curriculum chunks loaded`);
    } else {
      console.log('⚠️  RAG enabled but no embeddings found. Run: node server/embeddings.js');
    }
  }

  server.listen(PORT, () => {
    console.log(`\n🎓 ARENA Tutor API running at http://localhost:${PORT}`);
    console.log('\nEndpoints:');
    console.log('  GET  /api/health     - Check Ollama connection');
    console.log('  GET  /api/curriculum - Get curriculum data');
    console.log('  POST /api/chat       - Chat with tutor (RAG-enhanced)');
    console.log(`\nUsing model: ${OLLAMA_MODEL}`);
    console.log(`Ollama host: ${OLLAMA_HOST}`);
    console.log(`RAG: ${USE_RAG ? 'enabled' : 'disabled'}`);
  });

  // Graceful shutdown with trace flushing
  async function gracefulShutdown(signal) {
    console.log(`\n[${signal}] Shutting down...`);

    // Flush any pending LangSmith traces
    await flushTraces();

    server.close(() => {
      console.log('Server closed.');
      process.exit(0);
    });

    // Force exit after 10s if graceful shutdown hangs
    setTimeout(() => {
      console.error('Forced exit after timeout');
      process.exit(1);
    }, 10000);
  }

  process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
  process.on('SIGINT', () => gracefulShutdown('SIGINT'));
}

main();
