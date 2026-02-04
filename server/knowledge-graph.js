/**
 * Knowledge Graph Builder & API Server
 * Uses Node.js server patterns for streaming and event loop efficiency
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// ============================================================================
// Knowledge Graph Data Structure
// ============================================================================

class KnowledgeGraph {
  constructor() {
    this.nodes = new Map();      // concept -> { data, metadata }
    this.edges = new Map();      // concept -> Set of prerequisite concepts
    this.reverseEdges = new Map(); // concept -> Set of dependent concepts
  }

  addNode(concept, data = {}) {
    if (!this.nodes.has(concept)) {
      this.nodes.set(concept, {
        id: concept,
        ...data,
        inDegree: 0,
        outDegree: 0
      });
      this.edges.set(concept, new Set());
      this.reverseEdges.set(concept, new Set());
    }
    return this;
  }

  addEdge(from, to) {
    // from is prerequisite of to
    this.addNode(from);
    this.addNode(to);

    this.edges.get(from).add(to);
    this.reverseEdges.get(to).add(from);

    // Update degrees
    this.nodes.get(from).outDegree++;
    this.nodes.get(to).inDegree++;

    return this;
  }

  getPrerequisites(concept) {
    return Array.from(this.reverseEdges.get(concept) || []);
  }

  getDependents(concept) {
    return Array.from(this.edges.get(concept) || []);
  }

  // Topological sort for curriculum ordering
  topologicalSort() {
    const visited = new Set();
    const result = [];

    const visit = (node) => {
      if (visited.has(node)) return;
      visited.add(node);

      for (const prereq of this.getPrerequisites(node)) {
        visit(prereq);
      }
      result.push(node);
    };

    for (const node of this.nodes.keys()) {
      visit(node);
    }

    return result;
  }

  // Find all paths from concept to root (no prerequisites)
  findLearningPaths(concept) {
    const paths = [];

    const dfs = (current, path) => {
      const prereqs = this.getPrerequisites(current);
      if (prereqs.length === 0) {
        paths.push([...path, current]);
        return;
      }

      for (const prereq of prereqs) {
        dfs(prereq, [...path, current]);
      }
    };

    dfs(concept, []);
    return paths.map(p => p.reverse());
  }

  // Get nodes by complexity level (using in-degree as proxy)
  getNodesByLevel() {
    const levels = {};
    const sorted = this.topologicalSort();

    const nodeLevel = new Map();

    for (const node of sorted) {
      const prereqs = this.getPrerequisites(node);
      const level = prereqs.length === 0
        ? 0
        : Math.max(...prereqs.map(p => nodeLevel.get(p) || 0)) + 1;

      nodeLevel.set(node, level);

      if (!levels[level]) levels[level] = [];
      levels[level].push(node);
    }

    return levels;
  }

  toJSON() {
    const nodes = [];
    const edges = [];

    for (const [id, data] of this.nodes) {
      nodes.push({ id, ...data });
    }

    for (const [from, tos] of this.edges) {
      for (const to of tos) {
        edges.push({ source: from, target: to });
      }
    }

    return { nodes, edges };
  }
}

// ============================================================================
// Build Graph from Curriculum Data
// ============================================================================

function buildGraphFromCurriculum(labeledObjects) {
  const graph = new KnowledgeGraph();

  // Define concept prerequisites (domain knowledge)
  const conceptPrerequisites = {
    // Foundational
    'tensor': [],
    'linear_transformation': ['tensor'],
    'neural_network': ['tensor', 'linear_transformation'],

    // Optimization chain
    'loss_function': ['neural_network'],
    'gradient_descent': ['loss_function'],
    'backpropagation': ['gradient_descent'],

    // Activation functions
    'relu': ['linear_transformation'],
    'sigmoid': ['linear_transformation'],
    'activation_function': ['relu', 'sigmoid'],
    'nonlinearity': ['activation_function'],

    // PyTorch implementation
    'nn.module': ['neural_network', 'backpropagation'],
    'nn.parameter': ['nn.module'],
    'forward_pass': ['nn.module'],
    'backward_pass': ['forward_pass', 'backpropagation'],
    'optimizer': ['backward_pass', 'gradient_descent'],

    // Advanced
    'transformer': ['nn.module', 'attention'],
    'attention': ['linear_transformation', 'softmax'],
    'embedding': ['tensor'],
    'softmax': ['activation_function'],
    'logits': ['softmax'],

    // Tensor manipulation
    'einops': ['tensor'],
    'einsum': ['tensor', 'linear_transformation'],

    // Loss functions
    'mse': ['loss_function'],
    'cross_entropy': ['loss_function', 'softmax'],

    // Systems Thinking mappings
    'feedback_loop': [],
    'stock': ['feedback_loop'],
    'flow': ['stock'],
    'system': ['stock', 'flow', 'feedback_loop'],
    'delay': ['feedback_loop'],
  };

  // Add all concepts and their prerequisites
  for (const [concept, prereqs] of Object.entries(conceptPrerequisites)) {
    graph.addNode(concept);
    for (const prereq of prereqs) {
      graph.addEdge(prereq, concept);
    }
  }

  // Enrich with learning object metadata
  const conceptObjects = new Map();
  for (const obj of labeledObjects) {
    for (const concept of obj.concepts) {
      if (!conceptObjects.has(concept)) {
        conceptObjects.set(concept, []);
      }
      conceptObjects.get(concept).push({
        id: obj.id,
        contentType: obj.content_type,
        timestamp: obj.timestamp,
        cognitiveLoad: obj.cognitive_load_estimate
      });
    }
  }

  // Update nodes with learning object references
  for (const [concept, objects] of conceptObjects) {
    if (graph.nodes.has(concept)) {
      const node = graph.nodes.get(concept);
      node.learningObjects = objects;
      node.objectCount = objects.length;
      node.avgCognitiveLoad = objects.reduce((sum, o) => sum + o.cognitiveLoad, 0) / objects.length;
    }
  }

  return graph;
}

// ============================================================================
// HTTP Server (Node.js Patterns)
// ============================================================================

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

function sendJSON(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  });
  res.end(JSON.stringify(data, null, 2));
}

function createServer(graph) {
  return http.createServer(async (req, res) => {
    // Handle CORS preflight
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type'
      });
      res.end();
      return;
    }

    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathname = url.pathname;

    try {
      // Routes
      if (pathname === '/api/graph' && req.method === 'GET') {
        // Full graph data for visualization
        sendJSON(res, graph.toJSON());
      }

      else if (pathname === '/api/levels' && req.method === 'GET') {
        // Nodes organized by complexity level
        sendJSON(res, graph.getNodesByLevel());
      }

      else if (pathname === '/api/order' && req.method === 'GET') {
        // Topological order for curriculum
        sendJSON(res, {
          order: graph.topologicalSort(),
          total: graph.nodes.size
        });
      }

      else if (pathname.startsWith('/api/concept/') && req.method === 'GET') {
        const concept = pathname.replace('/api/concept/', '');
        const node = graph.nodes.get(concept);

        if (node) {
          sendJSON(res, {
            ...node,
            prerequisites: graph.getPrerequisites(concept),
            dependents: graph.getDependents(concept),
            learningPaths: graph.findLearningPaths(concept)
          });
        } else {
          sendJSON(res, { error: 'Concept not found' }, 404);
        }
      }

      else if (pathname === '/api/curriculum-weeks' && req.method === 'GET') {
        // Week-by-week curriculum view
        const levels = graph.getNodesByLevel();
        const weeks = [];

        let weekNum = 1;
        let conceptsInWeek = [];
        let loadInWeek = 0;
        const MAX_LOAD_PER_WEEK = 3.0;

        for (const level of Object.keys(levels).sort((a, b) => a - b)) {
          for (const concept of levels[level]) {
            const node = graph.nodes.get(concept);
            const load = node.avgCognitiveLoad || 0.5;

            if (loadInWeek + load > MAX_LOAD_PER_WEEK && conceptsInWeek.length > 0) {
              weeks.push({
                week: weekNum,
                concepts: conceptsInWeek,
                totalLoad: loadInWeek
              });
              weekNum++;
              conceptsInWeek = [];
              loadInWeek = 0;
            }

            conceptsInWeek.push({
              id: concept,
              level: parseInt(level),
              ...node
            });
            loadInWeek += load;
          }
        }

        // Don't forget last week
        if (conceptsInWeek.length > 0) {
          weeks.push({
            week: weekNum,
            concepts: conceptsInWeek,
            totalLoad: loadInWeek
          });
        }

        sendJSON(res, { weeks });
      }

      else if (pathname === '/' || pathname === '/index.html') {
        // Serve the UI
        const uiPath = path.join(__dirname, '..', 'ui', 'index.html');
        if (fs.existsSync(uiPath)) {
          res.writeHead(200, { 'Content-Type': 'text/html' });
          fs.createReadStream(uiPath).pipe(res);
        } else {
          sendJSON(res, { message: 'Knowledge Graph API', endpoints: [
            '/api/graph',
            '/api/levels',
            '/api/order',
            '/api/concept/:name',
            '/api/curriculum-weeks'
          ]});
        }
      }

      else {
        sendJSON(res, { error: 'Not found' }, 404);
      }

    } catch (err) {
      console.error('Server error:', err);
      sendJSON(res, { error: err.message }, 500);
    }
  });
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  // Load labeled curriculum data
  const dataPath = path.join(__dirname, '..', 'labeled_curriculum.json');

  if (!fs.existsSync(dataPath)) {
    console.error('Run curriculum_parser.py first to generate labeled_curriculum.json');
    process.exit(1);
  }

  const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  console.log(`Loaded ${data.objects.length} learning objects`);

  // Build knowledge graph
  const graph = buildGraphFromCurriculum(data.objects);
  console.log(`Built graph with ${graph.nodes.size} concepts`);

  // Start server
  const PORT = process.env.PORT || 3001;
  const server = createServer(graph);

  server.listen(PORT, () => {
    console.log(`\n🧠 Knowledge Graph API running at http://localhost:${PORT}`);
    console.log('\nEndpoints:');
    console.log('  GET /api/graph           - Full graph (nodes + edges)');
    console.log('  GET /api/levels          - Concepts by complexity level');
    console.log('  GET /api/order           - Topological curriculum order');
    console.log('  GET /api/concept/:name   - Single concept with paths');
    console.log('  GET /api/curriculum-weeks - Week-by-week breakdown');
  });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    console.log('\nShutting down...');
    server.close(() => process.exit(0));
  });
  process.on('SIGINT', () => {
    console.log('\nShutting down...');
    server.close(() => process.exit(0));
  });
}

// Export for testing
module.exports = { KnowledgeGraph, buildGraphFromCurriculum };

// Run if main
if (require.main === module) {
  main();
}
