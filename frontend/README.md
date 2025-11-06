# Student Loan Q&A Chatbot - Frontend

A modern Next.js frontend for interacting with the streaming RAG API for student loan questions.

## Features

- Real-time streaming responses using Server-Sent Events (SSE)
- Clean, modern UI with Tailwind CSS
- Multiple retrieval method selection
- Confidence scores and source count display
- Responsive design

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- The backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build for Production

```bash
npm run build
npm start
```

## Configuration

The API endpoint is configured in `next.config.js`. Update the rewrite rule if your backend is running on a different port or host.

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx      # Root layout
│   ├── page.tsx        # Main chat page
│   └── globals.css     # Global styles
├── components/
│   ├── ChatMessage.tsx # Message display component
│   └── ChatInput.tsx   # Input component
├── lib/
│   └── streaming.ts    # SSE streaming client
└── package.json
```

## Usage

1. Select a retrieval method from the dropdown (Production, Naive, Semantic, or Hybrid)
2. Type your question in the input field
3. Press Enter or click Send
4. Watch the response stream in real-time

The chatbot will display:
- The streaming response
- Number of sources used
- Confidence score (if available)

