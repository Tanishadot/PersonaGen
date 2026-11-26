import { render, screen } from '@testing-library/react';
import axios from 'axios';
import App from './App';

jest.mock('axios');

beforeEach(() => {
  // Stub the backend so the test makes no network calls.
  axios.get.mockImplementation((url) => {
    if (url.endsWith('/health')) {
      return Promise.resolve({ data: { status: 'healthy', models_loaded: true } });
    }
    if (url.endsWith('/personas')) {
      return Promise.resolve({ data: { personas: [] } });
    }
    if (url.endsWith('/metrics')) {
      return Promise.resolve({
        data: {
          average_empathy_score: 0.75,
          total_conversations: 3,
          response_diversity: 0.5,
          recent_emotions: [],
        },
      });
    }
    return Promise.reject(new Error(`Unexpected request: ${url}`));
  });
});

test('renders the app and loads data from the backend', async () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: 'PersonaGen' })).toBeInTheDocument();
  expect(screen.getByText('Welcome to PersonaGen')).toBeInTheDocument();

  // Health check succeeds, then metrics load and render.
  expect(await screen.findByText('Backend Ready')).toBeInTheDocument();
  expect(await screen.findByText('Avg Empathy: 0.750')).toBeInTheDocument();
  expect(screen.getByText('Total Messages: 3')).toBeInTheDocument();
});
