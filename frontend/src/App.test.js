import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('axios', () => ({
  get: jest.fn(() => Promise.resolve({ data: { status: 'healthy', models_loaded: true, personas: [] } })),
  post: jest.fn(),
  delete: jest.fn(),
}));

test('renders the PersonaGen header', () => {
  render(<App />);
  expect(screen.getByRole('heading', { name: 'PersonaGen' })).toBeInTheDocument();
});
