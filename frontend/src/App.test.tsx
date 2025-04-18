// ...existing code...
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom"; // Import jest-dom for extended matchers
import App from "./App";

describe("Deep Research Agent UI", () => {
  it("renders the main UI", () => {
    render(<App />);
    expect(screen.getByText(/Deep Research Agent/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Type your research prompt/i)).toBeInTheDocument();
  });

  it("disables send button if no prompt or models", () => {
    render(<App />);
    const sendBtn = screen.getByRole("button", { name: /send/i });
    expect(sendBtn).toBeDisabled();
  });

  // You can add more tests for model selection, spinner, and streaming logic
});