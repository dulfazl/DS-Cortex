declare module "plotly.js-dist-min" {
  const Plotly: {
    react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
    purge: (el: HTMLElement) => void;
    Plots: { resize: (el: HTMLElement) => void };
  };
  export default Plotly;
}
