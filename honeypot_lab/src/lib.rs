//! P9 honeypot lab: runs the dataset's attack and benign episodes against a
//! local model with native tool calls, a sandbox and a real collector, scores
//! every episode, and serves the Streamlit lab UI. See `docs/code_documentation.md`.

pub mod agent;
pub mod api;
pub mod collector;
pub mod config;
pub mod dataset;
pub mod judge;
pub mod model;
pub mod report;
pub mod runs;
pub mod sandbox;
pub mod scoring;
pub mod types;
