# PriceWise - AI-Powered Price Analysis Platform

A sophisticated data analysis application that combines natural language processing with advanced analytics to provide insights into pricing data.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **AI-Powered Analysis**: GPT-4 powered code generation for data analysis
- **Dynamic Column Detection**: Automatically detects and maps data columns
- **Large Dataset Support**: Handles datasets up to 2+ million rows efficiently
- **Interactive Visualizations**: Generate charts and graphs dynamically
- **Memory Optimization**: Smart sampling and efficient processing for large datasets

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **AI Integration**: OpenAI GPT-4o-mini
- **Database**: DuckDB (in-memory)
- **Visualization**: Matplotlib, Plotly
- **Forecasting**: Prophet
- **Machine Learning**: Scikit-learn

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- 8GB+ RAM (for large datasets)

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/pricewise.git
cd pricewise

# Run with Docker Compose
docker-compose up --build
```

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/pricewise.git
cd pricewise

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up configuration
mkdir .streamlit
# Create .streamlit/secrets.toml with your OpenAI API key

# Run the application
streamlit run app/main.py
```

## ⚙️ Configuration

### OpenAI API Setup
Create `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "your-openai-api-key-here"
```

### Environment Variables (Alternative)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## 📊 Usage

1. **Upload Data**: Support for CSV, Excel, and Parquet files
2. **Ask Questions**: Use natural language like "Show me price trends by location"
3. **Get Insights**: AI generates analysis code and visualizations
4. **Explore Results**: Interactive tables and charts

## 🔒 Security Features

- **Data Privacy**: No sensitive data sent to external services
- **Column Anonymization**: Generic column mapping for privacy
- **Local Processing**: Core analysis happens on your machine
- **Secure API**: OpenAI API calls only for code generation

## 📁 Project Structure

```
pricewise/
├── app/
│   ├── main.py              # Main Streamlit application
│   ├── router.py            # Query routing logic
│   ├── nl2code.py           # AI code generation
│   ├── analytics/           # Analysis modules
│   ├── charts.py            # Chart generation
│   └── commentary.py        # AI commentary
├── data/                    # Sample data files
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── README.md               # This file
```

## 🚀 Deployment Options

### Docker
- **Production Ready**: Containerized deployment
- **Easy Scaling**: Docker Compose for multiple instances
- **Consistent Environment**: Same behavior across platforms

### PyInstaller
- **Standalone Executable**: No Python installation required
- **Windows/Mac/Linux**: Cross-platform distribution
- **Offline Capable**: Works without internet connection

### Script-Based
- **Quick Setup**: Automated environment setup
- **Cross-Platform**: Windows, macOS, and Linux support
- **Easy Maintenance**: Simple update process

## 🔧 Development

### Setting Up Development Environment
```bash
git clone https://github.com/yourusername/pricewise.git
cd pricewise
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Format code
black app/
# Lint code
flake8 app/
# Type checking
mypy app/
```

## 📈 Performance

- **Small Datasets** (<100K rows): Full analysis
- **Medium Datasets** (100K-500K rows): Optimized processing
- **Large Datasets** (500K-2M rows): Smart sampling + full analysis
- **Very Large Datasets** (>2M rows): Stratified sampling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pricewise/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pricewise/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/pricewise/wiki)

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- Streamlit for the web framework
- Pandas and NumPy communities
- All contributors and users

---

**⭐ Star this repository if you find it helpful!**
