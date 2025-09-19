#!/usr/bin/env python3
"""
Real HFT Data Download Tool
==========================

Command-line tool for downloading and generating real HFT data.
Integrated with the existing HFT system architecture.

Usage Examples:
    python download_real_data.py --source enhanced_yahoo --symbol AAPL --period 5d
    python download_real_data.py --source synthetic_lob --symbol TEST --records 10000
    python download_real_data.py --source crypto_lob --symbol BTCUSDT --records 20000
    python download_real_data.py --source all --symbol AAPL
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_engine.data_sources.real_data_downloader import RealDataDownloader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main function for command-line data downloading"""
    parser = argparse.ArgumentParser(
        description='Download real HFT data from various sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source enhanced_yahoo --symbol AAPL --period 1d --interval 1m
  %(prog)s --source synthetic_lob --symbol TEST --records 5000 --levels 10
  %(prog)s --source crypto_lob --symbol BTCUSDT --records 10000
  %(prog)s --list-sources
  %(prog)s --source all --symbol AAPL --quick
        """
    )

    # Main options
    parser.add_argument('--source', type=str,
                       choices=['enhanced_yahoo', 'synthetic_lob', 'crypto_lob', 'all'],
                       default='enhanced_yahoo',
                       help='Data source to use')

    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Trading symbol')

    parser.add_argument('--output-dir', type=str, default='data/real_data',
                       help='Output directory for downloaded data')

    # Yahoo Finance specific options
    parser.add_argument('--period', type=str, default='5d',
                       help='Period for Yahoo Finance data (1d, 5d, 1mo, etc.)')

    parser.add_argument('--interval', type=str, default='1m',
                       help='Interval for Yahoo Finance data (1m, 5m, 15m, etc.)')

    # Synthetic data options
    parser.add_argument('--records', type=int, default=10000,
                       help='Number of records for synthetic data')

    parser.add_argument('--levels', type=int, default=10,
                       help='Number of order book levels for synthetic data')

    parser.add_argument('--start-price', type=float, default=100.0,
                       help='Starting price for synthetic data')

    # Utility options
    parser.add_argument('--list-sources', action='store_true',
                       help='List available data sources and exit')

    parser.add_argument('--quick', action='store_true',
                       help='Use smaller datasets for quick testing')

    parser.add_argument('--info-only', action='store_true',
                       help='Show data info without downloading')

    args = parser.parse_args()

    # Initialize downloader
    try:
        downloader = RealDataDownloader(args.output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize downloader: {e}")
        return 1

    # List sources and exit
    if args.list_sources:
        print("\n" + "="*70)
        print("📊 AVAILABLE REAL HFT DATA SOURCES")
        print("="*70)

        sources = downloader.get_available_sources()
        for source_id, info in sources.items():
            print(f"\n🔸 {info['name']} ({source_id})")
            print(f"   Description: {info['description']}")
            if 'max_period' in info:
                print(f"   Max Period: {info['max_period']}")
            if 'levels' in info:
                print(f"   Default Levels: {info['levels']}")
            if 'frequency' in info:
                print(f"   Frequency: {info['frequency']}")

        print(f"\n📁 Data will be saved to: {args.output_dir}")
        return 0

    # Adjust parameters for quick mode
    if args.quick:
        args.records = min(args.records, 1000)
        args.period = "1d"
        logger.info("🚀 Quick mode enabled - using smaller datasets")

    print(f"\n🚀 Starting HFT data download...")
    print(f"Source: {args.source}")
    print(f"Symbol: {args.symbol}")
    print(f"Output: {args.output_dir}")

    downloaded_files = []
    data_info = {}

    try:
        if args.source == 'enhanced_yahoo' or args.source == 'all':
            logger.info("📈 Downloading enhanced Yahoo Finance data...")
            data = downloader.download_enhanced_yahoo_data(
                symbol=args.symbol,
                period=args.period,
                interval=args.interval
            )

            file_path = f"{args.output_dir}/{args.symbol}_{args.period}_{args.interval}_enhanced.csv"
            downloaded_files.append(file_path)
            data_info[file_path] = downloader.get_data_info(data)

        if args.source == 'synthetic_lob' or args.source == 'all':
            logger.info("🔧 Generating synthetic LOB data...")
            records = min(args.records, 5000) if args.source == 'all' else args.records

            data = downloader.generate_synthetic_lob_data(
                symbol=args.symbol,
                start_price=args.start_price,
                num_records=records,
                levels=args.levels
            )

            file_path = f"{args.output_dir}/{args.symbol}_synthetic_lob_{records}.csv"
            downloaded_files.append(file_path)
            data_info[file_path] = downloader.get_data_info(data)

        if args.source == 'crypto_lob' or args.source == 'all':
            logger.info("₿ Generating crypto LOB data...")
            records = min(args.records, 5000) if args.source == 'all' else args.records

            # Use crypto-appropriate symbol and price
            crypto_symbol = args.symbol if 'BTC' in args.symbol.upper() else 'BTCUSDT'
            crypto_price = 45000.0 if crypto_symbol.startswith('BTC') else args.start_price

            data = downloader.generate_crypto_lob_data(
                symbol=crypto_symbol,
                start_price=crypto_price,
                num_records=records
            )

            file_path = f"{args.output_dir}/{crypto_symbol}_crypto_lob_{records}.csv"
            downloaded_files.append(file_path)
            data_info[file_path] = downloader.get_data_info(data)

        # Display results summary
        print(f"\n✅ Download completed successfully!")
        print(f"📁 Generated {len(downloaded_files)} datasets:")

        for file_path in downloaded_files:
            info = data_info.get(file_path, {})
            filename = Path(file_path).name

            print(f"\n📊 {filename}")
            print(f"   📈 Records: {info.get('num_records', 'N/A'):,}")
            print(f"   📋 Columns: {info.get('num_columns', 'N/A')}")
            print(f"   💾 Size: {info.get('memory_usage_mb', 0):.2f} MB")

            if 'date_range' in info and info['date_range']['start'] != 'N/A':
                print(f"   📅 Range: {info['date_range']['start']} → {info['date_range']['end']}")

            if 'price_stats' in info:
                ps = info['price_stats']
                print(f"   💰 Price: ${ps['min']:.2f} - ${ps['max']:.2f} (avg: ${ps['mean']:.2f})")

        print(f"\n🎯 Next Steps:")
        print(f"   1. Use with HFT pipeline:")
        print(f"      python run_full_pipeline.py --data_path {downloaded_files[0]}")
        print(f"   2. View specific dataset:")
        print(f"      python -c \"import pandas as pd; print(pd.read_csv('{downloaded_files[0]}').head())\"")
        print(f"   3. Analyze data quality:")
        print(f"      python -c \"import pandas as pd; df=pd.read_csv('{downloaded_files[0]}'); print(df.describe())\"")

        print(f"\n💡 Pro Tips:")
        print(f"   • Use --quick flag for faster testing")
        print(f"   • Try --source all to generate multiple datasets")
        print(f"   • Synthetic data is great for backtesting strategies")
        print(f"   • Enhanced Yahoo data includes realistic microstructure features")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    exit(main())