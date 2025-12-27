.PHONY: help backfill backfill-us backfill-nl signals signals-from-parquet daily daily-us daily-nl test-signals clean-signals view-signals paper-track paper-summary paper-add rebuild-events

# Default target
help:
	@echo "📊 Live Signal Generation Pipeline (Parquet-Based)"
	@echo "=================================================="
	@echo ""
	@echo "  make daily         - Full hourly pipeline: backfill + rebuild parquet + signals"
	@echo "  make daily-us      - US market only: backfill + rebuild + signals"
	@echo "  make daily-nl      - NL market only: backfill + rebuild + signals"
	@echo ""
	@echo "  make backfill      - Step 1: Fetch latest yfinance data (ALL markets)"
	@echo "  make backfill-us   - Step 1: Fetch latest US data only"
	@echo "  make backfill-nl   - Step 1: Fetch latest NL data only"
	@echo ""
	@echo "  make rebuild-events - Step 2: Rebuild parquet from database"
	@echo "  make signals        - Step 3: Generate signals from parquet"
	@echo ""
	@echo "  make view-signals  - View latest signals in readable format"
	@echo "  make test-signals  - Test signal generation with custom date"
	@echo "  make clean-signals - Clean old signal files (keep last 30 days)"
	@echo ""
	@echo "📊 Paper Trading Commands"
	@echo "========================"
	@echo ""
	@echo "  make paper-track   - Update all open trades + show summary"
	@echo "  make paper-add DATE=YYYY-MM-DD - Add signals to paper trade log"
	@echo "  make paper-summary - Show all paper trading performance"
	@echo "  make paper-summary START_DATE=YYYY-MM-DD - Show performance from date"
	@echo "  make paper-summary START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD - Show period"
	@echo ""

# US tickers (143 liquid stocks - expanded universe for more signals!)
# Big 7 mega-caps: AAPL MSFT GOOGL GOOG AMZN NVDA META TSLA (added 2025-12-26 - 83% win rate!)
# Growth/Tech: SNOW CRWD ZS DDOG NET SHOP COIN PLTR RBLX U MRVL WDAY TEAM FTNT MNST ZM DOCU OKTA
# Consumer: UBER LYFT DASH ABNB
# EV/Auto: RIVN LCID NIO XPEV LI
# Fintech: SOFI HOOD NU
# Biotech: MRNA BNTX
# Energy: ENPH FSLR
# S&P 100 core: BRK.B V JPM WMT LLY UNH XOM MA AVGO JNJ PG HD COST ABBV ORCL NFLX CVX MRK KO BAC AMD PEP TMO ADBE CSCO ACN CRM MCD LIN ABT INTC WFC DHR NKE CMCSA TXN QCOM DIS VZ PM IBM INTU UNP AMGN CAT GE RTX SPGI LOW HON NEE UPS AMAT PFE BLK ISRG SYK BA T ELV AXP DE BKNG ADI GILD MS LMT TJX CI PLD VRTX SBUX MDLZ MMC GS ADP TMUS BMY C NOW REGN ZTS SO SCHW MO AMT ETN BDX CB PANW DUK SLB BSX COP AON MMM PNC MU ITW USB
US_TICKERS := AAPL MSFT GOOGL GOOG AMZN NVDA META TSLA SNOW CRWD ZS DDOG NET SHOP COIN PLTR RBLX U MRVL WDAY TEAM FTNT MNST ZM DOCU OKTA UBER LYFT DASH ABNB RIVN LCID NIO XPEV LI SOFI HOOD NU MRNA BNTX ENPH FSLR BRK.B V JPM WMT LLY UNH XOM MA AVGO JNJ PG HD COST ABBV ORCL NFLX CVX MRK KO BAC AMD PEP TMO ADBE CSCO ACN CRM MCD LIN ABT INTC WFC DHR NKE CMCSA TXN QCOM DIS VZ PM IBM INTU UNP AMGN CAT GE RTX SPGI LOW HON NEE UPS AMAT PFE BLK ISRG SYK BA T ELV AXP DE BKNG ADI GILD MS LMT TJX CI PLD VRTX SBUX MDLZ MMC GS ADP TMUS BMY C NOW REGN ZTS SO SCHW MO AMT ETN BDX CB PANW DUK SLB BSX COP AON MMM PNC MU ITW USB

# NL tickers (Top 25)
NL_TICKERS := ABN.AS AD.AS AGN.AS AKZA.AS ALFEN.AS ALLFG.AS ASML.AS ASRNL.AS BAMNB.AS BESI.AS HEIA.AS INGA.AS INPST.AS KPN.AS LIGHT.AS NN.AS OCI.AS PHIA.AS RAND.AS REN.AS SBMO.AS SHELL.AS TOM2.AS UNA.AS WKL.AS

# Timestamp for daily signals
DATE := $(shell date +%Y-%m-%d)
ASOF := $(shell date +"%Y-%m-%d 10:00:00")

# Full daily pipeline: backfill + rebuild parquet + generate signals
daily: backfill rebuild-events signals
	@echo ""
	@echo "✅ Daily pipeline complete (ALL markets)!"
	@echo "📁 Signals saved to: data/signals/breakout_signals_$(DATE).csv"

# Daily pipeline - US only
daily-us: backfill-us rebuild-events signals
	@echo ""
	@echo "✅ Daily pipeline complete (US market only)!"
	@echo "📁 Signals saved to: data/signals/breakout_signals_$(DATE).csv"

# Daily pipeline - NL only
daily-nl: backfill-nl rebuild-events signals
	@echo ""
	@echo "✅ Daily pipeline complete (NL market only)!"
	@echo "📁 Signals saved to: data/signals/breakout_signals_$(DATE).csv"

# Backfill latest hourly data from Yahoo Finance (all markets)
backfill: backfill-us backfill-nl
	@echo ""
	@echo "✅ Backfill complete for ALL markets!"

# Backfill US market only
backfill-us:
	@echo "🔄 Fetching latest US market data (93 tickers)..."
	@.venv/bin/python scripts/backfill_ohlcv_hourly.py --tickers $(US_TICKERS) --period 5d
	@echo "✅ US backfill complete!"

# Backfill NL market only
backfill-nl:
	@echo "🔄 Fetching latest NL market data (25 tickers)..."
	@.venv/bin/python scripts/backfill_ohlcv_hourly.py --tickers $(NL_TICKERS) --period 5d
	@echo "✅ NL backfill complete!"

# Generate signals from parquet file (default - proven 80% win rate)
signals:
	@echo "🎯 Generating signals from parquet (validated method)..."
	@mkdir -p data/signals
	@.venv/bin/python scripts/live_tracker_breakouts.py \
		--config config/live_universe.json \
		--events data/ml_datasets/accum_distrib_events.parquet \
		--asof "$(ASOF)" \
		--lookback_hours 24 \
		--output data/signals/breakout_signals_$(DATE).csv \
		--summary_out data/signals/signal_summary_$(DATE).json
	@echo ""
	@echo "✅ Signal generation complete!"
	@echo ""
	@if [ -f data/signals/breakout_signals_$(DATE).csv ]; then \
		.venv/bin/python scripts/display_signals.py data/signals/breakout_signals_$(DATE).csv; \
	fi

# Rebuild events parquet file from database (run after backfill)
rebuild-events:
	@echo "🔄 Rebuilding events parquet from database..."
	@echo ""
	@.venv/bin/python scripts/build_accum_distrib_events.py \
		--output data/ml_datasets/accum_distrib_events.parquet
	@echo ""
	@echo "✅ Parquet file rebuilt!"
	@echo "📁 Saved to: data/ml_datasets/accum_distrib_events.parquet"

# Test signal generation with custom date/time
test-signals:
	@read -p "Enter asof datetime (YYYY-MM-DD HH:MM:SS): " asof; \
	read -p "Enter lookback hours (default: 24): " lookback; \
	lookback=$${lookback:-24}; \
	testdate=$$(echo $$asof | cut -d' ' -f1); \
	echo ""; \
	echo "🧪 Testing signal generation..."; \
	echo "   asof: $$asof"; \
	echo "   lookback: $$lookback hours"; \
	echo ""; \
	.venv/bin/python scripts/live_tracker_breakouts.py \
		--config config/live_universe.json \
		--events data/ml_datasets/accum_distrib_events.parquet \
		--asof "$$asof" \
		--lookback_hours $$lookback \
		--output data/signals/test_signals_$$testdate.csv \
		--summary_out data/signals/test_summary_$$testdate.json

# Clean old signal files (keep last 30 days)
clean-signals:
	@echo "🧹 Cleaning signal files older than 30 days..."
	@find data/signals -name "breakout_signals_*.csv" -mtime +30 -delete
	@find data/signals -name "signal_summary_*.json" -mtime +30 -delete
	@echo "✅ Cleanup complete!"

# View latest signals in readable format
view-signals:
	@if [ -f data/signals/breakout_signals_$(DATE).csv ]; then \
		.venv/bin/python scripts/display_signals.py data/signals/breakout_signals_$(DATE).csv; \
	else \
		echo "❌ No signals found for today ($(DATE))"; \
		echo ""; \
		echo "Available signal files:"; \
		ls -t data/signals/breakout_signals_*.csv 2>/dev/null | head -5 || echo "  No signal files found"; \
	fi

# Paper trading commands
paper-track:
	@echo "🔄 Updating paper trades..."
	@.venv/bin/python scripts/paper_trade_tracker.py --update-only

paper-add:
	@echo "📝 Adding today's signals to paper trade log..."
	@.venv/bin/python scripts/paper_trade_tracker.py --date $(DATE)

paper-summary:
	@echo "📊 Paper trading summary..."
ifdef START_DATE
ifdef END_DATE
	@.venv/bin/python scripts/paper_trade_tracker.py --update-only --start-date $(START_DATE) --end-date $(END_DATE)
else
	@.venv/bin/python scripts/paper_trade_tracker.py --update-only --start-date $(START_DATE)
endif
else
	@.venv/bin/python scripts/paper_trade_tracker.py --update-only
endif
