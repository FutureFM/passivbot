# D-TREND sobre Nautilus Trader

Réplica del sistema "D-TREND" de @Toptick_capital (hilo de X, 5 oct 2026), con datos públicos
de Binance USDⓈ-M y con [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader)
como motor de backtest y de ejecución en vivo.

Este directorio es un proyecto independiente. No usa código de Passivbot.

## 1. Qué dice el hilo y dónde está en el código

| Hilo | Implementación |
|---|---|
| Tres patas: TS momentum, XS momentum, XS carry | `[[legs]]` en `configs/dtrend.toml`; `model/system.py` |
| 4 factores TS: distancia a banda Bollinger 2SD, breakout (Carver), returns, EWMAC | `model/rules.py`: `bollinger`, `breakout`, `returns`, `ewmac` |
| Los mismos factores como momentum cross-sectional | pata `mode = "xs"`: forecast menos la media del universo (`forecasts.cross_sectional_demean`) |
| Carry = media de 3 días del funding | `rules.carry` (lookback 3), pata `xs_carry` |
| "±20 forecasts" | 10 reglas TS + 10 XS + 1 carry = 21 |
| Escalado de forecast a abs 10, tope ±20 | `forecasts.scale_and_cap` (escalar causal, agrupado entre monedas) |
| FDM e IDM | `forecasts.forecast_diversification_multiplier`, `system.instrument_diversification_multiplier` |
| Volatility targeting por posición, τ = 50 % | `posición = F/10 · τ · IDM · w_i / σ_i` |
| Igual peso por riesgo entre patas (Narang) | `risk.leg_weighting = "equal_risk"`: cada pata se escala a τ con su vol realizada, y luego un multiplicador de diversificación entre patas |
| Suavizado L/4 de Carver | `smooth` por regla (por defecto `lookback / 4`) |
| Buffers de rebalanceo | `execution.buffer_fraction = 0.10` (buffer de Carver) |
| Costes 9.5 bp por operación, funding incluido | `execution.cost_per_trade`; el funding entra en cada retorno |
| Top 50 monedas, base sin sesgo de supervivencia | universo top 50 por volumen; lista de símbolos de data.binance.vision (incluye deslistados) |
| Tamaño con el equity del cierre t−1, compuesto | backtest de investigación y estrategia Nautilus |
| Deciles de forecast vs retorno ajustado por vol, T+1..T+8; tests de IC | `research/analysis.py`: `forecast_deciles`, `information_coefficient` |
| Sharpe bruto/neto ± error estándar, por año y por pata | `summary.csv`, `sharpe_by_year.csv` |
| Turnover (Carver) y regla "speed limit" 0.15 SR | `turnover_x_yr`, `speed_limit.csv`, `--speed-limit` |
| Gráfico de forecasts por moneda | `latest_forecasts.png` |

## 2. Arquitectura

```
Binance (data.binance.vision + fapi REST, público)
        │  dtrend download
        ▼
data/binance/{klines,funding}/*.parquet
        │  load_panel  (universo causal top-N)
        ▼
model/system.run_model(panel, cfg)   ← función pura y causal
        │  pesos objetivo + anchura de buffer por moneda
        ├──► research/   backtest rápido, estadísticas, gráficos
        ├──► nautilus/backtest.py   BacktestEngine + módulo de funding
        └──► nautilus/live.py       TradingNode Binance USDT-M
```

- Las tres rutas usan la misma función `run_model`. Por eso operan los mismos objetivos.
- Un test verifica que `run_model` es causal: con datos hasta t, da el mismo objetivo en t.
- Nautilus 1.231 no liquida el funding de perpetuos en backtest. `nautilus/funding.py` lo hace:
  antes de procesar la barra que cierra en T, liquida los eventos de funding ≤ T sobre la
  posición abierta.
- La estrategia calcula el tamaño solo con el estado del exchange (posiciones, balance) y con
  datos de mercado. Un reinicio produce las mismas órdenes.

## 3. Instalación

Python 3.12 o superior.

```bash
cd dtrend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest                      # 29 tests, sin red
dtrend demo                 # pipeline completo con datos sintéticos
```

## 4. Uso

```bash
# 1. Descargar datos públicos (sin claves API). Todos los perps USDT, incluidos los deslistados.
dtrend download configs/dtrend.toml

# 2. Investigación: Sharpe, costes, turnover, deciles, IC, gráficos -> results/research/
dtrend research configs/dtrend.toml
dtrend research configs/dtrend.toml --speed-limit      # quitar reglas que cuestan > 0.15 SR/año

# 3. Backtest por eventos en Nautilus -> results/nautilus/
dtrend backtest configs/dtrend.toml
dtrend backtest configs/dtrend.toml --mode live_equivalent   # recalcula el modelo en cada cierre (lento)

# 4. En vivo. Por defecto: TESTNET y dry-run (solo registra órdenes).
export BINANCE_API_KEY=... BINANCE_API_SECRET=...
dtrend live configs/dtrend.toml --environment testnet
dtrend live configs/dtrend.toml --environment testnet --send-orders
```

Para dinero real hacen falta las dos opciones `--send-orders` y `--i-understand-real-money`.
Ejecute `dtrend download` cada día antes de las 00:00 UTC, o al arrancar el nodo, para tener el
historial al día.

## 5. Añadir una estrategia

Una estrategia nueva es una regla nueva. Siga estos pasos:

1. Escriba la función en `src/dtrend/model/rules.py`. Debe devolver un DataFrame (tiempo × moneda)
   y usar solo datos hasta el cierre t.
2. Añada el nombre a `RULE_KINDS` en `config.py` y una rama en `raw_forecast`.
3. Añádala a una pata en el TOML: `{ kind = "mi_regla", lookback = 20 }`. Use `mode = "xs"` para la
   versión cross-sectional. Cree una pata nueva con `[[legs]]` si es un factor distinto.
4. Ejecute `dtrend research` y mire `information_coefficient.csv` y `forecast_deciles.png`
   antes de mirar el Sharpe (el hilo construye las reglas con scatter plots e IC, no con backtests).

El escalado a 10, el tope a 20, el FDM, el IDM, el vol targeting y el peso por riesgo se aplican
automáticamente.

## 6. Diferencias con el hilo y límites conocidos

- **Market cap.** El hilo usa el top 50 por capitalización. Binance no publica capitalización.
  El universo usa el volumen en USDT de 30 días como sustituto.
- **Parámetros no publicados.** El hilo no da los lookbacks exactos, el span de vol ni los pesos
  de las reglas. Los valores por defecto son los estándar de Carver. Todos están en el TOML.
- **IDM de las patas XS.** Las patas XS son neutrales al mercado, y su IDM teórico supera 2.5.
  Con el tope de Carver (`max_idm = 2.5`), una pata XS sola queda por debajo de τ. En el libro
  combinado, el escalado por riesgo de cada pata (`leg_scale_bounds`) corrige este efecto.
- **Funding en Nautilus.** El precio de liquidación es el último cierre conocido, no el mark price.
- **Datos.** La descarga solo se probó con un servidor HTTP falso (`tests/test_downloader.py`).
  Este entorno de desarrollo no tenía acceso de red a Binance.
- **En vivo.** La configuración del nodo se construye en un test, pero no se conectó a Binance.
  Pruebe primero en TESTNET o DEMO con `dry-run`.
- **Speed limit.** La selección de reglas por speed limit usa la muestra completa (in-sample).
  El hilo dice que esta regla reduce el Sharpe. Por eso está desactivada por defecto.

## 7. Fuentes que cita el hilo

Robert Carver (*Advanced Futures Trading Strategies*, *Systematic Trading*, blog), Rishi K. Narang
(*Inside the Black Box*), y los artículos sobre funding de @dima_quant, @ScottPh77711570 y @pedma7.
