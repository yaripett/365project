import { useEffect, useState } from "react";
import "./App.css";

const LABELS = ["Passing yds", "Rushing yds", "Interceptions", "Pass TDs",
  "Rush TDs", "1st downs (pass)", "1st downs (rush)", "Total yds"];

// Average of the team's last 4 games before the given week (matches training)
function formBefore(games, week) {
  const prior = games.filter((g) => g.w < week).slice(-4);
  if (prior.length < 4) return null;
  return LABELS.map((_, i) => prior.reduce((sum, g) => sum + g.s[i], 0) / 4);
}

function predictHomeWin(m, hf, af) {
  let z = m.intercept;
  hf.forEach((v, i) => {
    z += (m.coef[i] * (v - af[i] - m.mean[i])) / m.scale[i];
  });
  return 1 / (1 + Math.exp(-z));
}

export default function App() {
  const [model, setModel] = useState(null);
  const [error, setError] = useState("");
  const [seasonSel, setSeasonSel] = useState(null);
  const [weekSel, setWeekSel] = useState(99);
  const [homeSel, setHomeSel] = useState("");
  const [awaySel, setAwaySel] = useState("");

  useEffect(() => {
    fetch("/model.json")
      .then((r) => r.json())
      .then(setModel)
      .catch(() => setError("Could not load model.json. Put it in the public folder."));
  }, []);

  if (error) return <p className="status">{error}</p>;
  if (!model) return <p className="status">Loading…</p>;

  const seasons = Object.keys(model.teams).map(Number).sort((a, b) => a - b);
  const season = seasonSel ?? seasons[seasons.length - 1];
  const seasonData = model.teams[season];
  const teams = Object.keys(seasonData).sort();
  const maxWeek = Math.max(...Object.values(seasonData).flat().map((g) => g.w));
  const week = Math.min(Math.max(weekSel, 5), maxWeek + 1);
  const home = teams.includes(homeSel) ? homeSel : teams[0];
  const away = teams.includes(awaySel) ? awaySel : teams[1];

  const hf = home !== away ? formBefore(seasonData[home], week) : null;
  const af = home !== away ? formBefore(seasonData[away], week) : null;
  const pHome = hf && af ? predictHomeWin(model, hf, af) : null;
  const winner = pHome === null ? null : pHome >= 0.5 ? home : away;
  const pWin = pHome === null ? null : Math.max(pHome, 1 - pHome);

  return (
    <div className="app">
      <header>
        <h1>NFL Matchup Predictor</h1>
        <p>Held-out accuracy ({model.trainThrough + 1}+ games): {(model.accuracy * 100).toFixed(1)}% on {model.testGames} games</p>
      </header>

      <main>
        <section className="card controls">
          <label>Season
            <select value={season} onChange={(e) => setSeasonSel(Number(e.target.value))}>
              {seasons.map((s) => <option key={s}>{s}</option>)}
            </select>
          </label>
          <label>Game in week {week}
            <input type="range" min={5} max={maxWeek + 1} value={week}
              onChange={(e) => setWeekSel(Number(e.target.value))} />
          </label>
          <label>Home team
            <select value={home} onChange={(e) => setHomeSel(e.target.value)}>
              {teams.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>
          <label>Away team
            <select value={away} onChange={(e) => setAwaySel(e.target.value)}>
              {teams.map((t) => <option key={t}>{t}</option>)}
            </select>
          </label>
        </section>

        {home === away && <p className="notice">Pick two different teams.</p>}
        {home !== away && pHome === null && (
          <p className="notice">A team needs 4 games before this week. Try a later week.</p>
        )}

        {pHome !== null && (
          <section className="card result">
            <div className="versus">
              <span>{away}</span><em>AT</em><span>{home}</span>
            </div>
            <h2>Predicted winner: <strong>{winner}</strong></h2>
            <p className="prob">{(pWin * 100).toFixed(1)}% win probability</p>
            <div className="bar">
              <div className="away" style={{ width: `${(1 - pHome) * 100}%` }}>{away} {((1 - pHome) * 100).toFixed(0)}%</div>
              <div className="home" style={{ width: `${pHome * 100}%` }}>{home} {(pHome * 100).toFixed(0)}%</div>
            </div>
            {season <= model.trainThrough && (
              <p className="warn">The model trained on {season} games, so this is not a true held-out prediction.</p>
            )}
            <table>
              <thead><tr><th>Last-4 avg</th><th>{away}</th><th>{home}</th></tr></thead>
              <tbody>
                {LABELS.map((l, i) => (
                  <tr key={l}><td>{l}</td><td>{af[i].toFixed(1)}</td><td>{hf[i].toFixed(1)}</td></tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </main>
      <footer>Simple rolling averages and one chronological split. NFL games are noisy; treat probabilities as rough estimates.</footer>
    </div>
  );
}
