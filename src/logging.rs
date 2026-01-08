use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

static START_TIME: OnceLock<Instant> = OnceLock::new();

/// Initialize the logger with custom formatting showing elapsed time.
///
/// If verbose is true, sets log level to Info, otherwise to Warn.
/// Output format: [HH:MM:SS] LEVEL: message
/// All output goes to stderr.
pub fn init_logger(verbose: bool) {
    START_TIME.set(Instant::now()).ok();

    let level = if verbose {
        log::LevelFilter::Info
    } else {
        log::LevelFilter::Warn
    };

    env_logger::Builder::from_default_env()
        .filter_level(level)
        .format(|buf, record| {
            let elapsed = START_TIME.get().unwrap().elapsed();
            let hours = elapsed.as_secs() / 3600;
            let minutes = (elapsed.as_secs() % 3600) / 60;
            let seconds = elapsed.as_secs() % 60;

            writeln!(
                buf,
                "[{:02}:{:02}:{:02}] {}: {}",
                hours,
                minutes,
                seconds,
                record.level(),
                record.args()
            )
        })
        .target(env_logger::Target::Stderr)
        .init();
}
