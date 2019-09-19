#![warn(
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_import_braces,
    unused_qualifications
)]
#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

//! Cranky is a library and tool for evaluating query effectiveness for information retrieval.
//!
//! It is an attempt to replace tools like [`trec_eval`](https://github.com/usnistgov/trec_eval),
//! [`gdeval`](https://trec.nist.gov/data/web/12/gdeval.pl),
//! and [`ndeval`](https://trec.nist.gov/data/web/12/ndeval.c).
//! The goal is to be fast but also provide a codebase that is easier to maintain, as well as
//! provide interface to use in Rust and (eventually) C/C++ libraries.
//! From there, we can create bindings to basically any other language.

use failure::{format_err, Error, ResultExt};
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::io::{prelude::*, BufReader};
use std::rc::Rc;
use std::str::FromStr;

/// Represents a query ID in TREC format.
///
/// The IDs are strings, and therefore can be costly to copy.
/// However, there are typically many records with the same query ID.
/// We use this fact by storing a reference counted pointer instead.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qid(pub Rc<String>);

/// Represents a query iteration.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Iter(pub Rc<String>);

/// Represents a query run.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Run(pub Rc<String>);

/// Document TREC ID.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Docid(pub String);

/// Floating point result score.
#[derive(Debug, PartialEq)]
pub struct Score(pub f32);

/// Document rank.
#[derive(Debug, PartialEq)]
pub struct Rank(pub u32);

/// Document relevance used as gold standard.
#[derive(Debug, PartialEq, Eq)]
pub enum Relevance {
    /// Judged document with a certain relevance score.
    Judged(u8),
    /// Document present in the reference file but left unjudged.
    Unjudged,
    /// Document not present.
    Missing,
}

impl fmt::Display for Relevance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Relevance::*;
        match self {
            &Judged(rel) => write!(f, "{}", rel),
            Unjudged => write!(f, "-1"),
            Missing => write!(f, "-2"),
        }
    }
}

impl Default for Relevance {
    fn default() -> Self {
        Self::Missing
    }
}

impl From<Rc<String>> for Qid {
    fn from(qid: Rc<String>) -> Self {
        Self(qid)
    }
}

impl From<Rc<String>> for Iter {
    fn from(id: Rc<String>) -> Self {
        Self(id)
    }
}

impl From<Rc<String>> for Run {
    fn from(id: Rc<String>) -> Self {
        Self(id)
    }
}

impl FromStr for Score {
    type Err = Error;

    fn from_str(score: &str) -> Result<Self, Self::Err> {
        let score: f32 = score.parse()?;
        Ok(Self(score))
    }
}

impl FromStr for Rank {
    type Err = Error;

    fn from_str(rank: &str) -> Result<Self, Self::Err> {
        let rank: u32 = rank.parse()?;
        Ok(Self(rank))
    }
}

impl FromStr for Relevance {
    type Err = Error;

    fn from_str(rank: &str) -> Result<Self, Self::Err> {
        let judgement: i32 = rank.parse()?;
        match judgement {
            -1 => Ok(Self::Unjudged),
            rel => Ok(Self::Judged(
                u8::try_from(rel).context("Error parsing judgement")?,
            )),
        }
    }
}

/// Result record.
///
/// # Examples
///
/// The separator could be any number of whitespaces.
///
/// ```
/// # use cranky::{Record, ResultRecord};
/// # fn main() -> Result<(), failure::Error> {
/// let record_line = "030  Q0\tZF08-175-870  0 \t 4238   prise1";
/// let record: ResultRecord = Record::parse_record(record_line, None)?;
/// # Ok(())
/// # }
/// ```
///
/// You can use [`StringIdFactory`](struct.StringIdFactory.html) to reuse string-based IDs.
///
/// ```
/// # use cranky::{Record, ResultRecord, StringIdFactory};
/// # fn main() -> Result<(), failure::Error> {
/// let mut id_factory = StringIdFactory::new();
/// let record_line = "030  Q0\tZF08-175-870  0 \t 4238   prise1";
/// let record: ResultRecord = Record::parse_record(record_line, Some(&mut id_factory))?;
/// # Ok(())
/// # }
/// ```
///
/// When formating to a string, all separators are tabs:
///
/// ```
/// # use cranky::{Record, ResultRecord, StringIdFactory};
/// # fn main() -> Result<(), failure::Error> {
/// # let mut id_factory = StringIdFactory::new();
/// # let record_line = "030  Q0\tZF08-175-870  0 \t 4238   prise1";
/// # let record: ResultRecord = Record::parse_record(record_line, Some(&mut id_factory))?;
/// assert_eq!(
///     &format!("{}", record),
///     "030\tQ0\tZF08-175-870\t0\t4238\tprise1"
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ResultRecord {
    /// Query TREC ID.
    pub qid: Qid,
    /// Document TREC ID.
    pub docid: Docid,
    /// Rank of the document in query result set.
    /// The lower the number, the higher the document is ranked.
    pub rank: Rank,
    /// Score of the document in query result set.
    /// The higher the number, the higher the document is ranked.
    pub score: Score,
    /// Iteration.
    pub iter: Iter,
    /// Run ID.
    pub run: Run,
    relevance: Relevance,
}

impl Eq for ResultRecord {}

impl Ord for ResultRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.iter, &self.qid, &self.docid).cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialOrd for ResultRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.iter, &self.qid, &self.docid).partial_cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialEq for ResultRecord {
    fn eq(&self, other: &Self) -> bool {
        (&self.iter, &self.qid, &self.docid) == (&other.iter, &other.qid, &other.docid)
    }
}

impl ResultRecord {
    fn with_relevance(self, relevance: Relevance) -> Self {
        Self { relevance, ..self }
    }
}

/// Judgement record.
#[derive(Debug)]
pub struct JudgementRecord {
    /// Query TREC ID.
    pub qid: Qid,
    /// Document TREC ID.
    pub docid: Docid,
    /// Iteration.
    pub iter: Iter,
    /// Gold standard relevance.
    pub relevance: Relevance,
}

impl Eq for JudgementRecord {}

impl Ord for JudgementRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.iter, &self.qid, &self.docid).cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialOrd for JudgementRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.iter, &self.qid, &self.docid).partial_cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialEq for JudgementRecord {
    fn eq(&self, other: &Self) -> bool {
        (&self.iter, &self.qid, &self.docid) == (&other.iter, &other.qid, &other.docid)
    }
}

/// Result or judgement record.
pub trait Record: Sized + fmt::Display {
    /// Parses record from a line of text.
    /// `record_factory` is optionally used for reusing string-based IDs;
    /// in case of its absense, all strings will be copied to the record.
    fn parse_record(
        record_line: &str,
        record_factory: Option<&mut StringIdFactory>,
    ) -> Result<Self, Error>;
}

fn rcid(id_factory: &mut Option<&mut StringIdFactory>, id: &str) -> Rc<String> {
    id_factory
        .as_mut()
        .map_or_else(|| Rc::new(id.to_string()), |f| f.get(id))
}

impl fmt::Display for ResultRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.qid.0, self.iter.0, self.docid.0, self.rank.0, self.score.0, self.run.0
        )
    }
}

impl Record for ResultRecord {
    fn parse_record(
        record_line: &str,
        mut id_factory: Option<&mut StringIdFactory>,
    ) -> Result<Self, Error> {
        let fields: Vec<&str> = record_line.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(format_err!(
                "Invalid number of colums {}; expected 6",
                fields.len()
            ));
        }
        let qid = Qid(rcid(&mut id_factory, fields[0]));
        let iter = Iter(rcid(&mut id_factory, fields[1]));
        let docid = Docid(String::from(fields[2]));
        let rank: Rank = fields[3].parse()?;
        let score: Score = fields[4].parse()?;
        let run = Run(rcid(&mut id_factory, fields[5]));
        Ok(Self {
            qid,
            iter,
            docid,
            rank,
            score,
            run,
            relevance: Relevance::default(),
        })
    }
}

impl fmt::Display for JudgementRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}\t{}\t{}\t{}",
            self.qid.0, self.iter.0, self.docid.0, self.relevance
        )
    }
}

impl Record for JudgementRecord {
    fn parse_record(
        record_line: &str,
        mut id_factory: Option<&mut StringIdFactory>,
    ) -> Result<Self, Error> {
        let fields: Vec<&str> = record_line.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(format_err!(
                "Invalid number of colums {}; expected 4",
                fields.len()
            ));
        }
        let qid = Qid(rcid(&mut id_factory, fields[0]));
        let iter = Iter(rcid(&mut id_factory, fields[1]));
        let docid = Docid(String::from(fields[2]));
        let relevance: Relevance = fields[3].parse()?;
        Ok(Self {
            qid,
            iter,
            docid,
            relevance,
        })
    }
}

/// Abstraction over a set of results from a single file in TREC format.
pub struct ResultSet(pub Vec<ResultRecord>);

/// Read records.
pub fn read_records<R, T>(reader: R) -> Result<Vec<T>, Error>
where
    R: Read,
    T: Record,
{
    let reader = BufReader::new(reader);
    let mut id_factory = StringIdFactory::new();
    let res: Result<Vec<_>, Error> = reader
        .lines()
        .map(|line| T::parse_record(&line?, Some(&mut id_factory)))
        .collect();
    Ok(res?)
}

impl ResultSet {
    /// Reads file to memory.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(read_records(reader)?))
    }

    /// Consumes a result set and judgements, and returns a new result set with applied judgements.
    pub fn apply_judgements(mut self, mut judgements: Judgements) -> Self {
        self.0.sort();
        judgements.0.sort();
        let compare = |res: &ResultRecord, rel: &JudgementRecord| {
            (&res.iter, &res.qid, &res.docid).cmp(&(&rel.iter, &rel.qid, &rel.docid))
        };
        Self(
            self.0
                .into_iter()
                .merge_join_by(judgements.0, compare)
                .filter_map(|either| match either {
                    Left(result) => Some(result.with_relevance(Relevance::Missing)),
                    Right(_) => None,
                    Both(result, judgement) => Some(result.with_relevance(judgement.relevance)),
                })
                .collect(),
        )
    }
}

/// Abstraction over a set of relevance judgements.
pub struct Judgements(pub Vec<JudgementRecord>);

impl Judgements {
    /// Reads file to memory.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(read_records(reader)?))
    }
}

/// A convenience structure used to produce String IDs.
/// It stores all previously used query IDs. When possible, it reuses a string.
pub struct StringIdFactory {
    ids: HashMap<String, Rc<String>>,
}

impl StringIdFactory {
    /// Constructs a new empty factory.
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
        }
    }

    /// Construct a new ID reusing a string if possible.
    pub fn get(&mut self, qid: &str) -> Rc<String> {
        if let Some(qid) = self.ids.get(qid) {
            Rc::clone(qid)
        } else {
            let new_qid = Rc::new(qid.to_string());
            self.ids.insert(qid.to_string(), Rc::clone(&new_qid));
            new_qid
        }
    }
}

impl Default for StringIdFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use std::io::Cursor;

    const TREC_FILE: &str = "321 Q0  9171    0   10.9951 R0
321 Q0  4a7c    1   10.9951 R0
321 Q0  a5df    2   10.929  R0
336 Q0  d6ed    0   12.7334 R0
336 Q0  6ddf    1   12.5704 R0
336 Q0  29a7    14  11.4663 R0";

    const QRELS: &str = "321 Q0  9171    0
321 Q0  4a7c    -1
321 Q0  a5df    2
336 Q0  6ddf    1
336 Q0  29a7    0";

    fn rec(
        qid: &str,
        iter: &str,
        docid: &str,
        rank: u32,
        score: f32,
        run: &str,
        rel: Relevance,
    ) -> ResultRecord {
        ResultRecord {
            qid: Qid(Rc::new(qid.to_string())),
            iter: Iter(Rc::new(iter.to_string())),
            docid: Docid(docid.to_string()),
            rank: Rank(rank),
            score: Score(score),
            run: Run(Rc::new(run.to_string())),
            relevance: rel,
        }
    }

    fn rel(qid: &str, iter: &str, docid: &str, rel: Relevance) -> JudgementRecord {
        JudgementRecord {
            qid: Qid(Rc::new(qid.to_string())),
            iter: Iter(Rc::new(iter.to_string())),
            docid: Docid(docid.to_string()),
            relevance: rel,
        }
    }

    #[test]
    fn test_trec_file() {
        let cursor = Cursor::new(&TREC_FILE);
        let trec_file = ResultSet::from_reader(cursor).expect("Could not parse TREC file");
        assert_eq!(
            trec_file.0,
            vec![
                rec("321", "Q0", "9171", 0, 10.9951, "R0", Relevance::Missing),
                rec("321", "Q0", "4a7c", 1, 10.9951, "R0", Relevance::Missing),
                rec("321", "Q0", "a5df", 2, 10.929, "R0", Relevance::Missing),
                rec("336", "Q0", "d6ed", 0, 12.7334, "R0", Relevance::Missing),
                rec("336", "Q0", "6ddf", 1, 12.5704, "R0", Relevance::Missing),
                rec("336", "Q0", "29a7", 14, 11.4663, "R0", Relevance::Missing),
            ]
        );
    }

    #[test]
    fn test_qrels() {
        let cursor = Cursor::new(&QRELS);
        let trec_file = Judgements::from_reader(cursor).expect("Could not parse TREC file");
        assert_eq!(
            trec_file.0,
            vec![
                rel("321", "Q0", "9171", Relevance::Judged(0)),
                rel("321", "Q0", "4a7c", Relevance::Unjudged),
                rel("321", "Q0", "a5df", Relevance::Judged(2)),
                rel("336", "Q0", "6ddf", Relevance::Judged(1)),
                rel("336", "Q0", "29a7", Relevance::Judged(0)),
            ]
        );
    }

    #[test]
    fn test_apply_judgements() {
        let cursor = Cursor::new(&TREC_FILE);
        let results = ResultSet::from_reader(cursor).expect("Could not parse TREC file");
        let cursor = Cursor::new(&QRELS);
        let judgements = Judgements::from_reader(cursor).expect("Could not parse TREC file");
        assert_eq!(
            results.apply_judgements(judgements).0,
            vec![
                rec("321", "Q0", "4a7c", 1, 10.9951, "R0", Relevance::Unjudged),
                rec("321", "Q0", "9171", 0, 10.9951, "R0", Relevance::Judged(0)),
                rec("321", "Q0", "a5df", 2, 10.929, "R0", Relevance::Judged(2)),
                rec("336", "Q0", "29a7", 14, 11.4663, "R0", Relevance::Judged(0)),
                rec("336", "Q0", "6ddf", 1, 12.5704, "R0", Relevance::Judged(1)),
                rec("336", "Q0", "d6ed", 0, 12.7334, "R0", Relevance::Missing),
            ]
        );
    }
}
