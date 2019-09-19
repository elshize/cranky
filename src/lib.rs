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
#[derive(Debug)]
pub struct ResultRecord {
    qid: Qid,
    docid: Docid,
    rank: Rank,
    score: Score,
    iter: Iter,
    run: Run,
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

/// Result record.
#[derive(Debug)]
pub struct RelevanceRecord {
    qid: Qid,
    docid: Docid,
    iter: Iter,
    relevance: Relevance,
}

impl Eq for RelevanceRecord {}

impl Ord for RelevanceRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.iter, &self.qid, &self.docid).cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialOrd for RelevanceRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.iter, &self.qid, &self.docid).partial_cmp(&(&other.iter, &other.qid, &other.docid))
    }
}

impl PartialEq for RelevanceRecord {
    fn eq(&self, other: &Self) -> bool {
        (&self.iter, &self.qid, &self.docid) == (&other.iter, &other.qid, &other.docid)
    }
}

struct RecordFactory {
    qid_factory: StringIdFactory<Qid>,
    iter_factory: StringIdFactory<Iter>,
    run_factory: StringIdFactory<Run>,
}

trait Record: Sized {
    fn parse_record(record_line: &str, record_factory: &mut RecordFactory) -> Result<Self, Error>;
}

impl Record for ResultRecord {
    fn parse_record(record_line: &str, record_factory: &mut RecordFactory) -> Result<Self, Error> {
        let fields: Vec<&str> = record_line.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(format_err!(
                "Invalid number of colums {}; expected 6",
                fields.len()
            ));
        }
        let qid = record_factory.qid_factory.get(fields[0]);
        let iter = record_factory.iter_factory.get(fields[1]);
        let docid = Docid(String::from(fields[2]));
        let rank: Rank = fields[3].parse()?;
        let score: Score = fields[4].parse()?;
        let run = record_factory.run_factory.get(fields[5]);
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

impl Record for RelevanceRecord {
    fn parse_record(record_line: &str, record_factory: &mut RecordFactory) -> Result<Self, Error> {
        let fields: Vec<&str> = record_line.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(format_err!(
                "Invalid number of colums {}; expected 4",
                fields.len()
            ));
        }
        let qid = record_factory.qid_factory.get(fields[0]);
        let iter = record_factory.iter_factory.get(fields[1]);
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

impl RecordFactory {
    fn new() -> Self {
        Self {
            qid_factory: StringIdFactory::new(),
            iter_factory: StringIdFactory::new(),
            run_factory: StringIdFactory::new(),
        }
    }
}

/// Abstraction over a set of results from a single file in TREC format.
pub struct ResultSet(pub Vec<ResultRecord>);

fn read_records<R, T>(reader: R) -> Result<Vec<T>, Error>
where
    R: Read,
    T: Record,
{
    let reader = BufReader::new(reader);
    let mut record_factory = RecordFactory::new();
    let res: Result<Vec<_>, Error> = reader
        .lines()
        .map(|line| T::parse_record(&line?, &mut record_factory))
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
        let compare = |res: &ResultRecord, rel: &RelevanceRecord| {
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
pub struct Judgements(pub Vec<RelevanceRecord>);

impl Judgements {
    /// Reads file to memory.
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Error> {
        Ok(Self(read_records(reader)?))
    }
}

/// A convenience structure used to produce `Qid` objects.
///
/// It stores all previously used query IDs.
/// When possible, it reuses a string.
struct StringIdFactory<T> {
    ids: HashMap<String, Rc<String>>,
    phantom: std::marker::PhantomData<T>,
}

impl<T> StringIdFactory<T>
where
    T: From<Rc<String>>,
{
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Construct a new ID reusing a string if possible.
    fn get(&mut self, qid: &str) -> T {
        if let Some(qid) = self.ids.get(qid) {
            T::from(Rc::clone(qid))
        } else {
            let new_qid = Rc::new(qid.to_string());
            self.ids.insert(qid.to_string(), Rc::clone(&new_qid));
            T::from(new_qid)
        }
    }
}

impl<T> Default for StringIdFactory<T>
where
    T: From<Rc<String>>,
{
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

    fn rel(qid: &str, iter: &str, docid: &str, rel: Relevance) -> RelevanceRecord {
        RelevanceRecord {
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
