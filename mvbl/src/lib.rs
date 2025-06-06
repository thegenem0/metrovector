use ebml_iterable::{TagIterator, TagWriter};

#[cfg(feature = "futures")]
use ebml_iterable::nonblocking::TagIteratorAsync;

pub use ebml_iterable::WriteOptions;
pub use ebml_iterable::iterator;
pub mod errors;
pub mod spec;

use spec::MvblSpec;

///
/// Alias for [`ebml_iterable::TagIterator`] using [`MvblSpec`] as the tag specification
///
/// This implements Rust's standard [`Iterator`] trait. The struct can be created with the `new` function on any source that implements the [`std::io::Read`] trait.
/// The iterator outputs [`MvblSpec`] variants containing the tag data. See the [ebml-iterable](https://crates.io/crates/ebml_iterable) docs for more information if needed.
///
/// Note: The `with_capacity` method can be used to construct a `[MvblIterator]` with a specified default buffer size.  
/// This is only useful as a microoptimization to memory management if you know the maximum tag size of the file you're reading.
///
pub type MvblIterator<R> = TagIterator<R, MvblSpec>;

///
/// Alias for [`ebml_iterable::TagIteratorAsync`] using [`MvblSpec`] as the generic type.
///
/// This Can be transformed into a [`Stream`] using [`into_stream`]. The struct can be created with the [`new()`] function on any source that implements the [`futures::AsyncRead`] trait.
/// The stream outputs [`MvblSpec`] variants containing the tag data. See the [ebml-iterable](https://crates.io/crates/ebml_iterable) docs for more information if needed.
///
#[cfg(feature = "futures")]
pub type MvblIteratorAsync<R> = TagIteratorAsync<R, MvblSpec>;

///
/// Alias for [`ebml_iterable::TagWriter`].
///
/// This can be used to write `mvbl` files from tag data. This struct can be created with the `new` function on any source that implements the [`std::io::Write`] trait.
/// See the [ebml-iterable](https://crates.io/crates/ebml_iterable) docs for more information if needed.
///
pub type MvblWriter<W> = TagWriter<W>;

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::{
        MvblIterator, MvblWriter,
        spec::{Master, MvblSpec},
    };

    #[test]
    fn basic_tag_stream_write_and_iterate() {
        let tags_to_write: Vec<MvblSpec> = vec![
            MvblSpec::Ebml(Master::Start),
            MvblSpec::EbmlVersion(1),
            MvblSpec::EbmlReadVersion(1),
            MvblSpec::EbmlMaxIdLength(4),
            MvblSpec::EbmlMaxSizeLength(8),
            MvblSpec::DocType("MetroVectorBlockLanguage".to_string()),
            MvblSpec::DocTypeVersion(1),
            MvblSpec::DocTypeReadVersion(1),
            MvblSpec::Ebml(Master::End),
            MvblSpec::Segment(Master::Start),
            MvblSpec::Cart(Master::Start),
            MvblSpec::CartHeader(Master::Start),
            MvblSpec::VectorCount(999),
            MvblSpec::CartHeader(Master::End),
            MvblSpec::DeletedVectorOffsets(vec![0, 0, 0, 0]),
            MvblSpec::Cart(Master::End),
            MvblSpec::Segment(Master::End),
        ];

        let mut dest = Cursor::new(Vec::new());
        let mut writer = MvblWriter::new(&mut dest);

        for tag in &tags_to_write {
            writer.write(tag).expect("Test shouldn't error on write");
        }

        let mut src = Cursor::new(dest.into_inner());
        let reader = MvblIterator::new(&mut src, &[]);

        let tags_read_back: Vec<MvblSpec> = reader.map(|t| t.unwrap()).collect();
        assert_eq!(MvblSpec::Ebml(Master::Start), tags_read_back[0]);
        assert_eq!(MvblSpec::EbmlVersion(1), tags_read_back[1]);
        assert_eq!(MvblSpec::EbmlReadVersion(1), tags_read_back[2]);
        assert_eq!(MvblSpec::EbmlMaxIdLength(4), tags_read_back[3]);
        assert_eq!(MvblSpec::EbmlMaxSizeLength(8), tags_read_back[4]);
        assert_eq!(
            MvblSpec::DocType("MetroVectorBlockLanguage".to_string()),
            tags_read_back[5]
        );
        assert_eq!(MvblSpec::DocTypeVersion(1), tags_read_back[6]);
        assert_eq!(MvblSpec::DocTypeReadVersion(1), tags_read_back[7]);
        assert_eq!(MvblSpec::Ebml(Master::End), tags_read_back[8]);
        assert_eq!(MvblSpec::Segment(Master::Start), tags_read_back[9]);
        assert_eq!(MvblSpec::Cart(Master::Start), tags_read_back[10]);
        assert_eq!(MvblSpec::CartHeader(Master::Start), tags_read_back[11]);
        assert_eq!(MvblSpec::VectorCount(999), tags_read_back[12]);
        assert_eq!(MvblSpec::CartHeader(Master::End), tags_read_back[13]);
        assert_eq!(
            MvblSpec::DeletedVectorOffsets(vec![0, 0, 0, 0]),
            tags_read_back[14]
        );
        assert_eq!(MvblSpec::Cart(Master::End), tags_read_back[15]);
        assert_eq!(MvblSpec::Segment(Master::End), tags_read_back[16]);
    }
}
