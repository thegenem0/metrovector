//!
//! Provides the [`MvblSpec`] enum, which implements [`EbmlSpecification`] and [`EbmlTag`].
//!
//! This is used in conjuction with the [ebml_iterable](https://crates.io/crates/ebml_iterable) library to be able to read and write MVBL formatted files based on raw tag data.
//! These can easily be converted to and from the regular enum variants using `into()` and `try_from()` to make working with the iterator stream easier.
//!

use ebml_iterable::specs::easy_ebml;
pub use ebml_iterable::specs::{EbmlSpecification, EbmlTag, Master, TagDataType};

easy_ebml! {
    #[derive(Clone, PartialEq, Debug)]
    pub enum MvblSpec {
        // --- EBML HEADER ---
        Ebml:                                           Master      = 0x1a45dfa3,
        Ebml/EbmlVersion:                               UnsignedInt = 0x4286,
        Ebml/EbmlReadVersion:                           UnsignedInt = 0x42f7,
        Ebml/EbmlMaxIdLength:                           UnsignedInt = 0x42f2,
        Ebml/EbmlMaxSizeLength:                         UnsignedInt = 0x42f3,
        Ebml/DocType:                                   Utf8        = 0x4282,
        Ebml/DocTypeVersion:                            UnsignedInt = 0x4287,
        Ebml/DocTypeReadVersion:                        UnsignedInt = 0x4285,


        // --- EBML SEGMENT ---
        Segment : Master = 0x18538067,

        // --- MVBL ROOT ---
        Segment/Cart : Master = 0x5A00,

        // --- Cart Header ---
        Segment/Cart/CartHeader : Master = 0x5A01,
        Segment/Cart/CartHeader/CartUUID : Utf8 = 0x5A02,
        Segment/Cart/CartHeader/VectorCount : UnsignedInt = 0x5A03,
        Segment/Cart/CartHeader/Timestamp : UnsignedInt = 0x5A04,
        Segment/Cart/CartHeader/DataChecksum : Binary = 0x5A05,

        // --- Vector data ---
        Segment/Cart/VectorData : Master = 0x5B00,
        Segment/Cart/VectorData/Quantization : Master = 0x5B01,
        Segment/Cart/VectorData/Quantization/Type : Utf8 = 0x5B02,
        Segment/Cart/VectorData/Quantization/Dimensions : UnsignedInt = 0x5B03,
        Segment/Cart/VectorData/Quantization/SubVectors : UnsignedInt = 0x5B04,
        Segment/Cart/VectorData/Quantization/BitsPerCodebook : Binary = 0x5B05,
        Segment/Cart/VectorData/Quantization/Codebooks : Binary = 0x5B06,
        Segment/Cart/VectorData/CompressedVectors : Binary = 0x5B07,

        // --- Index data ---
        Segment/Cart/IndexData : Master = 0x5C00,
        Segment/Cart/IndexData/IVFIndex : Master = 0x5C01,
        Segment/Cart/IndexData/IVFIndex/Centroids : Binary = 0x5C02,
        Segment/Cart/IndexData/IVFIndex/InvertedLists : Binary = 0x5C03,

        // --- Metadata store ---
        Segment/Cart/MetadataStore : Master = 0x5D00,
        Segment/Cart/MetadataStore/Metadata : Master = 0x5D01,
        Segment/Cart/MetadataStore/Metadata/ItemOffset : UnsignedInt = 0x5D02,
        Segment/Cart/MetadataStore/Metadata/Field : Master = 0x5D03,
        Segment/Cart/MetadataStore/Metadata/FieldName : Utf8 = 0x5D04,
        Segment/Cart/MetadataStore/Metadata/FieldValue: Utf8 = 0x5D05,
        Segment/Cart/MetadataStore/Metadata/FieldValueInt : UnsignedInt = 0x5D06,

        // --- Tombstone log ---
        Segment/Cart/DeletedVectorOffsets : Binary = 0x5E00,
    }
}
