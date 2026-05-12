use std::{
    cell::{Ref, RefCell},
    collections::BTreeMap,
    sync::Arc,
    time::Duration,
};

use enum_iterator::Sequence;
use vulkano::{
    ValidationError,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::Device,
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    sync::PipelineStage,
};

/// Profiling system to measure the execution time of different GPU operations
pub struct Profiler {
    query_pool: Arc<QueryPool>,
    records: RefCell<ProfilerRecords>,
}

impl Profiler {
    pub fn new(device: Arc<Device>) -> Self {
        let records = ProfilerRecords::new(device.physical_device().properties().timestamp_period);
        let query_pool = QueryPool::new(
            device,
            QueryPoolCreateInfo {
                query_count: Profiler::query_count(),
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap();
        Self {
            query_pool,
            records: RefCell::new(records),
        }
    }

    fn query_count() -> u32 {
        Self::QUERY_COUNT
    }

    /// Write the timestamp to a profiler stage
    pub fn write(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        stage: ProfilerStage,
    ) -> Result<(), Box<ValidationError>> {
        unsafe {
            command_buffer.write_timestamp(
                self.query_pool.clone(),
                stage.query_id(),
                stage.pipeline_stage(),
            )
        }?;
        self.records.borrow_mut().had_writes = true;
        Ok(())
    }

    pub fn read_results(&self) {
        if self.records.borrow().had_writes {
            {
                let results = &mut self.records.borrow_mut().timestamp_buffer;
                self.query_pool
                    .get_results(
                        0..Self::query_count(),
                        results.as_mut_slice(),
                        QueryResultFlags::empty(),
                    )
                    .unwrap();
            }
            self.records.borrow_mut().update_times();
        }
        unsafe { self.query_pool.reset(0..Self::query_count()) }.unwrap();
    }

    pub fn records(&self) -> Ref<ProfilerRecords> {
        self.records.borrow()
    }

    // These consts below need to assign an integer to every variant in ProfilerStage

    const PRE_VISBUFFER_RASTER: u32 = 0;
    const POST_VISBUFFER_RASTER: u32 = 1;
    const PRE_VISBUFFER_PROCESS: u32 = 2;
    const POST_VISBUFFER_PROCESS: u32 = 3;
    const POST_VISBUFFER_SHADE: u32 = 4;
    const POST_TEXEL_COUNT: u32 = 5;
    const POST_EMPTY_CULL: u32 = 6;
    const POST_CULL: u32 = 7;
    const POST_PREFIX_SUM: u32 = 8;
    const POST_TEXEL_BIN: u32 = 9;

    const QUERY_COUNT: u32 = 10;
}

/// Enumeration of the stages where a timestamp can be written
pub enum ProfilerStage {
    PreVisbufferRaster,
    PostVisbufferRaster,
    PreVisbufferProcess,
    PostTexelCount,
    PostEmptyCull,
    PostCull,
    PostPrefixSum,
    PostTexelBin,
    PostVisbufferProcess,
    PostVisbufferShade,
}

impl ProfilerStage {
    /// Maps each variant of ProfilerStage onto its query id integer
    fn query_id(&self) -> u32 {
        match self {
            ProfilerStage::PreVisbufferRaster => Profiler::PRE_VISBUFFER_RASTER,
            ProfilerStage::PostVisbufferRaster => Profiler::POST_VISBUFFER_RASTER,
            ProfilerStage::PreVisbufferProcess => Profiler::PRE_VISBUFFER_PROCESS,
            ProfilerStage::PostVisbufferProcess => Profiler::POST_VISBUFFER_PROCESS,
            ProfilerStage::PostVisbufferShade => Profiler::POST_VISBUFFER_SHADE,
            ProfilerStage::PostTexelCount => Profiler::POST_TEXEL_COUNT,
            ProfilerStage::PostEmptyCull => Profiler::POST_EMPTY_CULL,
            ProfilerStage::PostCull => Profiler::POST_CULL,
            ProfilerStage::PostPrefixSum => Profiler::POST_PREFIX_SUM,
            ProfilerStage::PostTexelBin => Profiler::POST_TEXEL_BIN,
        }
    }

    /// Maps which pipeline stage a ProfilerStage should be recorded on
    fn pipeline_stage(&self) -> PipelineStage {
        match self {
            ProfilerStage::PreVisbufferRaster => PipelineStage::TopOfPipe,
            ProfilerStage::PostVisbufferRaster => PipelineStage::BottomOfPipe,
            _ => PipelineStage::ComputeShader,
        }
    }
}

/// Categories of time measurements
/// 
/// These essentially are the steps for which we can measure execution times
#[derive(Copy, Clone, Ord, Eq, PartialEq, PartialOrd, Sequence, Debug)]
pub enum ProfilerCategory {
    VisbufferRasterization,
    VisbufferProcess,
    VisbufferShade,
    TexelCount,
    EmptyCull,
    Cull,
    PrefixSum,
    TexelBin,
}

pub struct ProfilerRecords {
    last_durations: BTreeMap<ProfilerCategory, Duration>,
    timestamp_period: f32,
    results_available: bool,
    had_writes: bool,
    timestamp_buffer: Vec<u32>,
}

impl ProfilerRecords {
    fn new(timestamp_period: f32) -> Self {
        let mut timestamp_buffer = vec![];
        timestamp_buffer.resize(Profiler::query_count() as usize, 0);
        Self {
            last_durations: BTreeMap::new(),
            timestamp_period,
            results_available: false,
            had_writes: false,
            timestamp_buffer,
        }
    }

    /// This function is responsible for mapping ProfilerStages onto ProfilerCategories
    fn update_times(&mut self) {
        self.update_time(
            ProfilerCategory::VisbufferRasterization,
            ProfilerStage::PreVisbufferRaster,
            ProfilerStage::PostVisbufferRaster,
        );
        self.update_time(
            ProfilerCategory::VisbufferProcess,
            ProfilerStage::PreVisbufferProcess,
            ProfilerStage::PostVisbufferProcess,
        );
        self.update_time(
            ProfilerCategory::VisbufferShade,
            ProfilerStage::PostVisbufferProcess,
            ProfilerStage::PostVisbufferShade,
        );
        self.update_time(
            ProfilerCategory::TexelCount,
            ProfilerStage::PreVisbufferProcess,
            ProfilerStage::PostTexelCount,
        );
        self.update_time(
            ProfilerCategory::EmptyCull,
            ProfilerStage::PostTexelCount,
            ProfilerStage::PostEmptyCull,
        );
        self.update_time(
            ProfilerCategory::Cull,
            ProfilerStage::PostTexelCount,
            ProfilerStage::PostCull,
        );
        self.update_time(
            ProfilerCategory::PrefixSum,
            ProfilerStage::PostCull,
            ProfilerStage::PostPrefixSum,
        );
        self.update_time(
            ProfilerCategory::TexelBin,
            ProfilerStage::PostPrefixSum,
            ProfilerStage::PostTexelBin,
        );
        self.results_available = true;
    }

    fn update_time(
        &mut self,
        category: ProfilerCategory,
        start: ProfilerStage,
        end: ProfilerStage,
    ) {
        let duration = self.duration_between(start, end);
        if duration.is_some() {
            self.last_durations.insert(category, duration.unwrap());
        }
    }

    fn duration_between(&self, start: ProfilerStage, end: ProfilerStage) -> Option<Duration> {
        let start_time = self.timestamp_buffer[start.query_id() as usize];
        let end_time = self.timestamp_buffer[end.query_id() as usize];
        if start_time > end_time {
            None
        } else {
            let difference = end_time / 1000 - start_time / 1000;
            Some(Duration::from_secs_f32(
                self.timestamp_period * difference as f32 / 1000000f32,
            ))
        }
    }

    pub fn last_durations(&self) -> Option<&BTreeMap<ProfilerCategory, Duration>> {
        if self.results_available {
            Some(&self.last_durations)
        } else {
            None
        }
    }
}
