use enum_iterator::Sequence;
use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Device;
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::PipelineStage;
use vulkano::{ValidationError, VulkanError};

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
        2
    }

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
                        QueryResultFlags::WAIT,
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

    const PRE_VISBUFFER: u32 = 0;
    const POST_VISBUFFER: u32 = 1;
}

pub enum ProfilerStage {
    PreVisbuffer,
    PostVisbuffer,
}

impl ProfilerStage {
    fn query_id(&self) -> u32 {
        match self {
            ProfilerStage::PreVisbuffer => Profiler::PRE_VISBUFFER,
            ProfilerStage::PostVisbuffer => Profiler::POST_VISBUFFER,
        }
    }

    fn pipeline_stage(&self) -> PipelineStage {
        match self {
            ProfilerStage::PreVisbuffer => PipelineStage::TopOfPipe,
            ProfilerStage::PostVisbuffer => PipelineStage::BottomOfPipe,
        }
    }
}

#[derive(Copy, Clone, Ord, Eq, PartialEq, PartialOrd, Sequence, Debug)]
pub enum ProfilerCategory {
    VisbufferRasterization,
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

    fn update_times(&mut self) {
        self.update_time(
            ProfilerCategory::VisbufferRasterization,
            ProfilerStage::PreVisbuffer,
            ProfilerStage::PostVisbuffer,
        );
        self.results_available = true;
    }

    fn update_time(
        &mut self,
        category: ProfilerCategory,
        start: ProfilerStage,
        end: ProfilerStage,
    ) {
        let duration =
            self.duration_between(start, end);
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
