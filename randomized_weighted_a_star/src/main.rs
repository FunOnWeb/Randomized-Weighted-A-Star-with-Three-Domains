// Copyright 2026 Evan Munoz. All rights reserved.
// Usage of this code is bounded by the MIT License.

extern crate cpu_time;
extern crate rand;
use std::any::Any;
use std::cmp::Eq;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use cpu_time::ProcessTime;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::env;
use std::fs;

fn randomized_weighted_a_star<S: Eq + ManualHash + Hash + Clone, I: Any, N: Searchable<S, I> + Clone>(starting_node: N, is_goal: fn(&N) -> bool, heuristic_param: fn(&S, &I) -> u32, mut heuristic_weights: Vec<f64>, map_info: &I) -> (Option<usize>, Option<u32>, Vec<N>, NodeInfo, f64, Option<<StdRng as SeedableRng>::Seed>) {
    let start_of_process = ProcessTime::now();
    let mut node_info = NodeInfo {
        nodes_generated: 0,
        nodes_expanded: 0,
        duplicates_detected: 0,
        costly_nodes_not_generated: 0,
        costly_nodes_not_expanded: 0,
        nodes_generated_progress: Vec::new(),
        time_elapsed_progress: Vec::new(),
    };
    let mut node_history: Vec<N> = Vec::from([starting_node.clone(), starting_node]);
    let mut next_in_chain_history: Vec<usize> = Vec::from([0, 0]);
    let mut frontiers_index_tracker: Vec<Option<Vec<usize>>> = Vec::from([None, Some(Vec::new())]);
    let mut frontiers: Vec<IndexTrackingMinHeap<S, I, N>> = Vec::new();
    if !heuristic_weights.contains(&1.0) {
        heuristic_weights.push(1.0);
    }
    heuristic_weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    for heuristic_weight_index in 0..heuristic_weights.len() {
        frontiers_index_tracker[1].as_mut().unwrap().push(0);
        frontiers.push(IndexTrackingMinHeap::<S, I, N>::from(1, heuristic_weight_index, heuristic_weights[heuristic_weight_index]));
    }
    let mut closed_set = HashClosedSet::<S, I, N>::from(1, &node_history);
    let mut current_solution: Option<usize> = None;
    let mut seed: <StdRng as SeedableRng>::Seed = Default::default();
    rand::rng().fill(&mut seed);
    let mut rng = StdRng::from_seed(seed);
    while frontiers[0].len() > 0 && start_of_process.elapsed().as_secs_f64() < 180.0 {
        let random_weight_index = rng.random_range(0..heuristic_weights.len());
        let node_to_expand = frontiers[random_weight_index].peek().unwrap();
        for heuristic_weight_index in 0..heuristic_weights.len() {
            frontiers[heuristic_weight_index].delete(frontiers_index_tracker[node_to_expand].as_ref().unwrap()[heuristic_weight_index], &node_history, &mut frontiers_index_tracker);
        }
        frontiers_index_tracker[node_to_expand] = None;
        if current_solution.is_none() || node_history[node_to_expand].admissible_f_val() < node_history[current_solution.unwrap()].get_path_cost() {
            let mut children: Vec<N> = node_history[node_to_expand].expand(&node_history, &next_in_chain_history, &closed_set, node_to_expand, heuristic_param, map_info, &mut node_info, &current_solution);
            loop {
                let child = children.pop();
                match child {
                    Some(node) => {
                        node_history.push(node);
                        next_in_chain_history.push(0);
                        let index_of_new_node = node_history.len() - 1;
                        closed_set.insert(index_of_new_node, &node_history, &mut next_in_chain_history);
                        if is_goal(&node_history[index_of_new_node]) {
                            frontiers_index_tracker.push(None);
                            if current_solution.is_none() || node_history[index_of_new_node].get_path_cost() < node_history[current_solution.unwrap()].get_path_cost() {
                                current_solution = Some(index_of_new_node);
                                let current_solution_path_cost = node_history[current_solution.unwrap()].get_path_cost();
                                node_info.nodes_generated_progress.push((node_info.nodes_generated, current_solution_path_cost));
                                node_info.time_elapsed_progress.push((start_of_process.elapsed().as_secs_f64(), current_solution_path_cost));
                            }
                        } else {
                            frontiers_index_tracker.push(Some(Vec::new()));
                            for heuristic_weight_index in 0..heuristic_weights.len() {
                                frontiers_index_tracker[index_of_new_node].as_mut().unwrap().push(frontiers[heuristic_weight_index].len());
                                frontiers[heuristic_weight_index].push(index_of_new_node, &node_history, &mut frontiers_index_tracker);
                            }
                        }
                    }
                    None => break,
                }
            }
            node_info.nodes_expanded += 1;
        } else {
            node_info.costly_nodes_not_expanded += 1;
        }
    }
    let mut error_bound: Option<u32> = None;
    if let Some(solution) = current_solution {
        error_bound = Some(0);
        if frontiers[0].len() != 0 {
            error_bound = Some(node_history[solution].admissible_f_val() - node_history[frontiers[0].peek().unwrap()].admissible_f_val());
        }
    }
    (current_solution, error_bound, node_history, node_info, start_of_process.elapsed().as_secs_f64(), Some(seed))
}

fn anytime_weighted_a_star<S: Eq + ManualHash + Hash + Clone, I: Any, N: Searchable<S, I> + Clone>(starting_node: N, is_goal: fn(&N) -> bool, heuristic_param: fn(&S, &I) -> u32, heuristic_weight: f64, map_info: &I) -> (Option<usize>, Option<u32>, Vec<N>, NodeInfo, f64, Option<<StdRng as SeedableRng>::Seed>) {
    let start_of_process = ProcessTime::now();
    let mut node_info = NodeInfo {
        nodes_generated: 0,
        nodes_expanded: 0,
        duplicates_detected: 0,
        costly_nodes_not_generated: 0,
        costly_nodes_not_expanded: 0,
        nodes_generated_progress: Vec::new(),
        time_elapsed_progress: Vec::new(),
    };
    let mut node_history: Vec<N> = Vec::from([starting_node.clone(), starting_node]);
    let mut next_in_chain_history: Vec<usize> = Vec::from([0, 0]);
    let mut frontier = MinNodeHeap::from(1, heuristic_weight);
    let mut closed_set = HashClosedSet::<S, I, N>::from(1, &node_history);
    let mut current_solution: Option<usize> = None;
    while frontier.len() > 0 && start_of_process.elapsed().as_secs_f64() < 180.0 {
        let node_to_expand = frontier.pop(&node_history).unwrap();
        if current_solution.is_none() || node_history[node_to_expand].admissible_f_val() < node_history[current_solution.unwrap()].get_path_cost() {
            let mut children: Vec<N> = node_history[node_to_expand].expand(&node_history, &next_in_chain_history, &closed_set, node_to_expand, heuristic_param, map_info, &mut node_info, &current_solution);
            loop {
                let child = children.pop();
                match child {
                    Some(node) => {
                        node_history.push(node);
                        next_in_chain_history.push(0);
                        let index_of_new_node = node_history.len() - 1;
                        closed_set.insert(index_of_new_node, &node_history, &mut next_in_chain_history);
                        if is_goal(&node_history[index_of_new_node]) {
                            if current_solution.is_none() || node_history[index_of_new_node].get_path_cost() < node_history[current_solution.unwrap()].get_path_cost() {
                                current_solution = Some(index_of_new_node);
                                let current_solution_path_cost = node_history[current_solution.unwrap()].get_path_cost();
                                node_info.nodes_generated_progress.push((node_info.nodes_generated, current_solution_path_cost));
                                node_info.time_elapsed_progress.push((start_of_process.elapsed().as_secs_f64(), current_solution_path_cost));
                            }
                        } else {
                            frontier.push(index_of_new_node, &node_history);
                        }
                    }
                    None => break,
                }
            }
            node_info.nodes_expanded += 1;
        } else {
            node_info.costly_nodes_not_expanded += 1;
        }
    }
    let mut error_bound: Option<u32> = None;
    if let Some(solution) = current_solution {
        error_bound = Some(0);
        if frontier.len() != 0 {
            if heuristic_weight == 1.0 {
                error_bound = Some(node_history[solution].admissible_f_val() - node_history[frontier.peek().unwrap()].admissible_f_val());
            } else {
                let mut min_f_val = node_history[frontier.get(0).unwrap()].admissible_f_val();
                for index in 1..frontier.len() {
                    let f_val = node_history[frontier.get(index).unwrap()].admissible_f_val();
                    if f_val < min_f_val {
                        min_f_val = f_val;
                    }
                }
                error_bound = Some(node_history[solution].admissible_f_val() - min_f_val);
            }
        }
    }
    (current_solution, error_bound, node_history, node_info, start_of_process.elapsed().as_secs_f64(), None)
}

fn weighted_a_star<S: Eq + ManualHash + Hash + Clone, I: Any, N: Searchable<S, I> + Clone>(starting_node: N, is_goal: fn(&N) -> bool, heuristic_param: fn(&S, &I) -> u32, heuristic_weight: f64, map_info: &I) -> (Option<usize>, Option<u32>, Vec<N>, NodeInfo, f64, Option<<StdRng as SeedableRng>::Seed>) {
    let start_of_process = ProcessTime::now();
    let mut node_info = NodeInfo {
        nodes_generated: 0,
        nodes_expanded: 0,
        duplicates_detected: 0,
        costly_nodes_not_generated: 0,
        costly_nodes_not_expanded: 0,
        nodes_generated_progress: Vec::new(),
        time_elapsed_progress: Vec::new(),
    };
    let mut node_history: Vec<N> = Vec::from([starting_node.clone(), starting_node]);
    let mut next_in_chain_history: Vec<usize> = Vec::from([0, 0]);
    let mut frontier = MinNodeHeap::from(1, heuristic_weight);
    let mut closed_set = HashClosedSet::<S, I, N>::from(1, &node_history);
    while frontier.len() > 0 {
        let node_to_expand = frontier.pop(&node_history).unwrap();
        let mut children: Vec<N> = node_history[node_to_expand].expand(&node_history, &next_in_chain_history, &closed_set, node_to_expand, heuristic_param, map_info, &mut node_info, &None);
        loop {
            let child = children.pop();
            match child {
                Some(node) => {
                    node_history.push(node);
                    next_in_chain_history.push(0);
                    let index_of_new_node = node_history.len() - 1;
                    closed_set.insert(index_of_new_node, &node_history, &mut next_in_chain_history);
                    if is_goal(&node_history[index_of_new_node]) {
                        let solution_path_cost = node_history[index_of_new_node].get_path_cost();
                        node_info.nodes_generated_progress.push((node_info.nodes_generated, solution_path_cost));
                        node_info.time_elapsed_progress.push((start_of_process.elapsed().as_secs_f64(), solution_path_cost));
                        let error_bound: Option<u32>;
                        if heuristic_weight == 1.0 {
                            error_bound = Some(0);
                        } else {
                            let mut min_f_val = node_history[frontier.get(0).unwrap()].admissible_f_val();
                            for index in 1..frontier.len() {
                                let f_val = node_history[frontier.get(index).unwrap()].admissible_f_val();
                                if f_val < min_f_val {
                                    min_f_val = f_val;
                                }
                            }
                            error_bound = Some(node_history[index_of_new_node].admissible_f_val() - min_f_val);
                        }
                        return (Some(index_of_new_node), error_bound, node_history, node_info, start_of_process.elapsed().as_secs_f64(), None);
                    } else {
                        frontier.push(index_of_new_node, &node_history);
                    }
                }
                None => break,
            }
        }
        node_info.nodes_expanded += 1;
    }
    (None, None, node_history, node_info, start_of_process.elapsed().as_secs_f64(), None)
}

pub trait Searchable<S: Eq + ManualHash + Hash, I: Any> {
    //fn create_starting_node(&S) -> Self;
    fn get_state_ref(&self) -> &S;
    fn get_path_cost(&self) -> u32;
    fn expand(&self, node_history: &[Self], next_in_chain_history: &[usize], closed_set: &HashClosedSet<S, I, Self>, parent_index: usize, heuristic_function: fn(&S, &I) -> u32, map_info: &I, node_info: &mut NodeInfo, current_solution: &Option<usize>) -> Vec<Self> where Self: Sized;
    fn admissible_f_val(&self) -> u32;
    fn inadmissible_f_val(&self, heuristic_weight: f64) -> f64;
}

pub trait ManualHash {
    fn manual_hash(&self) -> u64;
}

pub trait ActionPath {
    fn get_parent_index(&self) -> usize;
    fn get_action(&self) -> Action;
}

fn get_next_directions(wrapped_direction: &Option<Direction>) -> Vec<Direction> {
    match wrapped_direction {
        Some(direction) => match direction {
            Direction::East => Vec::from([Direction::East, Direction::North, Direction::South]),
            Direction::North => Vec::from([Direction::East, Direction::North, Direction::West]),
            Direction::West => Vec::from([Direction::North, Direction::West, Direction::South]),
            Direction::South => Vec::from([Direction::East, Direction::West, Direction::South]),
        },
        None => Vec::from([Direction::East, Direction::North, Direction::West, Direction::South]),
    }
}

#[derive(Clone)]
struct VacuumNode {
    parent_index: usize,
    state: VacuumState,
    path_cost: u32,
    heuristic: u32,
    action: Action,
}

impl VacuumNode {
    fn create_new_child(&self, new_state: VacuumState, list_of_children: &mut Vec<VacuumNode>, node_history: &[VacuumNode], next_in_chain_history: &[usize], closed_set: &HashClosedSet<VacuumState, MapInfo, VacuumNode>, parent_index: usize, new_path_cost: u32, new_heuristic: u32, node_info: &mut NodeInfo, action: Action, current_solution: &Option<usize>) {
        if current_solution.is_none() || new_path_cost + new_heuristic < node_history[current_solution.unwrap()].path_cost {
            let node_with_new_state = closed_set.get_node(&new_state, node_history, next_in_chain_history);
            let should_create_child = match node_with_new_state {
                Some(node) => new_path_cost < node_history[node].path_cost,
                None => true,
            };
            if should_create_child {
                list_of_children.push(VacuumNode {
                    parent_index: parent_index,
                    state: new_state,
                    path_cost: new_path_cost,
                    heuristic: new_heuristic,
                    action: action,
                });
                node_info.nodes_generated += 1;
            } else {
                node_info.duplicates_detected += 1;
            }
        } else {
            node_info.costly_nodes_not_generated += 1;
        }
    }
}

impl Searchable<VacuumState, MapInfo> for VacuumNode {
    fn get_state_ref(&self) -> &VacuumState {
        &self.state
    }

    fn get_path_cost(&self) -> u32 {
        self.path_cost
    }

    fn expand(&self, node_history: &[Self], next_in_chain_history: &[usize], closed_set: &HashClosedSet<VacuumState, MapInfo, VacuumNode>, parent_index: usize, heuristic_function: fn(&VacuumState, &MapInfo) -> u32, map_info: &MapInfo, node_info: &mut NodeInfo, current_solution: &Option<usize>) -> Vec<Self> {
        let mut children = Vec::new();
        let new_path_cost = self.path_cost + 1;
        if self.state.dirt_pos.contains(&self.state.vac_pos) {
            let mut new_state = self.state.clone();
            new_state.dirt_pos.retain(|pos| pos != &self.state.vac_pos);
            let new_heuristic = heuristic_function(&new_state, map_info);
            self.create_new_child(new_state, &mut children, node_history, next_in_chain_history, closed_set, parent_index, new_path_cost, new_heuristic, node_info, Action::Vacuum, current_solution);
        } else {
            let directions = match self.action {
                Action::Vacuum => get_next_directions(&None),
                Action::Move(dir) => get_next_directions(&Some(dir)),
            };
            for direction in directions.iter() {
                if self.state.vac_pos.move_stays_within_bounds(direction, map_info) && !self.state.vac_pos.move_runs_into_blockage(direction, map_info) {
                    let new_state = VacuumState {
                        vac_pos: self.state.vac_pos.get_new_coordinates(direction),
                        dirt_pos: self.state.dirt_pos.clone(),
                    };
                    self.create_new_child(new_state, &mut children, node_history, next_in_chain_history, closed_set, parent_index, new_path_cost, self.heuristic, node_info, Action::Move(direction.clone()), current_solution);
                }
            }
        }
        children
    }

    fn admissible_f_val(&self) -> u32 {
        self.path_cost + self.heuristic
    }

    fn inadmissible_f_val(&self, heuristic_weight: f64) -> f64 {
        (self.path_cost as f64) + (heuristic_weight * (self.heuristic as f64))
    }
}

impl ActionPath for VacuumNode {
    fn get_parent_index(&self) -> usize {
        self.parent_index
    }
    
    fn get_action(&self) -> Action {
        self.action.clone()
    }
}

#[derive(Clone, Hash)]
struct VacuumState {
    vac_pos: Coordinates,
    dirt_pos: Vec<Coordinates>,
}

impl PartialEq for VacuumState {
    fn eq(&self, other: &Self) -> bool {
        self.vac_pos == other.vac_pos && self.dirt_pos == other.dirt_pos
    }
}

impl Eq for VacuumState {}

impl ManualHash for VacuumState {
    fn manual_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Hash)]
struct Coordinates {
    x: u8,
    y: u8,
}

impl Coordinates {
    pub fn new(x_param: u8, y_param: u8) -> Coordinates {
        Coordinates {
            x: x_param,
            y: y_param,
        }
    }
    
    pub fn move_stays_within_bounds(&self, direction: &Direction, map_info: &MapInfo) -> bool {
        match direction {
            Direction::East => self.x + 1 < map_info.num_columns,
            Direction::North => self.y >= 1,
            Direction::West => self.x >= 1,
            Direction::South => self.y + 1 < map_info.num_rows,
        }
    }
    
    pub fn move_runs_into_blockage(&self, direction: &Direction, map_info: &MapInfo) -> bool {
        map_info.blockage_pos.contains(&self.get_new_coordinates(direction))
    }
    
    pub fn get_new_coordinates(&self, direction: &Direction) -> Coordinates {
        match direction {
            Direction::East => Coordinates::new(self.x + 1, self.y),
            Direction::North => Coordinates::new(self.x, self.y - 1),
            Direction::West => Coordinates::new(self.x - 1, self.y),
            Direction::South => Coordinates::new(self.x, self.y + 1),
        }
    }
}

impl PartialEq for Coordinates {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Eq for Coordinates {}

#[derive(Clone)]
pub enum Action {
    Vacuum,
    Move(Direction),
}

#[derive(Clone, Copy)]
pub enum Direction {
    East,
    North,
    West,
    South,
}

pub struct MapInfo {
    num_columns: u8,
    num_rows: u8,
    blockage_pos: HashSet<Coordinates>,
}

#[derive(Clone)]
struct SlidingTileNode {
    parent_index: usize,
    state: SlidingTileState,
    path_cost: u32,
    heuristic: u32,
    action: Option<Direction>,
}

impl Searchable<SlidingTileState, TileHeuristicInfo> for SlidingTileNode {
    fn get_state_ref(&self) -> &SlidingTileState {
        &self.state
    }

    fn get_path_cost(&self) -> u32 {
        self.path_cost
    }

    fn expand(&self, node_history: &[Self], next_in_chain_history: &[usize], closed_set: &HashClosedSet<SlidingTileState, TileHeuristicInfo, SlidingTileNode>, parent_index: usize, heuristic_function: fn(&SlidingTileState, &TileHeuristicInfo) -> u32, _map_info: &TileHeuristicInfo, node_info: &mut NodeInfo, current_solution: &Option<usize>) -> Vec<Self> {
        let mut children = Vec::new();
        let new_path_cost = self.path_cost + 1;
        let directions = get_next_directions(&self.action);
        for direction in directions.iter() {
            if self.state.move_stays_within_bounds(direction) {
                let heuristic_info = TileHeuristicInfo {
                    parent_heuristic: self.heuristic,
                    index_of_moved_tile: self.state.index_of_blank_space,
                    direction_of_movement: direction.clone(),
                };
                let new_state = self.state.move_tile(direction);
                let heuristic = heuristic_function(&new_state, &heuristic_info);
                if current_solution.is_none() || new_path_cost + heuristic < node_history[current_solution.unwrap()].path_cost {
                    let node_with_new_state = closed_set.get_node(&new_state, node_history, next_in_chain_history);
                    let should_create_child = match node_with_new_state {
                        Some(node) => new_path_cost < node_history[node].path_cost,
                        None => true,
                    };
                    if should_create_child {
                        children.push(SlidingTileNode {
                            parent_index: parent_index,
                            state: new_state,
                            path_cost: new_path_cost,
                            heuristic: heuristic,
                            action: Some(direction.clone()),
                        });
                        node_info.nodes_generated += 1;
                    } else {
                        node_info.duplicates_detected += 1;
                    }
                } else {
                    node_info.costly_nodes_not_generated += 1;
                }
            }
        }
        children
    }

    fn admissible_f_val(&self) -> u32 {
        self.path_cost + self.heuristic
    }

    fn inadmissible_f_val(&self, heuristic_weight: f64) -> f64 {
        (self.path_cost as f64) + (heuristic_weight * (self.heuristic as f64))
    }
}

impl ActionPath for SlidingTileNode {
    fn get_parent_index(&self) -> usize {
        self.parent_index
    }
    
    fn get_action(&self) -> Action {
        Action::Move(self.action.unwrap().clone())
    }
}

#[derive(Clone)]
struct InverseSlidingNode {
    parent_index: usize,
    state: SlidingTileState,
    path_cost: u32,
    heuristic: u32,
    action: Option<Direction>,
}

impl Searchable<SlidingTileState, TileHeuristicInfo> for InverseSlidingNode {
    fn get_state_ref(&self) -> &SlidingTileState {
        &self.state
    }

    fn get_path_cost(&self) -> u32 {
        self.path_cost
    }

    fn expand(&self, node_history: &[Self], next_in_chain_history: &[usize], closed_set: &HashClosedSet<SlidingTileState, TileHeuristicInfo, InverseSlidingNode>, parent_index: usize, heuristic_function: fn(&SlidingTileState, &TileHeuristicInfo) -> u32, _map_info: &TileHeuristicInfo, node_info: &mut NodeInfo, current_solution: &Option<usize>) -> Vec<Self> {
        let mut children = Vec::new();
        let directions = get_next_directions(&self.action);
        for direction in directions.iter() {
            if self.state.move_stays_within_bounds(direction) {
                let heuristic_info = TileHeuristicInfo {
                    parent_heuristic: self.heuristic,
                    index_of_moved_tile: self.state.index_of_blank_space,
                    direction_of_movement: direction.clone(),
                };
                let new_state = self.state.move_tile(direction);
                let new_path_cost = self.path_cost + (360360 / (new_state.board.get(heuristic_info.index_of_moved_tile) as u32));
                let heuristic = heuristic_function(&new_state, &heuristic_info);
                if current_solution.is_none() || new_path_cost + heuristic < node_history[current_solution.unwrap()].path_cost {
                    let node_with_new_state = closed_set.get_node(&new_state, node_history, next_in_chain_history);
                    let should_create_child = match node_with_new_state {
                        Some(node) => new_path_cost < node_history[node].path_cost,
                        None => true,
                    };
                    if should_create_child {
                        children.push(InverseSlidingNode {
                            parent_index: parent_index,
                            state: new_state,
                            path_cost: new_path_cost,
                            heuristic: heuristic,
                            action: Some(direction.clone()),
                        });
                        node_info.nodes_generated += 1;
                    } else {
                        node_info.duplicates_detected += 1;
                    }
                } else {
                    node_info.costly_nodes_not_generated += 1;
                }
            }
        }
        children
    }

    fn admissible_f_val(&self) -> u32 {
        self.path_cost + self.heuristic
    }

    fn inadmissible_f_val(&self, heuristic_weight: f64) -> f64 {
        (self.path_cost as f64) + (heuristic_weight * (self.heuristic as f64))
    }
}

impl ActionPath for InverseSlidingNode {
    fn get_parent_index(&self) -> usize {
        self.parent_index
    }
    
    fn get_action(&self) -> Action {
        Action::Move(self.action.unwrap().clone())
    }
}

#[derive(Clone, Copy, Hash)]
struct Board(u64);

impl Board {
    fn get(&self, index: u8) -> u8 {
        ((self.0 >> (index * 4)) & 0b1111) as u8
    }

    fn set(&mut self, index: u8, item: u8) {
        let cleared_bits_self = self.0 & !(0b1111 << (index * 4));
        self.0 = cleared_bits_self | ((item as u64) << (index * 4));
    }
}

#[derive(Clone, Hash)]
struct SlidingTileState {
    index_of_blank_space: u8,
    board: Board,
}

impl SlidingTileState {
    fn get_row_of_blank_space(&self) -> u8 {
        self.index_of_blank_space / 4
    }

    fn get_column_of_blank_space(&self) -> u8 {
        self.index_of_blank_space % 4
    }

    pub fn move_stays_within_bounds(&self, direction: &Direction) -> bool {
        match direction {
            Direction::East => self.get_column_of_blank_space() + 1 < 4,
            Direction::North => self.get_row_of_blank_space() >= 1,
            Direction::West => self.get_column_of_blank_space() >= 1,
            Direction::South => self.get_row_of_blank_space() + 1 < 4,
        }
    }
    
    pub fn move_tile(&self, direction: &Direction) -> SlidingTileState {
        let mut new_row_of_blank_space = self.get_row_of_blank_space();
        let mut new_column_of_blank_space = self.get_column_of_blank_space();
        match direction {
            Direction::East => new_column_of_blank_space += 1,
            Direction::North => new_row_of_blank_space -= 1,
            Direction::West => new_column_of_blank_space -= 1,
            Direction::South => new_row_of_blank_space += 1,
        }
        let mut new_board = self.board;
        let new_index_of_blank_space = (new_row_of_blank_space * 4) + new_column_of_blank_space;
        new_board.set(self.index_of_blank_space, new_board.get(new_index_of_blank_space));
        new_board.set(new_index_of_blank_space, 0);
        SlidingTileState {
            index_of_blank_space: new_index_of_blank_space,
            board: new_board,
        }
    }
}

impl PartialEq for SlidingTileState {
    fn eq(&self, other: &Self) -> bool {
        self.board.0 == other.board.0
    }
}

impl Eq for SlidingTileState {}

impl ManualHash for SlidingTileState {
    fn manual_hash(&self) -> u64 {
        self.board.0
    }
}

struct TileHeuristicInfo {
    parent_heuristic: u32,
    index_of_moved_tile: u8,
    direction_of_movement: Direction,
}

pub struct NodeInfo {
    nodes_generated: u32,
    nodes_expanded: u32,
    duplicates_detected: u32,
    costly_nodes_not_generated: u32,
    costly_nodes_not_expanded: u32,
    nodes_generated_progress: Vec<(u32, u32)>,
    time_elapsed_progress: Vec<(f64, u32)>
}

struct MinNodeHeap<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> {
    node_list: Vec<usize>,
    heuristic_weight: f64,
    _marker: PhantomData<(S, I, N)>,
}

impl<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> MinNodeHeap<S, I, N> {
    fn from(starting_node_index: usize, heuristic_weight: f64) -> MinNodeHeap<S, I, N> {
        MinNodeHeap::<S, I, N> {
            node_list: Vec::from([starting_node_index]),
            heuristic_weight: heuristic_weight,
            _marker: PhantomData,
        }
    }

    fn peek(&self) -> Option<usize> {
        if self.len() == 0 {
            None
        } else {
            Some(self.node_list[0])
        }
    }

    fn get(&self, index: usize) -> Option<usize> {
        if index >= self.len() {
            None
        } else {
            Some(self.node_list[index])
        }
    }

    fn len(&self) -> usize {
        self.node_list.len()
    }

    fn upheap(&mut self, index_to_move: usize, node_history: &[N]) {
        if index_to_move >= self.len() {
            panic!("Invalid index.");
        } else if self.len() != 1 {
            let value_to_move = self.node_list[index_to_move];
            let value_to_move_g_val = node_history[self.node_list[index_to_move]].get_path_cost();
            let value_to_move_f_val = node_history[self.node_list[index_to_move]].inadmissible_f_val(self.heuristic_weight);
            let mut current_index = index_to_move;
            let mut parent_index = (current_index - 1) / 2;
            while value_to_move_f_val < node_history[self.node_list[parent_index]].inadmissible_f_val(self.heuristic_weight) || (value_to_move_f_val == node_history[self.node_list[parent_index]].inadmissible_f_val(self.heuristic_weight) && value_to_move_g_val > node_history[self.node_list[parent_index]].get_path_cost()) {
                self.node_list[current_index] = self.node_list[parent_index];
                current_index = parent_index;
                if current_index == 0 {
                    break;
                } else {
                    parent_index = (current_index - 1) / 2;
                }
            }
            self.node_list[current_index] = value_to_move;
        }
    }

    fn push(&mut self, new_node_index: usize, node_history: &[N]) {
        self.node_list.push(new_node_index);
        self.upheap(self.len() - 1, node_history);
    }

    fn min_inadmissible_f_val(&self, index_one: usize, index_two: usize, node_history: &[N]) -> Option<usize> {
        if index_one >= self.len() {
            None
        } else if index_two >= self.len() || (node_history[self.node_list[index_one]].inadmissible_f_val(self.heuristic_weight) < node_history[self.node_list[index_two]].inadmissible_f_val(self.heuristic_weight) || (node_history[self.node_list[index_one]].inadmissible_f_val(self.heuristic_weight) == node_history[self.node_list[index_two]].inadmissible_f_val(self.heuristic_weight) && node_history[self.node_list[index_one]].get_path_cost() > node_history[self.node_list[index_two]].get_path_cost())) {
            Some(index_one)
        } else {
            Some(index_two)
        }
    }

    fn downheap(&mut self, index_to_move: usize, node_history: &[N]) {
        if index_to_move >= self.len() {
            panic!("Invalid index.");
        } else if self.len() != 1 {
            let value_to_move = self.node_list[index_to_move];
            let value_to_move_g_val = node_history[self.node_list[index_to_move]].get_path_cost();
            let value_to_move_f_val = node_history[self.node_list[index_to_move]].inadmissible_f_val(self.heuristic_weight);
            let mut current_index = index_to_move;
            let mut child_index = self.min_inadmissible_f_val(2 * current_index + 1, 2 * current_index + 2, node_history);
            while child_index.is_some() && (value_to_move_f_val > node_history[self.node_list[child_index.unwrap()]].inadmissible_f_val(self.heuristic_weight) || (value_to_move_f_val == node_history[self.node_list[child_index.unwrap()]].inadmissible_f_val(self.heuristic_weight) && value_to_move_g_val < node_history[self.node_list[child_index.unwrap()]].get_path_cost())) {
                self.node_list[current_index] = self.node_list[child_index.unwrap()];
                current_index = child_index.unwrap();
                child_index = self.min_inadmissible_f_val(2 * current_index + 1, 2 * current_index + 2, node_history);
            }
            self.node_list[current_index] = value_to_move;
        }
    }

    fn pop(&mut self, node_history: &[N]) -> Option<usize> {
        if self.len() <= 1 {
            self.node_list.pop()
        } else {
            let len = self.len();
            self.node_list.swap(0, len - 1);
            let index_of_min_node = self.node_list.pop();
            self.downheap(0, node_history);
            index_of_min_node
        }
    }
}

struct IndexTrackingMinHeap<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> {
    node_list: Vec<usize>,
    heuristic_weight_index: usize,
    heuristic_weight: f64,
    _marker: PhantomData<(S, I, N)>,
}

impl<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> IndexTrackingMinHeap<S, I, N> {
    fn from(starting_node_index: usize, heuristic_weight_index: usize, heuristic_weight: f64) -> IndexTrackingMinHeap<S, I, N> {
        IndexTrackingMinHeap::<S, I, N> {
            node_list: Vec::from([starting_node_index]),
            heuristic_weight_index: heuristic_weight_index,
            heuristic_weight: heuristic_weight,
            _marker: PhantomData,
        }
    }

    fn peek(&self) -> Option<usize> {
        if self.len() == 0 {
            None
        } else {
            Some(self.node_list[0])
        }
    }

    fn get(&self, index: usize) -> Option<usize> {
        if index >= self.len() {
            None
        } else {
            Some(self.node_list[index])
        }
    }

    fn len(&self) -> usize {
        self.node_list.len()
    }

    fn swap(&mut self, index_one: usize, index_two: usize, frontiers_index_tracker: &mut [Option<Vec<usize>>]) {
        frontiers_index_tracker[self.node_list[index_one]].as_mut().unwrap()[self.heuristic_weight_index] = index_two;
        frontiers_index_tracker[self.node_list[index_two]].as_mut().unwrap()[self.heuristic_weight_index] = index_one;
        self.node_list.swap(index_one, index_two);
    }

    fn upheap(&mut self, index_to_move: usize, node_history: &[N], frontiers_index_tracker: &mut [Option<Vec<usize>>]) {
        if index_to_move >= self.len() {
            panic!("Invalid index.");
        } else if self.len() != 1 {
            let value_to_move = self.node_list[index_to_move];
            let value_to_move_g_val = node_history[self.node_list[index_to_move]].get_path_cost();
            let value_to_move_f_val = node_history[self.node_list[index_to_move]].inadmissible_f_val(self.heuristic_weight);
            let mut current_index = index_to_move;
            let mut parent_index = (current_index - 1) / 2;
            while value_to_move_f_val < node_history[self.node_list[parent_index]].inadmissible_f_val(self.heuristic_weight) || (value_to_move_f_val == node_history[self.node_list[parent_index]].inadmissible_f_val(self.heuristic_weight) && value_to_move_g_val > node_history[self.node_list[parent_index]].get_path_cost()) {
                frontiers_index_tracker[self.node_list[parent_index]].as_mut().unwrap()[self.heuristic_weight_index] = current_index;
                self.node_list[current_index] = self.node_list[parent_index];
                current_index = parent_index;
                if current_index == 0 {
                    break;
                } else {
                    parent_index = (current_index - 1) / 2;
                }
            }
            frontiers_index_tracker[value_to_move].as_mut().unwrap()[self.heuristic_weight_index] = current_index;
            self.node_list[current_index] = value_to_move;
        }
    }

    fn push(&mut self, new_node_index: usize, node_history: &[N], frontiers_index_tracker: &mut [Option<Vec<usize>>]) {
        self.node_list.push(new_node_index);
        self.upheap(self.len() - 1, node_history, frontiers_index_tracker);
    }

    fn min_inadmissible_f_val(&self, index_one: usize, index_two: usize, node_history: &[N]) -> Option<usize> {
        if index_one >= self.len() {
            None
        } else if index_two >= self.len() || (node_history[self.node_list[index_one]].inadmissible_f_val(self.heuristic_weight) < node_history[self.node_list[index_two]].inadmissible_f_val(self.heuristic_weight) || (node_history[self.node_list[index_one]].inadmissible_f_val(self.heuristic_weight) == node_history[self.node_list[index_two]].inadmissible_f_val(self.heuristic_weight) && node_history[self.node_list[index_one]].get_path_cost() > node_history[self.node_list[index_two]].get_path_cost())) {
            Some(index_one)
        } else {
            Some(index_two)
        }
    }

    fn downheap(&mut self, index_to_move: usize, node_history: &[N], frontiers_index_tracker: &mut [Option<Vec<usize>>]) {
        if index_to_move >= self.len() {
            panic!("Invalid index.");
        } else if self.len() != 1 {
            let value_to_move = self.node_list[index_to_move];
            let value_to_move_g_val = node_history[self.node_list[index_to_move]].get_path_cost();
            let value_to_move_f_val = node_history[self.node_list[index_to_move]].inadmissible_f_val(self.heuristic_weight);
            let mut current_index = index_to_move;
            let mut child_index = self.min_inadmissible_f_val(2 * current_index + 1, 2 * current_index + 2, node_history);
            while child_index.is_some() && (value_to_move_f_val > node_history[self.node_list[child_index.unwrap()]].inadmissible_f_val(self.heuristic_weight) || (value_to_move_f_val == node_history[self.node_list[child_index.unwrap()]].inadmissible_f_val(self.heuristic_weight) && value_to_move_g_val < node_history[self.node_list[child_index.unwrap()]].get_path_cost())) {
                frontiers_index_tracker[self.node_list[child_index.unwrap()]].as_mut().unwrap()[self.heuristic_weight_index] = current_index;
                self.node_list[current_index] = self.node_list[child_index.unwrap()];
                current_index = child_index.unwrap();
                child_index = self.min_inadmissible_f_val(2 * current_index + 1, 2 * current_index + 2, node_history);
            }
            frontiers_index_tracker[value_to_move].as_mut().unwrap()[self.heuristic_weight_index] = current_index;
            self.node_list[current_index] = value_to_move;
        }
    }

    fn delete(&mut self, index_to_delete: usize, node_history: &[N], frontiers_index_tracker: &mut [Option<Vec<usize>>]) {
        if index_to_delete >= self.len() {
            panic!("Invalid index.");
        } else if index_to_delete == 0 && self.len() == 1 {
            self.node_list.pop();
        } else {
            self.swap(index_to_delete, self.len() - 1, frontiers_index_tracker);
            self.node_list.pop();
            if index_to_delete == 0 {
                self.downheap(index_to_delete, node_history, frontiers_index_tracker);
            } else if index_to_delete != self.len() {
                let parent_index = (index_to_delete - 1) / 2;
                let child_index = self.min_inadmissible_f_val(2 * index_to_delete + 1, 2 * index_to_delete + 2, node_history);
                if node_history[self.node_list[index_to_delete]].inadmissible_f_val(self.heuristic_weight) < node_history[self.node_list[parent_index]].inadmissible_f_val(self.heuristic_weight) {
                    self.upheap(index_to_delete, node_history, frontiers_index_tracker);
                } else if child_index.is_some() && node_history[self.node_list[index_to_delete]].inadmissible_f_val(self.heuristic_weight) > node_history[self.node_list[child_index.unwrap()]].inadmissible_f_val(self.heuristic_weight) {
                    self.downheap(index_to_delete, node_history, frontiers_index_tracker);
                }
            }
        }
    }
}

pub struct HashClosedSet<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> {
    vector: Vec<usize>,
    size: u64,
    num_elements: u64,
    _marker: PhantomData<(S, I, N)>,
}

impl<S: Eq + ManualHash + Hash, I: Any, N: Searchable<S, I>> HashClosedSet<S, I, N> {  
    fn from(value: usize, node_history: &[N]) -> HashClosedSet<S, I, N> {
        let mut closed_set = HashClosedSet::<S, I, N> {
            vector: vec![0; 500000003],
            size: 500000003,
            num_elements: 0,
            _marker: PhantomData,
        };
        let index = (node_history[value].get_state_ref().manual_hash() % 500000003) as usize;
        closed_set.vector[index] = value;
        closed_set
    }
    
    fn get_node(&self, state: &S, node_history: &[N], next_in_chain_history: &[usize]) -> Option<usize> {
        let index = (state.manual_hash() % self.size) as usize;
        let mut current_value = self.vector[index];
        while current_value != 0 {
            if *node_history[current_value].get_state_ref() == *state {
                return Some(current_value);
            }
            current_value = next_in_chain_history[current_value];
        }
        None
    }

    fn insert(&mut self, value: usize, node_history: &[N], next_in_chain_history: &mut [usize]) {
        let index = (node_history[value].get_state_ref().manual_hash() % self.size) as usize;
        let mut current_value = self.vector[index];
        if current_value == 0 {
            self.vector[index] = value;
            self.num_elements += 1;
            if self.num_elements == self.size / 2 {
                self.increase_size(node_history, next_in_chain_history);
            }
        } else {
            let mut previous_value = current_value;
            loop {
                if current_value == 0 {
                    next_in_chain_history[previous_value] = value;
                    self.num_elements += 1;
                    if self.num_elements == self.size / 2 {
                        self.increase_size(node_history, next_in_chain_history);
                    }
                    break;
                } else if *node_history[current_value].get_state_ref() == *node_history[value].get_state_ref() {
                    if self.vector[index] == current_value {
                        self.vector[index] = value;
                    } else {
                        next_in_chain_history[previous_value] = value;
                    }
                    let value_after_current_value = next_in_chain_history[current_value];
                    next_in_chain_history[value] = value_after_current_value;
                    break;
                }
                previous_value = current_value;
                current_value = next_in_chain_history[current_value];
            }
        }
    }

    fn increase_size(&mut self, node_history: &[N], next_in_chain_history: &mut [usize]) {
        let new_size = self.size * 2;
        let mut new_vector = vec![0; new_size as usize];
        for vector_index in 0..(self.size as usize) {
            let mut current_value_to_insert = self.vector[vector_index];
            loop {
                if current_value_to_insert != 0 {
                    let new_index = (node_history[current_value_to_insert].get_state_ref().manual_hash() % new_size) as usize;
                    let mut current_value = new_vector[new_index];
                    if current_value == 0 {
                        new_vector[new_index] = current_value_to_insert;
                    } else {
                        let mut previous_value = current_value;
                        loop {
                            if current_value == 0 {
                                next_in_chain_history[previous_value] = current_value_to_insert;
                                break;
                            }
                            previous_value = current_value;
                            current_value = next_in_chain_history[current_value];
                        }
                    }
                    let previous_value_to_insert = current_value_to_insert;
                    current_value_to_insert = next_in_chain_history[current_value_to_insert];
                    next_in_chain_history[previous_value_to_insert] = 0;
                } else {
                    break;
                }
            }
        }
        self.size = new_size;
        self.vector = new_vector;
    }
}

fn is_goal_for_vacuum(vacuum_node: &VacuumNode) -> bool {
    vacuum_node.heuristic == 0
}

fn diamond_admissible_heuristic(vacuum_state: &VacuumState, map_info: &MapInfo) -> u32 {
    let mut min_dist_from_top_left: u32 = (map_info.num_columns as u32) + (map_info.num_rows as u32);
    let mut max_dist_from_top_left: u32 = 0;
    let mut min_dist_from_bottom_left: u32 = (map_info.num_columns as u32) + (map_info.num_rows as u32);
    let mut max_dist_from_bottom_left: u32 = 0;
    for dirt_pos in vacuum_state.dirt_pos.iter() {
        let sum = (dirt_pos.x as u32) + (dirt_pos.y as u32);
        let difference = ((dirt_pos.x as u32) + (map_info.num_rows as u32)) - (dirt_pos.y as u32);
        if sum < min_dist_from_top_left {
            min_dist_from_top_left = sum;
        }
        if sum > max_dist_from_top_left {
            max_dist_from_top_left = sum;
        }
        if difference < min_dist_from_bottom_left {
            min_dist_from_bottom_left = difference;
        }
        if difference > max_dist_from_bottom_left {
            max_dist_from_bottom_left = difference;
        }
    }
    if min_dist_from_top_left > max_dist_from_top_left || min_dist_from_bottom_left > max_dist_from_bottom_left {
        0
    } else {
        let max_distance: u32 = std::cmp::max(max_dist_from_top_left - min_dist_from_top_left,
                                              max_dist_from_bottom_left - min_dist_from_bottom_left);
        if max_distance == 0 {
            1
        } else {
            max_distance + (vacuum_state.dirt_pos.len() as u32)
        }
    }
}

fn create_starting_vacuum_state(num_columns: u8, num_rows: u8, map_details: &[&str]) -> (VacuumState, HashSet<Coordinates>) {
    let mut starting_state = VacuumState {
        vac_pos: Coordinates { x: 0, y: 0 },
        dirt_pos: Vec::<Coordinates>::new(),
    };
    let mut blockage_pos = HashSet::new();
    let mut row = 0;
    while row < num_rows {
        let mut column = 0;
        while column < num_columns {
            if map_details[row as usize].as_bytes()[column as usize] as char == '@' {
                starting_state.vac_pos.x = column;
                starting_state.vac_pos.y = row;
            } else if map_details[row as usize].as_bytes()[column as usize] as char == '*' {
                let new_dirt = Coordinates { x: column, y: row };
                starting_state.dirt_pos.push(new_dirt);
            } else if map_details[row as usize].as_bytes()[column as usize] as char == '#' {
                let new_blockage = Coordinates { x: column, y: row };
                blockage_pos.insert(new_blockage);
            }
            column += 1;
        }
        row += 1;
    }
    (starting_state, blockage_pos)
}

fn is_goal_for_tile(sliding_tile_node: &SlidingTileNode) -> bool {
    sliding_tile_node.heuristic == 0
}

fn initialize_manhattan_admissible_heuristic(sliding_tile_state: &SlidingTileState) -> u32 {
    let mut heuristic: u32 = 0;
    for tile_index in 0..=15 {
        let tile_num = sliding_tile_state.board.get(tile_index);
        if tile_num != 0 {
            let row_of_tile = tile_index / 4;
            let column_of_tile = tile_index % 4;
            let row_of_destination = tile_num / 4;
            let column_of_destination = tile_num % 4;
            if row_of_tile > row_of_destination {
                heuristic += (row_of_tile - row_of_destination) as u32;
            } else {
                heuristic += (row_of_destination - row_of_tile) as u32;
            }
            if column_of_tile > column_of_destination {
                heuristic += (column_of_tile - column_of_destination) as u32;
            } else {
                heuristic += (column_of_destination - column_of_tile) as u32;
            }
        }
    }
    heuristic
}

fn manhattan_admissible_heuristic(sliding_tile_state: &SlidingTileState, heuristic_info: &TileHeuristicInfo) -> u32 {
    let mut heuristic: u32 = heuristic_info.parent_heuristic;
    let num_of_moved_tile = sliding_tile_state.board.get(heuristic_info.index_of_moved_tile);
    let row_of_moved_tile = heuristic_info.index_of_moved_tile / 4;
    let column_of_moved_tile = heuristic_info.index_of_moved_tile % 4;
    let row_of_destination = num_of_moved_tile / 4;
    let column_of_destination = num_of_moved_tile % 4;
    match heuristic_info.direction_of_movement {
        Direction::East => {
            if column_of_destination > column_of_moved_tile {
                heuristic += 1;
            } else {
                heuristic -= 1;
            }
        },
        Direction::North => {
            if row_of_destination < row_of_moved_tile {
                heuristic += 1;
            } else {
                heuristic -= 1;
            }
        },
        Direction::West => {
            if column_of_destination < column_of_moved_tile {
                heuristic += 1;
            } else {
                heuristic -= 1;
            }
        },
        Direction::South => {
            if row_of_destination > row_of_moved_tile {
                heuristic += 1;
            } else {
                heuristic -= 1;
            }
        },
    }
    heuristic
}

fn is_goal_for_inverse_tile(inverse_sliding_node: &InverseSlidingNode) -> bool {
    inverse_sliding_node.heuristic == 0
}

fn initialize_inverse_manhattan_admissible_heuristic(sliding_tile_state: &SlidingTileState) -> u32 {
    let mut heuristic: u32 = 0;
    for tile_index in 0..=15 {
        let tile_num = sliding_tile_state.board.get(tile_index) as u32;
        if tile_num != 0 {
            let row_of_tile = (tile_index / 4) as u32;
            let column_of_tile = (tile_index % 4) as u32;
            let row_of_destination = tile_num / 4;
            let column_of_destination = tile_num % 4;
            if row_of_tile > row_of_destination {
                heuristic += (360360 / tile_num) * (row_of_tile - row_of_destination);
            } else {
                heuristic += (360360 / tile_num) * (row_of_destination - row_of_tile);
            }
            if column_of_tile > column_of_destination {
                heuristic += (360360 / tile_num) * (column_of_tile - column_of_destination) ;
            } else {
                heuristic += (360360 / tile_num) * (column_of_destination - column_of_tile);
            }
        }
    }
    heuristic
}

fn inverse_manhattan_admissible_heuristic(sliding_tile_state: &SlidingTileState, heuristic_info: &TileHeuristicInfo) -> u32 {
    let mut heuristic: u32 = heuristic_info.parent_heuristic;
    let num_of_moved_tile = sliding_tile_state.board.get(heuristic_info.index_of_moved_tile);
    let row_of_moved_tile = heuristic_info.index_of_moved_tile / 4;
    let column_of_moved_tile = heuristic_info.index_of_moved_tile % 4;
    let row_of_destination = num_of_moved_tile / 4;
    let column_of_destination = num_of_moved_tile % 4;
    let heuristic_change = 360360 / (num_of_moved_tile as u32);
    match heuristic_info.direction_of_movement {
        Direction::East => {
            if column_of_destination > column_of_moved_tile {
                heuristic += heuristic_change;
            } else {
                heuristic -= heuristic_change;
            }
        },
        Direction::North => {
            if row_of_destination < row_of_moved_tile {
                heuristic += heuristic_change;
            } else {
                heuristic -= heuristic_change;
            }
        },
        Direction::West => {
            if column_of_destination < column_of_moved_tile {
                heuristic += heuristic_change;
            } else {
                heuristic -= heuristic_change;
            }
        },
        Direction::South => {
            if row_of_destination > row_of_moved_tile {
                heuristic += heuristic_change;
            } else {
                heuristic -= heuristic_change;
            }
        },
    }
    heuristic
}

fn create_starting_tile_state(map_details: &[&str]) -> SlidingTileState {
    let mut starting_state = SlidingTileState {
        index_of_blank_space: 0,
        board: Board(0)
    };
    for tile_index in 0..=15 {
        starting_state.board.set(tile_index, map_details[tile_index as usize].parse::<u8>().unwrap());
        if starting_state.board.get(tile_index) == 0 {
            starting_state.index_of_blank_space = tile_index;
        }
    }
    starting_state
}

enum NodeHistoryType {
    VacuumWorld(Vec<VacuumNode>),
    SlidingPuzzle(Vec<SlidingTileNode>),
    InverseSlidingPuzzle(Vec<InverseSlidingNode>),
}

fn run_algorithm_with_domain(heuristic_weight_or_weights: Vec<f64>, algorithm_of_choice: &str, domain_of_choice: &str, raw_world_details: String) -> (Option<usize>, Option<u32>, NodeHistoryType, NodeInfo, f64, Option<<StdRng as SeedableRng>::Seed>) {
    if domain_of_choice == "vacuum_world" {
        let mut world_details = raw_world_details.split("\n");
        let num_columns = world_details.next().unwrap().trim().parse::<u8>().unwrap();
        let num_rows = world_details.next().unwrap().trim().parse::<u8>().unwrap();
        let (starting_state, blockage_pos) = create_starting_vacuum_state(num_columns, num_rows, &world_details.collect::<Vec<&str>>());
        let map_info = MapInfo {
            num_columns: num_columns,
            num_rows: num_rows,
            blockage_pos: blockage_pos,
        };
        let starting_node = VacuumNode {
            parent_index: 0, // Placeholder
            heuristic: diamond_admissible_heuristic(&starting_state, &map_info),
            state: starting_state,
            path_cost: 0,
            action: Action::Vacuum, // Placeholder
        };
        let (ending_node, error_bound, closed_list, node_info, process_cpu_time, seed) = run_algorithm::<VacuumState, MapInfo, VacuumNode>(starting_node, is_goal_for_vacuum, diamond_admissible_heuristic, heuristic_weight_or_weights, &map_info, algorithm_of_choice);
        (ending_node, error_bound, NodeHistoryType::VacuumWorld(closed_list), node_info, process_cpu_time, seed)
    } else if domain_of_choice == "sliding_puzzle" {
        let world_details = raw_world_details.split(" ");
        let starting_state = create_starting_tile_state(&world_details.collect::<Vec<&str>>());
        let starting_node = SlidingTileNode {
            parent_index: 0, // Placeholder
            heuristic: initialize_manhattan_admissible_heuristic(&starting_state),
            state: starting_state,
            path_cost: 0,
            action: None, // Placeholder
        };
        let placeholder_heuristic_info = TileHeuristicInfo {
            parent_heuristic: 0,
            index_of_moved_tile: 0,
            direction_of_movement: Direction::East,
        };
        let (ending_node, error_bound, closed_list, node_info, process_cpu_time, seed) = run_algorithm::<SlidingTileState, TileHeuristicInfo, SlidingTileNode>(starting_node, is_goal_for_tile, manhattan_admissible_heuristic, heuristic_weight_or_weights, &placeholder_heuristic_info, algorithm_of_choice);
        (ending_node, error_bound, NodeHistoryType::SlidingPuzzle(closed_list), node_info, process_cpu_time, seed)
    } else if domain_of_choice == "inverse_sliding_puzzle" {
        let world_details = raw_world_details.split(" ");
        let starting_state = create_starting_tile_state(&world_details.collect::<Vec<&str>>());
        let starting_node = InverseSlidingNode {
            parent_index: 0, // Placeholder
            heuristic: initialize_inverse_manhattan_admissible_heuristic(&starting_state),
            state: starting_state,
            path_cost: 0,
            action: None, // Placeholder
        };
        let placeholder_heuristic_info = TileHeuristicInfo {
            parent_heuristic: 0,
            index_of_moved_tile: 0,
            direction_of_movement: Direction::East,
        };
        let (ending_node, error_bound, closed_list, node_info, process_cpu_time, seed) = run_algorithm::<SlidingTileState, TileHeuristicInfo, InverseSlidingNode>(starting_node, is_goal_for_inverse_tile, inverse_manhattan_admissible_heuristic, heuristic_weight_or_weights, &placeholder_heuristic_info, algorithm_of_choice);
        (ending_node, error_bound, NodeHistoryType::InverseSlidingPuzzle(closed_list), node_info, process_cpu_time, seed)
    } else {
        panic!("Invalid domain choice.");
    }
}

fn run_algorithm<S: Eq + ManualHash + Hash + Clone, I: Any, N: Searchable<S, I> + Clone>(starting_node: N, is_goal: fn(&N) -> bool, heuristic_param: fn(&S, &I) -> u32, heuristic_weight_or_weights: Vec<f64>, map_info: &I, algorithm_of_choice: &str) -> (Option<usize>, Option<u32>, Vec<N>, NodeInfo, f64, Option<<StdRng as SeedableRng>::Seed>) {
    if algorithm_of_choice == "weighted_a_star" {
        weighted_a_star::<S, I, N>(starting_node, is_goal, heuristic_param, heuristic_weight_or_weights[0], map_info)
    } else if algorithm_of_choice == "anytime_weighted_a_star" {
        anytime_weighted_a_star::<S, I, N>(starting_node, is_goal, heuristic_param, heuristic_weight_or_weights[0], map_info)
    } else if algorithm_of_choice == "randomized_weighted_a_star" {
        randomized_weighted_a_star::<S, I, N>(starting_node, is_goal, heuristic_param, heuristic_weight_or_weights, map_info)
    } else {
        panic!("Invalid algorithm choice.");
    }
}

fn push_action(action_vec: &mut Vec<char>, action: &Action) {
    match action {
        Action::Vacuum => action_vec.push('V'),
        Action::Move(direction) => match direction {
            Direction::East => action_vec.push('E'),
            Direction::North => action_vec.push('N'),
            Direction::West => action_vec.push('W'),
            Direction::South => action_vec.push('S'),
        }
    }
}

fn print_actions<N: ActionPath>(ending_node: usize, node_history: &[N]) {
    let mut reverse_actions: Vec<char> = Vec::new();
    push_action(&mut reverse_actions, &node_history[ending_node].get_action());
    let mut current_node_index = node_history[ending_node].get_parent_index();
    while current_node_index > 0 {
        push_action(&mut reverse_actions, &node_history[current_node_index].get_action());
        current_node_index = node_history[current_node_index].get_parent_index();
    }
    let mut action_index = reverse_actions.len();
    while action_index != 0 {
        println!("{}", reverse_actions[action_index - 1]);
        action_index -= 1;
    }
}

fn print_results(path_cost: Option<u32>, error_bound: Option<u32>, node_info: &NodeInfo, seconds_elapsed: f64, seed: Option<<StdRng as SeedableRng>::Seed>) {
    println!("Nodes generated: {}", node_info.nodes_generated);
    println!("Nodes expanded: {}", node_info.nodes_expanded);
    println!("Duplicates detected: {}", node_info.duplicates_detected);
    println!("State representations: {}", node_info.nodes_generated + node_info.duplicates_detected);
    println!("Costly nodes stopped from generation: {}", node_info.costly_nodes_not_generated);
    println!("Costly nodes stopped from expansion: {}", node_info.costly_nodes_not_expanded);
    match path_cost {
        Some(cost) => println!("Solution cost: {}", cost),
        None => println!("Solution cost: N/A"),
    }
    match error_bound {
        Some(bound) => println!("Error bound: {}", bound),
        None => println!("Error bound: N/A"),
    }
    println!("Seconds elapsed: {}", seconds_elapsed);
    println!("Nodes generated against path cost: {:?}", node_info.nodes_generated_progress);
    println!("Seconds elapsed against path cost: {:?}", node_info.time_elapsed_progress);
    match seed {
        Some(unwrapped_seed) => print!("Seed: {:?}", unwrapped_seed),
        None => print!("Seed: N/A"),
    }
}

fn main() {
    //let mut world_details = "10\n10\n*#_____*__\n__*__#__*_\n___*#_#__#\n_*_____*__\n_*__*___@_\n__#*_____#\n_______*__\n#______##_\n__#_#___*_\n_*_#_*_##_".split("\n");
    let args: Vec<String> = env::args().collect();
    let input_method_of_choice = args[1].clone();
    let algorithm_of_choice = args[2].clone();
    let domain_of_choice = args[3].clone();
    let (raw_world_details, raw_heuristic_weight_or_weights);
    if input_method_of_choice == "arguments" {
        raw_world_details = args[4].clone();
        raw_heuristic_weight_or_weights = args[5].clone();
    } else if input_method_of_choice == "files" {
        raw_world_details = fs::read_to_string(args[4].clone()).unwrap();
        raw_heuristic_weight_or_weights = fs::read_to_string(args[5].clone()).unwrap();
    } else {
        panic!("Invalid input method choice.");
    }
    let heuristic_weight_or_weights_iter = raw_heuristic_weight_or_weights.split(" ");
    let mut heuristic_weight_or_weights = Vec::new();
    for weight in heuristic_weight_or_weights_iter {
        heuristic_weight_or_weights.push(weight.parse::<f64>().unwrap());
    }
    let (ending_node, error_bound, wrapped_node_history, node_info, seconds_elapsed, seed) = run_algorithm_with_domain(heuristic_weight_or_weights, &algorithm_of_choice, &domain_of_choice, raw_world_details);
    match ending_node {
        Some(node) => match wrapped_node_history {
            NodeHistoryType::VacuumWorld(list) => {
                //print_actions(node, &list);
                print_results(Some(list[node].get_path_cost()), error_bound, &node_info, seconds_elapsed, seed);
            },
            NodeHistoryType::SlidingPuzzle(list) => {
                //print_actions(node, &list);
                print_results(Some(list[node].get_path_cost()), error_bound, &node_info, seconds_elapsed, seed);
            },
            NodeHistoryType::InverseSlidingPuzzle(list) => {
                //print_actions(node, &list);
                print_results(Some(list[node].get_path_cost()), error_bound, &node_info, seconds_elapsed, seed);
            },
        },
        None => print_results(None, None, &node_info, seconds_elapsed, seed),
    }
}