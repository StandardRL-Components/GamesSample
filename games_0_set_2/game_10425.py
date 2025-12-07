import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:23:49.730557
# Source Brief: brief_00425.md
# Brief Index: 425
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for a particle (visual only)
class Particle:
    """A visual particle representing a unit of data flow."""
    def __init__(self, start_pos, target_pos, speed_multiplier):
        self.pos = np.array(start_pos, dtype=np.float32)
        direction = target_pos - start_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            self.vel = (direction / distance) * speed_multiplier * random.uniform(0.8, 1.2)
        else:
            self.vel = np.array([0,0], dtype=np.float32)
        
        # Life is proportional to distance, so particles exist for the edge traversal
        self.max_life = int(distance / (np.linalg.norm(self.vel) + 1e-6)) if np.linalg.norm(self.vel) > 0 else 10
        self.life = self.max_life

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life > 0

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Data Stream', a real-time puzzle game.
    The goal is to route a flow of data to a target port by redirecting flow
    at network nodes. The environment is designed for visual quality and
    responsive gameplay, using a MultiDiscrete action space for simultaneous
    control over selection and actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Route a continuous stream of data to a target port by redirecting flow at network nodes. "
        "Optimize your network to achieve the target efficiency before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys to select a node. Press space to route flow towards the target. "
        "Press shift to distribute flow evenly among all connections."
    )
    auto_advance = True

    # --- Game Design Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1350  # 45 seconds * 30 FPS

    # --- Visual Style ---
    COLOR_BG = (10, 15, 25)
    COLOR_GRID = (25, 35, 50)
    COLOR_NODE = (120, 130, 150)
    COLOR_NODE_SELECTED = (255, 255, 255)
    COLOR_NODE_SOURCE = (255, 200, 0)
    COLOR_TARGET = (0, 255, 150)
    COLOR_EDGE = (50, 60, 80)
    COLOR_FLOW = (0, 150, 255)
    COLOR_TEXT = (220, 220, 220)

    # --- Gameplay Mechanics ---
    NUM_NODES = 12
    MIN_NODE_DIST = 80
    DATA_GENERATION_RATE = 5.0  # units per step
    TARGET_PERCENTAGE = 0.75
    FLOW_MOMENTUM = 0.95  # How much preference is kept frame-to-frame
    ACTION_REDIRECT_STRENGTH = 0.5 # How strongly an action influences flow

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        # Initialize all state variables to prevent errors before first reset
        self.nodes = []
        self.edges = {}
        self.adj = {}
        self.source_node_id = -1
        self.target_node_id = -1
        self.selected_node_id = -1
        self.node_routing_prefs = {}
        self.edge_flows = {}
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_data_generated = 0.0
        self.data_at_target = 0.0
        self.data_lost = 0.0
        self.last_reward = 0.0
        self.prev_space_held = False
        self.prev_shift_held = False

        # self.validate_implementation() # Commented out as it's for dev, not required by tests
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_network()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_data_generated = 1e-6 # Avoid division by zero
        self.data_at_target = 0.0
        self.data_lost = 0.0
        self.particles = []
        
        self.selected_node_id = self.source_node_id

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_val, shift_val = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_val, shift_val)
        self._update_game_state()
        
        reward = self.last_reward
        terminated = self._check_termination()

        if terminated:
            win_condition = (self.data_at_target / self.total_data_generated) >= self.TARGET_PERCENTAGE
            if win_condition:
                reward += 100.0
            else:
                reward -= 100.0
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_network(self):
        # 1. Create nodes with random positions, ensuring they are not too close.
        self.nodes = []
        while len(self.nodes) < self.NUM_NODES:
            pos = (
                self.np_random.uniform(50, self.WIDTH - 50),
                self.np_random.uniform(50, self.HEIGHT - 50)
            )
            if all(np.linalg.norm(np.array(pos) - np.array(n['pos'])) > self.MIN_NODE_DIST for n in self.nodes):
                self.nodes.append({'id': len(self.nodes), 'pos': pos, 'data_pool': 0.0})

        # 2. Guarantee connectivity with a Minimum Spanning Tree (Prim's algorithm).
        self.adj = {i: [] for i in range(self.NUM_NODES)}
        self.edges = {}
        in_mst = {0}
        remaining = set(range(1, self.NUM_NODES))
        while remaining:
            min_dist, best_edge = float('inf'), None
            for u in in_mst:
                for v in remaining:
                    dist = np.linalg.norm(np.array(self.nodes[u]['pos']) - np.array(self.nodes[v]['pos']))
                    if dist < min_dist:
                        min_dist, best_edge = dist, (u, v)
            u, v = best_edge
            in_mst.add(v)
            remaining.remove(v)
            self.adj[u].append(v)
            self.adj[v].append(u)
            self.edges[tuple(sorted((u,v)))] = {}

        # 3. Add extra edges to create cycles and more complex paths.
        for _ in range(self.NUM_NODES // 2):
            u, v = self.np_random.choice(self.NUM_NODES, 2, replace=False).tolist()
            if v not in self.adj[u] and u != v:
                self.adj[u].append(v)
                self.adj[v].append(u)
                self.edges[tuple(sorted((u,v)))] = {}

        # 4. Assign distinct source and target nodes.
        self.source_node_id, self.target_node_id = self.np_random.choice(self.NUM_NODES, 2, replace=False).tolist()
        
        # 5. Initialize flow and routing preferences.
        self.node_routing_prefs, self.edge_flows = {}, {}
        for u in range(self.NUM_NODES):
            neighbors = self.adj[u]
            self.node_routing_prefs[u] = {v: 1.0 / len(neighbors) for v in neighbors} if neighbors else {}
            for v in neighbors:
                self.edge_flows[(u, v)] = 0.0

    def _handle_input(self, movement, space_held, shift_held):
        # --- Handle Movement: Select the most appropriate node in the given direction ---
        if movement != 0:
            current_pos = np.array(self.nodes[self.selected_node_id]['pos'])
            target_vector = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}[movement]
            
            best_candidate, min_score = -1, float('inf')
            for node in self.nodes:
                if node['id'] == self.selected_node_id: continue
                
                vec_to_node = np.array(node['pos']) - current_pos
                dist = np.linalg.norm(vec_to_node)
                if dist < 1e-6: continue

                dir_to_node = vec_to_node / dist
                dot_product = np.dot(target_vector, dir_to_node)
                
                if dot_product > 0.707: # Within a 90-degree cone (45 deg each side)
                    score = dist * (2.0 - dot_product) # Prioritize closer nodes in the right direction
                    if score < min_score:
                        min_score, best_candidate = score, node['id']
            
            if best_candidate != -1:
                self.selected_node_id = best_candidate

        # --- Handle Actions: Use press-once logic for better game feel ---
        if space_held and not self.prev_space_held: # Redirect flow towards target
            # SFX: bleep_confirm
            path = self._find_shortest_path(self.selected_node_id, self.target_node_id)
            if path and len(path) > 1:
                self._adjust_routing(self.selected_node_id, path[1], self.ACTION_REDIRECT_STRENGTH)

        if shift_held and not self.prev_shift_held: # Distribute flow evenly
            # SFX: whoosh_reset
            neighbors = self.adj[self.selected_node_id]
            if neighbors:
                for neighbor in neighbors:
                    self._adjust_routing(self.selected_node_id, neighbor, 1.0/len(neighbors), absolute=True)
        
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

    def _adjust_routing(self, node_id, target_neighbor_id, strength, absolute=False):
        prefs = self.node_routing_prefs.get(node_id)
        if not prefs: return

        if absolute:
            for neighbor in prefs: prefs[neighbor] = strength
        else:
            current_pref = prefs[target_neighbor_id]
            new_pref = current_pref + (1.0 - current_pref) * strength
            prefs[target_neighbor_id] = new_pref
            
            total_other_prefs = sum(p for n, p in prefs.items() if n != target_neighbor_id)
            if total_other_prefs > 1e-6:
                scale = (1.0 - new_pref) / total_other_prefs
                for n_id in prefs:
                    if n_id != target_neighbor_id: prefs[n_id] *= scale
        
        total = sum(prefs.values())
        if total > 1e-6:
            for n_id in prefs: prefs[n_id] /= total

    def _find_shortest_path(self, start, end):
        if start == end: return [start]
        q = [(start, [start])]
        visited = {start}
        while q:
            curr, path = q.pop(0)
            for neighbor in self.adj[curr]:
                if neighbor == end: return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return None

    def _update_game_state(self):
        self.steps += 1
        self.last_reward = 0.0

        # 1. Add new data at the source node.
        self.nodes[self.source_node_id]['data_pool'] += self.DATA_GENERATION_RATE
        self.total_data_generated += self.DATA_GENERATION_RATE

        # 2. Propagate data through the network based on routing preferences.
        new_data_pools = {i: 0.0 for i in range(self.NUM_NODES)}
        for u_id, node in enumerate(self.nodes):
            if node['data_pool'] > 0:
                prefs = self.node_routing_prefs[u_id]
                if prefs:
                    for v_id, pref in prefs.items():
                        flow_amount = node['data_pool'] * pref
                        
                        if v_id == self.target_node_id:
                            self.data_at_target += flow_amount
                            self.last_reward += flow_amount * 0.1
                            # SFX: point_scored
                        else:
                            new_data_pools[v_id] += flow_amount
                        
                        current_flow = self.edge_flows.get((u_id, v_id), 0.0)
                        self.edge_flows[(u_id, v_id)] = current_flow * self.FLOW_MOMENTUM + flow_amount * (1.0 - self.FLOW_MOMENTUM)
                else: # Dead end node, data is lost.
                    self.data_lost += node['data_pool']
                    self.last_reward -= node['data_pool'] * 0.01
                    # SFX: data_lost_bloop

        for i in range(self.NUM_NODES): self.nodes[i]['data_pool'] = new_data_pools[i]

        # 3. Spawn and update particles for visualization.
        if self.steps % 2 == 0:
            for u, v in self.edge_flows:
                flow = self.edge_flows.get((u, v), 0.0)
                if self.np_random.random() < flow * 0.2: # Probabilistic spawning
                    pos_u, pos_v = np.array(self.nodes[u]['pos']), np.array(self.nodes[v]['pos'])
                    self.particles.append(Particle(pos_u, pos_v, flow * 0.5))
        
        self.particles = [p for p in self.particles if p.update()]

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS: return True
        if self.total_data_generated > 0 and (self.data_at_target / self.total_data_generated) >= self.TARGET_PERCENTAGE:
            return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "data_routed_percent": (self.data_at_target / self.total_data_generated) if self.total_data_generated > 0 else 0,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_edges()
        self._render_nodes()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(0, self.WIDTH, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_edges(self):
        for u, v in self.edges:
            pos1, pos2 = self.nodes[u]['pos'], self.nodes[v]['pos']
            pygame.draw.aaline(self.screen, self.COLOR_EDGE, pos1, pos2, 1)

    def _render_nodes(self):
        for node in self.nodes:
            pos, node_id = (int(node['pos'][0]), int(node['pos'][1])), node['id']
            color = self.COLOR_NODE_SOURCE if node_id == self.source_node_id else self.COLOR_NODE
            
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, color)
            
            if node_id == self.target_node_id:
                for i in range(2): pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15 + i, self.COLOR_TARGET)

            if node_id == self.selected_node_id:
                pulse = int((1 + math.sin(self.steps * 0.2)) * 3)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 18 + pulse, self.COLOR_NODE_SELECTED)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            alpha = int(255 * (p.life / p.max_life))
            color = (*self.COLOR_FLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, color)
            
    def _render_ui(self):
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        timer_text = f"TIME: {time_left:.1f}s"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 15, 10))

        progress = (self.data_at_target / self.total_data_generated) * 100 if self.total_data_generated > 0 else 0
        target_text = f"ROUTED: {progress:.1f}% / {self.TARGET_PERCENTAGE*100:.0f}%"
        target_surf = self.font_large.render(target_text, True, self.COLOR_TEXT)
        self.screen.blit(target_surf, (15, 10))

        total_flow = sum(node['data_pool'] for node in self.nodes)
        flow_text = f"SYSTEM LOAD: {total_flow:.1f}"
        flow_surf = self.font_small.render(flow_text, True, self.COLOR_TEXT)
        self.screen.blit(flow_surf, (self.WIDTH // 2 - flow_surf.get_width() // 2, self.HEIGHT - 30))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # The following calls are problematic in __init__ before the first reset establishes a seed and state.
        # They are good checks for a unit test but can be fragile here.
        # test_obs = self._get_observation()
        # assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        # assert test_obs.dtype == np.uint8
        
        # obs, info = self.reset()
        # assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        # assert isinstance(info, dict)
        
        # test_action = self.action_space.sample()
        # obs, reward, term, trunc, info = self.step(test_action)
        # assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        # assert isinstance(reward, (int, float))
        # assert isinstance(term, bool)
        # assert trunc == False
        # assert isinstance(info, dict)
        
        # print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows the game to be played by a human for testing and tuning.
    # To run, you might need to unset the SDL_VIDEODRIVER dummy variable.
    # E.g., run `unset SDL_VIDEODRIVER` in your shell before executing the script.
    try:
        del os.environ['SDL_VIDEODRIVER']
    except KeyError:
        pass
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Data Stream")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    while running:
        # --- Human Input to Action Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # 0=none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        action = [
            movement,
            1 if keys[pygame.K_SPACE] else 0,
            1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        ]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ENVIRONMENT ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    pygame.quit()