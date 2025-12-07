import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selected crystal. "
        "Each move attempt cycles to the next crystal. "
        "Light all paths before time runs out!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated crystal cavern. "
        "Strategically place crystals to illuminate all paths within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_WALL = (30, 35, 50)
    COLOR_UNLIT_PATH = (70, 80, 100)
    COLOR_LIT_PATH = (255, 255, 150)
    COLOR_LIT_PATH_GLOW = (255, 220, 80, 50)
    CRYSTAL_COLORS = [
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Yellow
    ]
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_SELECT_GLOW = (255, 255, 255)

    # Game parameters
    NUM_NODES = 25
    NUM_CRYSTALS = 3
    NEAREST_NEIGHBORS_K = 3
    LIGHT_RADIUS = 90
    CRYSTAL_MOVE_SPEED = 4  # Pixels per frame, for smooth movement

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Set headless mode for Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Isometric projection parameters
        self.tile_width_half = 32
        self.tile_height_half = 16
        self.origin_x = self.WIDTH // 2
        self.origin_y = self.HEIGHT // 2 - 50

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_condition_met = False
        self.time_left = 0
        self.nodes = []
        self.edges = set()
        self.adj = {}
        self.directional_neighbors = []
        self.path_segments = []
        self.path_lit_states = []
        self.crystals = []
        self.selected_crystal_idx = 0
        self.last_reward = 0.0


    def _iso_transform(self, grid_x, grid_y):
        iso_x = self.origin_x + (grid_x - grid_y) * self.tile_width_half
        iso_y = self.origin_y + (grid_x + grid_y) * self.tile_height_half
        return int(iso_x), int(iso_y)

    def _generate_layout(self):
        # 1. Generate random grid points
        grid_extents = 8
        potential_points = [(gx, gy) for gx in range(grid_extents) for gy in range(grid_extents)]
        self.np_random.shuffle(potential_points)
        grid_nodes = potential_points[:self.NUM_NODES]

        self.nodes = [{'grid_pos': pos, 'iso_pos': self._iso_transform(pos[0], pos[1])} for pos in grid_nodes]
        
        # 2. Build k-NN graph
        self.edges = set()
        for i in range(self.NUM_NODES):
            distances = []
            for j in range(self.NUM_NODES):
                if i == j:
                    continue
                dist = math.hypot(self.nodes[i]['iso_pos'][0] - self.nodes[j]['iso_pos'][0],
                                  self.nodes[i]['iso_pos'][1] - self.nodes[j]['iso_pos'][1])
                distances.append((dist, j))
            
            distances.sort()
            for k in range(self.NEAREST_NEIGHBORS_K):
                if k < len(distances):
                    j = distances[k][1]
                    # Add edge in a canonical way to avoid duplicates
                    edge = tuple(sorted((i, j)))
                    self.edges.add(edge)

        # 3. Ensure connectivity
        for _ in range(self.NUM_NODES): # Failsafe loop
            # Find components
            q = []
            visited = [False] * self.NUM_NODES
            components = []
            for i in range(self.NUM_NODES):
                if not visited[i]:
                    component = []
                    q.append(i)
                    visited[i] = True
                    head = 0
                    while head < len(q):
                        u = q[head]
                        head += 1
                        component.append(u)
                        for v1, v2 in self.edges:
                            if v1 == u and not visited[v2]:
                                visited[v2] = True
                                q.append(v2)
                            elif v2 == u and not visited[v1]:
                                visited[v1] = True
                                q.append(v1)
                    components.append(component)
            
            if len(components) == 1:
                break

            # Connect closest pair of nodes from different components
            min_dist = float('inf')
            closest_pair = None
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    for u in components[i]:
                        for v in components[j]:
                            dist = math.hypot(self.nodes[u]['iso_pos'][0] - self.nodes[v]['iso_pos'][0],
                                              self.nodes[u]['iso_pos'][1] - self.nodes[v]['iso_pos'][1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = tuple(sorted((u, v)))
            if closest_pair:
                self.edges.add(closest_pair)
        
        # 4. Create adjacency list and path segments
        self.adj = {i: [] for i in range(self.NUM_NODES)}
        self.path_segments = []
        for u, v in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            mid_x = (self.nodes[u]['iso_pos'][0] + self.nodes[v]['iso_pos'][0]) / 2
            mid_y = (self.nodes[u]['iso_pos'][1] + self.nodes[v]['iso_pos'][1]) / 2
            self.path_segments.append({'nodes': (u, v), 'midpoint': (mid_x, mid_y)})
        
        # 5. Determine directional neighbors for movement
        self.directional_neighbors = []
        for i in range(self.NUM_NODES):
            neighbors = {'n': None, 's': None, 'e': None, 'w': None}
            p_i = self.nodes[i]['grid_pos']
            
            best_n, best_s, best_e, best_w = (float('inf'), None), (float('inf'), None), (float('inf'), None), (float('inf'), None)

            for neighbor_idx in self.adj[i]:
                p_n = self.nodes[neighbor_idx]['grid_pos']
                dx, dy = p_n[0] - p_i[0], p_n[1] - p_i[1]

                # Map to screen-like directions
                # Up-Right (East), Down-Right (South), Down-Left (West), Up-Left (North)
                if dy < 0 and dx >= 0: # North-East quadrant -> East
                    if dx + abs(dy) < best_e[0]: best_e = (dx + abs(dy), neighbor_idx)
                if dy >= 0 and dx > 0: # South-East quadrant -> South
                    if dx + dy < best_s[0]: best_s = (dx + dy, neighbor_idx)
                if dy > 0 and dx <= 0: # South-West quadrant -> West
                    if abs(dx) + dy < best_w[0]: best_w = (abs(dx) + dy, neighbor_idx)
                if dy <= 0 and dx < 0: # North-West quadrant -> North
                    if abs(dx) + abs(dy) < best_n[0]: best_n = (abs(dx) + abs(dy), neighbor_idx)
            
            neighbors['n'] = best_n[1]
            neighbors['s'] = best_s[1]
            neighbors['e'] = best_e[1]
            neighbors['w'] = best_w[1]
            self.directional_neighbors.append(neighbors)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Loop until a valid starting configuration is generated
        while True:
            self._generate_layout()

            # Place crystals
            self.crystals = []
            start_nodes = self.np_random.choice(self.NUM_NODES, self.NUM_CRYSTALS, replace=False)
            for i in range(self.NUM_CRYSTALS):
                node_idx = start_nodes[i]
                pos = self.nodes[node_idx]['iso_pos']
                self.crystals.append({
                    'node_idx': node_idx,
                    'pos': list(pos),
                    'target_pos': list(pos),
                    'color': self.CRYSTAL_COLORS[i % len(self.CRYSTAL_COLORS)],
                })

            # Update lighting based on initial crystal placement
            self.path_lit_states = self._update_lighting()

            # Check condition: ensure at least 50% lit at start
            lit_count = sum(self.path_lit_states)
            if len(self.path_lit_states) == 0 or (lit_count / len(self.path_lit_states)) >= 0.5:
                break  # Exit loop if condition is met or there are no paths

        # Once a valid layout is found, finalize the reset
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        self.time_left = self.MAX_STEPS
        self.selected_crystal_idx = 0

        return self._get_observation(), self._get_info()

    def _update_lighting(self):
        new_lit_states = []
        for segment in self.path_segments:
            is_lit = False
            for crystal in self.crystals:
                dist = math.hypot(segment['midpoint'][0] - crystal['pos'][0],
                                  segment['midpoint'][1] - crystal['pos'][1])
                if dist < self.LIGHT_RADIUS:
                    is_lit = True
                    break
            new_lit_states.append(is_lit)
        return new_lit_states

    def _update_crystal_positions(self):
        for crystal in self.crystals:
            dx = crystal['target_pos'][0] - crystal['pos'][0]
            dy = crystal['target_pos'][1] - crystal['pos'][1]
            dist = math.hypot(dx, dy)
            if dist < self.CRYSTAL_MOVE_SPEED:
                crystal['pos'] = list(crystal['target_pos'])
            else:
                crystal['pos'][0] += (dx / dist) * self.CRYSTAL_MOVE_SPEED
                crystal['pos'][1] += (dy / dist) * self.CRYSTAL_MOVE_SPEED

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        self.time_left -= 1
        
        old_lit_count = sum(self.path_lit_states)
        
        # Handle movement action
        if movement > 0:
            crystal = self.crystals[self.selected_crystal_idx]
            current_node_idx = crystal['node_idx']
            
            # Map action to N, W, S, E
            # 1=up -> N, 2=down -> S, 3=left -> W, 4=right -> E
            direction_map = {1: 'n', 2: 's', 3: 'w', 4: 'e'}
            target_dir = direction_map.get(movement)

            if target_dir:
                target_node_idx = self.directional_neighbors[current_node_idx][target_dir]
                if target_node_idx is not None:
                    # Check if another crystal is already at the target node
                    is_occupied = False
                    for i, c in enumerate(self.crystals):
                        if i != self.selected_crystal_idx and c['node_idx'] == target_node_idx:
                            is_occupied = True
                            break
                    
                    if not is_occupied:
                        crystal['node_idx'] = target_node_idx
                        crystal['target_pos'] = list(self.nodes[target_node_idx]['iso_pos'])
                        # sfx: crystal move
            
            # Cycle selection on any arrow key press
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % self.NUM_CRYSTALS
            # sfx: select cycle

        # Interpolate crystal movement
        self._update_crystal_positions()

        # Update lighting state
        self.path_lit_states = self._update_lighting()
        new_lit_count = sum(self.path_lit_states)

        # Calculate reward
        reward = 0
        if new_lit_count > old_lit_count:
            reward += (new_lit_count - old_lit_count) * 0.1  # Reward for new lit paths
        if new_lit_count < old_lit_count:
            reward += (old_lit_count - new_lit_count) * -0.01 # Small penalty for un-lighting paths
        
        self.last_reward = reward
        self.score += reward

        # Check termination
        terminated = False
        truncated = False
        self.win_condition_met = (len(self.path_segments) > 0) and (new_lit_count == len(self.path_segments))

        if self.win_condition_met:
            reward += 100  # Win bonus
            self.score += 100
            terminated = True
            self.game_over = True
        elif self.time_left <= 0:
            reward -= 100  # Loss penalty
            self.score -= 100
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        lit_percentage = 0
        if len(self.path_lit_states) > 0:
            lit_percentage = sum(self.path_lit_states) / len(self.path_lit_states)
        
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "lit_percentage": lit_percentage,
            "last_reward": self.last_reward
        }

    def _render_game(self):
        # Render cavern hull
        if len(self.nodes) > 2:
            hull_points = []
            
            # Find convex hull (Graham scan)
            points = [n['iso_pos'] for n in self.nodes]
            points.sort()
            
            def cross_product(p1, p2, p3):
                return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

            upper_hull = []
            for p in points:
                while len(upper_hull) >= 2 and cross_product(upper_hull[-2], upper_hull[-1], p) <= 0:
                    upper_hull.pop()
                upper_hull.append(p)

            lower_hull = []
            for p in reversed(points):
                while len(lower_hull) >= 2 and cross_product(lower_hull[-2], lower_hull[-1], p) <= 0:
                    lower_hull.pop()
                lower_hull.append(p)
            
            hull_points = upper_hull[:-1] + lower_hull[:-1]
            if len(hull_points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, hull_points, self.COLOR_WALL)
                pygame.gfxdraw.aapolygon(self.screen, hull_points, self.COLOR_WALL)


        # Render paths
        for i, segment in enumerate(self.path_segments):
            p1 = self.nodes[segment['nodes'][0]]['iso_pos']
            p2 = self.nodes[segment['nodes'][1]]['iso_pos']
            
            if self.path_lit_states[i]:
                # Glow effect for lit paths
                pygame.draw.aaline(self.screen, self.COLOR_LIT_PATH_GLOW, p1, p2, 5)
                pygame.draw.aaline(self.screen, self.COLOR_LIT_PATH_GLOW, p1, p2, 3)
                pygame.draw.aaline(self.screen, self.COLOR_LIT_PATH, p1, p2, 1)
            else:
                pygame.draw.aaline(self.screen, self.COLOR_UNLIT_PATH, p1, p2, 1)
        
        # Render crystals
        for i, crystal in enumerate(self.crystals):
            pos = (int(crystal['pos'][0]), int(crystal['pos'][1]))
            color = crystal['color']
            
            # Draw crystal body
            points = [
                (pos[0], pos[1] - 8), (pos[0] + 5, pos[1]),
                (pos[0], pos[1] + 8), (pos[0] - 5, pos[1])
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

            # Selection indicator
            if i == self.selected_crystal_idx and not self.game_over:
                pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                radius = int(10 + pulse * 4)
                alpha = int(100 + pulse * 100)
                glow_color = (*self.COLOR_SELECT_GLOW, alpha)
                
                # Use a temporary surface for alpha blending
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, glow_color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_SELECT_GLOW)


    def _render_ui(self):
        info = self._get_info()
        
        # Lit Percentage
        lit_percent_str = f"Lit: {info['lit_percentage']:.0%}"
        text_surf = self.font_ui.render(lit_percent_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        time_str = f"Time: {max(0, self.time_left / self.FPS):.1f}"
        text_surf = self.font_ui.render(time_str, True, self.COLOR_UI_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)
        
        # Game Over Message
        if self.game_over:
            if self.win_condition_met:
                msg = "CAVERN ILLUMINATED"
                color = self.COLOR_LIT_PATH
            else:
                msg = "TIME'S UP"
                color = (200, 50, 50)
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)


    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # --- Verification part ---
    print("Verifying environment implementation...")
    env = GameEnv()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    print("✓ Implementation verified successfully")
    env.close()
    
    # --- Manual Play Example ---
    # This part requires a window.
    try:
        # Unset the dummy driver to allow window creation
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Crystal Cavern")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        # Map keys to actions
        key_to_action = {
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
        }

        print("\n--- Manual Play ---")
        print("Controls: Arrow keys to move. Q to quit. R to reset.")

        while running:
            movement_action = 0 # No-op by default
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in key_to_action:
                        movement_action = key_to_action[event.key]
                    if event.key == pygame.K_q:
                        running = False
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("Environment reset.")

            # For manual play, we only send an action on key press
            action = [movement_action, 0, 0] # Space and Shift are not used
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Draw the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Episode finished. Total Reward: {total_reward:.2f}")
                print("Press 'R' to reset.")

            clock.tick(GameEnv.FPS)

        env.close()
    except pygame.error as e:
        print(f"\nPygame display could not be initialized: {e}")
        print("Manual play is unavailable. The environment is still valid for training.")