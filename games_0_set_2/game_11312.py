import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:47:17.515290
# Source Brief: brief_01312.md
# Brief Index: 1312
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Navigate a shifting 3D wireframe world, avoid patrolling enemies by becoming ethereal, and reach the exit."
    user_guide = "Use arrow keys to move between nodes. Press space to become ethereal and pass through enemies. Press shift to change the 3D orientation."
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 5000
    PLAYER_SPEED = 0.05  # Progress per step

    # --- COLORS ---
    COLOR_BG = (10, 5, 25)
    COLOR_WIRE = (40, 30, 80)
    COLOR_PLAYER_SOLID = (0, 180, 255)
    COLOR_PLAYER_ETHEREAL = (180, 100, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_EXIT = (50, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- GYM SPACES ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- STATE VARIABLES ---
        self.steps = 0
        self.score = 0
        self.level = 0
        self.game_over = False
        self.last_win = False
        
        self.player_pos_3d = np.array([0.0, 0.0, 0.0])
        self.player_node_from = 0
        self.player_node_to = 0
        self.player_move_progress = 1.0
        self.player_is_ethereal = False
        
        self.gravity_idx = 0
        self.gravity_vectors = [
            (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, -1), (0, 0, 1)
        ]

        self.enemies = []
        self.particles = []
        
        self.tetra_vertices = []
        self.tetra_edges = []
        self.tetra_adj = {}
        self.exit_node = -1
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.reward_this_step = 0.0
        
        # Note: self.np_random is initialized in super().reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if not self.last_win:
            self.level = 1
            self.score = 0
        else:
            self.level += 1

        self.steps = 0
        self.game_over = False
        self.last_win = False
        self.player_is_ethereal = False
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.gravity_idx = 0

        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0.0

        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_particles()
        
        self._check_events_and_collisions()
        
        reward = self._calculate_reward()
        self.score += reward
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not self.game_over: # Max steps reached
            self.reward_this_step -= 5 # Penalty for timeout
            reward -= 5

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_level(self):
        # --- Create Tetrahedron Geometry ---
        s = 100 + self.level * 2 # Scale
        amp = min(s * 0.4, self.level * 5) # Perturbation amplitude
        
        base_vertices = [
            np.array([1.0, 1.0, 1.0]), np.array([-1.0, -1.0, 1.0]),
            np.array([-1.0, 1.0, -1.0]), np.array([1.0, -1.0, -1.0])
        ]
        self.tetra_vertices = [
            v * s + self.np_random.uniform(-amp, amp, 3) for v in base_vertices
        ]
        self.tetra_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.tetra_adj = {i: [] for i in range(4)}
        for v1, v2 in self.tetra_edges:
            self.tetra_adj[v1].append(v2)
            self.tetra_adj[v2].append(v1)

        # --- Place Player, Exit, and Enemies ---
        nodes = list(range(4))
        self.np_random.shuffle(nodes)
        self.player_node_from = nodes[0]
        self.exit_node = nodes[1]
        
        self.player_node_to = self.player_node_from
        self.player_pos_3d = self.tetra_vertices[self.player_node_from]
        self.player_move_progress = 1.0

        num_enemies = 1 + (self.level - 1) // 3
        enemy_speed = 0.01 + 0.005 * ((self.level - 1) // 5)
        self.enemies = []
        
        available_edges = [e for e in self.tetra_edges if self.player_node_from not in e]
        for _ in range(num_enemies):
            if not available_edges: continue
            edge_idx = self.np_random.integers(len(available_edges))
            edge = available_edges.pop(edge_idx)
            
            self.enemies.append({
                "path": [edge[0], edge[1]],
                "path_idx": 0,
                "progress": self.np_random.random(),
                "speed": enemy_speed * self.np_random.uniform(0.8, 1.2),
                "pos_3d": np.array([0.0, 0.0, 0.0]),
                "state": "PATROL", # PATROL, CHASE, RETURN
                "chase_timer": 0,
                "pulse": self.np_random.random() * math.pi * 2,
            })

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Form Shift (Space) ---
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            self.player_is_ethereal = not self.player_is_ethereal

        # --- Gravity Flip (Shift) ---
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed:
            self.gravity_idx = (self.gravity_idx + 1) % len(self.gravity_vectors)
            p_pos_2d = self._project_3d_to_2d(self.player_pos_3d)
            for _ in range(30):
                angle = self.np_random.random() * 2 * math.pi
                speed = self.np_random.random() * 3 + 1
                self.particles.append({
                    "pos": list(p_pos_2d),
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": self.np_random.integers(20, 40),
                    "color": random.choice([self.COLOR_PLAYER_ETHEREAL, (255,255,255), self.COLOR_WIRE])
                })

        # --- Movement ---
        if movement != 0 and self.player_move_progress >= 1.0:
            target_node = self._determine_move_target(movement)
            if target_node is not None:
                self.player_node_to = target_node
                self.player_move_progress = 0.0

        self.last_space_held, self.last_shift_held = space_held, shift_held

    def _determine_move_target(self, movement_action):
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)} # Up, Down, Left, Right
        desired_vec = np.array(move_map[movement_action])
        
        current_pos_2d = self._project_3d_to_2d(self.tetra_vertices[self.player_node_from])
        neighbors = self.tetra_adj[self.player_node_from]
        
        best_neighbor = None
        max_dot = -2 # Use a value lower than -1

        for neighbor_idx in neighbors:
            neighbor_pos_2d = self._project_3d_to_2d(self.tetra_vertices[neighbor_idx])
            direction_vec = neighbor_pos_2d - current_pos_2d
            
            dist = np.linalg.norm(direction_vec)
            if dist < 1e-6: continue
            
            direction_vec /= dist
            dot_product = np.dot(direction_vec, desired_vec)
            
            if dot_product > max_dot:
                max_dot = dot_product
                best_neighbor = neighbor_idx
        
        if max_dot > 0.3:
            return best_neighbor
        return None

    def _update_player(self):
        if self.player_move_progress < 1.0:
            self.player_move_progress += self.PLAYER_SPEED
            self.player_move_progress = min(1.0, self.player_move_progress)
            
            p_from = self.tetra_vertices[self.player_node_from]
            p_to = self.tetra_vertices[self.player_node_to]
            self.player_pos_3d = p_from + (p_to - p_from) * self.player_move_progress
            
            if self.player_move_progress >= 1.0:
                self.player_node_from = self.player_node_to
    
    def _update_enemies(self):
        for enemy in self.enemies:
            # --- FSM Logic ---
            player_dist = self._graph_distance(self.player_node_from, self._get_closest_node_to_pos(enemy["pos_3d"]))
            
            if enemy["state"] == "PATROL":
                if player_dist < 5 and not self.player_is_ethereal:
                    enemy["state"] = "CHASE"
                    enemy["chase_timer"] = 120
            elif enemy["state"] == "CHASE":
                if self.player_is_ethereal or player_dist >= 5 or enemy["chase_timer"] <= 0:
                    enemy["state"] = "RETURN"
                    path_start_node = self._get_closest_node_to_pos(enemy["pos_3d"])
                    path_end_node = enemy["path"][0] if enemy["path"] else self._get_closest_node_to_pos(enemy["pos_3d"])
                    enemy["path"] = self._find_path(path_start_node, path_end_node)
                    enemy["path_idx"] = 0
                else:
                    enemy["chase_timer"] -= 1
                    new_path = self._find_path(self._get_closest_node_to_pos(enemy["pos_3d"]), self.player_node_from)
                    if new_path:
                        enemy["path"] = new_path
                        enemy["path_idx"] = 0

            elif enemy["state"] == "RETURN":
                if enemy["progress"] >= 1.0:
                    if self._get_closest_node_to_pos(enemy["pos_3d"]) == enemy["path"][-1]:
                        enemy["state"] = "PATROL"
                        edge = self.np_random.choice(self.tetra_adj[enemy["path"][-1]])
                        enemy["path"] = [enemy["path"][-1], edge]
                        enemy["path_idx"] = 0

            # --- Movement Logic ---
            enemy["progress"] += enemy["speed"]
            if enemy["progress"] >= 1.0:
                enemy["progress"] = 0.0
                enemy["path_idx"] += 1
                if enemy["path"] and enemy["path_idx"] >= len(enemy["path"]) - 1:
                    current_node = enemy["path"][-1]
                    if enemy["state"] == "PATROL":
                        adj_nodes = self.tetra_adj[current_node]
                        prev_node = enemy["path"][-2] if len(enemy["path"]) > 1 else -1
                        possible_next = [n for n in adj_nodes if n != prev_node]
                        next_node = self.np_random.choice(possible_next) if possible_next else self.np_random.choice(adj_nodes)
                        enemy["path"] = [current_node, next_node]
                    enemy["path_idx"] = 0

            # --- Position Update (FIXED) ---
            if not enemy["path"] or enemy["path_idx"] + 1 >= len(enemy["path"]):
                if enemy["path"]:
                    last_node_idx = enemy["path"][-1]
                    enemy["pos_3d"] = self.tetra_vertices[last_node_idx]
                enemy["pulse"] += 0.1
                continue

            from_node_idx = enemy["path"][enemy["path_idx"]]
            to_node_idx = enemy["path"][enemy["path_idx"] + 1]
            p_from = self.tetra_vertices[from_node_idx]
            p_to = self.tetra_vertices[to_node_idx]
            enemy["pos_3d"] = p_from + (p_to - p_from) * enemy["progress"]
            enemy["pulse"] += 0.1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

    def _check_events_and_collisions(self):
        # --- Player-Exit Collision ---
        if self.player_node_from == self.exit_node and self.player_move_progress >= 1.0:
            self.reward_this_step += 100.0
            self.game_over = True
            self.last_win = True

        # --- Player-Enemy Collision ---
        if not self.player_is_ethereal:
            player_edge = tuple(sorted((self.player_node_from, self.player_node_to)))
            for enemy in self.enemies:
                if not enemy["path"] or enemy["path_idx"] + 1 >= len(enemy["path"]):
                    continue
                enemy_from_node = enemy["path"][enemy["path_idx"]]
                enemy_to_node = enemy["path"][enemy["path_idx"] + 1]
                enemy_edge = tuple(sorted((enemy_from_node, enemy_to_node)))
                
                if player_edge == enemy_edge:
                    dist = abs(self.player_move_progress - enemy["progress"])
                    if self.player_node_from != enemy_from_node:
                        dist = abs(self.player_move_progress - (1.0 - enemy["progress"]))
                    
                    if dist < 0.1: # Collision threshold
                        self.reward_this_step -= 10.0
                        self.game_over = True
                        self.last_win = False
                        return

    def _calculate_reward(self):
        continuous_reward = 0.0
        is_near_enemy = False
        for enemy in self.enemies:
            dist = self._graph_distance(self.player_node_from, self._get_closest_node_to_pos(enemy["pos_3d"]))
            if dist < 3:
                is_near_enemy = True
                break
        
        if self.player_is_ethereal and not is_near_enemy:
            continuous_reward += 0.01
        elif not self.player_is_ethereal and is_near_enemy:
            continuous_reward -= 0.1
        
        dist_before = self._graph_distance(self.player_node_to, self.exit_node)
        dist_after = self._graph_distance(self.player_node_from, self.exit_node)
        if dist_before < dist_after:
            continuous_reward += 0.05

        return self.reward_this_step + continuous_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        projected_vertices = [self._project_3d_to_2d(v) for v in self.tetra_vertices]
        
        for v1_idx, v2_idx in self.tetra_edges:
            p1 = projected_vertices[v1_idx]
            p2 = projected_vertices[v2_idx]
            pygame.draw.aaline(self.screen, self.COLOR_WIRE, p1, p2)
        
        exit_pos_2d = projected_vertices[self.exit_node]
        self._draw_glowing_rect(self.screen, exit_pos_2d, 15, self.COLOR_EXIT, 3)

        for enemy in self.enemies:
            pos_2d = self._project_3d_to_2d(enemy["pos_3d"])
            size = 8 + 2 * math.sin(enemy["pulse"])
            self._draw_glowing_circle(self.screen, pos_2d, size, self.COLOR_ENEMY, 3)

        player_pos_2d = self._project_3d_to_2d(self.player_pos_3d)
        player_color = self.COLOR_PLAYER_ETHEREAL if self.player_is_ethereal else self.COLOR_PLAYER_SOLID
        self._draw_glowing_circle(self.screen, player_pos_2d, 10, player_color, 4)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 40.0))))
            color = (*p["color"], alpha)
            radius = int(5 * (1 - p["life"] / 40.0))
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

    def _render_ui(self):
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        if self.game_over:
            msg = "LEVEL COMPLETE" if self.last_win else "GAME OVER"
            color = self.COLOR_EXIT if self.last_win else self.COLOR_ENEMY
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_is_ethereal": self.player_is_ethereal,
        }

    def _project_3d_to_2d(self, p_3d):
        iso_x = p_3d[0] - p_3d[2]
        iso_y = (p_3d[0] + p_3d[2]) / 2 - p_3d[1]
        
        scale = 1.0
        screen_x = self.WIDTH / 2 + iso_x * scale
        screen_y = self.HEIGHT / 2 + iso_y * scale
        return np.array([screen_x, screen_y])

    def _graph_distance(self, start_node, end_node):
        if start_node == end_node: return 0
        q = deque([(start_node, 0)])
        visited = {start_node}
        while q:
            curr, dist = q.popleft()
            if curr == end_node:
                return dist
            for neighbor in self.tetra_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, dist + 1))
        return float('inf')

    def _find_path(self, start_node, end_node):
        if start_node == end_node: return [start_node]
        q = deque([(start_node, [start_node])])
        visited = {start_node}
        while q:
            curr, path = q.popleft()
            for neighbor in self.tetra_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = list(path)
                    new_path.append(neighbor)
                    if neighbor == end_node:
                        return new_path
                    q.append((neighbor, new_path))
        return []

    def _get_closest_node_to_pos(self, pos_3d):
        dists = [np.linalg.norm(pos_3d - v) for v in self.tetra_vertices]
        return np.argmin(dists)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_layers):
        pos_int = (int(pos[0]), int(pos[1]))
        for i in range(glow_layers, 0, -1):
            alpha = 100 / (i**2)
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius + i*2), glow_color)
        pygame.gfxdraw.aacircle(surface, pos_int[0], pos_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, pos_int[0], pos_int[1], int(radius), color)
        
    def _draw_glowing_rect(self, surface, center_pos, size, color, glow_layers):
        for i in range(glow_layers, 0, -1):
            alpha = 100 / (i**2)
            glow_color = (*color, int(alpha))
            r = pygame.Rect(0, 0, size + i*4, size + i*4)
            r.center = center_pos
            pygame.draw.rect(surface, glow_color, r, border_radius=3)
        
        r_main = pygame.Rect(0, 0, size, size)
        r_main.center = center_pos
        pygame.draw.rect(surface, color, r_main, border_radius=2)

if __name__ == '__main__':
    # Pygame setup for interactive mode
    # This check is to ensure we are not in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        del os.environ["SDL_VIDEODRIVER"]
    
    pygame.init()
    pygame.font.init()

    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tetrahedral Transience")
    clock = pygame.time.Clock()

    movement = 0
    space_held = 0
    shift_held = 0

    print("\n--- Controls ---")
    print("Arrows: Move")
    print("Space: Toggle Ethereal Form")
    print("Shift: Flip Gravity")
    print("R: Reset Environment")
    print("----------------\n")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            # Render the final frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Pause for 2 seconds on game over before auto-reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()