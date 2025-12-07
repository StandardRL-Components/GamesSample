import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move in isometric directions. Survive and find the glowing exit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a procedurally generated crypt, battling bats and seeking the exit in this isometric survival horror game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 50, 50
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    MAX_STEPS = 1000
    INITIAL_PLAYER_HEALTH = 3
    INITIAL_ENEMY_COUNT = 5

    # Colors
    COLOR_BG = (10, 8, 12)
    COLOR_FLOOR = (40, 35, 45)
    COLOR_FLOOR_OUTLINE = (55, 50, 60)
    COLOR_WALL = (25, 20, 30)
    COLOR_WALL_OUTLINE = (40, 35, 45)
    COLOR_PLAYER = (255, 0, 80)
    COLOR_PLAYER_GLOW = (255, 0, 80, 50)
    COLOR_ENEMY = (230, 230, 255)
    COLOR_EXIT = (255, 220, 50)
    COLOR_EXIT_GLOW = (255, 220, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_DAMAGE = (255, 50, 50)
    COLOR_KILL_EFFECT = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.floor_tiles = []
        self.enemies = []
        self.particles = []
        
        # We need to reset to initialize the np_random generator
        # before validation can use it.
        self.reset()
        # self.validate_implementation() # Commented out for submission, but useful for local testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_health = self.INITIAL_PLAYER_HEALTH
        self.enemy_spawn_prob = 0.01

        self._generate_map()
        self._place_entities()
        
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # In case step is called after termination, return a final state
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0, True, False, info

        reward = -0.01  # Penalty for taking a step
        
        movement = action[0]
        self._handle_player_movement(movement)
        
        # Automatic attack
        attack_reward = self._handle_player_attack()
        reward += attack_reward

        # Enemy turn and collision damage
        damage_reward = self._handle_enemies()
        reward += damage_reward

        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_spawn_prob = min(0.1, self.enemy_spawn_prob + 0.01)

        # Random enemy spawn
        if self.np_random.random() < self.enemy_spawn_prob:
            self._spawn_enemy()
            
        # self._update_particles() # This was in original code but is not used
        
        terminated = self._check_termination()
        
        if terminated:
            if np.array_equal(self.player_pos, self.exit_pos):
                reward += 100
            elif self.player_health <= 0:
                reward -= 100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        px, py = self.player_pos
        dx, dy = 0, 0
        
        # Isometric movement mapping
        if movement == 1: dx, dy = -1, 0  # Up -> NW
        elif movement == 2: dx, dy = 1, 0   # Down -> SE
        elif movement == 3: dx, dy = 0, 1   # Left -> SW
        elif movement == 4: dx, dy = 0, -1  # Right -> NE

        if movement != 0:
            new_px, new_py = px + dx, py + dy
            if self._is_valid_pos(new_px, new_py):
                self.player_pos = (new_px, new_py)

    def _handle_player_attack(self):
        reward = 0
        enemies_to_remove = []
        px, py = self.player_pos
        
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            # Check adjacency
            if abs(px - ex) + abs(py - ey) == 1:
                enemies_to_remove.append(enemy)
                reward += 5
                self.score += 1
                # Sound: Enemy defeated
                ex_screen, ey_screen = self._world_to_screen(ex, ey)
                self._create_particles((ex_screen, ey_screen), self.COLOR_KILL_EFFECT, 20, 2.0)
        
        if enemies_to_remove:
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        
        return reward

    def _handle_enemies(self):
        reward = 0
        for enemy in self.enemies:
            # Move enemy along patrol path
            if enemy['patrol_path']:
                enemy['patrol_idx'] = (enemy['patrol_idx'] + 1) % len(enemy['patrol_path'])
                enemy['pos'] = enemy['patrol_path'][enemy['patrol_idx']]

            # Check for collision with player
            if np.array_equal(enemy['pos'], self.player_pos):
                self.player_health -= 1
                reward -= 1
                # Sound: Player hurt
                px_screen, py_screen = self._world_to_screen(self.player_pos[0], self.player_pos[1])
                self._create_particles((px_screen, py_screen), self.COLOR_DAMAGE, 30, 3.0)
        
        self.player_health = max(0, self.player_health)
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "player_health": self.player_health}

    def _check_termination(self):
        return (
            self.player_health <= 0 or
            np.array_equal(self.player_pos, self.exit_pos) or
            self.steps >= self.MAX_STEPS
        )

    def _generate_map(self):
        self.grid.fill(0)  # 0 for wall
        self.floor_tiles = []
        
        digger_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        num_tiles_to_carve = int(self.GRID_WIDTH * self.GRID_HEIGHT * 0.4)

        for _ in range(num_tiles_to_carve):
            x, y = digger_pos
            if self.grid[x, y] == 0:
                self.grid[x, y] = 1  # 1 for floor
                self.floor_tiles.append((x, y))
            
            direction = self.np_random.choice(['N', 'S', 'E', 'W'])
            if direction == 'N': digger_pos[1] = max(1, y - 1)
            elif direction == 'S': digger_pos[1] = min(self.GRID_HEIGHT - 2, y + 1)
            elif direction == 'E': digger_pos[0] = min(self.GRID_WIDTH - 2, x + 1)
            elif direction == 'W': digger_pos[0] = max(1, x - 1)
    
    def _place_entities(self):
        start_idx = self.np_random.integers(0, len(self.floor_tiles))
        self.player_pos = self.floor_tiles[start_idx]

        # Find a distant exit
        max_dist = -1
        self.exit_pos = self.player_pos
        for tile in self.floor_tiles:
            dist = abs(tile[0] - self.player_pos[0]) + abs(tile[1] - self.player_pos[1])
            if dist > max_dist:
                max_dist = dist
                self.exit_pos = tile
        
        self.enemies = []
        for _ in range(self.INITIAL_ENEMY_COUNT):
            self._spawn_enemy()
    
    def _spawn_enemy(self):
        if not self.floor_tiles: return
        
        pos = None
        for _ in range(100): # Try 100 times to find a valid spot
            idx = self.np_random.integers(0, len(self.floor_tiles))
            candidate_pos = self.floor_tiles[idx]
            dist_to_player = abs(candidate_pos[0] - self.player_pos[0]) + abs(candidate_pos[1] - self.player_pos[1])
            if dist_to_player > 5 and not np.array_equal(candidate_pos, self.exit_pos):
                pos = candidate_pos
                break
        if pos is None: # Fallback if no suitable spot found
            pos = self.floor_tiles[self.np_random.integers(len(self.floor_tiles))]

        patrol_path = [pos]
        for _ in range(2):
            last_pos = patrol_path[-1]
            neighbors = [(last_pos[0]+dx, last_pos[1]+dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
            valid_neighbors = [n for n in neighbors if self._is_valid_pos(n[0], n[1])]
            if valid_neighbors:
                # np.random.choice on a list of tuples returns a numpy array, so we index manually
                chosen_neighbor = valid_neighbors[self.np_random.integers(len(valid_neighbors))]
                patrol_path.append(chosen_neighbor)
            else:
                patrol_path.append(last_pos)

        self.enemies.append({'pos': pos, 'patrol_path': patrol_path, 'patrol_idx': 0})

    def _is_valid_pos(self, x, y):
        return 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[x, y] == 1

    def _world_to_screen(self, gx, gy):
        sx = (gx - gy) * self.TILE_WIDTH / 2
        sy = (gx + gy) * self.TILE_HEIGHT / 2
        return int(sx), int(sy)

    def _draw_iso_poly(self, surface, sx, sy, color, outline_color=None):
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2),
            (sx, sy + self.TILE_HEIGHT),
            (sx - self.TILE_WIDTH / 2, sy + self.TILE_HEIGHT / 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _draw_iso_rect(self, surface, sx, sy, color, height_offset=0, glow_color=None):
        h = self.TILE_HEIGHT * 1.2
        w = self.TILE_WIDTH * 0.3
        
        if glow_color:
            glow_surf = pygame.Surface((w * 3, h * 3), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, glow_color, glow_surf.get_rect())
            surface.blit(glow_surf, (sx - w, sy - h - height_offset), special_flags=pygame.BLEND_RGBA_ADD)

        rect_points = [
            (sx, sy - h - height_offset),
            (sx + w/2, sy - h/2 - height_offset),
            (sx, sy - height_offset),
            (sx - w/2, sy - h/2 - height_offset)
        ]
        pygame.gfxdraw.filled_polygon(surface, rect_points, color)
        pygame.gfxdraw.aapolygon(surface, rect_points, tuple(min(255, c+30) for c in color[:3]))

    def _render_game(self):
        # Camera offset to center player
        player_sx, player_sy = self._world_to_screen(*self.player_pos)
        offset_x = self.SCREEN_WIDTH / 2 - player_sx
        offset_y = self.SCREEN_HEIGHT / 2 - player_sy

        # Determine visible grid range
        view_dist_x = int(self.SCREEN_WIDTH / self.TILE_WIDTH) + 4
        view_dist_y = int(self.SCREEN_HEIGHT / self.TILE_HEIGHT) + 4
        
        min_gx = max(0, self.player_pos[0] - view_dist_x)
        max_gx = min(self.GRID_WIDTH, self.player_pos[0] + view_dist_x)
        min_gy = max(0, self.player_pos[1] - view_dist_y)
        max_gy = min(self.GRID_HEIGHT, self.player_pos[1] + view_dist_y)

        # Draw floor and walls
        for gy in range(int(min_gy), int(max_gy)):
            for gx in range(int(min_gx), int(max_gx)):
                sx, sy = self._world_to_screen(gx, gy)
                sx += offset_x
                sy += offset_y
                
                if self.grid[gx, gy] == 1: # Floor
                    self._draw_iso_poly(self.screen, sx, sy, self.COLOR_FLOOR, self.COLOR_FLOOR_OUTLINE)
                else: # Wall
                    self._draw_iso_poly(self.screen, sx, sy, self.COLOR_WALL, self.COLOR_WALL_OUTLINE)
        
        # Draw entities (sorted by y for correct overlap)
        entities_to_draw = []
        # Exit
        if self.exit_pos:
            ex, ey = self.exit_pos
            sx, sy = self._world_to_screen(ex, ey)
            entities_to_draw.append({'type': 'exit', 'pos': (sx + offset_x, sy + offset_y), 'sort_key': ex + ey})
        # Enemies
        for enemy in self.enemies:
            ex, ey = enemy['pos']
            sx, sy = self._world_to_screen(ex, ey)
            entities_to_draw.append({'type': 'enemy', 'pos': (sx + offset_x, sy + offset_y), 'sort_key': ex + ey})
        # Player
        entities_to_draw.append({'type': 'player', 'pos': (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2), 'sort_key': self.player_pos[0] + self.player_pos[1]})

        entities_to_draw.sort(key=lambda e: e['sort_key'])

        for entity in entities_to_draw:
            sx, sy = entity['pos']
            if entity['type'] == 'exit':
                self._draw_iso_rect(self.screen, sx, sy, self.COLOR_EXIT, height_offset=self.TILE_HEIGHT/2, glow_color=self.COLOR_EXIT_GLOW)
            elif entity['type'] == 'enemy':
                bob = math.sin(self.steps * 0.3 + sx) * 3
                self._draw_iso_rect(self.screen, sx, sy, self.COLOR_ENEMY, height_offset=self.TILE_HEIGHT/2 + bob)
            elif entity['type'] == 'player':
                bob = math.sin(self.steps * 0.2) * 2
                self._draw_iso_rect(self.screen, sx, sy, self.COLOR_PLAYER, height_offset=self.TILE_HEIGHT/2 + bob, glow_color=self.COLOR_PLAYER_GLOW)

        self._update_and_draw_particles(offset_x, offset_y)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Health
        health_text = self.font_small.render("HEALTH:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 30))
        for i in range(self.INITIAL_PLAYER_HEALTH):
            color = self.COLOR_PLAYER if i < self.player_health else self.COLOR_WALL
            pygame.draw.rect(self.screen, color, (80 + i * 25, 32, 20, 15))

        if self.game_over:
            msg = "YOU ESCAPED!" if np.array_equal(self.player_pos, self.exit_pos) else "YOU DIED"
            color = self.COLOR_EXIT if np.array_equal(self.player_pos, self.exit_pos) else self.COLOR_DAMAGE
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 50))
            self.screen.blit(end_text, text_rect)

    def _create_particles(self, pos, color, count, max_speed=1.0):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(20, 40), 'color': color})

    def _update_and_draw_particles(self, offset_x, offset_y):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(p['life'] * 6)))
                color = p['color'] + (alpha,)
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (pos[0]-2 + offset_x, pos[1]-2 + offset_y), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment.
    # To see the game, comment out the os.environ line at the top.
    
    # Check if we are in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. No visual output will be shown.")
        print("To play the game visually, comment out the line:")
        print("os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')")
        
        # A simple test loop for headless mode
        env = GameEnv()
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        step_count = 0
        while not terminated and step_count < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        print(f"Headless test finished after {step_count} steps. Final reward: {total_reward}")
        env.close()

    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen_main = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Crypt Escape")
        
        terminated = False
        total_reward = 0
        
        # Game loop
        running = True
        clock = pygame.time.Clock()
        
        while running:
            action = np.array([0, 0, 0])  # Default to no-op
            
            # Process events once per frame
            key_pressed_this_frame = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    key_pressed_this_frame = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        terminated = False
                    elif not terminated:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_UP]: action[0] = 1
                        elif keys[pygame.K_DOWN]: action[0] = 2
                        elif keys[pygame.K_LEFT]: action[0] = 3
                        elif keys[pygame.K_RIGHT]: action[0] = 4

            if not terminated and key_pressed_this_frame and action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Terminated: {terminated}")
            
            # Render the observation to the main screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen_main.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(30) # Limit FPS for human playability
            
        env.close()