import os
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Space to attack in your last moved direction. "
        "Reach the blue portal to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A grid-based dungeon crawler. Navigate the maze, defeat enemies, collect gold, and find the exit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.SCREEN_WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.SCREEN_HEIGHT // self.TILE_SIZE

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (10, 5, 5)
        self.COLOR_WALL = (40, 40, 50)
        self.COLOR_FLOOR = (70, 60, 50)
        self.COLOR_HERO = (50, 200, 50)
        self.COLOR_ENEMY = (200, 50, 50)
        self.COLOR_GOLD = (255, 223, 0)
        self.COLOR_EXIT = (50, 100, 255)
        self.COLOR_UI_BG = (20, 20, 30, 200)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.COLOR_HEALTH_FULL = (40, 180, 40)
        self.COLOR_HEALTH_EMPTY = (80, 20, 20)

        # Game state variables are initialized in reset()
        self.grid = []
        self.hero_pos = [0, 0]
        self.hero_health = 0
        self.hero_max_health = 5
        self.hero_facing = (0, 1)  # (dx, dy)
        self.enemies = []
        self.gold_piles = []
        self.exit_pos = [0, 0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        self.last_dist_to_exit = 0

        # Initialize state
        # The first reset call needs a seed to create self.np_random
        self.reset(seed=0)
        
        # Run validation check
        self.validate_implementation()

    def _generate_dungeon(self):
        # 1: Wall, 0: Floor
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Randomized DFS for maze generation
        stack = []
        start_x, start_y = (1, 1)
        self.grid[start_x, start_y] = 0
        stack.append((start_x, start_y))

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and self.grid[nx, ny] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                idx = self.np_random.integers(len(neighbors))
                nx, ny = neighbors[idx]
                self.grid[nx, ny] = 0
                self.grid[x + (nx - x) // 2, y + (ny - y) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _populate_dungeon(self):
        floor_tiles = np.argwhere(self.grid == 0).tolist()
        self.np_random.shuffle(floor_tiles)

        # Place hero
        self.hero_pos = floor_tiles.pop(0)
        self.hero_health = self.hero_max_health
        self.hero_facing = (0, 1)

        # Place exit (furthest from hero)
        max_dist = -1
        for tile in floor_tiles:
            dist = abs(tile[0] - self.hero_pos[0]) + abs(tile[1] - self.hero_pos[1])
            if dist > max_dist:
                max_dist = dist
                self.exit_pos = tile
        floor_tiles.remove(self.exit_pos)
        self.last_dist_to_exit = max_dist

        # Place gold
        self.gold_piles = []
        for _ in range(10):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            self.gold_piles.append(pos)
        
        # Place enemies
        self.enemies = []
        num_enemies = 10
        for _ in range(num_enemies):
            if not floor_tiles: break
            pos = floor_tiles.pop(0)
            # Ensure enemy is not too close to the start
            if abs(pos[0] - self.hero_pos[0]) + abs(pos[1] - self.hero_pos[1]) > 3:
                dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                patrol_dir = dirs[self.np_random.integers(len(dirs))]
                p1 = pos
                p2 = [pos[0] + patrol_dir[0], pos[1] + patrol_dir[1]]
                if self.grid[p2[0], p2[1]] == 1: # If patrol hits a wall, reverse
                    p2 = [pos[0] - patrol_dir[0], pos[1] - patrol_dir[1]]
                
                self.enemies.append({
                    "pos": p1,
                    "patrol_1": p1,
                    "patrol_2": p2,
                    "target": "patrol_2",
                    "health": 1
                })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        self._generate_dungeon()
        self._populate_dungeon()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        player_acted = False

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Player Action Phase ---
        if movement != 0:
            player_acted = True
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            if dx != 0 or dy != 0:
                self.hero_facing = (dx, dy)
            next_pos = [self.hero_pos[0] + dx, self.hero_pos[1] + dy]
            if self.grid[next_pos[0], next_pos[1]] == 0:
                self.hero_pos = next_pos
        
        if space_held:
            player_acted = True
            # Attack in facing direction
            attack_pos = [self.hero_pos[0] + self.hero_facing[0], self.hero_pos[1] + self.hero_facing[1]]
            for enemy in self.enemies[:]:
                if enemy["pos"] == attack_pos:
                    # sfx: player_attack_hit
                    self.enemies.remove(enemy)
                    reward += 1.0
                    self.score += 10
                    self._create_particles(attack_pos, self.COLOR_ENEMY, 15)
                    break
            else:
                # sfx: player_attack_miss
                self._create_particles(attack_pos, (100,100,100), 5, speed=0.5)


        if movement == 0 and not space_held: # Explicit wait action
            player_acted = True
            
        # --- World Update Phase (only if player acted) ---
        if player_acted:
            # Update enemies
            for enemy in self.enemies:
                target_pos = enemy[enemy["target"]]
                if enemy["pos"] == target_pos:
                    enemy["target"] = "patrol_1" if enemy["target"] == "patrol_2" else "patrol_2"
                    target_pos = enemy[enemy["target"]]
                
                dx = np.sign(target_pos[0] - enemy["pos"][0])
                dy = np.sign(target_pos[1] - enemy["pos"][1])
                
                # Move enemy only if the next tile is a floor
                next_enemy_pos = [enemy["pos"][0] + dx, enemy["pos"][1] + dy]
                if self.grid[next_enemy_pos[0], next_enemy_pos[1]] == 0:
                    enemy["pos"] = next_enemy_pos
            
            # Check for collisions and pickups
            for enemy in self.enemies:
                if enemy["pos"] == self.hero_pos:
                    # sfx: player_hurt
                    self.hero_health -= 1
                    reward -= 0.5 # Small penalty for taking damage
                    self._create_particles(self.hero_pos, self.COLOR_HERO, 20)
                    
            if self.hero_pos in self.gold_piles:
                # sfx: gold_pickup
                self.gold_piles.remove(self.hero_pos)
                reward += 5.0
                self.score += 50
                self._create_particles(self.hero_pos, self.COLOR_GOLD, 20)

            # Distance-to-exit reward
            current_dist = abs(self.hero_pos[0] - self.exit_pos[0]) + abs(self.hero_pos[1] - self.exit_pos[1])
            if current_dist < self.last_dist_to_exit:
                reward += 0.1
            elif current_dist > self.last_dist_to_exit:
                reward -= 0.1
            self.last_dist_to_exit = current_dist

        # Update step counter
        self.steps += 1
        
        # --- Termination Check ---
        if self.hero_health <= 0:
            # sfx: game_over_death
            reward = -100.0
            terminated = True
            self.game_over = True
        
        if self.hero_pos == self.exit_pos:
            # sfx: game_over_win
            reward = 100.0
            terminated = True
            self.game_over = True
            self.score += 1000

        if self.steps >= 1000:
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, pos, color, count, speed=2.0):
        px, py = (pos[0] + 0.5) * self.TILE_SIZE, (pos[1] + 0.5) * self.TILE_SIZE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = self.np_random.uniform(0.5, 1.5) * speed
            self.particles.append({
                "x": px, "y": py,
                "vx": math.cos(angle) * vel, "vy": math.sin(angle) * vel,
                "lifetime": self.np_random.integers(10, 21), # .integers is exclusive of high value
                "color": color
            })
    
    def _render_dungeon(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_WALL if self.grid[x, y] == 1 else self.COLOR_FLOOR
                pygame.draw.rect(self.screen, color, rect)
    
    def _render_entities(self):
        anim_offset = math.sin(self.steps * 0.2) * 2

        # Exit portal
        ex, ey = (self.exit_pos[0] + 0.5) * self.TILE_SIZE, (self.exit_pos[1] + 0.5) * self.TILE_SIZE
        for i in range(4):
            radius = self.TILE_SIZE * 0.4 + math.sin(self.steps * 0.1 + i * math.pi / 4) * 2
            alpha = 100 + math.sin(self.steps * 0.15 + i) * 50
            color = (*self.COLOR_EXIT, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(ex), int(ey), int(radius - i * 2), color)

        # Gold
        for pos in self.gold_piles:
            gx, gy = (pos[0] + 0.5) * self.TILE_SIZE, (pos[1] + 0.5) * self.TILE_SIZE
            sparkle_size = 2 + math.sin(self.steps * 0.3 + pos[0])
            pygame.draw.rect(self.screen, self.COLOR_GOLD, (gx - sparkle_size, gy - sparkle_size, sparkle_size*2, sparkle_size*2))

        # Enemies
        for enemy in self.enemies:
            ex, ey = (enemy["pos"][0] + 0.5) * self.TILE_SIZE, (enemy["pos"][1] + 0.5) * self.TILE_SIZE + anim_offset
            size = self.TILE_SIZE * 0.6
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (ex - size/2, ey - size/2, size, size))

        # Hero
        hx, hy = (self.hero_pos[0] + 0.5) * self.TILE_SIZE, (self.hero_pos[1] + 0.5) * self.TILE_SIZE + anim_offset / 2
        size = self.TILE_SIZE * 0.7
        pygame.draw.rect(self.screen, self.COLOR_HERO, (hx - size/2, hy - size/2, size, size))
        # Facing indicator
        fx = hx + self.hero_facing[0] * size * 0.4
        fy = hy + self.hero_facing[1] * size * 0.4
        pygame.draw.circle(self.screen, (255, 255, 255), (int(fx), int(fy)), 2)

    def _render_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                size = int(p['lifetime'] * 0.2)
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), max(0, size))

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Health Bar
        health_text = self.font_small.render("HP", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (10, 10))
        bar_width = 100
        bar_height = 15
        for i in range(self.hero_max_health):
            color = self.COLOR_HEALTH_FULL if i < self.hero_health else self.COLOR_HEALTH_EMPTY
            pygame.draw.rect(self.screen, color, (45 + i * (bar_width / self.hero_max_health + 2), 10, bar_width / self.hero_max_health, bar_height))

        # Gold/Score
        score_text = self.font_small.render(f"GOLD: {self.score}", True, self.COLOR_UI_TEXT)
        text_rect = score_text.get_rect(right=self.SCREEN_WIDTH - 10, top=10)
        self.screen.blit(score_text, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.hero_health > 0 and self.hero_pos == self.exit_pos else "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_dungeon()
        self._render_entities()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.hero_health,
            "hero_pos": self.hero_pos,
            "exit_pos": self.exit_pos,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        # Test game-specific assertions
        self.reset()
        assert self.hero_health <= self.hero_max_health
        assert self.grid[self.hero_pos[0], self.hero_pos[1]] == 0
        assert self.grid[self.exit_pos[0], self.exit_pos[1]] == 0
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Monkey-patch the render method for human playback
        GameEnv.metadata["render_modes"].append("human")
        # Unset the dummy video driver if we are in human mode
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]

        def render_human(self):
            if not hasattr(self, 'window'):
                pygame.display.init()
                self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            
            # Get the observation frame
            frame = self._get_observation()
            # Pygame uses (width, height), numpy uses (height, width)
            frame = np.transpose(frame, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            self.window.blit(surf, (0, 0))
            pygame.display.flip()
            self.clock.tick(30) # Limit to 30 fps for human viewing
        GameEnv.render = render_human

    env = GameEnv(render_mode=render_mode)
    obs, info = env.reset(seed=42)
    
    # --- Interactive human play loop ---
    if render_mode == "human":
        terminated = False
        while not terminated:
            env.render()
            
            movement = 0 # no-op
            space = 0
            shift = 0
            
            action_taken = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    action_taken = True
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space = 1
                    elif event.key == pygame.K_w: # Wait action
                        movement, space, shift = 0, 0, 0
                    else:
                        action_taken = False

            if action_taken:
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        
        # Show final screen for a moment
        env.render()
        pygame.time.wait(2000)

    env.close()