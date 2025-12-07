
# Generated: 2025-08-28T01:55:57.134351
# Source Brief: brief_04282.md
# Brief Index: 4282

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character. Collect yellow gems, avoid red enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Grab 20 gems before the 30-second timer runs out, but watch out for patrolling enemies!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    GRID_SIZE = (20, 20)
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT_SECONDS = 30
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    GEMS_TO_WIN = 20
    NUM_ENEMIES = 5
    NUM_INITIAL_GEMS = 5

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_GEM = (255, 255, 0)
    COLOR_GEM_GLOW = (255, 255, 150)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 150, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 0)

    # Movement timings (in frames)
    PLAYER_MOVE_COOLDOWN = 4 # Player can move every 4 frames
    ENEMY_MOVE_COOLDOWN = 12 # Enemies are slower than the player

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Calculate grid and tile dimensions
        self.tile_size = self.SCREEN_HEIGHT // self.GRID_SIZE[1]
        self.grid_width = self.GRID_SIZE[0] * self.tile_size
        self.grid_height = self.GRID_SIZE[1] * self.tile_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2

        # Initialize state variables
        self.player_pos = None
        self.player_visual_pos = None
        self.gems = None
        self.enemies = None
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.player_move_timer = 0
        self.enemy_move_timer = 0
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()

    def _get_grid_coords(self, grid_pos):
        """Converts grid coordinates (x, y) to pixel coordinates."""
        x, y = grid_pos
        return (
            self.grid_offset_x + x * self.tile_size + self.tile_size // 2,
            self.grid_offset_y + y * self.tile_size + self.tile_size // 2,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.player_move_timer = 0
        self.enemy_move_timer = 0
        self.particles = []

        # Generate initial positions
        occupied_positions = set()

        # Player
        self.player_pos = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        occupied_positions.add(self.player_pos)
        self.player_visual_pos = np.array(self._get_grid_coords(self.player_pos), dtype=float)

        # Gems
        self.gems = set()
        while len(self.gems) < self.NUM_INITIAL_GEMS:
            pos = (
                self.np_random.integers(0, self.GRID_SIZE[0]),
                self.np_random.integers(0, self.GRID_SIZE[1]),
            )
            if pos not in occupied_positions:
                self.gems.add(pos)
                occupied_positions.add(pos)

        # Enemies
        self.enemies = []
        while len(self.enemies) < self.NUM_ENEMIES:
            pos = (
                self.np_random.integers(0, self.GRID_SIZE[0]),
                self.np_random.integers(0, self.GRID_SIZE[1]),
            )
            if pos not in occupied_positions:
                # Assign a random patrol axis (0 for horizontal, 1 for vertical)
                patrol_axis = self.np_random.integers(0, 2)
                self.enemies.append({
                    "pos": pos,
                    "visual_pos": np.array(self._get_grid_coords(pos), dtype=float),
                    "patrol_axis": patrol_axis,
                    "direction": 1 if self.np_random.random() > 0.5 else -1
                })
                occupied_positions.add(pos)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        self.time_remaining -= 1
        self.player_move_timer -= 1
        self.enemy_move_timer -= 1

        reward = 0
        
        # --- Update Game Logic ---
        
        # 1. Move Player
        dist_before = self._get_dist_to_nearest_gem()
        
        if self.player_move_timer <= 0:
            moved = self._move_player(movement)
            if moved:
                self.player_move_timer = self.PLAYER_MOVE_COOLDOWN
        
        dist_after = self._get_dist_to_nearest_gem()
        
        # Continuous reward for moving towards a gem
        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 1.0
            else:
                reward -= 0.1

        # 2. Move Enemies
        if self.enemy_move_timer <= 0:
            self._move_enemies()
            self.enemy_move_timer = self.ENEMY_MOVE_COOLDOWN

        # 3. Handle Interactions & State Changes
        # Gem collection
        if self.player_pos in self.gems:
            # SFX: Gem collected!
            self.gems.remove(self.player_pos)
            self.score += 1
            reward += 10.0
            self._create_particle_burst(self._get_grid_coords(self.player_pos), 20, self.COLOR_GEM)
            self._spawn_gem()

        # Enemy collision
        for enemy in self.enemies:
            if self.player_pos == enemy["pos"]:
                # SFX: Player hit!
                self.game_over = True
                reward = -100.0
                self._create_particle_burst(self.player_visual_pos, 50, self.COLOR_PLAYER)
                break
        
        # 4. Update Animations
        self._update_visuals()
        self._update_particles()
        
        # 5. Check Termination Conditions
        terminated = self.game_over
        if self.time_remaining <= 0:
            terminated = True
        if self.score >= self.GEMS_TO_WIN:
            # SFX: Victory!
            terminated = True
            reward += 50.0 # Victory bonus
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_player(self, movement):
        px, py = self.player_pos
        moved = False
        if movement == 1 and py > 0:  # Up
            py -= 1
            moved = True
        elif movement == 2 and py < self.GRID_SIZE[1] - 1:  # Down
            py += 1
            moved = True
        elif movement == 3 and px > 0:  # Left
            px -= 1
            moved = True
        elif movement == 4 and px < self.GRID_SIZE[0] - 1:  # Right
            px += 1
            moved = True
        
        if moved:
            self.player_pos = (px, py)
        return moved
        
    def _move_enemies(self):
        for enemy in self.enemies:
            ex, ey = enemy["pos"]
            
            if enemy["patrol_axis"] == 0: # Horizontal patrol
                new_ex = ex + enemy["direction"]
                if not (0 <= new_ex < self.GRID_SIZE[0]):
                    enemy["direction"] *= -1
                    new_ex = ex + enemy["direction"]
                enemy["pos"] = (new_ex, ey)
            else: # Vertical patrol
                new_ey = ey + enemy["direction"]
                if not (0 <= new_ey < self.GRID_SIZE[1]):
                    enemy["direction"] *= -1
                    new_ey = ey + enemy["direction"]
                enemy["pos"] = (ex, new_ey)

    def _spawn_gem(self):
        if len(self.gems) >= self.GEMS_TO_WIN: return
        
        occupied = {self.player_pos} | self.gems | {e["pos"] for e in self.enemies}
        
        if len(occupied) >= self.GRID_SIZE[0] * self.GRID_SIZE[1]: return # No space left

        while True:
            pos = (
                self.np_random.integers(0, self.GRID_SIZE[0]),
                self.np_random.integers(0, self.GRID_SIZE[1]),
            )
            if pos not in occupied:
                self.gems.add(pos)
                break
    
    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return None
        px, py = self.player_pos
        min_dist = float('inf')
        for gx, gy in self.gems:
            dist = abs(px - gx) + abs(py - gy) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _update_visuals(self):
        # Interpolate player visual position
        target_pos = np.array(self._get_grid_coords(self.player_pos))
        self.player_visual_pos += (target_pos - self.player_visual_pos) * 0.5

        # Interpolate enemy visual positions
        for enemy in self.enemies:
            target_pos = np.array(self._get_grid_coords(enemy["pos"]))
            enemy["visual_pos"] += (target_pos - enemy["visual_pos"]) * 0.25

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_gems()
        self._render_enemies()
        if not self.game_over:
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_SIZE[0] + 1):
            start_pos = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y)
            end_pos = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y + self.grid_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_SIZE[1] + 1):
            start_pos = (self.grid_offset_x, self.grid_offset_y + y * self.tile_size)
            end_pos = (self.grid_offset_x + self.grid_width, self.grid_offset_y + y * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_player(self):
        pos = self.player_visual_pos.astype(int)
        radius = int(self.tile_size * 0.4)
        glow_radius = int(radius * (1.5 + 0.2 * math.sin(self.steps * 0.2)))
        
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 50))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_PLAYER_GLOW, 50))
        
        # Core circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_gems(self):
        size_mod = 0.5 + (math.sin(self.steps * 0.15) + 1) / 4 # 0.5 to 1.0
        radius = int(self.tile_size * 0.2 * size_mod)
        glow_radius = int(self.tile_size * 0.35 * size_mod)
        
        for gem_pos in self.gems:
            pos = self._get_grid_coords(gem_pos)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_GEM_GLOW, 100))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, (*self.COLOR_GEM_GLOW, 100))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_GEM)

    def _render_enemies(self):
        radius = int(self.tile_size * 0.35)
        for enemy in self.enemies:
            pos = enemy["visual_pos"].astype(int)
            # Simple square shape for enemies
            rect = pygame.Rect(0, 0, radius * 2, radius * 2)
            rect.center = pos
            glow_rect = rect.inflate(10, 10)

            # Glow
            glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.COLOR_ENEMY_GLOW, 80), glow_surface.get_rect(), border_radius=5)
            self.screen.blit(glow_surface, glow_rect.topleft)
            
            # Core
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"GEMS: {self.score}/{self.GEMS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left_sec = max(0, self.time_remaining / self.FPS)
        timer_color = self.COLOR_TEXT if time_left_sec > 5 else self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(f"TIME: {time_left_sec:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WON!" if self.score >= self.GEMS_TO_WIN else "GAME OVER"
            color = self.COLOR_PLAYER if self.score >= self.GEMS_TO_WIN else self.COLOR_ENEMY
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, end_rect.inflate(20, 20))
            self.screen.blit(end_text, end_rect)

    def _create_particle_burst(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(10, 25)
            self.particles.append({
                "pos": list(pos),
                "vel": velocity,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95 # Drag
            p["vel"][1] *= 0.95
            p["lifetime"] -= 1
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
            color = (*p["color"], alpha)
            radius = int(p["lifetime"] / p["max_lifetime"] * 4)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "gems_collected": self.score,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually, you need to map keyboard inputs to the action space
    # This is a simple example of how to do that.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Grid Grab")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Input ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # The MultiDiscrete action space requires all parts
        action = [movement, 0, 0] # space and shift are not used in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # In a real scenario, you might wait for a key press to reset
            # For this demo, we'll just show the final screen for a bit
            # then allow 'r' to reset.
            pass

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()