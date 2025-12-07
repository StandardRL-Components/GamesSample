
# Generated: 2025-08-27T20:20:09.103627
# Source Brief: brief_02420.md
# Brief Index: 2420

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a standard block. "
        "Shift+Space to place a reinforced block. Build to survive the waves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Build a fortress of blocks to withstand increasingly difficult waves of projectile attacks. "
        "Survive 10 waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_WIDTH = SCREEN_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    
    FORTRESS_ROWS = 3 # The bottom N rows are the fortress area

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_BLOCK_STANDARD = (120, 130, 140)
    COLOR_BLOCK_REINFORCED = (80, 90, 100)
    COLOR_PROJECTILE = (255, 80, 80)
    COLOR_PARTICLE = (255, 200, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_HEALTH_FULL = (0, 200, 0)
    COLOR_HEALTH_EMPTY = (200, 0, 0)
    COLOR_FORTRESS_ZONE = (25, 38, 50)

    # Game parameters
    MAX_WAVES = 10
    MAX_STEPS = 3000
    MAX_FORTRESS_HEALTH = 20
    BUILD_TIME_FRAMES = 30 * 8 # 8 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # Initialize state variables to be set in reset()
        self.grid = None
        self.cursor_pos = None
        self.projectiles = None
        self.particles = None
        self.game_phase = None
        self.wave_number = None
        self.score = None
        self.fortress_health = None
        self.build_timer = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = False
        self.damage_this_wave = False
        self.np_random = None

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Grid: [type, health]. 0=empty, 1=standard, 2=reinforced
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS, 2), dtype=np.int32)
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.projectiles = []
        self.particles = []
        
        self.game_phase = "BUILD"
        self.wave_number = 1
        self.score = 0
        self.fortress_health = self.MAX_FORTRESS_HEALTH
        
        self.build_timer = self.BUILD_TIME_FRAMES
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        self.damage_this_wave = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        if self.game_phase == "BUILD":
            self.build_timer -= 1
            if self.build_timer <= 0:
                self.game_phase = "ATTACK"
                self._spawn_wave()
        
        elif self.game_phase == "ATTACK":
            reward += self._update_projectiles()
            if not self.projectiles: # Wave is over
                # Wave survival rewards
                if not self.damage_this_wave:
                    reward += 5.0 # No damage bonus
                    self.score += 5
                reward += 1.0 # Wave survival bonus
                self.score += 1

                self.wave_number += 1
                if self.wave_number > self.MAX_WAVES:
                    self.game_over = True # VICTORY
                    reward += 50.0
                else:
                    self.game_phase = "BUILD"
                    self.build_timer = self.BUILD_TIME_FRAMES
                    self.damage_this_wave = False

        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and self.fortress_health <= 0:
            reward = -100.0 # Heavy penalty for fortress destruction
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement (wraps around) ---
        if movement == 1: # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_ROWS) % self.GRID_ROWS
        elif movement == 2: # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_ROWS
        elif movement == 3: # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_COLS) % self.GRID_COLS
        elif movement == 4: # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_COLS
        
        # --- Block Placement (only in BUILD phase on key press) ---
        space_pressed = space_held and not self.prev_space_held
        if self.game_phase == "BUILD" and space_pressed:
            cx, cy = self.cursor_pos
            # Cannot build in the top row (spawn area) or on existing blocks
            if cy > 0 and self.grid[cy, cx, 0] == 0:
                block_type = 2 if shift_held else 1
                block_health = 2 if block_type == 2 else 1
                self.grid[cy, cx, 0] = block_type
                self.grid[cy, cx, 1] = block_health
                # sfx: place_block.wav
        
        self.prev_space_held = space_held

    def _spawn_wave(self):
        num_projectiles = 2 + self.wave_number
        speed = 2.0 + (self.wave_number - 1) * 0.2

        for _ in range(num_projectiles):
            spawn_x = self.np_random.uniform(self.CELL_WIDTH, self.SCREEN_WIDTH - self.CELL_WIDTH)
            spawn_y = -self.CELL_HEIGHT
            
            # Target a random point in the fortress zone
            target_x = self.np_random.uniform(0, self.SCREEN_WIDTH)
            target_y = self.np_random.uniform(self.SCREEN_HEIGHT - self.FORTRESS_ROWS * self.CELL_HEIGHT, self.SCREEN_HEIGHT)
            
            angle = math.atan2(target_y - spawn_y, target_x - spawn_x)
            
            self.projectiles.append({
                "pos": np.array([spawn_x, spawn_y], dtype=float),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float)
            })
        # sfx: wave_start.wav

    def _update_projectiles(self):
        step_reward = 0
        projectiles_to_remove = []

        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            px, py = proj["pos"]
            
            # Check for collision with grid cells
            grid_x, grid_y = int(px / self.CELL_WIDTH), int(py / self.CELL_HEIGHT)
            
            if 0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS:
                if self.grid[grid_y, grid_x, 0] > 0: # Hit a block
                    self.grid[grid_y, grid_x, 1] -= 1
                    if self.grid[grid_y, grid_x, 1] <= 0:
                        self.grid[grid_y, grid_x, 0] = 0 # Destroy block
                        # sfx: block_destroy.wav
                    
                    self._create_impact_particles(proj["pos"])
                    projectiles_to_remove.append(proj)
                    step_reward += 0.1 # Deflection reward
                    self.score += 0.1
                    # sfx: block_hit.wav
                    continue

            # Check for collision with fortress base (if it passes through)
            if py >= self.SCREEN_HEIGHT - self.FORTRESS_ROWS * self.CELL_HEIGHT:
                self.fortress_health -= 1
                self.damage_this_wave = True
                self._create_impact_particles(proj["pos"], count=20, color=(255, 50, 50))
                projectiles_to_remove.append(proj)
                # sfx: fortress_hit.wav
                continue

            # Check for out of bounds (sides or bottom)
            if not (0 <= px < self.SCREEN_WIDTH and py < self.SCREEN_HEIGHT):
                projectiles_to_remove.append(proj)

        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return step_reward

    def _create_impact_particles(self, pos, count=15, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": pos.copy(),
                "vel": np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                "life": self.np_random.integers(15, 25),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"] *= 0.95 # Damping
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _check_termination(self):
        if self.fortress_health <= 0 or self.steps >= self.MAX_STEPS or self.wave_number > self.MAX_WAVES:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Fortress zone
        fortress_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.FORTRESS_ROWS * self.CELL_HEIGHT, self.SCREEN_WIDTH, self.FORTRESS_ROWS * self.CELL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS_ZONE, fortress_rect)

        # Grid lines
        for r in range(self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, r * self.CELL_HEIGHT), (self.SCREEN_WIDTH, r * self.CELL_HEIGHT))
        for c in range(self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (c * self.CELL_WIDTH, 0), (c * self.CELL_WIDTH, self.SCREEN_HEIGHT))

        # Blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block_type = self.grid[r, c, 0]
                if block_type > 0:
                    color = self.COLOR_BLOCK_STANDARD if block_type == 1 else self.COLOR_BLOCK_REINFORCED
                    rect = pygame.Rect(c * self.CELL_WIDTH, r * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
                    pygame.draw.rect(self.screen, color, rect)
                    # Indicate damage on reinforced blocks
                    if block_type == 2 and self.grid[r, c, 1] == 1:
                         pygame.draw.rect(self.screen, (255,255,255,50), rect, 2)

        # Cursor
        if self.game_phase == "BUILD":
            cx, cy = self.cursor_pos
            cursor_rect = pygame.Rect(cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT, self.CELL_WIDTH, self.CELL_HEIGHT)
            pygame.gfxdraw.box(self.screen, cursor_rect, self.COLOR_CURSOR)

        # Projectiles
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 5, self.COLOR_PROJECTILE)

        # Particles
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = max(1, int(p["life"] / 5))
            # Use a surface with SRCALPHA for per-pixel alpha
            particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)
            alpha = int(255 * (p["life"] / 25))
            color = p["color"]
            pygame.draw.rect(particle_surf, color + (alpha,), (0, 0, size, size))
            self.screen.blit(particle_surf, (pos[0]-size//2, pos[1]-size//2))

    def _render_ui(self):
        # UI Background with alpha
        ui_bg_surf = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        ui_bg_surf.fill((0,0,0,150))
        self.screen.blit(ui_bg_surf, (0,0))

        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 8))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        wave_text_rect = wave_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        wave_text_rect.y = 8
        self.screen.blit(wave_text, wave_text_rect)

        # Fortress Health
        health_text = self.font_small.render("FORTRESS HP:", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - 210, 8))
        
        # Health Bar
        health_ratio = max(0, self.fortress_health / self.MAX_FORTRESS_HEALTH)
        health_bar_width = 100
        current_health_width = int(health_bar_width * health_ratio)
        health_bar_bg = pygame.Rect(self.SCREEN_WIDTH - 110, 8, health_bar_width, 16)
        health_bar_fg = pygame.Rect(self.SCREEN_WIDTH - 110, 8, current_health_width, 16)
        
        health_color = (
            self.COLOR_HEALTH_EMPTY[0] + (self.COLOR_HEALTH_FULL[0] - self.COLOR_HEALTH_EMPTY[0]) * health_ratio,
            self.COLOR_HEALTH_EMPTY[1] + (self.COLOR_HEALTH_FULL[1] - self.COLOR_HEALTH_EMPTY[1]) * health_ratio,
            self.COLOR_HEALTH_EMPTY[2] + (self.COLOR_HEALTH_FULL[2] - self.COLOR_HEALTH_EMPTY[2]) * health_ratio,
        )
        
        pygame.draw.rect(self.screen, (50, 50, 50), health_bar_bg)
        pygame.draw.rect(self.screen, health_color, health_bar_fg)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, health_bar_bg, 1)

        # Phase Status Text
        if self.game_phase == "BUILD":
            timer_text = self.font_large.render(f"BUILD PHASE: {math.ceil(self.build_timer/30)}s", True, self.COLOR_TEXT)
            timer_text_rect = timer_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(timer_text, timer_text_rect)
            
            timer_ratio = self.build_timer / self.BUILD_TIME_FRAMES
            bar_y = self.SCREEN_HEIGHT - 5
            bar_width = self.SCREEN_WIDTH * timer_ratio
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (0, bar_y, bar_width, 5))

        elif self.game_phase == "ATTACK":
            status_text = self.font_large.render("INCOMING!", True, self.COLOR_PROJECTILE)
            status_text_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(status_text, status_text_rect)
            
        # Game Over / Victory Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            color = (100, 255, 100) if self.wave_number > self.MAX_WAVES else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            end_text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "fortress_health": self.fortress_health,
            "phase": self.game_phase,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Ensure Pygame runs headlessly for the env
    env = GameEnv()
    
    # Re-enable video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fortress Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping for human player ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30)
        
    env.close()