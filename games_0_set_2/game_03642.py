
# Generated: 2025-08-27T23:58:08.831050
# Source Brief: brief_03642.md
# Brief Index: 3642

        
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
    """
    A real-time tower defense game where the player builds a fortress to withstand waves of enemies.
    The goal is to survive 20 increasingly difficult waves by strategically placing blocks.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. Press space to build a block."
    )

    # Short, user-facing description of the game
    game_description = (
        "Build a block fortress to withstand increasingly difficult waves of enemy attacks."
    )

    # Frames only advance when an action is received
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 20
    CELL_SIZE = SCREEN_HEIGHT // GRID_HEIGHT
    GAME_AREA_WIDTH = GRID_WIDTH * CELL_SIZE # 400px
    UI_AREA_WIDTH = SCREEN_WIDTH - GAME_AREA_WIDTH # 240px

    MAX_STEPS = 2500
    MAX_WAVES = 20
    INITIAL_FORTRESS_HEALTH = 100
    INITIAL_BLOCKS = 50
    BLOCKS_PER_WAVE = 15
    INTER_WAVE_STEPS = 60 # Steps between waves for preparation

    # --- Colors ---
    COLOR_BG = (34, 40, 49) # Dark grey-blue
    COLOR_GRID = (57, 62, 70) # Medium grey
    COLOR_UI_BG = (42, 50, 58)
    COLOR_UI_DIVIDER = (80, 90, 100)
    COLOR_TEXT = (238, 238, 238) # Light grey/white
    COLOR_FORTRESS = (255, 215, 0) # Gold
    COLOR_BLOCK = (0, 173, 181) # Teal
    COLOR_ENEMY = (214, 52, 71) # Red
    COLOR_PROJECTILE = (79, 138, 139) # Muted teal
    COLOR_CURSOR = (255, 255, 255) # White

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
        self.font_main = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_title = pygame.font.SysFont('Consolas', 28, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        
        # Game state variables are initialized in reset()
        self.reset()

        # Critical self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        # Fortress
        self.fortress_health = self.INITIAL_FORTRESS_HEALTH
        self.fortress_pos = [(self.GRID_WIDTH // 2 - 1, self.GRID_HEIGHT - 1), (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)]
        self.fortress_flash_timer = 0
        
        # Player state
        self.blocks_available = self.INITIAL_BLOCKS
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 5]
        
        # World state
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        # Wave mechanics
        self.wave_number = 0
        self.wave_active = False
        self.inter_wave_timer = 0
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        self.steps += 1
        
        # Unpack action
        movement, space_held, _ = action
        
        # --- Handle Player Input ---
        self._handle_input(movement, space_held == 1)
        
        # --- Update Game Logic ---
        self._update_particles()
        self.fortress_flash_timer = max(0, self.fortress_flash_timer - 1)
        
        if self.wave_active:
            # Main combat phase
            self._update_enemies()
            projectile_reward = self._update_projectiles()
            reward += projectile_reward
            
            if not self.enemies:
                # Wave cleared
                reward += 10.0 # Wave survival reward
                self.score += 100 * self.wave_number
                self.wave_active = False
                self.inter_wave_timer = self.INTER_WAVE_STEPS
                self.blocks_available += self.BLOCKS_PER_WAVE
        else:
            # Inter-wave preparation phase
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                if self.wave_number >= self.MAX_WAVES:
                    self.win = True
                    self.game_over = True
                    reward += 100.0 # Game win reward
                else:
                    self._start_new_wave()
        
        # Continuous survival reward
        reward += 0.1
        self.score += 1
        
        # --- Check Termination ---
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated always False
            self._get_info()
        )

    def _start_new_wave(self):
        self.wave_number += 1
        self.wave_active = True
        
        # Spawn enemies
        num_enemies = self.wave_number
        spawn_locations = random.sample(range(self.GRID_WIDTH), k=num_enemies)
        
        for x in spawn_locations:
            self.enemies.append({
                "pos": [x + 0.5, 0.5],
                "fire_cooldown": random.randint(30, 60)
            })

    def _handle_input(self, movement, place_block):
        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1
        
        # Place block
        if place_block and self.blocks_available > 0:
            cx, cy = self.cursor_pos
            is_fortress_pos = (cx, cy) in self.fortress_pos
            if self.grid[cx, cy] == 0 and not is_fortress_pos:
                self.grid[cx, cy] = 1
                self.blocks_available -= 1
                # sfx: block_place.wav
                self._create_particles(cx + 0.5, cy + 0.5, 10, self.COLOR_BLOCK)

    def _update_enemies(self):
        fortress_center_x = self.GRID_WIDTH / 2
        for enemy in self.enemies:
            # --- Movement ---
            target_pos = [fortress_center_x, self.GRID_HEIGHT - 1]
            dx = target_pos[0] - enemy["pos"][0]
            dy = target_pos[1] - enemy["pos"][1]
            
            # Simple pathfinding: prioritize down, then sideways
            next_grid_y = math.floor(enemy["pos"][1] + 0.1)
            if next_grid_y < self.GRID_HEIGHT - 1 and self.grid[int(enemy["pos"][0]), next_grid_y + 1] == 0:
                enemy["pos"][1] += 0.05
            else:
                if dx > 0.1 and self.grid[int(enemy["pos"][0] + 1), int(enemy["pos"][1])] == 0:
                    enemy["pos"][0] += 0.05
                elif dx < -0.1 and self.grid[int(enemy["pos"][0] - 1), int(enemy["pos"][1])] == 0:
                    enemy["pos"][0] -= 0.05
            
            # --- Firing ---
            enemy["fire_cooldown"] -= 1
            if enemy["fire_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                projectile_speed = 0.1 + (self.wave_number // 5) * 0.05
                self.projectiles.append({
                    "pos": list(enemy["pos"]),
                    "vel": [0, projectile_speed] # Fire straight down
                })
                enemy["fire_cooldown"] = 100 - min(80, self.wave_number * 2)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for proj in self.projectiles:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
            
            px, py = int(proj["pos"][0]), int(proj["pos"][1])

            # Check boundaries
            if not (0 <= px < self.GRID_WIDTH and 0 <= py < self.GRID_HEIGHT):
                projectiles_to_remove.append(proj)
                continue
            
            # Check collision with blocks
            if self.grid[px, py] == 1:
                projectiles_to_remove.append(proj)
                self.grid[px, py] = 0 # Block is destroyed
                self._create_particles(px + 0.5, py + 0.5, 20, self.COLOR_BLOCK)
                # sfx: block_destroy.wav
                reward += 1.0 # Reward for destroying projectile
                continue
            
            # Check collision with fortress
            if (px, py) in self.fortress_pos:
                projectiles_to_remove.append(proj)
                self.fortress_health -= 5
                self.fortress_flash_timer = 10
                self._create_particles(px + 0.5, py + 0.5, 20, self.COLOR_FORTRESS)
                # sfx: fortress_hit.wav
                
        self.projectiles = [p for p in self.projectiles if p not in projectiles_to_remove]
        return reward

    def _check_termination(self):
        if self.fortress_health <= 0:
            self.fortress_health = 0
            self.game_over = True
            # sfx: game_over.wav
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.win:
            return True
        return False

    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Area ---
        self._render_grid()
        self._render_fortress()
        self._render_blocks()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        
        # --- Render UI Area ---
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fortress_health": self.fortress_health,
            "wave": self.wave_number,
            "blocks_available": self.blocks_available,
        }

    # --- Rendering Helpers ---
    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (x * self.CELL_SIZE, 0)
            end_pos = (x * self.CELL_SIZE, self.SCREEN_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (0, y * self.CELL_SIZE)
            end_pos = (self.GAME_AREA_WIDTH, y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_fortress(self):
        color = self.COLOR_ENEMY if self.fortress_flash_timer > 0 else self.COLOR_FORTRESS
        for pos in self.fortress_pos:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(c*0.8 for c in color), rect, 2)

    def _render_blocks(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 1:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_BLOCK, rect)
                    pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_BLOCK), rect.inflate(-4,-4))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_x = int(enemy["pos"][0] * self.CELL_SIZE)
            pos_y = int(enemy["pos"][1] * self.CELL_SIZE)
            size = self.CELL_SIZE - 4
            rect = pygame.Rect(pos_x - size // 2, pos_y - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=3)
            
    def _render_projectiles(self):
        for proj in self.projectiles:
            pos_x = int(proj["pos"][0] * self.CELL_SIZE)
            pos_y = int(proj["pos"][1] * self.CELL_SIZE)
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 3, self.COLOR_PROJECTILE)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        alpha = 100 if self.grid[cx,cy] == 0 and (cx,cy) not in self.fortress_pos else 30
        pygame.draw.rect(s, self.COLOR_CURSOR + (alpha,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=3)
        pygame.draw.rect(s, self.COLOR_CURSOR + (alpha+50,), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 2, border_radius=3)
        self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        ui_x = self.GAME_AREA_WIDTH
        # Background and divider
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, 0, self.UI_AREA_WIDTH, self.SCREEN_HEIGHT))
        pygame.draw.line(self.screen, self.COLOR_UI_DIVIDER, (ui_x, 0), (ui_x, self.SCREEN_HEIGHT), 2)
        
        # --- Text Rendering Helper ---
        def draw_text(text, font, y, color=self.COLOR_TEXT, center=True):
            surface = font.render(text, True, color)
            x_pos = ui_x + (self.UI_AREA_WIDTH - surface.get_width()) / 2 if center else ui_x + 20
            self.screen.blit(surface, (x_pos, y))

        # --- UI Elements ---
        draw_text("STATUS", self.font_title, 20)
        
        y_offset = 80
        # Wave
        draw_text(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", self.font_main, y_offset, center=False)
        y_offset += 40
        
        # Health
        draw_text("FORTRESS HP:", self.font_main, y_offset, center=False)
        health_pct = self.fortress_health / self.INITIAL_FORTRESS_HEALTH
        health_bar_width = self.UI_AREA_WIDTH - 40
        health_color = (76, 175, 80) if health_pct > 0.5 else (255, 193, 7) if health_pct > 0.2 else (244, 67, 54)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (ui_x + 20, y_offset + 30, health_bar_width, 20))
        pygame.draw.rect(self.screen, health_color, (ui_x + 20, y_offset + 30, health_bar_width * health_pct, 20))
        y_offset += 70

        # Blocks
        draw_text(f"BLOCKS: {self.blocks_available}", self.font_main, y_offset, center=False)
        y_offset += 40
        
        # Score
        draw_text(f"SCORE: {self.score}", self.font_main, y_offset, center=False)
        y_offset += 70

        # Game Over / Win message
        if self.game_over:
            s = pygame.Surface((self.GAME_AREA_WIDTH, 120), pygame.SRCALPHA)
            s.fill((50, 50, 50, 200))
            self.screen.blit(s, (0, self.SCREEN_HEIGHT/2 - 60))

            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_FORTRESS if self.win else self.COLOR_ENEMY
            msg_surf = self.font_title.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.GAME_AREA_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    # --- Effects ---
    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            self.particles.append({
                "pos": [x * self.CELL_SIZE, y * self.CELL_SIZE],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": random.randint(10, 20),
                "color": color
            })

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["life"] * 0.2))
            alpha = int(255 * (p["life"] / 20))
            color = p["color"] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        
        print("✓ Implementation validated successfully")

# --- Example Usage ---
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # The environment is designed for headless rendering (rgb_array), 
    # but we can display the frames to play.
    
    # Re-initialize pygame with a display
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Fortress Defense")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY INSTRUCTIONS")
    print(env.user_guide)
    print("="*30 + "\n")

    running = True
    while running:
        # --- Action Mapping ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        # --- Step Environment ---
        # Since auto_advance is False, we only step on an action.
        # For a smoother manual play experience, we can step continuously.
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)

        # --- Render ---
        # The observation is already the rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        clock.tick(30) # Limit FPS for manual play

    env.close()