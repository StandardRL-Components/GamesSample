
# Generated: 2025-08-27T12:27:13.639130
# Source Brief: brief_00050.md
# Brief Index: 50

        
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
        "Controls: Arrow keys to move cursor. Space to plant a seed. Shift to harvest a mature crop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a flourishing grid garden. Plant seeds, wait for them to grow, and harvest them to fill the entire grid. Plan carefully to not run out of seeds!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 10
    MAX_STEPS = 1000
    STARTING_SEEDS = 15
    GROWTH_TIME = 5 # Steps to mature

    # Colors
    COLOR_BG = (34, 40, 49)
    COLOR_GRID = (57, 62, 70)
    COLOR_HARVESTED = (94, 71, 51)
    COLOR_CURSOR = (255, 211, 105)
    COLOR_TEXT = (238, 238, 238)
    COLOR_TEXT_SHADOW = (20, 20, 20)
    
    # Cell states
    STATE_EMPTY = 0
    STATE_GROWING = 1
    STATE_HARVESTED = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 32)
        
        # Game state variables
        self.grid = None
        self.growth_timers = None
        self.cursor_pos = None
        self.seed_count = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_message = ""
        self.particles = []
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.STATE_EMPTY, dtype=np.uint8)
        self.growth_timers = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int16)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.seed_count = self.STARTING_SEEDS
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        plant_action = action[1] == 1
        harvest_action = action[2] == 1
        
        reward = 0
        
        # --- 1. Handle Input and Actions ---
        self._move_cursor(movement)
        
        if plant_action:
            reward += self._plant_seed()
            
        if harvest_action:
            reward += self._harvest_crop()

        # --- 2. Update Game State ---
        # Advance growth timers for all growing plants
        growing_mask = (self.grid == self.STATE_GROWING)
        self.growth_timers[growing_mask] += 1
        
        self.steps += 1
        
        # --- 3. Check for Termination ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        # --- 4. Update Particles ---
        self._update_particles()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

    def _plant_seed(self):
        x, y = self.cursor_pos
        if self.grid[y, x] == self.STATE_EMPTY and self.seed_count > 0:
            self.seed_count -= 1
            self.grid[y, x] = self.STATE_GROWING
            self.growth_timers[y, x] = 0
            # SFX: Plant seed
            self._create_particles(x, y, (126, 214, 223), 5, -1) # Plant effect
            return 0 # No immediate reward for planting
        return 0

    def _harvest_crop(self):
        x, y = self.cursor_pos
        if self.grid[y, x] == self.STATE_GROWING and self.growth_timers[y, x] >= self.GROWTH_TIME:
            self.grid[y, x] = self.STATE_HARVESTED
            
            # Base reward for harvesting
            harvest_reward = 10
            
            # Adjacency bonus
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE:
                    if self.grid[ny, nx] == self.STATE_HARVESTED:
                        harvest_reward += 1
            
            self.score += harvest_reward
            # SFX: Harvest crop
            self._create_particles(x, y, (255, 190, 11), 20, 2) # Harvest effect
            return harvest_reward
        return 0

    def _check_termination(self):
        # Win condition: grid is full
        if np.all(self.grid == self.STATE_HARVESTED):
            self.win_message = "GARDEN COMPLETE!"
            self.score += 100
            return True, 100

        # Loss condition: no seeds and no growing crops left
        num_growing = np.sum(self.grid == self.STATE_GROWING)
        if self.seed_count == 0 and num_growing == 0:
            self.win_message = "OUT OF SEEDS"
            self.score -= 100
            return True, -100
            
        # Max steps reached
        if self.steps >= self.MAX_STEPS:
            self.win_message = "TIME UP"
            return True, 0
            
        return False, 0
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        grid_pixel_size = self.HEIGHT - 40
        self.cell_size = grid_pixel_size // self.GRID_SIZE
        offset_x = (self.WIDTH - grid_pixel_size) // 2
        offset_y = (self.HEIGHT - grid_pixel_size) // 2

        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    offset_x + x * self.cell_size,
                    offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                # Draw cell background
                state = self.grid[y, x]
                if state == self.STATE_HARVESTED:
                    pygame.draw.rect(self.screen, self.COLOR_HARVESTED, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Draw growing crop
                if state == self.STATE_GROWING:
                    growth_ratio = min(1.0, self.growth_timers[y, x] / self.GROWTH_TIME)
                    
                    # Color brightens with growth
                    plant_color = tuple(int(c * (0.6 + 0.4 * growth_ratio)) for c in (76, 175, 80))
                    
                    # Size increases with growth
                    plant_size = int(self.cell_size * 0.8 * growth_ratio)
                    plant_rect = pygame.Rect(0, 0, plant_size, plant_size)
                    plant_rect.center = rect.center
                    pygame.draw.rect(self.screen, plant_color, plant_rect, border_radius=max(1, int(plant_size * 0.2)))
                    
                    # Add a pulsating glow when mature
                    if growth_ratio >= 1.0:
                        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1
                        glow_color = (255, 255, 150)
                        glow_size = int(plant_size * (1.1 + 0.2 * pulse))
                        
                        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surface, (*glow_color, 40), (glow_size // 2, glow_size // 2), glow_size // 2)
                        pygame.draw.circle(glow_surface, (*glow_color, 60), (glow_size // 2, glow_size // 2), glow_size // 3)
                        self.screen.blit(glow_surface, (plant_rect.centerx - glow_size // 2, plant_rect.centery - glow_size // 2), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw particles
        for p in self.particles:
            p_x, p_y, p_vx, p_vy, p_life, p_color = p
            pygame.draw.circle(self.screen, p_color, (int(p_x), int(p_y)), int(p_life))

        # Draw cursor
        cursor_rect = pygame.Rect(
            offset_x + self.cursor_pos[0] * self.cell_size,
            offset_y + self.cursor_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
    
    def _render_ui(self):
        # Render text with a shadow
        def draw_text(text, font, color, pos, shadow_color):
            text_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Draw seed count
        seed_text = f"Seeds: {self.seed_count}"
        draw_text(seed_text, self.font_medium, self.COLOR_TEXT, (15, 10), self.COLOR_TEXT_SHADOW)
        
        # Draw score
        score_text = f"Score: {self.score}"
        score_size = self.font_medium.size(score_text)
        draw_text(score_text, self.font_medium, self.COLOR_TEXT, (self.WIDTH - score_size[0] - 15, 10), self.COLOR_TEXT_SHADOW)

        # Draw game over message
        if self.game_over and self.win_message:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_size = self.font_large.size(self.win_message)
            draw_text(self.win_message, self.font_large, self.COLOR_CURSOR,
                      ((self.WIDTH - msg_size[0]) // 2, (self.HEIGHT - msg_size[1]) // 2),
                      self.COLOR_TEXT_SHADOW)

    def _create_particles(self, grid_x, grid_y, color, count, speed_mult):
        grid_pixel_size = self.HEIGHT - 40
        offset_x = (self.WIDTH - grid_pixel_size) // 2
        offset_y = (self.HEIGHT - grid_pixel_size) // 2
        
        px = offset_x + (grid_x + 0.5) * self.cell_size
        py = offset_y + (grid_y + 0.5) * self.cell_size
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.uniform(2, 4)
            self.particles.append([px, py, vx, vy, life, color])
            
    def _update_particles(self):
        new_particles = []
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 0.1  # life -= decay
            if p[4] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "seeds": self.seed_count,
            "cursor_pos": list(self.cursor_pos),
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Grid Garden")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # No-op
        plant = 0
        harvest = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            plant = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            harvest = 1

        action = [movement, plant, harvest]
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Game Step ---
        # Since auto_advance is False, we only step when there's an action.
        # For a playable experience, we need to send an action every frame.
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing a reset
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(10) # Limit frame rate for human playability

    env.close()