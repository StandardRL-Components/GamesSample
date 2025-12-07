
# Generated: 2025-08-28T00:44:02.903260
# Source Brief: brief_03878.md
# Brief Index: 3878

        
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


# Helper class for particle effects
class Particle:
    def __init__(self, x, y, vx, vy, life, color, size_start, size_end, gravity=0.1):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size_start = size_start
        self.size_end = size_end
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            lerp_factor = self.life / self.max_life
            current_size = int(self.size_end + (self.size_start - self.size_end) * lerp_factor)
            if current_size > 0:
                pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), current_size)

# Helper class for floating text (score popups)
class FloatingText:
    def __init__(self, text, x, y, life, font, color=(255, 255, 255)):
        self.text = text
        self.x = x
        self.y = y
        self.life = life
        self.max_life = life
        self.font = font
        self.color = color

    def update(self):
        self.life -= 1
        self.y -= 0.5  # Move upwards
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            alpha = max(0, min(255, alpha))
            text_surface = self.font.render(self.text, True, self.color)
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, (int(self.x - text_surface.get_width() / 2), int(self.y)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrows to move cursor. Space to plant a seed. Shift to water the selected tile."
    game_description = "Cultivate a grid garden by planting and watering seeds. Grow 10 plants to win, but don't run out of seeds! Watering adjacent mature plants gives combo points."
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 10
    CELL_SIZE = 32
    GRID_OFFSET_X = (WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_OFFSET_Y = (HEIGHT - GRID_SIZE * CELL_SIZE) // 2
    MAX_STEPS = 1000
    INITIAL_SEEDS = 25
    WIN_CONDITION_PLANTS = 10

    # Colors
    COLOR_BG = (48, 32, 24)  # Dark loamy color
    COLOR_SOIL = (112, 75, 56)
    COLOR_WET_SOIL = (80, 53, 40)
    COLOR_GRID = (130, 93, 75)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_WATER = (100, 150, 255)
    PLANT_COLORS = {
        1: (154, 205, 50),  # Seedling: yellowgreen
        2: (124, 252, 0),   # Sprout: lawngreen
        3: (50, 205, 50),   # Small Plant: limegreen
        4: (34, 139, 34),   # Mature Plant: forestgreen
    }
    FLOWER_COLOR = (255, 105, 180) # Hot pink for flowers

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_popup = pygame.font.Font(None, 22)
        self.font_game_over = pygame.font.Font(None, 72)
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [0, 0]
        self.seeds = 0
        self.mature_plants = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.floating_texts = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.seeds = self.INITIAL_SEEDS
        self.mature_plants = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.particles = []
        self.floating_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, plant_action, water_action = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1
        
        # Update animations
        self._update_effects()

        # Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        cx, cy = self.cursor_pos
        cell_state = self.grid[cy, cx]

        # Prioritize actions: Plant > Water
        if plant_action:
            if self.seeds > 0 and cell_state == 0:
                # sfx: plant_seed.wav
                self.grid[cy, cx] = 1  # Place seedling
                self.seeds -= 1
                reward += 0.1 # Small reward for valid placement
                self._create_effect("plant", cx, cy)
                self._create_floating_text(f"-1 Seed", cx, cy, (255,200,100))
            else:
                # sfx: action_fail.wav
                reward -= 0.05 # Small penalty for invalid plant action
        
        elif water_action:
            if 0 < cell_state < 4:  # Growing plant
                # sfx: water_plant.wav
                self.grid[cy, cx] += 1
                reward += 1.0
                self._create_effect("water", cx, cy)
                
                if self.grid[cy, cx] == 4:  # Reached maturity
                    # sfx: plant_mature.wav
                    self.mature_plants += 1
                    reward += 5.0
                    self.score += 5
                    self._create_effect("mature", cx, cy)
                    self._create_floating_text("+5", cx, cy, (255, 255, 100))
                    
                    # Check for combos
                    combo_count = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and self.grid[ny, nx] == 4:
                            combo_count += 1
                    if combo_count > 0:
                        # sfx: combo.wav
                        combo_reward = combo_count * 2
                        reward += combo_reward
                        self.score += combo_reward
                        self._create_floating_text(f"+{combo_reward} COMBO!", cx, cy, (255, 165, 0))

            else:  # Empty, wet, or mature soil
                # sfx: water_splash.wav
                reward -= 0.1
                if cell_state == 0:
                    self.grid[cy, cx] = -1 # Temporarily wet soil
                self._create_effect("water", cx, cy)

        # Update score with rewards, capped at 0 for negative rewards
        self.score += max(0, reward)

        # Revert wet soil back to normal soil for the next frame
        self.grid[self.grid == -1] = 0

        # Check for termination
        terminated = False
        if self.mature_plants >= self.WIN_CONDITION_PLANTS:
            # sfx: win_jingle.wav
            reward += 100
            self.game_over = True
            self.win = True
            terminated = True
        
        can_grow_more = np.any((self.grid > 0) & (self.grid < 4))
        if self.seeds <= 0 and not can_grow_more:
            # sfx: lose_sound.wav
            reward -= 100
            self.game_over = True
            self.win = False
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_effects()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "seeds": self.seeds,
            "mature_plants": self.mature_plants,
            "cursor_pos": list(self.cursor_pos),
        }

    def _render_game(self):
        # Draw grid cells
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                cell_state = self.grid[y, x]
                color = self.COLOR_WET_SOIL if cell_state == -1 else self.COLOR_SOIL
                pygame.draw.rect(self.screen, color, rect)

                # Draw plants
                if cell_state > 0:
                    self._draw_plant(self.screen, rect, cell_state)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            start_x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            start_y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.GRID_OFFSET_Y), (start_x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, start_y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, start_y))

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pulse = (math.sin(pygame.time.get_ticks() * 0.01) + 1) / 2
        line_width = int(2 + pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, line_width)

    def _draw_plant(self, surface, rect, stage):
        center_x, center_y = rect.center
        if stage == 1: # Seedling
            pygame.draw.circle(surface, self.PLANT_COLORS[1], rect.center, 3)
        elif stage == 2: # Sprout
            pygame.draw.line(surface, self.PLANT_COLORS[2], (center_x, center_y+4), (center_x-4, center_y-4), 2)
            pygame.draw.line(surface, self.PLANT_COLORS[2], (center_x, center_y+4), (center_x+4, center_y-4), 2)
        elif stage == 3: # Small Plant
            pygame.draw.circle(surface, self.PLANT_COLORS[3], (center_x - 5, center_y), 5)
            pygame.draw.circle(surface, self.PLANT_COLORS[3], (center_x + 5, center_y), 5)
            pygame.draw.circle(surface, self.PLANT_COLORS[3], (center_x, center_y - 5), 5)
        elif stage == 4: # Mature Plant
            pygame.draw.circle(surface, self.PLANT_COLORS[4], (center_x - 7, center_y+2), 7)
            pygame.draw.circle(surface, self.PLANT_COLORS[4], (center_x + 7, center_y+2), 7)
            pygame.draw.circle(surface, self.PLANT_COLORS[4], (center_x, center_y - 6), 7)
            pygame.draw.circle(surface, self.FLOWER_COLOR, rect.center, 5)
            pygame.gfxdraw.aacircle(surface, rect.centerx, rect.centery, 5, self.FLOWER_COLOR)

    def _render_ui(self):
        # Seed counter
        seed_text = self.font_ui.render(f"Seeds: {self.seeds}", True, self.COLOR_TEXT)
        self.screen.blit(seed_text, (15, 15))

        # Plant counter
        plant_text = self.font_ui.render(f"Plants: {self.mature_plants} / {self.WIN_CONDITION_PLANTS}", True, self.COLOR_TEXT)
        self.screen.blit(plant_text, (self.WIDTH - plant_text.get_width() - 15, 15))

        # Score display
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 35))

        # Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.win else "GAME OVER"
            end_color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _update_effects(self):
        self.particles = [p for p in self.particles if p.update()]
        self.floating_texts = [t for t in self.floating_texts if t.update()]

    def _render_effects(self):
        for p in self.particles:
            p.draw(self.screen)
        for t in self.floating_texts:
            t.draw(self.screen)
            
    def _get_cell_center(self, grid_x, grid_y):
        return (
            self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _create_effect(self, effect_type, grid_x, grid_y):
        center_x, center_y = self._get_cell_center(grid_x, grid_y)
        if effect_type == "water":
            for _ in range(15):
                self.particles.append(Particle(
                    x=center_x + random.uniform(-10, 10), y=center_y - 10,
                    vx=random.uniform(-1, 1), vy=random.uniform(-2, -0.5),
                    life=random.randint(15, 25), color=self.COLOR_WATER,
                    size_start=random.randint(3, 5), size_end=1, gravity=0.2
                ))
        elif effect_type == "plant":
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.particles.append(Particle(
                    x=center_x, y=center_y,
                    vx=math.cos(angle) * speed, vy=math.sin(angle) * speed,
                    life=random.randint(10, 20), color=self.COLOR_SOIL,
                    size_start=random.randint(3, 6), size_end=0, gravity=0.1
                ))
        elif effect_type == "mature":
             for _ in range(40):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 5)
                self.particles.append(Particle(
                    x=center_x, y=center_y,
                    vx=math.cos(angle) * speed, vy=math.sin(angle) * speed,
                    life=random.randint(20, 40), color=self.FLOWER_COLOR,
                    size_start=random.randint(2, 4), size_end=0, gravity=0
                ))
    
    def _create_floating_text(self, text, grid_x, grid_y, color):
        center_x, center_y = self._get_cell_center(grid_x, grid_y)
        self.floating_texts.append(
            FloatingText(text, center_x, center_y - 10, 40, self.font_popup, color)
        )

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Grid Garden")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        plant = 0
        water = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: plant = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: water = 1
        
        action = [movement, plant, water]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        # Since auto_advance is False, we need to control the step rate
        clock.tick(10) # Run at 10 steps per second for human playability

    env.close()