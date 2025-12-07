
# Generated: 2025-08-28T02:42:36.668383
# Source Brief: brief_04538.md
# Brief Index: 4538

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import defaultdict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
GRID_SIZE = 10
CELL_SIZE = 32
GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH) // 2 - 40
GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

MAX_STEPS = 6000
WIN_SCORE = 1000

# --- Colors ---
COLOR_BG = (25, 35, 45)
COLOR_GRID = (40, 55, 70)
COLOR_TEXT = (230, 240, 255)
COLOR_TEXT_SHADOW = (10, 15, 20)
COLOR_CURSOR = (255, 255, 0, 100) # Yellow, semi-transparent
COLOR_BARN = (210, 210, 220)
COLOR_BARN_ROOF = (180, 40, 40)
COLOR_PLOT_EMPTY = (90, 60, 40)

# --- Crop Definitions ---
CROP_DATA = {
    "carrot": {
        "seed_color": (255, 165, 0),
        "grow_time": 300,
        "ripe_time": 250,
        "value": 5,
        "colors": [(60, 180, 75), (255, 225, 25), (180, 40, 40)], # Planted, Ripe, Overripe
    },
    "lettuce": {
        "seed_color": (170, 255, 0),
        "grow_time": 200,
        "ripe_time": 200,
        "value": 3,
        "colors": [(128, 255, 128), (50, 205, 50), (107, 142, 35)],
    },
    "radish": {
        "seed_color": (230, 25, 75),
        "grow_time": 500,
        "ripe_time": 300,
        "value": 10,
        "colors": [(220, 150, 150), (230, 25, 75), (128, 0, 0)],
    },
}
CROP_TYPES = list(CROP_DATA.keys())

# --- Game States ---
STATE_EMPTY = 0
STATE_PLANTED = 1
STATE_GROWN = 2
STATE_OVERRIPE = 3


class Particle:
    def __init__(self, x, y, dx, dy, lifetime, color, size):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.color = color
        self.size = size

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1
        self.dx *= 0.98
        self.dy *= 0.98

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            s = self.size * (self.lifetime / self.max_lifetime)
            if s > 0:
                pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(s))


class Plot:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.rect = pygame.Rect(
            GRID_ORIGIN_X + col * CELL_SIZE,
            GRID_ORIGIN_Y + row * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        self.reset()

    def reset(self):
        self.state = STATE_EMPTY
        self.crop_type = None
        self.timer = 0

    def plant(self, crop_type):
        if self.state == STATE_EMPTY:
            self.state = STATE_PLANTED
            self.crop_type = crop_type
            self.timer = CROP_DATA[crop_type]["grow_time"]
            return True
        return False

    def harvest(self):
        if self.state in [STATE_GROWN, STATE_OVERRIPE]:
            value_multiplier = 0.5 if self.state == STATE_OVERRIPE else 1.0
            crop_info = (self.crop_type, value_multiplier)
            self.reset()
            return crop_info
        return None

    def update(self):
        if self.state == STATE_PLANTED:
            self.timer -= 1
            if self.timer <= 0:
                self.state = STATE_GROWN
                self.timer = CROP_DATA[self.crop_type]["ripe_time"]
        elif self.state == STATE_GROWN:
            self.timer -= 1
            if self.timer <= 0:
                self.state = STATE_OVERRIPE

    def draw(self, surface):
        pygame.draw.rect(surface, COLOR_PLOT_EMPTY, self.rect)
        if self.state != STATE_EMPTY:
            crop_info = CROP_DATA[self.crop_type]
            color = crop_info["colors"][self.state - 1]
            
            if self.state == STATE_PLANTED:
                progress = 1 - (self.timer / crop_info["grow_time"])
                radius = int(2 + (CELL_SIZE / 2 - 4) * progress)
            else:
                radius = int(CELL_SIZE / 2 - 2)
            
            pygame.gfxdraw.filled_circle(surface, self.rect.centerx, self.rect.centery, radius, color)
            pygame.gfxdraw.aacircle(surface, self.rect.centerx, self.rect.centery, radius, color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move cursor. Space to plant/harvest/sell. Shift to cycle selected seed."
    )

    game_description = (
        "A fast-paced farming game. Plant seeds, harvest crops, and sell them at the barn. Earn 1000 coins before time runs out!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_s = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        self.barn_rect = pygame.Rect(GRID_ORIGIN_X + GRID_WIDTH + 20, GRID_ORIGIN_Y + (GRID_HEIGHT - 80) // 2, 60, 80)
        
        self.grid = [[Plot(r, c) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
        self.particles = []
        
        self.reset()
        
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.steps_remaining = MAX_STEPS

        self.cursor_pos = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.inventory = defaultdict(int)
        self.selected_seed_idx = 0

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.grid[r][c].reset()
        
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Actions ---
        self._handle_movement(movement)
        
        if shift_press:
            self.selected_seed_idx = (self.selected_seed_idx + 1) % len(CROP_TYPES)
            # Small feedback for cycling seeds
            self._spawn_particles(self.cursor_world_pos[0], self.cursor_world_pos[1], 5, CROP_DATA[CROP_TYPES[self.selected_seed_idx]]["seed_color"], 1.5)

        if space_press:
            reward += self._handle_interaction()

        # --- Update Game State ---
        self.steps += 1
        self.steps_remaining -= 1
        
        unharvested_penalty = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                plot = self.grid[r][c]
                if plot.state in [STATE_GROWN, STATE_OVERRIPE]:
                    unharvested_penalty += 0.01
                plot.update()
        reward -= unharvested_penalty

        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

        # --- Check Termination ---
        terminated = False
        if self.score >= WIN_SCORE:
            self.win = True
            self.game_over = True
            terminated = True
            reward += 100
        elif self.steps_remaining <= 0:
            self.game_over = True
            terminated = True
            reward -= 10
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        # Cursor can be on the 10x10 grid or the 1x10 barn column
        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1

        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dy)
            self.cursor_pos[1] = (self.cursor_pos[1] + dx)

            # Clamp and wrap
            self.cursor_pos[0] = max(0, min(GRID_SIZE - 1, self.cursor_pos[0]))
            if self.cursor_pos[1] < 0: self.cursor_pos[1] = GRID_SIZE
            if self.cursor_pos[1] > GRID_SIZE: self.cursor_pos[1] = 0

    def _handle_interaction(self):
        reward = 0
        # If cursor is on the barn column
        if self.cursor_pos[1] == GRID_SIZE:
            # Sell crops
            sell_value = 0
            if sum(self.inventory.values()) > 0:
                # sfx: cash_register.wav
                for crop_type, count in self.inventory.items():
                    sell_value += CROP_DATA[crop_type]["value"] * count
                self.score += sell_value
                self.inventory.clear()
                reward += sell_value * 0.1 # Scaled reward for selling
                self._spawn_particles(self.barn_rect.centerx, self.barn_rect.centery, 30, (255, 215, 0), 3)
        else:
            # Interact with a plot
            plot = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
            
            # Plant
            if plot.state == STATE_EMPTY:
                selected_crop = CROP_TYPES[self.selected_seed_idx]
                if plot.plant(selected_crop):
                    # sfx: plant_seed.wav
                    self._spawn_particles(plot.rect.centerx, plot.rect.centery, 10, (152, 251, 152), 2)
            # Harvest
            else:
                harvest_result = plot.harvest()
                if harvest_result:
                    # sfx: harvest.wav
                    crop_type, value_multiplier = harvest_result
                    # For now, let's just add to inventory, value is realized on sale.
                    # Overripe crops are implicitly worth less.
                    # We can use the multiplier later if we want different item qualities.
                    self.inventory[crop_type] += 1
                    reward += 0.5 # Reward for any successful harvest
                    self._spawn_particles(plot.rect.centerx, plot.rect.centery, 20, CROP_DATA[crop_type]["colors"][1], 2.5)
        return reward
    
    @property
    def cursor_world_pos(self):
        if self.cursor_pos[1] == GRID_SIZE: # Barn column
             return self.barn_rect.center
        else: # Grid
            return (
                GRID_ORIGIN_X + self.cursor_pos[1] * CELL_SIZE + CELL_SIZE // 2,
                GRID_ORIGIN_Y + self.cursor_pos[0] * CELL_SIZE + CELL_SIZE // 2
            )

    def _spawn_particles(self, x, y, count, color, size):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            lifetime = random.randint(20, 40)
            self.particles.append(Particle(x, y, dx, dy, lifetime, color, size))

    def _render_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            s = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "YOU WIN!" if self.win else "TIME'S UP!"
            self._render_text(msg, self.screen.get_rect().center, self.font_l)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.screen, COLOR_GRID, (GRID_ORIGIN_X, GRID_ORIGIN_Y + i * CELL_SIZE), (GRID_ORIGIN_X + GRID_WIDTH, GRID_ORIGIN_Y + i * CELL_SIZE))
            pygame.draw.line(self.screen, COLOR_GRID, (GRID_ORIGIN_X + i * CELL_SIZE, GRID_ORIGIN_Y), (GRID_ORIGIN_X + i * CELL_SIZE, GRID_ORIGIN_Y + GRID_HEIGHT))

        # Draw plots
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self.grid[r][c].draw(self.screen)

        # Draw barn
        pygame.draw.rect(self.screen, COLOR_BARN, self.barn_rect)
        pygame.draw.polygon(self.screen, COLOR_BARN_ROOF, [(self.barn_rect.left, self.barn_rect.top), (self.barn_rect.right, self.barn_rect.top), (self.barn_rect.centerx, self.barn_rect.top - 20)])
        pygame.draw.rect(self.screen, (0,0,0), self.barn_rect, 2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
        # Draw cursor
        cursor_rect = pygame.Rect(0, 0, CELL_SIZE + 4, CELL_SIZE + 4)
        if self.cursor_pos[1] == GRID_SIZE:
             cursor_rect = self.barn_rect.inflate(8, 8)
        else:
             cursor_rect.center = self.cursor_world_pos
        
        pygame.draw.rect(self.screen, (255,255,0), cursor_rect, 2, border_radius=4)

    def _render_ui(self):
        # Score
        self._render_text(f"COINS: {self.score}", (20, 15), self.font_m)
        
        # Timer
        time_str = f"{self.steps_remaining / 60:.2f}"
        self._render_text(f"TIME: {time_str}", (SCREEN_WIDTH - 200, 15), self.font_m)

        # Inventory
        inv_y = SCREEN_HEIGHT - 30
        self._render_text("INVENTORY:", (20, inv_y), self.font_s)
        inv_x = 120
        for crop_type, count in self.inventory.items():
            if count > 0:
                color = CROP_DATA[crop_type]["colors"][1]
                pygame.draw.circle(self.screen, color, (inv_x, inv_y + 8), 8)
                self._render_text(f"x{count}", (inv_x + 12, inv_y), self.font_s)
                inv_x += 60

        # Selected Seed
        sel_x = SCREEN_WIDTH - 220
        self._render_text("SEED:", (sel_x, SCREEN_HEIGHT - 30), self.font_s)
        selected_crop_type = CROP_TYPES[self.selected_seed_idx]
        color = CROP_DATA[selected_crop_type]["seed_color"]
        pygame.draw.circle(self.screen, color, (sel_x + 60, inv_y + 8), 8)
        self._render_text(selected_crop_type.upper(), (sel_x + 75, inv_y), self.font_s)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "steps_remaining": self.steps_remaining,
            "cursor_pos": self.cursor_pos,
            "inventory": dict(self.inventory),
            "selected_seed": CROP_TYPES[self.selected_seed_idx]
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Simulator")
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print(env.user_guide)

    while not done:
        mov, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # In a real game loop, we'd wait for the next action.
        # Here we simulate it with a small delay for playability.
        clock.tick(10) # Limit speed for human play

    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()