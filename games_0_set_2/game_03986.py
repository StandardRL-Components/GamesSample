
# Generated: 2025-08-28T01:03:28.140170
# Source Brief: brief_03986.md
# Brief Index: 3986

        
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


# Helper class for particles to enhance visual feedback
class Particle:
    """A simple particle for visual effects like harvesting or selling."""
    def __init__(self, x, y, color, size, life, dx, dy, gravity=0.1):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.initial_life = life
        self.dx = dx
        self.dy = dy
        self.gravity = gravity

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.initial_life))
            alpha = max(0, min(255, alpha))
            # Use a surface to handle per-pixel alpha
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, self.size, self.size, self.size, (*self.color, alpha))
            surface.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)))

# Helper class for managing individual farm plots
class FarmTile:
    """Manages the state of a single tile on the farm grid."""
    def __init__(self, growth_duration):
        self.state = "tilled"  # tilled, growing, ready, barn
        self.growth_timer = 0
        self.max_growth_time = growth_duration
        self.pulse_phase = random.uniform(0, 2 * math.pi)

    def plant(self):
        if self.state == "tilled":
            self.state = "growing"
            self.growth_timer = self.max_growth_time
            return True
        return False

    def harvest(self):
        if self.state == "ready":
            self.state = "tilled"
            self.growth_timer = 0
            return True
        return False

    def update(self, dt):
        if self.state == "growing":
            self.growth_timer -= dt
            if self.growth_timer <= 0:
                self.state = "ready"
                self.growth_timer = 0
        elif self.state == "ready":
            self.pulse_phase = (self.pulse_phase + 5 * dt) % (2 * math.pi)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Hold Space on a plot to plant/harvest. Hold Shift at the barn to sell."
    )

    game_description = (
        "Manage a farm to harvest crops and sell produce, aiming to earn 1000 coins before the timer runs out."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 5
        self.TILE_SIZE = 64
        self.UI_HEIGHT = self.SCREEN_HEIGHT - (self.GRID_HEIGHT * self.TILE_SIZE)
        self.GRID_OFFSET_Y = self.UI_HEIGHT

        self.FPS = 30
        self.WIN_COINS = 1000
        self.GAME_DURATION_SECONDS = 300
        self.PLAYER_SPEED = 8.0
        self.CROP_GROWTH_SECONDS = 10
        self.BARN_POS = (0, 2)

        # --- Colors ---
        self.COLOR_BG = (139, 172, 15)
        self.COLOR_SOIL = (92, 53, 40)
        self.COLOR_SOIL_TILL = (143, 86, 59)
        self.COLOR_CROP_GROWING = (124, 181, 14)
        self.COLOR_CROP_READY = (255, 215, 0)
        self.COLOR_BARN = (180, 40, 40)
        self.COLOR_BARN_ROOF = (90, 90, 90)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BG = (40, 40, 40)
        self.COLOR_UI_SHADOW = (20, 20, 20)

        # --- Fonts ---
        try:
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.player_grid_pos = None
        self.player_pixel_pos = None
        self.farm_grid = None
        self.player_inventory = None
        self.coins = None
        self.time_remaining = None
        self.game_over = None
        self.steps = None
        self.action_cooldown = None
        self.particles = []
        self.player_bob = 0

        self.reset()
        self.validate_implementation()

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        y = grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2 + self.GRID_OFFSET_Y
        return pygame.Vector2(x, y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_grid_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.player_pixel_pos = self._grid_to_pixel(self.player_grid_pos)

        self.farm_grid = np.array(
            [
                [FarmTile(self.CROP_GROWTH_SECONDS) for _ in range(self.GRID_HEIGHT)]
                for _ in range(self.GRID_WIDTH)
            ]
        )
        self.farm_grid[self.BARN_POS[0]][self.BARN_POS[1]].state = "barn"

        for _ in range(3):
            x, y = self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)
            if self.farm_grid[x][y].state == "tilled":
                self.farm_grid[x][y].plant()
                self.farm_grid[x][y].growth_timer = self.np_random.uniform(1, self.CROP_GROWTH_SECONDS)

        self.player_inventory = {"crop": 0}
        self.coins = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        self.steps = 0
        self.action_cooldown = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = 1 / self.FPS
        self.steps += 1
        self.time_remaining -= dt
        self.player_bob = (self.player_bob + 4 * dt) % (2 * math.pi)
        if self.action_cooldown > 0:
            self.action_cooldown = max(0, self.action_cooldown - 1)

        reward = -0.01  # Small penalty for moving/existing

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self.farm_grid[x][y].update(dt)

        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_movement(movement)
        action_reward = self._handle_interactions(space_held, shift_held)
        reward += action_reward

        terminated = False
        terminal_reward = 0
        if self.coins >= self.WIN_COINS:
            terminal_reward = 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            terminal_reward = -100
            terminated = True
            self.game_over = True
        
        reward += terminal_reward
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_movement(self, movement):
        # Movement updates the target grid position, pixel position interpolates
        target_grid_pos = self.player_grid_pos.copy()
        if movement == 1: target_grid_pos[1] -= 1  # Up
        elif movement == 2: target_grid_pos[1] += 1  # Down
        elif movement == 3: target_grid_pos[0] -= 1  # Left
        elif movement == 4: target_grid_pos[0] += 1  # Right
        
        if movement != 0:
            target_grid_pos[0] = np.clip(target_grid_pos[0], 0, self.GRID_WIDTH - 1)
            target_grid_pos[1] = np.clip(target_grid_pos[1], 0, self.GRID_HEIGHT - 1)
            self.player_grid_pos = target_grid_pos

        target_pixel_pos = self._grid_to_pixel(self.player_grid_pos)
        move_vec = target_pixel_pos - self.player_pixel_pos
        if move_vec.length() > self.PLAYER_SPEED:
            self.player_pixel_pos += move_vec.normalize() * self.PLAYER_SPEED
        else:
            self.player_pixel_pos = target_pixel_pos

    def _handle_interactions(self, space_held, shift_held):
        if self.action_cooldown > 0:
            return 0
        
        reward = 0
        px, py = self.player_grid_pos
        current_tile = self.farm_grid[px][py]

        if space_held:
            if current_tile.state == "ready" and current_tile.harvest():
                self.player_inventory["crop"] += 1
                reward += 0.1
                self.action_cooldown = 3 # Harvest time
                # sfx: harvest sound
                self._create_particles(self.player_pixel_pos, self.COLOR_CROP_READY, 10, -2)
            elif current_tile.state == "tilled" and current_tile.plant():
                self.action_cooldown = 2 # Plant time
                # sfx: plant seed sound
            return reward

        if shift_held:
            if tuple(self.player_grid_pos) == self.BARN_POS and self.player_inventory["crop"] > 0:
                num_sold = self.player_inventory["crop"]
                self.coins += num_sold
                reward += 1.0 * num_sold
                self.player_inventory["crop"] = 0
                self.action_cooldown = 5 # Sell time
                # sfx: cash register sound
                self._create_particles(self.player_pixel_pos, self.COLOR_CROP_READY, num_sold * 2, -3, is_coin=True)
            return reward
        return 0

    def _create_particles(self, pos, color, count, initial_vy, is_coin=False):
        for _ in range(count):
            dx = self.np_random.uniform(-1.5, 1.5)
            dy = self.np_random.uniform(initial_vy - 1, initial_vy + 1)
            size = self.np_random.integers(3, 7) if not is_coin else 5
            life = self.np_random.integers(20, 40)
            p_color = (255, 223, 0) if is_coin else color
            self.particles.append(Particle(pos.x, pos.y, p_color, size, life, dx, dy))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_player()
        self._render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.coins, "steps": self.steps}

    def _render_grid(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE + self.GRID_OFFSET_Y, self.TILE_SIZE, self.TILE_SIZE)
                tile = self.farm_grid[x][y]
                
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_SOIL_TILL, rect.inflate(-4, -4))

                if tile.state == "barn":
                    self._render_barn(rect)
                    continue

                center_x, center_y = rect.center
                if tile.state == "growing":
                    progress = 1.0 - (tile.growth_timer / tile.max_growth_time)
                    radius = int((self.TILE_SIZE / 2 - 8) * progress)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, max(1, radius), self.COLOR_CROP_GROWING)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, max(1, radius), self.COLOR_CROP_GROWING)
                elif tile.state == "ready":
                    pulse = (math.sin(tile.pulse_phase) + 1) / 2
                    radius = int((self.TILE_SIZE / 2 - 8) + pulse * 3)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_CROP_READY)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_CROP_READY)

    def _render_barn(self, rect):
        pygame.draw.rect(self.screen, self.COLOR_BARN, rect)
        roof_points = [(rect.left, rect.centery), (rect.centerx, rect.top + 10), (rect.right, rect.centery)]
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, roof_points)
        door_rect = pygame.Rect(0, 0, rect.width // 2, rect.height // 2)
        door_rect.center = rect.center
        door_rect.y = rect.centery
        pygame.draw.rect(self.screen, (0,0,0), door_rect)

    def _render_player(self):
        pos_x, pos_y = int(self.player_pixel_pos.x), int(self.player_pixel_pos.y)
        radius = self.TILE_SIZE // 4
        bob_offset = int(math.sin(self.player_bob) * 3)

        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y + bob_offset, radius + 2, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y + bob_offset, radius + 2, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y + bob_offset, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y + bob_offset, radius, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        ui_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_SHADOW, (0, self.UI_HEIGHT - 2), (self.SCREEN_WIDTH, self.UI_HEIGHT - 2), 4)

        coin_text = f"COINS: {self.coins}/{self.WIN_COINS}"
        self._draw_text(coin_text, self.font_large, self.COLOR_UI_TEXT, 20, self.UI_HEIGHT // 2)

        mins, secs = divmod(max(0, self.time_remaining), 60)
        time_text = f"TIME: {int(mins):02}:{int(secs):02}"
        self._draw_text(time_text, self.font_large, self.COLOR_UI_TEXT, self.SCREEN_WIDTH - 20, self.UI_HEIGHT // 2, align="right")
        
        inv_text = f"HARVESTED: {self.player_inventory['crop']}"
        self._draw_text(inv_text, self.font_small, self.COLOR_UI_TEXT, 20, self.SCREEN_HEIGHT - 20, align="left")

    def _draw_text(self, text, font, color, x, y, align="left"):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "center": text_rect.center = (x, y)
        elif align == "right": text_rect.right = x
        else: text_rect.left = x
        text_rect.y = y - text_rect.height // 2
        
        shadow_surface = font.render(text, True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow_surface, text_rect.move(2, 2))
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Farm Manager")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        action.fill(0)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

    env.close()
    print("Game Over!")
    print(f"Final Score: {info['score']}")