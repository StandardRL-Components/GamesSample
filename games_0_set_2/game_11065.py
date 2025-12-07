import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for animated particles
class Particle:
    def __init__(self, x, y, color, size_mult=1.0):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = random.uniform(1, 3) * size_mult
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed
        self.life = random.randint(15, 30)  # Frames
        self.color = color
        self.size = random.uniform(2, 5) * size_mult

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), self.color)

# Helper class for platforms
class Platform:
    def __init__(self, grid_x, grid_y, size, base_color):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.size = size  # Logical size [1-5]
        self.visual_size = float(size)  # For smooth animation
        self.particles = []
        self.base_color = base_color
        self.last_change_timer = 0

    def change_size(self, delta):
        new_size = max(1, min(5, self.size + delta))
        changed = new_size != self.size
        self.size = new_size
        if changed:
            # sfx: platform_change
            self.last_change_timer = 30 # frames
            for _ in range(int(abs(delta) * 10)):
                self.particles.append(Particle(0, 0, (255, 255, 255), abs(delta)))
        return changed

    def update(self):
        # Lerp visual size towards logical size for smooth animation
        self.visual_size += (self.size - self.visual_size) * 0.2
        if self.last_change_timer > 0:
            self.last_change_timer -= 1
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

    def draw(self, surface, x, y, width, height):
        size_ratio = (self.visual_size - 1) / 4.0  # Normalize to [0, 1]
        platform_width = int(width * (0.4 + size_ratio * 0.6))
        platform_height = int(height * (0.4 + size_ratio * 0.6))
        px = x + (width - platform_width) // 2
        py = y + (height - platform_height) // 2
        
        # Color interpolation based on size
        color = tuple(int(c * (0.6 + 0.4 * size_ratio)) for c in self.base_color)
        
        rect = pygame.Rect(px, py, platform_width, platform_height)
        pygame.draw.rect(surface, color, rect, border_radius=4)
        pygame.draw.rect(surface, (255,255,255), rect, width=1, border_radius=4)

        for p in self.particles:
            p.draw(surface)


# Helper class for Gods
class God:
    GOD_NAMES = ["Itztli", "Xochitl", "Coyotl", "Ehecatl", "Tonatiuh", "Metztli"]
    def __init__(self, seed_rng):
        self.name = seed_rng.choice(self.GOD_NAMES)
        self.preferred_size = seed_rng.randint(1, 5)
        self.appeasement = 0  # 0-100
        self.is_appeased = False
        self.color = tuple(seed_rng.sample(range(50, 256), 3))

    def update_appeasement(self, platforms, total_platforms):
        if self.is_appeased:
            return
        
        matching_platforms = sum(1 for p in platforms if p.size == self.preferred_size)
        self.appeasement = (matching_platforms / total_platforms) * 100
        if self.appeasement >= 100:
            self.appeasement = 100
            if not self.is_appeased:
                self.is_appeased = True
                # sfx: god_appeased_fully


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    game_description = (
        "Appease ancient gods by reshaping a grid of sacred platforms. Each action you take ripples "
        "through the grid, affecting neighboring platforms."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selector. Press space to grow a platform or "
        "shift to shrink it. Growing/shrinking a platform has the opposite effect on its neighbors."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Visual & Game Constants ---
        self.GRID_SIZE = (5, 4) # 5 columns, 4 rows
        self.GAME_AREA_RECT = pygame.Rect(170, 40, 460, 350)
        self.MAX_TURNS = 200
        self.MAX_GYM_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BG_PATTERN = (30, 45, 60)
        self.COLOR_PLATFORM_BASE = (100, 80, 70)
        self.COLOR_SELECTOR = (255, 200, 0)
        self.COLOR_SUN = (255, 230, 100)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (40, 60, 80)
        self.COLOR_UI_BORDER = (60, 90, 120)
        self.COLOR_METER_BG = (10, 20, 30)
        self.COLOR_METER_GOOD = (0, 200, 100)
        
        # Fonts
        self.FONT_S = pygame.font.Font(None, 20)
        self.FONT_M = pygame.font.Font(None, 28)
        self.FONT_L = pygame.font.Font(None, 48)

        # --- State Variables ---
        self.num_gods_to_spawn = 2
        self.seed_rng = None
        self.platforms = []
        self.gods = []
        self.selector_pos = [0, 0]
        self.gym_steps = 0
        self.game_turns = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        # Initialize state by calling reset
        # self.reset() # This is typically called by the environment wrapper
        # self.validate_implementation() # This is for debugging and should not be in final code

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_rng = random.Random(seed)
        else:
            # Create a new random generator if no seed is provided
            # This ensures reset() is deterministic if seeded, and random otherwise.
            self.seed_rng = random.Random()

        self.gym_steps = 0
        self.game_turns = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.selector_pos = [0, 0]
        
        # Create Gods
        self.gods = []
        god_names = self.seed_rng.sample(God.GOD_NAMES, self.num_gods_to_spawn)
        preferred_sizes = self.seed_rng.sample(range(1, 6), self.num_gods_to_spawn)
        for i in range(self.num_gods_to_spawn):
            god = God(self.seed_rng)
            god.name = god_names[i]
            # Ensure unique preferred sizes for better gameplay
            god.preferred_size = preferred_sizes[i]
            self.gods.append(god)
            
        # Create Platforms
        self.platforms = []
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                p = Platform(x, y, self.seed_rng.randint(1, 5), self.COLOR_PLATFORM_BASE)
                self.platforms.append(p)
        
        self._update_all_gods_appeasement()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.gym_steps += 1
        reward = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._move_selector(movement)
        
        action_taken = False
        if space_held and not shift_held:
            # sfx: grow_action
            reward = self._perform_turn(1)  # Grow
            action_taken = True
        elif shift_held and not space_held:
            # sfx: shrink_action
            reward = self._perform_turn(-1) # Shrink
            action_taken = True
        
        if not action_taken:
            reward = -0.01 # Small penalty for inaction

        self.score += reward
        
        terminated = self._check_termination()
        truncated = self.gym_steps >= self.MAX_GYM_STEPS
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.win_condition:
                terminal_reward = 100.0
                # Increase difficulty for the next game
                self.num_gods_to_spawn = min(self.num_gods_to_spawn + 1, len(God.GOD_NAMES))
            else: # Loss
                terminal_reward = -100.0
            reward += terminal_reward
            self.score += terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _perform_turn(self, size_delta):
        self.game_turns += 1
        
        pre_action_appeasements = {g.name: g.appeasement for g in self.gods}
        
        selected_platform = self.platforms[self.selector_pos[1] * self.GRID_SIZE[0] + self.selector_pos[0]]
        
        # Main action
        changed_platforms = {selected_platform}
        selected_platform.change_size(size_delta)
        
        # Chain reaction
        sx, sy = self.selector_pos
        neighbors_pos = [(sx, sy - 1), (sx, sy + 1), (sx - 1, sy), (sx + 1, sy)]
        for nx, ny in neighbors_pos:
            if 0 <= nx < self.GRID_SIZE[0] and 0 <= ny < self.GRID_SIZE[1]:
                neighbor_platform = self.platforms[ny * self.GRID_SIZE[0] + nx]
                neighbor_platform.change_size(-size_delta) # Opposite reaction
                changed_platforms.add(neighbor_platform)
        
        self._update_all_gods_appeasement()

        # Calculate reward
        turn_reward = 0.0
        for g in self.gods:
            change = g.appeasement - pre_action_appeasements[g.name]
            turn_reward += change / 10.0 # Scaled continuous reward
            
            if g.is_appeased and pre_action_appeasements[g.name] < 100:
                turn_reward += 5.0 # Event reward for appeasing a god

        return turn_reward

    def _check_termination(self):
        self.win_condition = all(g.is_appeased for g in self.gods)
        if self.win_condition:
            return True
        if self.game_turns >= self.MAX_TURNS:
            return True
        return False

    def _get_observation(self):
        self._update_animations()
        self._render_background()
        self._render_game_area()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "game_turns": self.game_turns,
            "gym_steps": self.gym_steps,
            "gods_appeased": sum(1 for g in self.gods if g.is_appeased),
            "total_gods": len(self.gods),
        }

    def _update_animations(self):
        for p in self.platforms:
            p.update()
        self.clock.tick(self.metadata["render_fps"])

    def _move_selector(self, movement):
        if movement == 0: return # None
        # sfx: selector_move
        x, y = self.selector_pos
        if movement == 1: y -= 1 # Up
        elif movement == 2: y += 1 # Down
        elif movement == 3: x -= 1 # Left
        elif movement == 4: x += 1 # Right
        
        # Wrap around
        self.selector_pos[0] = x % self.GRID_SIZE[0]
        self.selector_pos[1] = y % self.GRID_SIZE[1]

    def _update_all_gods_appeasement(self):
        total = self.GRID_SIZE[0] * self.GRID_SIZE[1]
        for g in self.gods:
            g.update_appeasement(self.platforms, total)

    # --- RENDER METHODS ---
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_PATTERN, (i, 0), (i, self.screen_height), 1)
        for i in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLOR_BG_PATTERN, (0, i), (self.screen_width, i), 1)
    
    def _render_game_area(self):
        cell_w = self.GAME_AREA_RECT.width / self.GRID_SIZE[0]
        cell_h = self.GAME_AREA_RECT.height / self.GRID_SIZE[1]

        for p in self.platforms:
            px = self.GAME_AREA_RECT.left + p.grid_x * cell_w
            py = self.GAME_AREA_RECT.top + p.grid_y * cell_h
            
            # Update particle positions relative to the platform's center
            for particle in p.particles:
                particle.x = px + cell_w / 2 + particle.vx * (30 - particle.life) / 10
                particle.y = py + cell_h / 2 + particle.vy * (30 - particle.life) / 10
            
            p.draw(self.screen, px, py, cell_w, cell_h)
            
        self._render_selector(cell_w, cell_h)

    def _render_selector(self, cell_w, cell_h):
        sel_x, sel_y = self.selector_pos
        rect = pygame.Rect(
            self.GAME_AREA_RECT.left + sel_x * cell_w,
            self.GAME_AREA_RECT.top + sel_y * cell_h,
            cell_w,
            cell_h
        )
        # Glow effect
        for i in range(4, 0, -1):
            glow_rect = rect.inflate(i*2, i*2)
            alpha = 100 - i * 20
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_SELECTOR, alpha), s.get_rect(), border_radius=8)
            self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, 3, border_radius=8)

    def _render_ui(self):
        # Left UI Panel
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, 160, self.screen_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, (0, 0, 160, self.screen_height), 3)

        # Title
        title_surf = self.FONT_M.render("AZTEC'S FATE", True, self.COLOR_SUN)
        self.screen.blit(title_surf, (80 - title_surf.get_width()//2, 15))

        # God UI
        for i, god in enumerate(self.gods):
            self._render_god_ui(god, i)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.FONT_S.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, self.screen_height - 30))

        # Sun/Timer Bar
        self._render_sun()

    def _render_god_ui(self, god, index):
        y_offset = 60 + index * 90
        
        # God Name
        name_surf = self.FONT_S.render(god.name.upper(), True, god.color)
        self.screen.blit(name_surf, (10, y_offset))
        
        # Preferred Size Display
        pref_size_text = self.FONT_S.render("Prefers:", True, self.COLOR_UI_TEXT)
        self.screen.blit(pref_size_text, (10, y_offset + 20))
        
        size_box_w = 20
        size_box_h = 20
        size_ratio = (god.preferred_size - 1) / 4.0
        platform_w = int(size_box_w * (0.4 + size_ratio * 0.6))
        platform_h = int(size_box_h * (0.4 + size_ratio * 0.6))
        px = 80 + (size_box_w - platform_w) // 2
        py = y_offset + 20 + (size_box_h - platform_h) // 2
        pygame.draw.rect(self.screen, god.color, (px, py, platform_w, platform_h), border_radius=2)
        
        # Appeasement Meter
        meter_rect = pygame.Rect(10, y_offset + 45, 140, 20)
        pygame.draw.rect(self.screen, self.COLOR_METER_BG, meter_rect, border_radius=5)
        
        fill_width = (god.appeasement / 100) * meter_rect.width
        fill_rect = pygame.Rect(meter_rect.left, meter_rect.top, fill_width, meter_rect.height)
        
        color = self.COLOR_METER_GOOD if god.is_appeased else god.color
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, meter_rect, 2, border_radius=5)

    def _render_sun(self):
        sun_bar_rect = pygame.Rect(self.GAME_AREA_RECT.left, 10, self.GAME_AREA_RECT.width, 20)
        
        progress = self.game_turns / self.MAX_TURNS
        sun_x = sun_bar_rect.left + progress * sun_bar_rect.width
        sun_y = sun_bar_rect.centery
        
        # Sun glow
        for i in range(10, 0, -2):
            alpha = 150 - i * 12
            pygame.gfxdraw.filled_circle(self.screen, int(sun_x), int(sun_y), 6 + i, (*self.COLOR_SUN, alpha))
        pygame.gfxdraw.filled_circle(self.screen, int(sun_x), int(sun_y), 8, self.COLOR_SUN)
        pygame.gfxdraw.aacircle(self.screen, int(sun_x), int(sun_y), 8, self.COLOR_SUN)
        
        turns_left = self.MAX_TURNS - self.game_turns
        turns_text = self.FONT_S.render(f"TURNS: {turns_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(turns_text, (self.screen_width - turns_text.get_width() - 10, 12))

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
        self.screen.blit(overlay, (0, 0))
        
        message = "ALL GODS APPEASED" if self.win_condition else "THE SUN HAS SET"
        color = self.COLOR_METER_GOOD if self.win_condition else (200, 50, 50)
        
        text_surf = self.FONT_L.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        self.screen.blit(text_surf, text_rect)
        
        score_text = f"Final Score: {int(self.score)}"
        score_surf = self.FONT_M.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2 + 50))
        self.screen.blit(score_surf, score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage:
if __name__ == '__main__':
    # Set render_mode to "human" to see the game window
    # Set it to "rgb_array" for headless execution
    render_mode = "human"
    
    # The "dummy" video driver is for headless operation. 
    # If you want to see the window, you must unset it.
    if render_mode == "human":
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode=render_mode)
    
    # --- Manual Play Setup ---
    if env.render_mode == "human":
        pygame.display.set_caption("Aztec's Fate")
        live_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset(seed=42)
    done = False
    
    # Game loop for manual play
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # For human play, we need to handle events and keys
        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset(seed=random.randint(0, 10000))
                    
            # Handle continuous key presses for fluid movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            # For manual play, we need a small delay to make actions distinct
            # An RL agent would not need this.
            pygame.time.wait(50) 
        else: # For automated testing, just sample an action
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if env.render_mode == "human":
            # The observation is a numpy array, we need to convert it back to a surface
            # Transpose is needed because pygame and numpy have different axis orders
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            live_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
    env.close()