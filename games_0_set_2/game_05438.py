
# Generated: 2025-08-28T05:00:44.195142
# Source Brief: brief_05438.md
# Brief Index: 5438

        
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


class Crystal:
    def __init__(self, grid_x, grid_y, color_idx, tile_size, grid_offset):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.color_idx = color_idx
        self.pixel_x = grid_offset[0] + (grid_x + 0.5) * tile_size
        self.pixel_y = grid_offset[1] + (grid_y + 0.5) * tile_size
        self.target_pixel_x = self.pixel_x
        self.target_pixel_y = self.pixel_y
        self.is_dying = False
        self.death_timer = 0.0
        self.scale = 1.0

    def update(self, dt):
        # Animate movement
        self.pixel_x += (self.target_pixel_x - self.pixel_x) * 15 * dt
        self.pixel_y += (self.target_pixel_y - self.pixel_y) * 15 * dt

        # Animate death
        if self.is_dying:
            self.death_timer -= dt
            self.scale = max(0, self.death_timer / 0.25) # 0.25s death animation

class Particle:
    def __init__(self, x, y, color, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = 1.0

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 150 * dt  # Gravity
        self.lifetime -= dt
        self.size = max(0, self.size * self.lifetime)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to swap the selected crystal. Press space to select the next crystal, and shift to select the previous one."
    )

    game_description = (
        "A strategic puzzle game. Swap adjacent crystals to create lines of three or more of the same color. Clear all crystals before time runs out to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    NUM_CRYSTAL_TYPES = 5
    TIME_LIMIT_SECONDS = 60.0
    MAX_STEPS = TIME_LIMIT_SECONDS * 30 # Assuming 30 FPS

    # --- Colors ---
    COLOR_BG = (15, 10, 25)
    COLOR_GRID = (40, 30, 60)
    CRYSTAL_COLORS = [
        (255, 50, 50),   # Red
        (50, 255, 50),   # Green
        (80, 80, 255),   # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    CRYSTAL_HIGHLIGHTS = [pygame.Color(c).lerp((255,255,255), 0.6) for c in CRYSTAL_COLORS]
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_WARN = (255, 255, 0)
    COLOR_TIMER_CRIT = (255, 0, 0)

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
        self.font_small = pygame.font.SysFont("sans-serif", 18, bold=True)
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        self.tile_size = 36
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.tile_size) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.tile_size) // 2
        self.grid_offset = (self.grid_offset_x, self.grid_offset_y)

        self.game_state = "IDLE"
        self.animation_timer = 0.0
        self.swapping_crystals = []
        self.turn_reward = 0.0
        self.crystals = []
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.game_state = "AWAITING_INPUT"
        self.animation_timer = 0.0
        self.swapping_crystals = []
        self.particles = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_initial_grid()
        self.selected_crystal_idx = 0 if self.crystals else -1
        self.initial_crystal_count = len(self.crystals)

        return self._get_observation(), self._get_info()

    def _generate_initial_grid(self):
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Fill with random crystals, avoiding initial matches
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                possible_colors = list(range(1, self.NUM_CRYSTAL_TYPES + 1))
                # Check left
                if x > 1 and self.grid[x-1, y] == self.grid[x-2, y] and self.grid[x-1, y] in possible_colors:
                    possible_colors.remove(self.grid[x-1, y])
                # Check up
                if y > 1 and self.grid[x, y-1] == self.grid[x, y-2] and self.grid[x, y-1] in possible_colors:
                    possible_colors.remove(self.grid[x, y-1])
                
                self.grid[x, y] = self.np_random.choice(possible_colors) if possible_colors else 1

        self.crystals = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] > 0:
                    self.crystals.append(Crystal(x, y, self.grid[x, y] - 1, self.tile_size, self.grid_offset))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        dt = 1.0 / 30.0 # Fixed timestep for auto_advance
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - dt)
        reward = -0.01  # Small penalty for time passing

        # --- Update animations and state transitions ---
        self._update_animations_and_state(dt)

        # --- Process player input ---
        if self.game_state == "AWAITING_INPUT":
            input_reward = self._handle_input(action)
            reward += input_reward

        # --- Check for termination ---
        if not self.game_over:
            if self.time_remaining <= 0:
                self.game_over = True
                self.game_state = "GAME_OVER_LOSS"
                reward -= 50
            elif not self.crystals:
                self.game_over = True
                self.game_state = "GAME_OVER_WIN"
                reward += 50
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Handle selection
        if space_held and not self.prev_space_held and self.crystals:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystals)
        if shift_held and not self.prev_shift_held and self.crystals:
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + len(self.crystals)) % len(self.crystals)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Handle swap
        if movement != 0 and self.selected_crystal_idx != -1:
            c1 = self.crystals[self.selected_crystal_idx]
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            nx, ny = c1.grid_x + dx, c1.grid_y + dy

            if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                c2 = self._get_crystal_at_grid(nx, ny)
                if c2:
                    self.game_state = "SWAPPING"
                    self.animation_timer = 0.25 # Swap animation duration
                    self.swapping_crystals = [c1, c2]
                    self._swap_crystals(c1, c2)
        return reward

    def _update_animations_and_state(self, dt):
        if self.animation_timer > 0:
            self.animation_timer -= dt

        for c in self.crystals:
            c.update(dt)
        
        for p in self.particles[:]:
            p.update(dt)
            if p.lifetime <= 0:
                self.particles.remove(p)
        
        # --- State Machine Logic ---
        if self.game_state == "SWAPPING" and self.animation_timer <= 0:
            c1, c2 = self.swapping_crystals
            matches = self._find_matches_at([ (c1.grid_x, c1.grid_y), (c2.grid_x, c2.grid_y) ])
            if not matches:
                # Invalid move, swap back
                self.turn_reward = -1.0
                self.game_state = "SWAPPING_BACK"
                self.animation_timer = 0.25
                self._swap_crystals(c1, c2)
            else:
                self.turn_reward = 0.0
                self._start_clearing(matches)

        elif self.game_state == "SWAPPING_BACK" and self.animation_timer <= 0:
            self.score += self.turn_reward
            self.game_state = "AWAITING_INPUT"
            self.swapping_crystals = []

        elif self.game_state == "CLEARING" and self.animation_timer <= 0:
            self._apply_gravity()
            self.game_state = "FALLING"
            self.animation_timer = 0.3 # Falling animation

        elif self.game_state == "FALLING" and self.animation_timer <= 0:
            all_matches = self._find_all_matches()
            if all_matches:
                self.turn_reward += 5.0 # Cascade bonus
                self._start_clearing(all_matches)
            else:
                self.score += self.turn_reward
                self.game_state = "AWAITING_INPUT"

    def _start_clearing(self, matches):
        # sfx: match_clear.wav
        num_cleared = len(matches)
        self.turn_reward += num_cleared
        if num_cleared >= 4:
            self.turn_reward += 5.0 # Bonus for 4+

        for x, y in matches:
            crystal = self._get_crystal_at_grid(x, y)
            if crystal and not crystal.is_dying:
                crystal.is_dying = True
                crystal.death_timer = 0.25
                self.grid[x, y] = 0
                self._spawn_particles(crystal)
        
        self.game_state = "CLEARING"
        self.animation_timer = 0.25 # Death animation

    def _apply_gravity(self):
        # sfx: crystals_fall.wav
        self.crystals = [c for c in self.crystals if not c.is_dying]
        
        for x in range(self.GRID_WIDTH):
            empty_count = 0
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    crystal = self._get_crystal_at_grid(x, y)
                    if crystal:
                        self.grid[x, y + empty_count] = self.grid[x, y]
                        self.grid[x, y] = 0
                        crystal.grid_y += empty_count
                        crystal.target_pixel_y = self.grid_offset[1] + (crystal.grid_y + 0.5) * self.tile_size
        
        # Update selected index
        if self.crystals:
            self.selected_crystal_idx = min(self.selected_crystal_idx, len(self.crystals) - 1)
        else:
            self.selected_crystal_idx = -1

    def _find_matches_at(self, positions):
        all_matched = set()
        for x, y in positions:
            if self.grid[x, y] == 0: continue
            color = self.grid[x, y]
            
            # Horizontal
            h_line = [ (x, y) ]
            for i in range(x - 1, -1, -1):
                if self.grid[i, y] == color: h_line.append((i, y))
                else: break
            for i in range(x + 1, self.GRID_WIDTH):
                if self.grid[i, y] == color: h_line.append((i, y))
                else: break
            if len(h_line) >= 3: all_matched.update(h_line)

            # Vertical
            v_line = [ (x, y) ]
            for j in range(y - 1, -1, -1):
                if self.grid[x, j] == color: v_line.append((x, j))
                else: break
            for j in range(y + 1, self.GRID_HEIGHT):
                if self.grid[x, j] == color: v_line.append((x, j))
                else: break
            if len(v_line) >= 3: all_matched.update(v_line)
        return all_matched

    def _find_all_matches(self):
        all_pos = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        return self._find_matches_at(all_pos)

    def _get_crystal_at_grid(self, x, y):
        for c in self.crystals:
            if c.grid_x == x and c.grid_y == y and not c.is_dying:
                return c
        return None

    def _swap_crystals(self, c1, c2):
        # sfx: crystal_swap.wav
        self.grid[c1.grid_x, c1.grid_y], self.grid[c2.grid_x, c2.grid_y] = self.grid[c2.grid_x, c2.grid_y], self.grid[c1.grid_x, c1.grid_y]
        c1.grid_x, c2.grid_x = c2.grid_x, c1.grid_x
        c1.grid_y, c2.grid_y = c2.grid_y, c1.grid_y
        c1.target_pixel_x, c2.target_pixel_x = c2.target_pixel_x, c1.target_pixel_x
        c1.target_pixel_y, c2.target_pixel_y = c2.target_pixel_y, c1.target_pixel_y
        # Update selection if needed
        self.crystals.sort(key=lambda c: (c.grid_y, c.grid_x))
        try:
            self.selected_crystal_idx = self.crystals.index(c1 if c1 in self.swapping_crystals else c2)
        except ValueError: # If a selected crystal was cleared
            self.selected_crystal_idx = 0 if self.crystals else -1


    def _spawn_particles(self, crystal):
        color = self.CRYSTAL_COLORS[crystal.color_idx]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            size = random.uniform(3, 7)
            self.particles.append(Particle(crystal.pixel_x, crystal.pixel_y, color, size, angle, speed))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            start = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y)
            end = (self.grid_offset_x + x * self.tile_size, self.grid_offset_y + self.GRID_HEIGHT * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.grid_offset_x, self.grid_offset_y + y * self.tile_size)
            end = (self.grid_offset_x + self.GRID_WIDTH * self.tile_size, self.grid_offset_y + y * self.tile_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw crystals
        for i, c in enumerate(self.crystals):
            self._draw_crystal(c)
        
        # Draw selection highlight
        if self.selected_crystal_idx != -1 and self.game_state in ["AWAITING_INPUT", "SWAPPING", "SWAPPING_BACK"]:
            selected = self.crystals[self.selected_crystal_idx]
            size = int(self.tile_size * 0.8)
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 4
            rect = pygame.Rect(0, 0, size + pulse, size + pulse)
            rect.center = (int(selected.pixel_x), int(selected.pixel_y))
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=8)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, (int(p.x), int(p.y)), int(p.size))

    def _draw_crystal(self, crystal):
        size = int(self.tile_size * 0.8 * crystal.scale)
        if size <= 0: return
        
        pos_x, pos_y = int(crystal.pixel_x), int(crystal.pixel_y)
        color = self.CRYSTAL_COLORS[crystal.color_idx]
        highlight = self.CRYSTAL_HIGHLIGHTS[crystal.color_idx]
        
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (pos_x, pos_y)
        
        # Use gfxdraw for antialiasing
        pygame.gfxdraw.box(self.screen, rect, color)
        pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, size // 2, highlight)
        
        # Simple shine effect
        shine_rect = rect.copy()
        shine_rect.width = int(size * 0.8)
        shine_rect.height = int(size * 0.2)
        shine_rect.centerx = pos_x
        shine_rect.centery = pos_y - int(size * 0.2)
        pygame.draw.rect(self.screen, (255,255,255, 50), shine_rect, border_radius=4)


    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Crystals remaining
        crystals_left = len(self.crystals)
        crystals_text = self.font_small.render(f"CRYSTALS: {crystals_left}/{self.initial_crystal_count}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystals_text, (15, 40))

        # Timer
        time_color = self.COLOR_UI_TEXT
        if self.time_remaining < 10: time_color = self.COLOR_TIMER_CRIT
        elif self.time_remaining < 30: time_color = self.COLOR_TIMER_WARN
        time_str = f"{int(self.time_remaining // 60):02}:{int(self.time_remaining % 60):02}"
        time_text = self.font_small.render(f"TIME: {time_str}", True, time_color)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 15, 15))

        # Game Over message
        if self.game_state == "GAME_OVER_WIN":
            msg = self.font_large.render("YOU WIN!", True, (150, 255, 150))
            msg_rect = msg.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg, msg_rect)
        elif self.game_state == "GAME_OVER_LOSS":
            msg = self.font_large.render("TIME'S UP", True, (255, 150, 150))
            msg_rect = msg.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "crystals_left": len(self.crystals),
            "game_state": self.game_state
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    
    terminated = False
    clock = pygame.time.Clock()
    
    # --- Main Game Loop ---
    while not terminated:
        # --- Human Input to Action Mapping ---
        movement, space_held, shift_held = 0, 0, 0
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS
        
    env.close()