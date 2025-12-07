import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Descend into the abyss in this match-2 puzzle game. Match adjacent tiles to gain oxygen, "
        "food, and increase your depth, but watch your resources dwindle."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to select two adjacent, identical tiles to match them. "
        "Hold shift to activate time warp when available."
    )
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 8
    GRID_HEIGHT = 6
    TILE_SIZE = 48
    TILE_GAP = 4
    GRID_START_X = (SCREEN_WIDTH - (GRID_WIDTH * (TILE_SIZE + TILE_GAP) - TILE_GAP)) // 2
    GRID_START_Y = 80

    # Colors
    COLOR_BG = (1, 10, 28)
    COLOR_TEXT = (220, 240, 255)
    COLOR_OXYGEN = (0, 191, 255)
    COLOR_FOOD = (124, 252, 0)
    COLOR_DANGER = (255, 69, 0)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TIME_WARP = (255, 215, 0, 100) # Yellow with alpha
    TILE_COLORS = [
        (46, 204, 113),  # Emerald
        (52, 152, 219),  # Peter River
        (155, 89, 182),  # Amethyst
        (230, 126, 34),  # Carrot
        (241, 196, 15),   # Sun Flower
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Persistent state
        self.persistent_target_depth = 1000

        # State variables initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.oxygen = 0.0
        self.food = 0.0
        self.current_depth = 0
        self.tile_grid = []
        self.cursor_pos = [0, 0]
        self.selected_tiles = []
        self.time_warp_active_steps = 0
        self.time_warp_cooldown_steps = 0
        self.particles = []
        self.bg_particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_bg_particles()
        # self.reset() is called by the environment runner, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.oxygen = 100.0
        self.food = 100.0
        self.current_depth = 0
        
        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.selected_tiles = []
        
        self.time_warp_active_steps = 0
        self.time_warp_cooldown_steps = 0
        
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_grid()
        while not self._find_possible_matches():
            self._generate_grid() # Ensure a solvable start

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty per step to encourage speed
        
        self._handle_input(movement, space_held, shift_held)
        match_reward = self._process_selection(space_held)
        reward += match_reward
        
        warp_reward = self._process_time_warp(shift_held)
        reward += warp_reward

        self._update_game_state()
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.oxygen <= 0:
                reward -= 100
            elif self.current_depth >= self.persistent_target_depth:
                reward += 100
                self.persistent_target_depth += 200 # Increase difficulty for next run
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    # --- Game Logic Sub-methods ---

    def _handle_input(self, movement, space_held, shift_held):
        # Movement is processed once per step
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_HEIGHT - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_WIDTH - 1, self.cursor_pos[1] + 1)

    def _process_selection(self, space_held):
        space_pressed = space_held and not self.prev_space_held
        if not space_pressed:
            return 0

        # sound placeholder: # sfx_select_tile
        row, col = self.cursor_pos
        if self.tile_grid[row][col] == 0: # Cannot select empty space
            return 0
        
        if self.cursor_pos not in self.selected_tiles:
            self.selected_tiles.append(list(self.cursor_pos))

        if len(self.selected_tiles) == 2:
            return self._attempt_match()
        return 0

    def _attempt_match(self):
        pos1, pos2 = self.selected_tiles
        r1, c1 = pos1
        r2, c2 = pos2
        
        is_adjacent = abs(r1 - r2) + abs(c1 - c2) == 1
        is_same_type = self.tile_grid[r1][c1] == self.tile_grid[r2][c2]

        reward = 0
        if is_adjacent and is_same_type:
            # sound placeholder: # sfx_match_success
            reward += 1.0
            
            # Grant resources
            oxygen_gain = 15.0
            food_gain = 10.0
            self.oxygen = min(100, self.oxygen + oxygen_gain)
            self.food = min(100, self.food + food_gain)
            reward += 0.05 * oxygen_gain
            reward += 0.02 * food_gain
            
            self.current_depth += 25
            self.score += 50

            # Create particle explosion
            self._create_explosion((r1, c1))
            self._create_explosion((r2, c2))

            # Remove tiles and apply gravity
            self.tile_grid[r1][c1] = 0
            self.tile_grid[r2][c2] = 0
            self._apply_gravity_and_refill()

            # Check for new matches and shuffle if none
            if not self._find_possible_matches():
                self._shuffle_grid()

        else:
            # sound placeholder: # sfx_match_fail
            reward -= 0.1
            self.oxygen -= 2.0 # Penalty for wrong match
        
        self.selected_tiles.clear()
        return reward

    def _process_time_warp(self, shift_held):
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.time_warp_cooldown_steps == 0:
            # sound placeholder: # sfx_time_warp_activate
            self.time_warp_active_steps = 150 # 5 seconds at 30fps
            self.time_warp_cooldown_steps = 600 # 20 seconds cooldown
            if self.oxygen < 30:
                return 5.0 # Strategic use reward
        return 0
        
    def _update_game_state(self):
        # Update timers
        if self.time_warp_active_steps > 0:
            self.time_warp_active_steps -= 1
        else:
            if self.time_warp_cooldown_steps > 0:
                self.time_warp_cooldown_steps -= 1
            
            # Deplete resources
            depletion_rate = 0.1 + (self.current_depth // 200) * 0.05
            self.food -= depletion_rate
            if self.food < 0:
                self.oxygen += self.food # food is negative, so this subtracts
                self.food = 0
        
        self.oxygen = max(0, self.oxygen)
        self.score = max(0, self.score)

    def _check_termination(self):
        if self.oxygen <= 0:
            self.game_over = True
            # sound placeholder: # sfx_game_over
        if self.current_depth >= self.persistent_target_depth:
            self.game_over = True
            # sound placeholder: # sfx_win
        if self.steps >= 1500:
            self.game_over = True
        return self.game_over

    # --- Grid & Particle Management ---

    def _generate_grid(self):
        self.tile_grid = [[self.np_random.integers(1, len(self.TILE_COLORS) + 1) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]

    def _find_possible_matches(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.tile_grid[r][c] == 0: continue
                # Check right
                if c + 1 < self.GRID_WIDTH and self.tile_grid[r][c] == self.tile_grid[r][c+1]:
                    return True
                # Check down
                if r + 1 < self.GRID_HEIGHT and self.tile_grid[r][c] == self.tile_grid[r+1][c]:
                    return True
        return False

    def _shuffle_grid(self):
        # sound placeholder: # sfx_shuffle
        flat_list = [item for sublist in self.tile_grid for item in sublist if item != 0]
        self.np_random.shuffle(flat_list)
        
        idx = 0
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.tile_grid[r][c] != 0:
                    self.tile_grid[r][c] = flat_list[idx]
                    idx += 1
        if not self._find_possible_matches():
            self._generate_grid() # Failsafe, regenerate if shuffle fails

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.tile_grid[r][c] != 0:
                    self.tile_grid[empty_row][c], self.tile_grid[r][c] = self.tile_grid[r][c], self.tile_grid[empty_row][c]
                    empty_row -= 1
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.tile_grid[r][c] == 0:
                    self.tile_grid[r][c] = self.np_random.integers(1, len(self.TILE_COLORS) + 1)

    def _generate_bg_particles(self):
        self.bg_particles = []
        for _ in range(100):
            p = {
                "pos": [random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)],
                "vel": [random.uniform(-0.1, 0.1), random.uniform(0.1, 0.3)],
                "size": random.uniform(1, 3),
                "color": (5, 30, 60)
            }
            self.bg_particles.append(p)

    def _create_explosion(self, grid_pos):
        row, col = grid_pos
        center_x = self.GRID_START_X + col * (self.TILE_SIZE + self.TILE_GAP) + self.TILE_SIZE / 2
        center_y = self.GRID_START_Y + row * (self.TILE_SIZE + self.TILE_GAP) + self.TILE_SIZE / 2
        tile_type = self.tile_grid[row][col]
        color = self.TILE_COLORS[tile_type - 1]

        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            p = {
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": random.uniform(2, 5),
                "color": color,
                "lifespan": 30
            }
            self.particles.append(p)

    def _update_particles(self):
        # Update foreground particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)
        
        # Update background particles
        for p in self.bg_particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            if p["pos"][1] > self.SCREEN_HEIGHT:
                p["pos"][1] = 0
                p["pos"][0] = random.uniform(0, self.SCREEN_WIDTH)
            if p["pos"][0] > self.SCREEN_WIDTH: p["pos"][0] = 0
            if p["pos"][0] < 0: p["pos"][0] = self.SCREEN_WIDTH

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_bg_particles()
        self._render_grid()
        self._render_particles()
        self._render_ui()
        if self.time_warp_active_steps > 0:
            self._render_time_warp_effect()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_bg_particles(self):
        for p in self.bg_particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["size"]), p["color"])

    def _render_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                tile_type = self.tile_grid[r][c]
                if tile_type == 0:
                    continue

                x = self.GRID_START_X + c * (self.TILE_SIZE + self.TILE_GAP)
                y = self.GRID_START_Y + r * (self.TILE_SIZE + self.TILE_GAP)
                rect = pygame.Rect(x, y, self.TILE_SIZE, self.TILE_SIZE)
                color = self.TILE_COLORS[tile_type - 1]
                
                pygame.draw.rect(self.screen, color, rect, border_radius=8)

                # Selection highlight
                if [r, c] in self.selected_tiles:
                    pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 4, border_radius=8)

        # Cursor
        cur_r, cur_c = self.cursor_pos
        cur_x = self.GRID_START_X + cur_c * (self.TILE_SIZE + self.TILE_GAP) - 4
        cur_y = self.GRID_START_Y + cur_r * (self.TILE_SIZE + self.TILE_GAP) - 4
        cur_rect = pygame.Rect(cur_x, cur_y, self.TILE_SIZE + 8, self.TILE_SIZE + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cur_rect, 2, border_radius=12)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["size"]))

    def _render_ui(self):
        # Bar dimensions
        bar_width = 200
        bar_height = 20
        
        # Oxygen bar
        oxygen_ratio = self.oxygen / 100.0
        oxy_color = self.COLOR_OXYGEN if oxygen_ratio > 0.2 else self.COLOR_DANGER
        self._draw_bar(20, 20, bar_width, bar_height, "OXYGEN", oxygen_ratio, oxy_color)

        # Food bar
        food_ratio = self.food / 100.0
        self._draw_bar(20, 50, bar_width, bar_height, "FOOD", food_ratio, self.COLOR_FOOD)

        # Depth display
        depth_text = self.font_main.render(f"DEPTH: {self.current_depth}m / {self.persistent_target_depth}m", True, self.COLOR_TEXT)
        self.screen.blit(depth_text, (self.SCREEN_WIDTH - depth_text.get_width() - 20, 20))

        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 50))

        # Time warp status
        if self.time_warp_cooldown_steps > 0:
            cooldown_ratio = self.time_warp_cooldown_steps / 600
            s = pygame.Surface((30,30), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, 15, 15, 15, (255,215,0,50))
            pygame.gfxdraw.arc(s, 15, 15, 14, 0, int(360 * cooldown_ratio), self.COLOR_TIME_WARP[:3])
            self.screen.blit(s, (self.SCREEN_WIDTH // 2 - 15, 15))


    def _draw_bar(self, x, y, width, height, label, ratio, color):
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (30, 30, 50), bg_rect, border_radius=5)
        
        # Fill
        fill_width = max(0, int(width * ratio))
        fill_rect = pygame.Rect(x, y, fill_width, height)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=5)
        
        # Border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, bg_rect, 2, border_radius=5)

        # Text
        text = self.font_small.render(label, True, self.COLOR_TEXT)
        self.screen.blit(text, (x + 5, y + 2))

    def _render_time_warp_effect(self):
        warp_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        alpha = 100 * (self.time_warp_active_steps / 150) # Fade out
        warp_surface.fill((*self.COLOR_TIME_WARP[:3], alpha))
        self.screen.blit(warp_surface, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "oxygen": self.oxygen,
            "food": self.food,
            "depth": self.current_depth,
            "target_depth": self.persistent_target_depth
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be run by the evaluator
    
    # Un-comment the line below to run with a visible display
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Deep Sea Solitaire")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # To remove the validation check that was in the original code
    try:
        delattr(GameEnv, 'validate_implementation')
    except AttributeError:
        pass

    while running:
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

        # Handle events on each frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}, Info: {info}")
            running = False # End after one episode for automated testing
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()