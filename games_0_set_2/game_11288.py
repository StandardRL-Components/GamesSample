import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:45:23.222938
# Source Brief: brief_01288.md
# Brief Index: 1288
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player, controlling a submarine, must clear a
    field of colored debris by matching tiles. The goal is to clear the entire
    field before a descending storm timer runs out. Players must also avoid
    triggering mines hidden beneath the debris. The game rewards strategic,
    efficient clearing and penalizes inaction and mistakes.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Clear a field of colored debris by matching tiles, but avoid hidden mines before the storm timer runs out."
    )
    user_guide = (
        "Use the arrow keys to move the selector. Press space to clear a group of matching tiles and shift to slow down time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 11
    NUM_COLORS = 3
    MAX_STEPS = 5000
    GROUP_CLEAR_THRESHOLD = 3
    LARGE_GROUP_BONUS_THRESHOLD = 5

    # --- Colors ---
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (20, 40, 80)
    COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
    ]
    COLOR_DEBRIS = (100, 100, 110, 200)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_MINE_BODY = (40, 40, 40)
    COLOR_MINE_WARNING = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_SHADOW = (20, 20, 30)
    COLOR_UI_STORM_BAR = (200, 40, 40)
    COLOR_UI_STORM_BG = (50, 50, 70)
    COLOR_TIME_SLOW_ACTIVE = (100, 200, 255)
    COLOR_TIME_SLOW_COOLDOWN = (70, 90, 120)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # --- Game State Initialization ---
        self.tile_grid = []
        self.selector_pos = [0, 0]
        self.visual_selector_pos = [0.0, 0.0]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.storm_timer = 1.0
        self.base_storm_speed = 0.0005
        self.storm_speed_increase_rate = 0.00001
        self.time_slow_duration = 0
        self.time_slow_cooldown = 0
        self.particles = []
        self.difficulty_level = 1.0

        if render_mode == "human":
            pygame.display.set_caption("Debris Clear")
            self.human_screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.reset()
        # self.validate_implementation() # Commented out for submission, useful for local dev

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over and self._count_debris() == 0: # Won previous round
             self.difficulty_level = min(2.5, self.difficulty_level + 0.05)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.storm_timer = 1.0
        self.time_slow_duration = 0
        self.time_slow_cooldown = 0
        self.particles = []
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self._generate_level()
        
        grid_pixel_width = self.tile_size * self.GRID_SIZE
        grid_pixel_height = self.tile_size * self.GRID_SIZE
        offset_x = (self.SCREEN_WIDTH - grid_pixel_width) / 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_height) / 2
        
        self.visual_selector_pos = [
            offset_x + self.selector_pos[0] * self.tile_size,
            offset_y + self.selector_pos[1] * self.tile_size
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = -0.1  # Cost of living

        # --- Update Game State ---
        self._update_time_slow()
        self._update_storm()

        # --- Handle Actions ---
        reward += self._handle_movement(movement)
        reward += self._handle_time_slow(shift_held)
        reward += self._handle_clear_action(space_held)
        
        # --- Update Visuals ---
        self._update_particles()
        
        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self._count_debris() == 0:
                reward += 100 # Victory
                # SFX: Victory fanfare
            else:
                reward -= 100 # Failure
                # SFX: Failure sound
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Core Logic Sub-functions ---

    def _generate_level(self):
        self.tile_size = (self.SCREEN_HEIGHT * 0.8) / self.GRID_SIZE
        num_debris = int((self.GRID_SIZE ** 2) * min(0.9, 0.4 * self.difficulty_level))
        num_mines = int(num_debris * 0.1 * self.difficulty_level)

        self.tile_grid = [[{
            "color": self.np_random.integers(0, self.NUM_COLORS),
            "has_debris": False,
            "is_mine": False,
            "is_revealed_mine": False,
            "animation_scale": 1.0
        } for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        possible_debris_locs = list(np.ndindex((self.GRID_SIZE, self.GRID_SIZE)))
        self.np_random.shuffle(possible_debris_locs)
        
        debris_locs = possible_debris_locs[:num_debris]
        for r, c in debris_locs:
            self.tile_grid[r][c]["has_debris"] = True
        
        possible_mine_locs = [loc for loc in debris_locs if loc != (self.GRID_SIZE // 2, self.GRID_SIZE // 2)]
        self.np_random.shuffle(possible_mine_locs)
        mine_locs = possible_mine_locs[:num_mines]
        for r, c in mine_locs:
            self.tile_grid[r][c]["is_mine"] = True

    def _update_time_slow(self):
        if self.time_slow_cooldown > 0:
            self.time_slow_cooldown -= 1
        if self.time_slow_duration > 0:
            self.time_slow_duration -= 1

    def _update_storm(self):
        current_storm_speed = self.base_storm_speed + (self.steps // 500) * self.storm_speed_increase_rate
        if self.time_slow_duration > 0:
            current_storm_speed *= 0.5
        self.storm_timer = max(0, self.storm_timer - current_storm_speed)

    def _handle_movement(self, movement):
        if movement == 1: self.selector_pos[1] -= 1  # Up
        elif movement == 2: self.selector_pos[1] += 1  # Down
        elif movement == 3: self.selector_pos[0] -= 1  # Left
        elif movement == 4: self.selector_pos[0] += 1  # Right
        
        self.selector_pos[0] %= self.GRID_SIZE
        self.selector_pos[1] %= self.GRID_SIZE
        return 0

    def _handle_time_slow(self, shift_held):
        if shift_held and self.time_slow_cooldown == 0 and self.time_slow_duration == 0:
            self.time_slow_duration = 200 # Approx 6.6 seconds at 30fps
            self.time_slow_cooldown = 600 # Approx 20 seconds
            # SFX: Time slow activate
            lifespan = 40
            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                self.particles.append({
                    "pos": [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2],
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "color": self.COLOR_TIME_SLOW_ACTIVE, "size": random.uniform(2, 5),
                    "lifespan": lifespan, "start_lifespan": lifespan, "type": "glow"
                })
            return 0.1 # Small reward for using ability
        return 0

    def _handle_clear_action(self, space_held):
        if not space_held:
            return 0
        
        r, c = self.selector_pos[1], self.selector_pos[0]
        if not self.tile_grid[r][c]["has_debris"]:
            return 0 # Cannot clear an already clear tile

        # SFX: Select tile
        group = self._find_tile_group(c, r)
        if len(group) < self.GROUP_CLEAR_THRESHOLD:
            # SFX: Invalid move buzz
            return -0.2 # Penalty for failed attempt
        
        # Check for mine detonation
        for gr, gc in group:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                    if self.tile_grid[nr][nc]["is_revealed_mine"]:
                        self.game_over = True
                        # SFX: Mine explosion
                        self._create_explosion(gc, gr, self.COLOR_MINE_WARNING, 50)
                        return 0 # Terminal reward handled in main step loop

        # If safe, process clear and rewards
        reward = 0
        is_large_group = len(group) >= self.LARGE_GROUP_BONUS_THRESHOLD
        if is_large_group:
            reward += 10
            # SFX: Large group clear
        
        reward += len(group) # +1 per tile
        self.score += len(group) * (10 if is_large_group else 5)
        
        mined_adj_cleared = False
        for gr, gc in group:
            self.tile_grid[gr][gc]["has_debris"] = False
            self.tile_grid[gr][gc]["animation_scale"] = 0.0
            if self.tile_grid[gr][gc]["is_mine"]:
                self.tile_grid[gr][gc]["is_revealed_mine"] = True
                # SFX: Mine revealed
            
            # Check for bonus reward
            if not mined_adj_cleared:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                        if self.tile_grid[nr][nc]["is_revealed_mine"]:
                            mined_adj_cleared = True
                            break
            
            self._create_explosion(gc, gr, self.COLORS[self.tile_grid[gr][gc]["color"]], 15)
        
        if mined_adj_cleared:
            reward += 5
            
        return reward

    def _check_termination(self):
        if self.game_over: return True
        if self.steps >= self.MAX_STEPS: return True
        if self.storm_timer <= 0: return True
        if self._count_debris() == 0: return True
        
        # Check for no valid moves
        if not self._check_for_valid_moves():
            return True
        
        return False

    def _check_for_valid_moves(self):
        visited_for_check = set()
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.tile_grid[r][c]["has_debris"] and (r,c) not in visited_for_check:
                    group = self._find_tile_group(c,r)
                    for pos in group: visited_for_check.add(pos)
                    
                    if len(group) >= self.GROUP_CLEAR_THRESHOLD:
                        # Check if this move is lethal
                        is_lethal = False
                        for gr, gc in group:
                            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nr, nc = gr + dr, gc + dc
                                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                                    if self.tile_grid[nr][nc]["is_revealed_mine"]:
                                        is_lethal = True
                                        break
                            if is_lethal: break
                        if not is_lethal:
                            return True # Found a valid, non-lethal move
        return False

    def _find_tile_group(self, start_c, start_r):
        if not self.tile_grid[start_r][start_c]["has_debris"]:
            return []
            
        target_color = self.tile_grid[start_r][start_c]["color"]
        q = [(start_r, start_c)]
        visited = set([(start_r, start_c)])
        group = []

        while q:
            r, c = q.pop(0)
            group.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and (nr, nc) not in visited:
                    if self.tile_grid[nr][nc]["has_debris"] and self.tile_grid[nr][nc]["color"] == target_color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return group
    
    def _count_debris(self):
        return sum(row.count(True) for row in [[tile["has_debris"] for tile in r] for r in self.tile_grid])

    # --- Rendering and Observation ---

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # --- Submarine in center (visual only) ---
        sub_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        pygame.gfxdraw.filled_ellipse(self.screen, int(sub_pos[0]), int(sub_pos[1]), 20, 12, (255, 200, 50))
        pygame.gfxdraw.ellipse(self.screen, int(sub_pos[0]), int(sub_pos[1]), 20, 12, (255, 255, 150))
        pygame.gfxdraw.filled_circle(self.screen, int(sub_pos[0] + 10), int(sub_pos[1]), 5, (150, 220, 255))
        
        # --- Grid and Tiles ---
        grid_pixel_width = self.tile_size * self.GRID_SIZE
        grid_pixel_height = self.tile_size * self.GRID_SIZE
        offset_x = (self.SCREEN_WIDTH - grid_pixel_width) / 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_height) / 2
        
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                tile = self.tile_grid[r][c]
                tile_x = offset_x + c * self.tile_size
                tile_y = offset_y + r * self.tile_size
                
                # Animate tile clearing
                if tile["animation_scale"] < 1.0:
                    tile["animation_scale"] += 0.1
                
                # Base tile color
                radius = int(self.tile_size * 0.4)
                center_x = int(tile_x + self.tile_size / 2)
                center_y = int(tile_y + self.tile_size / 2)
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLORS[tile["color"]])
                pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLORS[tile["color"]])

                # Debris overlay
                if tile["has_debris"]:
                    s = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                    scale = (2 - tile["animation_scale"]) / 2 # Pop in effect
                    size = int(self.tile_size * 0.9 * scale)
                    rect = pygame.Rect((self.tile_size - size)/2, (self.tile_size - size)/2, size, size)
                    pygame.draw.rect(s, self.COLOR_DEBRIS, rect, border_radius=int(self.tile_size * 0.1))
                    self.screen.blit(s, (tile_x, tile_y))

                # Revealed mine
                if tile["is_revealed_mine"]:
                    glow_size = int(self.tile_size * 0.3 + 3 * math.sin(self.steps * 0.2))
                    glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, glow_size, (*self.COLOR_MINE_WARNING, glow_alpha))
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.tile_size * 0.3), self.COLOR_MINE_BODY)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, int(self.tile_size * 0.15), self.COLOR_MINE_WARNING)
        
        # --- Selector ---
        target_selector_x = offset_x + self.selector_pos[0] * self.tile_size
        target_selector_y = offset_y + self.selector_pos[1] * self.tile_size
        self.visual_selector_pos[0] = self.visual_selector_pos[0] * 0.6 + target_selector_x * 0.4
        self.visual_selector_pos[1] = self.visual_selector_pos[1] * 0.6 + target_selector_y * 0.4
        
        sel_rect = pygame.Rect(self.visual_selector_pos[0], self.visual_selector_pos[1], self.tile_size, self.tile_size)
        glow_size = int(self.tile_size + 8 + 4 * math.sin(self.steps * 0.15))
        glow_rect = pygame.Rect(0,0, glow_size, glow_size)
        glow_rect.center = sel_rect.center
        
        s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_SELECTOR, 60), s.get_rect(), border_radius=int(self.tile_size * 0.2))
        self.screen.blit(s, glow_rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, sel_rect, 3, border_radius=int(self.tile_size * 0.2))

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.05 # Gravity/buoyancy
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["start_lifespan"]))
            color = (*p["color"], alpha) if len(p["color"]) == 3 else (*p["color"][:3], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            if p["type"] == "spark":
                pygame.draw.circle(self.screen, color, pos, int(p["size"] * (p["lifespan"] / p["start_lifespan"])))
            elif p["type"] == "glow":
                 pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["size"]), color)


    def _create_explosion(self, c, r, color, count):
        grid_pixel_width = self.tile_size * self.GRID_SIZE
        grid_pixel_height = self.tile_size * self.GRID_SIZE
        offset_x = (self.SCREEN_WIDTH - grid_pixel_width) / 2
        offset_y = (self.SCREEN_HEIGHT - grid_pixel_height) / 2
        
        center_x = offset_x + c * self.tile_size + self.tile_size / 2
        center_y = offset_y + r * self.tile_size + self.tile_size / 2
        
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 3)
            lifespan = random.randint(20, 50)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "color": color, "size": random.uniform(1, 4),
                "lifespan": lifespan, "start_lifespan": lifespan, "type": "spark"
            })

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_UI_TEXT)
        shadow = self.font_large.render(f"{self.score:06d}", True, self.COLOR_UI_SHADOW)
        self.screen.blit(shadow, (12, self.SCREEN_HEIGHT - 42))
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - 40))

        # --- Storm Timer Bar ---
        bar_width = self.SCREEN_WIDTH * 0.6
        bar_height = 15
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_UI_STORM_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_STORM_BAR, (bar_x, bar_y, bar_width * self.storm_timer, bar_height), border_radius=4)
        
        # --- Time Slow Indicator ---
        slow_ind_pos = (self.SCREEN_WIDTH - 40, self.SCREEN_HEIGHT - 40)
        if self.time_slow_duration > 0:
            glow_size = int(20 + 5 * math.sin(self.steps * 0.3))
            pygame.gfxdraw.filled_circle(self.screen, slow_ind_pos[0], slow_ind_pos[1], glow_size, (*self.COLOR_TIME_SLOW_ACTIVE, 50))
            pygame.gfxdraw.filled_circle(self.screen, slow_ind_pos[0], slow_ind_pos[1], 15, self.COLOR_TIME_SLOW_ACTIVE)
        else:
            pygame.gfxdraw.filled_circle(self.screen, slow_ind_pos[0], slow_ind_pos[1], 15, self.COLOR_TIME_SLOW_COOLDOWN)
            if self.time_slow_cooldown > 0:
                angle = 360 * (self.time_slow_cooldown / 600)
                rect = pygame.Rect(slow_ind_pos[0]-15, slow_ind_pos[1]-15, 30, 30)
                pygame.draw.arc(self.screen, self.COLOR_TIME_SLOW_ACTIVE, rect, math.radians(90), math.radians(90 + angle), 3)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "storm_timer": self.storm_timer,
            "debris_remaining": self._count_debris(),
            "difficulty": self.difficulty_level
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # --- Example Usage & Human Play ---
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # To control with keyboard:
    # Arrow keys for movement, Space to clear, Left Shift to slow time
    
    while not done:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # --- Render for Human ---
        if "human" in env.metadata["render_modes"]:
            # The observation is already the rendered screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            env.human_screen.blit(surf, (0, 0))
            pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()