import gymnasium as gym
import os
import pygame
import math
import random
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player, a time-manipulating entity,
    navigates a grid of volcanic vents. The goal is to reach the core
    of a dormant volcano by teleporting between vents while avoiding
    overflowing lava. The player can accelerate or decelerate the
    lava flow cycles to create safe paths.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a grid of volcanic vents by teleporting between safe zones. "
        "Manipulate time to control lava flows and reach the volcano's core."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to teleport between vents. "
        "Hold space to accelerate time and shift to decelerate time."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 7
    GRID_HEIGHT = 5
    MAX_STEPS = 2000

    # Colors
    COLOR_BG = (15, 10, 25)
    COLOR_MOUNTAIN_1 = (25, 20, 40)
    COLOR_MOUNTAIN_2 = (35, 30, 50)
    COLOR_PLAYER = (0, 191, 255) # Bright Cyan
    COLOR_PLAYER_GLOW = (0, 100, 155)
    COLOR_LAVA = (255, 69, 0)
    COLOR_LAVA_GLOW = (200, 50, 0)
    COLOR_VENT = (70, 70, 80)
    COLOR_VENT_SAFE = (60, 220, 120) # Green
    COLOR_CORE = (148, 0, 211) # Violet
    COLOR_CORE_GLOW = (100, 0, 160)
    COLOR_TEXT = (240, 240, 240)

    # Time manipulation auras
    AURA_ACCEL = (255, 165, 0, 50) # Orange
    AURA_DECEL = (30, 144, 255, 50) # Dodger Blue

    # Game physics
    VENT_SPACING = 120
    VENT_RADIUS = 40
    SAFE_LAVA_LEVEL = 0.85
    INITIAL_ERUPTION_SPEED = 0.03
    ERUPTION_SPEED_INCREASE = 0.005 # More noticeable increase

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 36, bold=True)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = (0, 0)
        self.core_pos = (0, 0)
        self.vents = []
        self.particles = []
        self.game_time = 0.0
        self.eruption_speed = self.INITIAL_ERUPTION_SPEED
        self.time_manipulation_state = 0 # 0: normal, 1: accel, 2: decel
        self.last_time_manipulation_state = 0
        self.initial_dist_to_core = 1

        self._background_surface = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_time = 0.0
        self.eruption_speed = self.INITIAL_ERUPTION_SPEED
        self.time_manipulation_state = 0
        self.last_time_manipulation_state = 0
        self.particles.clear()

        # Pre-render background for performance (needs seeded RNG)
        self._background_surface = self._create_background()

        # --- World Generation ---
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1)
        self.core_pos = (self.GRID_WIDTH // 2, 0)
        self.initial_dist_to_core = self._manhattan_distance(self.player_pos, self.core_pos)

        self.vents = []
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                # Ensure start and end points have gentle cycles
                if (x, y) == self.player_pos or (x, y) == self.core_pos:
                    # Start at the bottom of the cycle for maximum initial safety
                    phase = 3 * math.pi / 2
                else:
                    phase = self.np_random.uniform(0, 2 * math.pi)

                self.vents.append({
                    "grid_pos": (x, y),
                    "phase": phase,
                    "lava_level": 0.0,
                    "is_safe": True
                })

        self._update_game_state(np.array([0, 0, 0])) # Initial update

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = self._update_game_state(action)

        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, action):
        """Main logic update function."""
        reward = 0.01  # Small survival reward

        # --- 1. Process Time Manipulation ---
        space_held = action[1] == 1
        shift_held = action[2] == 1

        if space_held:
            self.time_manipulation_state = 1 # Accelerate
            time_factor = 2.5
        elif shift_held:
            self.time_manipulation_state = 2 # Decelerate
            time_factor = 0.4
        else:
            self.time_manipulation_state = 0 # Normal
            time_factor = 1.0

        if self.time_manipulation_state != self.last_time_manipulation_state:
            reward -= 0.1 # Penalty for switching states
        self.last_time_manipulation_state = self.time_manipulation_state

        # --- 2. Update Game Clock & Difficulty ---
        self.game_time += time_factor
        if self.steps > 0 and self.steps % 200 == 0:
            self.eruption_speed += self.ERUPTION_SPEED_INCREASE

        # --- 3. Update Vent Lava Levels ---
        for vent in self.vents:
            # Sine wave for smooth rise and fall
            vent["lava_level"] = (math.sin(self.game_time * self.eruption_speed + vent["phase"]) + 1) / 2
            vent["is_safe"] = vent["lava_level"] < self.SAFE_LAVA_LEVEL

            # Spawn particles on eruption peak
            if vent["lava_level"] > 0.98 and math.cos(self.game_time * self.eruption_speed + vent["phase"]) < 0:
                 self._spawn_eruption_particles(vent["grid_pos"])

        # --- 4. Check for Failure in Current Vent ---
        current_vent = self._get_vent_at(self.player_pos)
        if not current_vent["is_safe"]:
            self.game_over = True
            reward -= 50.0
            return reward

        # --- 5. Process Movement ---
        movement = action[0]
        if movement != 0: # 0 is no-op
            old_dist = self._manhattan_distance(self.player_pos, self.core_pos)
            target_pos = list(self.player_pos)
            if movement == 1: target_pos[1] -= 1 # Up
            elif movement == 2: target_pos[1] += 1 # Down
            elif movement == 3: target_pos[0] -= 1 # Left
            elif movement == 4: target_pos[0] += 1 # Right

            target_pos = tuple(target_pos)
            target_vent = self._get_vent_at(target_pos)

            if target_vent:
                if target_vent["is_safe"]:
                    self.player_pos = target_pos
                    new_dist = self._manhattan_distance(self.player_pos, self.core_pos)
                    reward += 5.0 * (old_dist - new_dist) # Reward for getting closer
                else:
                    self.game_over = True
                    reward -= 50.0
                    return reward

        # --- 6. Check for Victory ---
        if self.player_pos == self.core_pos:
            self.game_over = True
            reward += 100.0

        return reward

    def _get_vent_at(self, grid_pos):
        """Finds a vent at a given grid coordinate."""
        for vent in self.vents:
            if vent["grid_pos"] == grid_pos:
                return vent
        return None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_observation(self):
        # --- 1. Clear Screen ---
        self.screen.blit(self._background_surface, (0, 0))

        # --- 2. Update & Render Particles ---
        self._update_and_draw_particles()

        # --- 3. Render Game Elements ---
        self._render_game_world()

        # --- 4. Render UI ---
        self._render_ui()

        # Convert to numpy array (H, W, C)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_background(self):
        """Creates a static background surface with mountain silhouettes."""
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg.fill(self.COLOR_BG)
        # Draw two layers of mountains
        self._draw_mountain_range(bg, self.COLOR_MOUNTAIN_2, 300, 150, 8)
        self._draw_mountain_range(bg, self.COLOR_MOUNTAIN_1, 400, 100, 6)
        return bg

    def _draw_mountain_range(self, surface, color, height, variation, num_peaks):
        points = [(0, self.SCREEN_HEIGHT)]
        for i in range(num_peaks + 1):
            x = self.SCREEN_WIDTH * i / num_peaks
            y = self.SCREEN_HEIGHT - height + self.np_random.integers(-variation, variation, endpoint=True)
            points.append((x, y))
        points.append((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_game_world(self):
        """Renders vents, lava, and the player."""
        screen_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

        # Camera zoom effect
        current_dist = self._manhattan_distance(self.player_pos, self.core_pos)
        dist_ratio = max(0, current_dist / self.initial_dist_to_core) if self.initial_dist_to_core > 0 else 0
        zoom = 1.0 - 0.4 * (1.0 - dist_ratio) # Zooms out as player gets closer

        scaled_spacing = int(self.VENT_SPACING * zoom)
        scaled_radius = int(self.VENT_RADIUS * zoom)

        # Render vents
        for vent in self.vents:
            dx = vent["grid_pos"][0] - self.player_pos[0]
            dy = vent["grid_pos"][1] - self.player_pos[1]
            pos_x = screen_center[0] + dx * scaled_spacing
            pos_y = screen_center[1] + dy * scaled_spacing

            # Cull off-screen vents
            if not (-scaled_radius < pos_x < self.SCREEN_WIDTH + scaled_radius and \
                    -scaled_radius < pos_y < self.SCREEN_HEIGHT + scaled_radius):
                continue

            is_core = vent["grid_pos"] == self.core_pos
            self._draw_vent(self.screen, (pos_x, pos_y), scaled_radius, vent["lava_level"], vent["is_safe"], is_core)

        # Render player
        self._draw_player(self.screen, screen_center, int(25 * zoom))

    def _draw_vent(self, surface, pos, radius, lava_level, is_safe, is_core):
        """Draws a single vent with lava and glow effects."""
        pos = (int(pos[0]), int(pos[1]))
        
        base_color = self.COLOR_CORE if is_core else self.COLOR_VENT
        glow_color = self.COLOR_CORE_GLOW if is_core else self.COLOR_LAVA_GLOW
        
        if lava_level > 0.1 and not is_core:
            glow_radius = int(radius * (1 + lava_level * 0.4))
            self._draw_glow_circle(surface, pos, glow_radius, glow_color)

        pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius, base_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, base_color)
        
        if lava_level > 0:
            lava_radius = int(max(1, radius * lava_level))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], lava_radius, self.COLOR_LAVA)

        if is_safe and not is_core:
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], radius + 2, self.COLOR_VENT_SAFE)

    def _draw_player(self, surface, pos, size):
        """Draws the player avatar and time manipulation aura."""
        pos = (int(pos[0]), int(pos[1]))

        if self.time_manipulation_state == 1: # Accelerate
            self._draw_glow_circle(surface, pos, int(size * 2.5), self.AURA_ACCEL)
        elif self.time_manipulation_state == 2: # Decelerate
            self._draw_glow_circle(surface, pos, int(size * 2.5), self.AURA_DECEL)
        
        self._draw_glow_circle(surface, pos, int(size * 1.5), self.COLOR_PLAYER_GLOW)

        points = [
            (pos[0], pos[1] - size),
            (pos[0] + size // 2, pos[1]),
            (pos[0], pos[1] + size),
            (pos[0] - size // 2, pos[1]),
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(surface, points, self.COLOR_PLAYER)

    def _draw_glow_circle(self, surface, pos, radius, color):
        """Renders a soft glowing circle."""
        temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        center = (radius, radius)
        
        for i in range(radius, 0, -2):
            alpha = int(color[3] * (1 - (i / radius))**2) if len(color) == 4 else int(128 * (1 - (i / radius))**2)
            
            if alpha > 0:
                draw_color = (color[0], color[1], color[2], alpha)
                pygame.gfxdraw.filled_circle(temp_surface, center[0], center[1], i, draw_color)
        
        surface.blit(temp_surface, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _spawn_eruption_particles(self, grid_pos):
        """Spawns particles for a lava eruption."""
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "grid_pos": grid_pos,
                "offset": [0, 0],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.uniform(20, 40),
                "color": self.np_random.choice([self.COLOR_LAVA, (255,140,0), (255,215,0)])
            })

    def _update_and_draw_particles(self):
        """Updates particle physics and renders them."""
        screen_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        current_dist = self._manhattan_distance(self.player_pos, self.core_pos)
        dist_ratio = max(0, current_dist / self.initial_dist_to_core) if self.initial_dist_to_core > 0 else 0
        zoom = 1.0 - 0.4 * (1.0 - dist_ratio)
        scaled_spacing = int(self.VENT_SPACING * zoom)
        
        for p in self.particles[:]:
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            p["offset"][0] += p["vel"][0]
            p["offset"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity

            dx = p["grid_pos"][0] - self.player_pos[0]
            dy = p["grid_pos"][1] - self.player_pos[1]
            
            base_x = screen_center[0] + dx * scaled_spacing
            base_y = screen_center[1] + dy * scaled_spacing
            
            pos_x = int(base_x + p["offset"][0])
            pos_y = int(base_y + p["offset"][1])
            
            size = max(1, int(p["life"] / 10))
            pygame.draw.circle(self.screen, p["color"], (pos_x, pos_y), size)

    def _render_ui(self):
        """Renders score, steps, and game over messages."""
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))
        
        if self.game_over:
            if self.player_pos == self.core_pos:
                msg = "CORE REACHED"
                color = self.COLOR_VENT_SAFE
            else:
                msg = "ABSORBED BY LAVA"
                color = self.COLOR_LAVA
            
            end_text = self.font_msg.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "dist_to_core": self._manhattan_distance(self.player_pos, self.core_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test observation space
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8

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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)

# Example usage:
if __name__ == '__main__':
    # This block will not run in the test environment, but is useful for manual play.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    total_reward = 0
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Volcanic Vent Navigator")
    clock = pygame.time.Clock()
    
    while running:
        action = np.array([0, 0, 0]) # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
                truncated = False

        if not (terminated or truncated):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    env.close()
    print(f"Game finished. Final score: {total_reward:.2f}")