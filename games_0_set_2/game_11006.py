import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


# Ensure Pygame runs in a headless mode for server-side execution.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw

class GameEnv(gym.Env):
    """
    Gymnasium environment where the agent plays as a fractal deity.
    The goal is to grow a fractal structure by managing Faith and Energy resources,
    applying different growth patterns to achieve ultimate complexity.
    """
    # --- Start of required attributes ---
    game_description = "Grow a celestial fractal by managing Faith and Energy, unlocking complex patterns to achieve ultimate complexity."
    user_guide = "Controls: Use ↑↓←→ arrow keys to select a growth pattern. Hold Space to apply it to the oldest part of the fractal, or Shift to apply it to a random point."
    auto_advance = True
    # --- End of required attributes ---

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    VICTORY_COMPLEXITY = 10000

    # Colors
    COLOR_BG_START = (5, 0, 15)
    COLOR_BG_END = (20, 0, 40)
    COLOR_FAITH = (50, 150, 255)
    COLOR_ENERGY = (50, 255, 150)
    COLOR_COMPLEXITY = (200, 100, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (20, 20, 50, 180) # RGBA
    COLOR_UI_HIGHLIGHT = (255, 255, 100)

    # Rewards
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0
    REWARD_PER_UNLOCK = 10.0
    REWARD_PER_10_COMPLEXITY = 0.1
    REWARD_PER_RESOURCE_TICK = 0.005

    # Game Mechanics
    RESOURCE_REGEN_RATE = 0.5
    COST_INCREASE_PER_1K_COMPLEXITY = 0.05
    INITIAL_FAITH = 100
    INITIAL_ENERGY = 100
    MAX_RESOURCES = 200

    # Pattern Definitions
    PATTERNS = [
        {"name": "Spike", "cost_faith": 5, "cost_energy": 10, "rules": [(0, 0.9)]},
        {"name": "Fork", "cost_faith": 15, "cost_energy": 15, "rules": [(-25, 0.7), (25, 0.7)]},
        {"name": "Trident", "cost_faith": 30, "cost_energy": 25, "rules": [(-40, 0.6), (0, 0.6), (40, 0.6)]},
        {"name": "Crystal", "cost_faith": 40, "cost_energy": 50, "rules": [(-90, 0.4), (90, 0.4), (0, 0.8)]},
        {"name": "Bloom", "cost_faith": 75, "cost_energy": 75, "rules": [(-60, 0.5), (0, 0.5), (60, 0.5), (120, 0.5), (180, 0.5), (240, 0.5)]}
    ]
    UNLOCK_THRESHOLDS = {100: 2, 500: 3, 1500: 4}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self._create_starfield()
        self._create_background_surface()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.faith = float(self.INITIAL_FAITH)
        self.energy = float(self.INITIAL_ENERGY)
        
        initial_length = 80.0
        start_pos = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT * 0.8], dtype=np.float64)
        end_pos = start_pos - np.array([0.0, initial_length], dtype=np.float64)
        self.fractal_segments = [(start_pos, end_pos, -90.0, 0, initial_length)]
        self.complexity = 1

        self.unlocked_patterns = [0, 1]
        self.selected_pattern_idx = 0
        self.last_movement_action = 0

        self.particles = []
        self.camera_zoom = 1.0
        self.camera_offset = np.array([0.0, 0.0], dtype=np.float64)
        self.last_reward_info = "Game Start"

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        self.steps += 1

        self.faith = min(self.MAX_RESOURCES, self.faith + self.RESOURCE_REGEN_RATE)
        self.energy = min(self.MAX_RESOURCES, self.energy + self.RESOURCE_REGEN_RATE)
        reward += 2 * self.REWARD_PER_RESOURCE_TICK

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self._handle_pattern_selection(movement)
        
        action_taken, growth_reward = False, 0.0
        if space_pressed or shift_pressed:
            action_taken, growth_reward = self._handle_pattern_application(space_pressed, shift_pressed)
            reward += growth_reward

        unlock_reward = self._update_unlocks()
        reward += unlock_reward
        self._update_camera()
        self._update_particles()
        
        if action_taken and unlock_reward > 0:
            self.last_reward_info = f"Pattern Unlocked! +{unlock_reward:.1f}"
        elif action_taken:
            self.last_reward_info = "Fractal Growth"
        else:
            self.last_reward_info = "Waiting..."

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.complexity >= self.VICTORY_COMPLEXITY:
                reward += self.REWARD_WIN
                self.last_reward_info = f"VICTORY! +{self.REWARD_WIN:.0f}"
            elif not self._can_afford_any_pattern():
                 reward += self.REWARD_LOSE
                 self.last_reward_info = f"STAGNATION! {self.REWARD_LOSE:.0f}"
            else:
                 self.last_reward_info = "Time Limit Reached"
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_pattern_selection(self, movement):
        if movement != 0 and movement != self.last_movement_action:
            num_unlocked = len(self.unlocked_patterns)
            if num_unlocked > 0:
                if movement in [1, 3]:  # Up or Left
                    self.selected_pattern_idx = (self.selected_pattern_idx - 1 + num_unlocked) % num_unlocked
                elif movement in [2, 4]:  # Down or Right
                    self.selected_pattern_idx = (self.selected_pattern_idx + 1) % num_unlocked
        self.last_movement_action = movement

    def _handle_pattern_application(self, space_pressed, shift_pressed):
        pattern_id = self.unlocked_patterns[self.selected_pattern_idx]
        pattern = self.PATTERNS[pattern_id]
        
        cost_multiplier = 1 + (self.COST_INCREASE_PER_1K_COMPLEXITY * (self.complexity // 1000))
        faith_cost = pattern['cost_faith'] * cost_multiplier
        energy_cost = pattern['cost_energy'] * cost_multiplier

        if self.faith < faith_cost or self.energy < energy_cost:
            return False, 0

        growth_points = [s[1:] for s in self.fractal_segments]
        if not growth_points: return False, 0

        target_point = None
        if space_pressed:
            target_point = growth_points[0]
        elif shift_pressed:
            idx = self.np_random.integers(len(growth_points))
            target_point = growth_points[idx]
        
        if target_point:
            self.faith -= faith_cost
            self.energy -= energy_cost
            
            old_complexity = self.complexity
            self._apply_pattern(pattern, target_point)
            complexity_gain = self.complexity - old_complexity
            
            growth_reward = (complexity_gain // 10) * self.REWARD_PER_10_COMPLEXITY
            return True, growth_reward
        
        return False, 0

    def _apply_pattern(self, pattern, growth_point):
        pos, angle, generation, length = growth_point
        
        for angle_offset, length_multiplier in pattern['rules']:
            new_angle_deg = angle + angle_offset
            new_angle_rad = math.radians(new_angle_deg)
            new_length = max(2.0, length * length_multiplier)
            
            new_end_pos = pos + np.array([math.cos(new_angle_rad), math.sin(new_angle_rad)], dtype=np.float64) * new_length
            
            new_segment = (pos.copy(), new_end_pos, new_angle_deg, generation + 1, new_length)
            self.fractal_segments.append(new_segment)
        
        self.complexity = len(self.fractal_segments)
        self._create_particles(pos, 20, self.COLOR_FAITH if pattern['cost_faith'] > pattern['cost_energy'] else self.COLOR_ENERGY)

    def _update_unlocks(self):
        reward = 0.0
        for threshold, pattern_idx in self.UNLOCK_THRESHOLDS.items():
            if self.complexity >= threshold and pattern_idx not in self.unlocked_patterns:
                self.unlocked_patterns.append(pattern_idx)
                reward += self.REWARD_PER_UNLOCK
        return reward

    def _check_termination(self):
        return self.complexity >= self.VICTORY_COMPLEXITY or self.steps >= self.MAX_STEPS or not self._can_afford_any_pattern()

    def _can_afford_any_pattern(self):
        cost_multiplier = 1 + (self.COST_INCREASE_PER_1K_COMPLEXITY * (self.complexity // 1000))
        for pattern_id in self.unlocked_patterns:
            pattern = self.PATTERNS[pattern_id]
            if self.faith >= pattern['cost_faith'] * cost_multiplier and self.energy >= pattern['cost_energy'] * cost_multiplier:
                return True
        return False

    def _get_observation(self):
        self.screen.blit(self.background_surface, (0, 0))
        self._render_stars()
        self._render_fractal()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "complexity": self.complexity, "faith": self.faith, "energy": self.energy, "unlocked_patterns": len(self.unlocked_patterns)}

    def _create_starfield(self):
        self.stars = [{"pos": [random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)], "size": random.uniform(0.5, 1.5), "brightness": random.randint(50, 150)} for _ in range(150)]

    def _create_background_surface(self):
        self.background_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = tuple(int(self.COLOR_BG_START[i] * (1 - interp) + self.COLOR_BG_END[i] * interp) for i in range(3))
            pygame.draw.line(self.background_surface, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_stars(self):
        for star in self.stars:
            star['pos'][0] = (star['pos'][0] - 0.05 * star['size']) % self.SCREEN_WIDTH
            c = star['brightness']
            pygame.gfxdraw.pixel(self.screen, int(star['pos'][0]), int(star['pos'][1]), (c, c, c, c))

    def _update_camera(self):
        if not self.fractal_segments: return
        
        all_points = np.array([p for seg in self.fractal_segments for p in seg[:2]])
        min_coords, max_coords = np.min(all_points, axis=0), np.max(all_points, axis=0)
        fractal_size = max_coords - min_coords
        
        if np.any(fractal_size < 1e-6): return

        padding = 100
        zoom_x = (self.SCREEN_WIDTH - padding) / fractal_size[0]
        zoom_y = (self.SCREEN_HEIGHT - padding) / fractal_size[1]
        target_zoom = min(zoom_x, zoom_y, 2.0)

        self.camera_zoom = self.camera_zoom * 0.95 + target_zoom * 0.05
        
        fractal_center = min_coords + fractal_size / 2.0
        screen_center = np.array([self.SCREEN_WIDTH / 2.0, self.SCREEN_HEIGHT / 2.0])
        target_offset = screen_center - fractal_center * self.camera_zoom
        
        self.camera_offset = self.camera_offset * 0.95 + target_offset * 0.05

    def _world_to_screen(self, pos):
        return pos * self.camera_zoom + self.camera_offset

    def _render_fractal(self):
        max_gen = max((s[3] for s in self.fractal_segments), default=1)
        
        for start_pos, end_pos, _, generation, _ in self.fractal_segments:
            p1, p2 = self._world_to_screen(start_pos), self._world_to_screen(end_pos)
            
            gen_ratio = min(1.0, generation / max(1, max_gen * 0.8))
            color = tuple(int(255 * (1 - gen_ratio) + self.COLOR_COMPLEXITY[i] * gen_ratio) for i in range(3))
            
            try:
                pygame.draw.aaline(self.screen, color, p1, p2)
            except (ValueError, TypeError):
                pass

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle, speed = random.uniform(0, 2 * math.pi), random.uniform(1, 4)
            self.particles.append({"pos": pos.copy(), "vel": np.array([math.cos(angle), math.sin(angle)]) * speed, "lifespan": random.randint(15, 30), "color": color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1

    def _render_particles(self):
        for p in self.particles:
            screen_pos = self._world_to_screen(p['pos'])
            alpha = max(0, 255 * (p['lifespan'] / 30))
            radius = max(1, p['lifespan'] / 10)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(screen_pos[0]), int(screen_pos[1]), int(radius), (*p['color'], int(alpha)))
            except (ValueError, TypeError):
                pass
            
    def _render_ui(self):
        bar_w, bar_h, bar_pad = 150, 15, 5
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, bar_w + 4, bar_h + 4))
        pygame.draw.rect(self.screen, self.COLOR_FAITH, (12, 12, int((self.faith / self.MAX_RESOURCES) * bar_w), bar_h))
        self.screen.blit(self.font_small.render("Faith", True, self.COLOR_UI_TEXT), (15, 12))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10 + bar_h + bar_pad, bar_w + 4, bar_h + 4))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (12, 12 + bar_h + bar_pad, int((self.energy / self.MAX_RESOURCES) * bar_w), bar_h))
        self.screen.blit(self.font_small.render("Energy", True, self.COLOR_UI_TEXT), (15, 12 + bar_h + bar_pad))

        score_text = self.font_medium.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 15, 10))
        comp_text = self.font_medium.render(f"COMPLEXITY: {self.complexity}", True, self.COLOR_COMPLEXITY)
        self.screen.blit(comp_text, (self.SCREEN_WIDTH - comp_text.get_width() - 15, 35))

        cost_multiplier = 1 + (self.COST_INCREASE_PER_1K_COMPLEXITY * (self.complexity // 1000))
        menu_h = 25 * len(self.unlocked_patterns) + 10
        menu_w = 200
        menu_surf = pygame.Surface((menu_w, menu_h), pygame.SRCALPHA)
        menu_surf.fill(self.COLOR_UI_BG)

        for i, pattern_id in enumerate(self.unlocked_patterns):
            pattern, y_pos = self.PATTERNS[pattern_id], 5 + i * 25
            color = self.COLOR_UI_HIGHLIGHT if i == self.selected_pattern_idx else self.COLOR_UI_TEXT
            menu_surf.blit(self.font_medium.render(f"{pattern['name']}", True, color), (10, y_pos))
            f_cost, e_cost = int(pattern['cost_faith'] * cost_multiplier), int(pattern['cost_energy'] * cost_multiplier)
            f_color = self.COLOR_FAITH if self.faith >= f_cost else (100, 100, 120)
            e_color = self.COLOR_ENERGY if self.energy >= e_cost else (100, 100, 120)
            menu_surf.blit(self.font_small.render(f"{f_cost}", True, f_color), (menu_w - 70, y_pos + 2))
            menu_surf.blit(self.font_small.render(f"{e_cost}", True, e_color), (menu_w - 35, y_pos + 2))

        self.screen.blit(menu_surf, (self.SCREEN_WIDTH - menu_w - 10, self.SCREEN_HEIGHT - menu_h - 10))

        status_text = self.font_medium.render(self.last_reward_info, True, self.COLOR_UI_TEXT)
        self.screen.blit(status_text, (self.SCREEN_WIDTH // 2 - status_text.get_width() // 2, self.SCREEN_HEIGHT - 30))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will use a graphical display, overriding the headless setting.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Fractal God")
    clock = pygame.time.Clock()
    
    running = True
    
    movement_action, space_held, shift_held = 0, 0, 0

    print("\n--- Controls ---")
    print(GameEnv.user_guide)
    print("R: Reset Environment | Q: Quit")
    print("----------------\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: obs, info = env.reset()
                if event.key == pygame.K_UP: movement_action = 1
                elif event.key == pygame.K_DOWN: movement_action = 2
                elif event.key == pygame.K_LEFT: movement_action = 3
                elif event.key == pygame.K_RIGHT: movement_action = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift_held = 1
        
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]: movement_action = 0
                if event.key == pygame.K_SPACE: space_held = 0
                if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift_held = 0

        action = [movement_action, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode Finished! Final Score: {info['score']:.2f}, Complexity: {info['complexity']}")
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()