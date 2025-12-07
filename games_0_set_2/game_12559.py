import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:04:13.219235
# Source Brief: brief_02559.md
# Brief Index: 2559
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent collects memory fragments across time.
    The agent must learn to recognize patterns, avoid corrupted fragments, and
    switch between time periods (Past, Present, Future) to collect a full
    set of 10 memories.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Collect memory fragments scattered across time. Switch between Past, Present, and Future "
        "to find the correct patterns, but beware of corrupted data."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to select a fragment. Press space to collect and shift to switch time periods."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30

    # Colors
    COLOR_BG_PAST = (10, 20, 40)
    COLOR_BG_PRESENT = (15, 40, 30)
    COLOR_BG_FUTURE = (30, 20, 40)
    TIME_PERIOD_COLORS = [COLOR_BG_PAST, COLOR_BG_PRESENT, COLOR_BG_FUTURE]
    TIME_PERIOD_NAMES = ["PAST", "PRESENT", "FUTURE"]

    COLOR_GRID = (50, 60, 80)
    COLOR_SAFE = (0, 255, 200)
    COLOR_CORRUPTED = (255, 50, 100)
    COLOR_GHOST = (100, 120, 140, 100)
    COLOR_SELECT = (255, 180, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BAR_BG = (40, 50, 70)
    COLOR_UI_BAR_FILL = (0, 200, 150)
    
    # Game parameters
    MAX_STEPS = 2000
    COLLECTION_GOAL = 10
    FRAGMENT_COUNT = 3

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
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        self.font_warning = pygame.font.Font(None, 18)

        # --- Internal State ---
        # These are initialized properly in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.time_period = 0  # 0: Past, 1: Present, 2: Future
        self.fragments = []
        self.collected_fragments = set()
        self.selected_fragment_idx = 0
        
        # Difficulty scaling
        self.total_fragments_collected = 0
        self.collections_completed = 0

        # Animation state
        self.particles = []
        self.current_bg_color = self.TIME_PERIOD_COLORS[0]
        self.selector_pos = [0, 0]
        self.last_action_time = {}

        # self.reset() is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.time_period = self.np_random.integers(0, 3)
        self.collected_fragments = set()
        self.selected_fragment_idx = 1 # Start in the middle
        
        # Reset difficulty if it's a full env reset
        if options is None or not options.get("soft_reset", False):
            self.total_fragments_collected = 0
            self.collections_completed = 0

        self.current_bg_color = self.TIME_PERIOD_COLORS[self.time_period]
        self._generate_fragments()
        
        # Initialize selector position
        target_x, target_y = self._get_fragment_screen_pos(self.selected_fragment_idx)
        self.selector_pos = [target_x, target_y]

        self.particles = []
        self.last_action_time = {'select': 0, 'switch': 0, 'collect': 0}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        
        # 1. Unnecessary Time Switch Penalty (checked before switch)
        selected_frag = self.fragments[self.selected_fragment_idx]
        if shift_held and selected_frag['time_period'] == self.time_period and not selected_frag['is_corrupted']:
            reward -= 1.0 # Penalize switching when the selected fragment is already collectible.

        # 2. Movement (Selection Change)
        # We add a small cooldown to prevent flickering selection from a held key
        if movement in [3, 4] and self.steps > self.last_action_time.get('select', 0) + 5:
            if movement == 3: # Left
                self.selected_fragment_idx = (self.selected_fragment_idx - 1) % self.FRAGMENT_COUNT
            elif movement == 4: # Right
                self.selected_fragment_idx = (self.selected_fragment_idx + 1) % self.FRAGMENT_COUNT
            self.last_action_time['select'] = self.steps
            # sfx: select_ui

        # 3. Time Period Switch (Shift)
        if shift_held and self.steps > self.last_action_time.get('switch', 0) + 15:
            self.time_period = (self.time_period + 1) % 3
            self.last_action_time['switch'] = self.steps
            self._create_time_switch_particles()
            # sfx: time_switch
        
        # 4. Collection (Space)
        if space_held and self.steps > self.last_action_time.get('collect', 0) + 15:
            self.last_action_time['collect'] = self.steps
            selected_frag = self.fragments[self.selected_fragment_idx]

            if selected_frag['time_period'] == self.time_period:
                if selected_frag['is_corrupted']:
                    # sfx: corrupt_error
                    reward -= 50.0
                    self.game_over = True
                    self._create_corruption_burst_particles(selected_frag['pos'])
                else:
                    # sfx: collect_success
                    reward += 1.0
                    self.total_fragments_collected += 1
                    
                    if selected_frag['shape'] not in self.collected_fragments:
                        self.collected_fragments.add(selected_frag['shape'])
                    
                    if len(self.collected_fragments) >= self.COLLECTION_GOAL:
                        reward += 50.0
                        self.game_over = True
                        self.collections_completed += 1
                        # sfx: collection_complete
                    else:
                        self._generate_fragments()
            else:
                # sfx: collect_fail_wrong_time
                reward -= 0.1 # Small penalty for trying to collect in wrong time period

        # --- Update Game State ---
        self._update_animations()

        # --- Check Termination ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collection_progress": len(self.collected_fragments),
            "time_period": self.TIME_PERIOD_NAMES[self.time_period],
        }

    def _render_all(self):
        # Background
        self.screen.fill(self.current_bg_color)
        self._draw_grid()

        # Particles (drawn behind fragments)
        self._draw_particles()

        # Game elements
        self._draw_fragments()
        self._draw_selector()

        # UI
        self._draw_ui()

    def _update_animations(self):
        # BG color interpolation
        target_bg = self.TIME_PERIOD_COLORS[self.time_period]
        self.current_bg_color = tuple(
            int(c + (t - c) * 0.1) for c, t in zip(self.current_bg_color, target_bg)
        )

        # Selector interpolation
        target_x, target_y = self._get_fragment_screen_pos(self.selected_fragment_idx)
        self.selector_pos[0] += (target_x - self.selector_pos[0]) * 0.25
        self.selector_pos[1] += (target_y - self.selector_pos[1]) * 0.25
        
        # Particle physics
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            p['vel'][1] += p.get('gravity', 0) # Apply gravity if present

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)

    def _draw_fragments(self):
        for i, frag in enumerate(self.fragments):
            is_ghost = frag['time_period'] != self.time_period
            pulse = math.sin(self.steps * 0.1 + i * 2) * 2
            size = 20 + pulse
            
            color = self.COLOR_CORRUPTED if frag['is_corrupted'] else self.COLOR_SAFE
            if is_ghost:
                color = self.COLOR_GHOST

            x, y = frag['pos']
            
            # Glow effect
            for i in range(4, 0, -1):
                glow_alpha = 30 if not is_ghost else 15
                glow_color = (*color[:3], glow_alpha) if len(color) == 4 else (*color, glow_alpha)
                self._draw_shape(frag['shape'], x, y, size + i * 3, glow_color, filled=True)

            # Main shape
            self._draw_shape(frag['shape'], x, y, size, color, filled=True)
            
            # Corruption flicker
            if frag['is_corrupted'] and not is_ghost and self.np_random.random() < 0.3:
                flicker_color = (255, 150, 150)
                self._draw_shape(frag['shape'], x, y, size + 3, flicker_color, filled=False, width=2)
                if self.np_random.random() < 0.1:
                    self._create_corruption_drip_particles(frag['pos'])

    def _draw_shape(self, shape_type, x, y, size, color, filled=False, width=1):
        x, y = int(x), int(y)
        size = int(size)
        if shape_type == 0: # Circle
            if filled:
                pygame.gfxdraw.filled_circle(self.screen, x, y, size, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, size, color)
        elif shape_type == 1: # Square
            points = [
                (x - size, y - size), (x + size, y - size),
                (x + size, y + size), (x - size, y + size)
            ]
            if filled:
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)
        elif shape_type == 2: # Triangle
            points = [
                (x, y - size * 1.1),
                (x - size, y + size * 0.7),
                (x + size, y + size * 0.7),
            ]
            if filled:
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _draw_selector(self):
        x, y = self.selector_pos
        size = 35
        # Draw multiple layers for a soft glow
        for i in range(5, 0, -1):
            alpha = 80 - i * 15
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(size + i*2), (*self.COLOR_SELECT, alpha))
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), size, self.COLOR_SELECT)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), size-1, self.COLOR_SELECT)

    def _draw_ui(self):
        # Collection Progress Bar
        bar_w, bar_h = 300, 20
        bar_x, bar_y = (self.SCREEN_WIDTH - bar_w) / 2, 20
        progress = len(self.collected_fragments) / self.COLLECTION_GOAL
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, bar_w * progress, bar_h), border_radius=5)
        
        # Time Period Text
        time_text_surf = self.font_large.render(self.TIME_PERIOD_NAMES[self.time_period], True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text_surf, (self.SCREEN_WIDTH - time_text_surf.get_width() - 20, 15))
        
        # Score and Steps Text
        score_text = f"Score: {self.score:.1f}"
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_UI_TEXT)
        steps_surf = self.font_main.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 20))
        self.screen.blit(steps_surf, (20, self.SCREEN_HEIGHT - 30))

        # Corruption warning
        selected_frag = self.fragments[self.selected_fragment_idx]
        if selected_frag['is_corrupted'] and selected_frag['time_period'] == self.time_period:
            warn_surf = self.font_warning.render("CORRUPTION DETECTED", True, self.COLOR_CORRUPTED)
            x, y = selected_frag['pos']
            self.screen.blit(warn_surf, (x - warn_surf.get_width() / 2, y + 40))

    def _generate_fragments(self):
        self.fragments.clear()
        
        # Difficulty scaling
        corruption_chance = min(0.9, 0.1 + (self.total_fragments_collected // 50) * 0.005)
        pattern_complexity = min(3, 1 + self.collections_completed)

        # Generate shapes based on complexity
        available_shapes = list(range(self.COLLECTION_GOAL))
        shapes = []
        if pattern_complexity == 1:
            shape = self.np_random.choice(available_shapes)
            shapes = [shape] * 3
        elif pattern_complexity == 2:
            s1, s2 = self.np_random.choice(available_shapes, 2, replace=False)
            base = [s1, s1, s2]
            self.np_random.shuffle(base)
            shapes = base
        else: # pattern_complexity == 3
            shapes = self.np_random.choice(available_shapes, 3, replace=False).tolist()

        for i in range(self.FRAGMENT_COUNT):
            x, y = self._get_fragment_screen_pos(i)
            self.fragments.append({
                'shape': shapes[i],
                'is_corrupted': self.np_random.random() < corruption_chance,
                'time_period': self.np_random.integers(0, 3),
                'pos': (x, y)
            })
        
        # Anti-softlock: Ensure at least one fragment is safe and accessible
        safe_indices = [i for i, f in enumerate(self.fragments) if not f['is_corrupted']]
        if not safe_indices:
            # If all are corrupted, make one safe
            idx_to_make_safe = self.np_random.integers(0, self.FRAGMENT_COUNT)
            self.fragments[idx_to_make_safe]['is_corrupted'] = False
            safe_indices = [idx_to_make_safe]

        # Ensure one of the safe fragments is in the current time period
        accessible_safe_frags = [i for i in safe_indices if self.fragments[i]['time_period'] == self.time_period]
        if not accessible_safe_frags:
            # If no safe fragment is in the current time, move one there
            frag_to_make_accessible = self.np_random.choice(safe_indices)
            self.fragments[frag_to_make_accessible]['time_period'] = self.time_period

    def _get_fragment_screen_pos(self, index):
        x = self.SCREEN_WIDTH * (index + 1) / (self.FRAGMENT_COUNT + 1)
        y = self.SCREEN_HEIGHT / 2
        return x, y

    def _draw_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'][:3], alpha)
            # Use a surface for alpha blending
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))


    def _create_time_switch_particles(self):
        color = self.TIME_PERIOD_COLORS[self.time_period]
        for _ in range(50):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.integers(1, 4)
            })
    
    def _create_corruption_drip_particles(self, pos):
        if len(self.particles) > 200: return # Limit particles
        lifespan = self.np_random.integers(15, 30)
        self.particles.append({
            'pos': [pos[0] + self.np_random.uniform(-5, 5), pos[1] + 20],
            'vel': [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(0.5, 1.5)],
            'lifespan': lifespan, 'max_lifespan': lifespan,
            'color': self.COLOR_CORRUPTED,
            'size': self.np_random.integers(2, 5),
            'gravity': 0.1
        })
        
    def _create_corruption_burst_particles(self, pos):
        for _ in range(100):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 5 + 2
            lifespan = self.np_random.integers(30, 60)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan, 'max_lifespan': lifespan,
                'color': self.COLOR_CORRUPTED,
                'size': self.np_random.integers(2, 5)
            })

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is not used by the evaluation system, which runs the environment headlessly.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for human play
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Memory Fragment Collector")
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- Action mapping from keyboard ---
        movement, space, shift = 0, 0, 0
        
        # This is a manual override for human play.
        # The agent will use the full action space.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- ENV RESET ---")
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {terminated or truncated}")

        if terminated or truncated:
            print(f"--- EPISODE FINISHED ---")
            print(f"Final Score: {info['score']:.2f}, Fragments: {info['collection_progress']}/{GameEnv.COLLECTION_GOAL}")
            obs, info = env.reset()

        # --- Render to screen ---
        # The observation is (H, W, C), but pygame needs (W, H) surface
        # and surfarray.make_surface expects (W, H, C)
        obs_swapped = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(obs_swapped)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.TARGET_FPS)

    env.close()